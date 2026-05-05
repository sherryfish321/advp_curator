from typing import Optional, List, Tuple, Dict
import re
import ast
import os
import fitz
from copy import deepcopy
import requests
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import CrossEncoder
from llama_cpp import Llama, LogitsProcessorList
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

# NOTE: define config
# Chroma DB params
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "")
CHROMA_DB_COLLECTION_NAME = os.environ.get("CHROMA_DB_COLLECTION_NAME", "")
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', ""))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", ""))

# Retrieval params
INFINITY_URL = f"http://localhost:{os.environ.get('INFINITY_URL_PORT', 0)}"
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "")
TOP_K = int(os.environ.get("TOP_K", 20))
TOP_K_RERANK = int(os.environ.get("TOP_K_RERANK", 5))
SIMILARITY_SCORE_THRESHOLD = float(os.environ.get('SIMILARITY_SCORE_THRESHOLD', 0))

# Generation params
LLAMA_CLIENT = OpenAI(base_url=f"http://localhost:{os.environ.get('LLAMA_URL_PORT', 0)}/v1", api_key="none")
MAX_NEW_TOKEN = int(os.environ.get("MAX_NEW_TOKEN", 32))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0))
TOP_P = float(os.environ.get("TOP_P", 1))

# params for getting choice
DETAIL_CHOICE_SIMILARITY_SCORE_THRESHOLD = float(os.environ.get("DETAIL_CHOICE_SIMILARITY_SCORE_THRESHOLD", 0.6))

class InfinityEmbeddings(Embeddings):
    def __init__(self, model: str, url: str = INFINITY_URL):
        self.model = model
        self.url = url

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(f"{self.url}/embeddings", json={
            "input": texts,
            "model": self.model
        })
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# NOTE: need to move this function to a separate file later
def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove tabs and newlines
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Normalise unicode minus/dash variants to ASCII hyphen before stripping
    text = text.replace('−', '-').replace('–', '-').replace('—', '-')
    # Remove special characters; keep punctuation needed for numeric/scientific values
    text = re.sub(r'[^a-z0-9\s\.\,\-\+\%\(\)\<\>\=\:]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_document(document: Document) -> Document:
    cleaned_text = clean_text(document.page_content)
    cleaned_document = Document(page_content=cleaned_text, metadata=document.metadata)
    return cleaned_document

def ingest_doc_from_pmc(pmid: int, pmcid: str,
                        chroma_db_path: str = CHROMA_DB_PATH, chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME, 
                        chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, print_progress: bool = False):
    documents = []
    metadata = []
    try:
        curr_doc = ""
        url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for d in data:
                doc = d["documents"]
                for p in doc:
                    passage = p["passages"]
                    for item in passage:
                        if "text" in item:
                            curr_doc += "\n\n" + item["text"]
        documents.append(curr_doc)
        metadata.append({"PMID": str(pmid), "PMCID": pmcid})
    except Exception as e:
        raise Exception(f"Failed to extract paper {pmid}_{pmcid} from PMC with error {e}")
    if print_progress:
        print(f"Finished loading {len(documents)} documents.")
        print()

    # split documents into chunks
    if print_progress:
        print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    splitted_documents = text_splitter.create_documents(texts=documents, metadatas=metadata)
    splitted_documents = [clean_document(doc) for doc in splitted_documents]
    if print_progress:
        print(f"Finished splitting to make {len(splitted_documents)} chunks.")
        print()

    # create Chroma vector store
    if print_progress:
        print("Creating Chroma vector store...")
    chroma_db = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=InfinityEmbeddings(model="NeuML/pubmedbert-base-embeddings"),
        collection_name=chroma_db_collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    # delete current collection contents before adding new documents
    collection = chroma_db._collection
    all_docs = collection.get(include=[])
    all_ids = all_docs["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    # add documents to Chroma vector store
    chroma_db.add_documents(splitted_documents)
    if print_progress:
        print("Finished creating Chroma vector store.")
        print()

def ingest_doc_from_pdf(pmid: int, filename: str,
                        chroma_db_path: str = CHROMA_DB_PATH, chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME, 
                        chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, print_progress: bool = False):
    documents = []
    metadata = []
    try:
        with fitz.open(f"{filename}") as doc:
            text = ""
            for page in doc:
                text += page.get_text() + "\n\n" # special indicator of pages
            documents.append(text)
            metadata.append({"PMID": str(pmid)})
    except Exception as e:
        raise Exception(f"Failed to extract paper {pmid} from {filename} with error {e}")
    if print_progress:
        print(f"Finished loading {len(documents)} documents.")
        print()

    # split documents into chunks
    if print_progress:
        print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    splitted_documents = text_splitter.create_documents(texts=documents, metadatas=metadata)
    splitted_documents = [clean_document(doc) for doc in splitted_documents]
    if print_progress:
        print(f"Finished splitting to make {len(splitted_documents)} chunks.")
        print()

    # create Chroma vector store
    if print_progress:
        print("Creating Chroma vector store...")
    chroma_db = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=InfinityEmbeddings(model="NeuML/pubmedbert-base-embeddings"),
        collection_name=chroma_db_collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    # delete current collection contents before adding new documents
    collection = chroma_db._collection
    all_docs = collection.get(include=[])
    all_ids = all_docs["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    # add documents to Chroma vector store
    chroma_db.add_documents(splitted_documents)
    if print_progress:
        print("Finished creating Chroma vector store.")
        print()

def make_embeddings(sentences: str | List[str]) -> torch.Tensor:
    if isinstance(sentences, str):
        sentences = [sentences]
    resp = requests.post(f"{INFINITY_URL}/embeddings", json={
        "input": sentences,
        "model": EMBEDDINGS_MODEL,
    })
    vecs = [item["embedding"] for item in sorted(resp.json()["data"], key=lambda x: x["index"])]
    return F.normalize(torch.tensor(vecs), p=2, dim=1)

def calculate_similarity_scores(sentences_1: str | List[str], sentences_2: str | List[str]) -> torch.Tensor:
    embeddings_1, embeddings_2 = make_embeddings(sentences_1), make_embeddings(sentences_2)
    return embeddings_1 @ embeddings_2.T 

def rerank(query: str, documents: List[str]) -> List[float]:
    resp = requests.post(f"{INFINITY_URL}/rerank", json={
        "query": query,
        "documents": documents,
        "model": RERANK_MODEL
    })
    results = resp.json()["results"]
    scores = [0.0] * len(documents)
    for item in results:
        scores[item["index"]] = item["relevance_score"]
    return scores

def combine_possible_info(lst: List[str]):
    return " + ".join([x for x in list(set(lst)) if len(x) > 0])

def combine_possible_info_multilist(multilst: List[List[str]]):
    final_lst = []
    for lst in multilst:
        final_lst.extend(lst)
    return " + ".join([x for x in list(set(final_lst)) if len(x) > 0])


class ADVPInformationRetriever:
    def __init__(self, referencing_col_df: pd.DataFrame, referencing_col_with_choice_df: Optional[pd.DataFrame] = None,
                 chroma_db_path: str = CHROMA_DB_PATH, chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME, 
                 top_k: int = TOP_K, top_k_rerank: int = TOP_K_RERANK, similarity_score_threshold: float = SIMILARITY_SCORE_THRESHOLD,
                 temperature: float = TEMPERATURE, top_p: float = TOP_P, max_new_tokens: int = MAX_NEW_TOKEN,
                 detail_choice_similarity_score_threshold: float = DETAIL_CHOICE_SIMILARITY_SCORE_THRESHOLD,
                 device: Optional[str] = None):
        # load ref col df
        self.referencing_col_lst = referencing_col_df["column"].to_list()
        # Definition-only context (no examples) — safe to show to the LLM.
        self.referencing_col_context_lst = referencing_col_df.apply(
            lambda x: x["column"] if pd.isna(x["description"]) else x["column"] + ": " + x["description"],
            axis=1,
        ).to_list()
        # Examples kept separate; used ONLY to strengthen retrieval and as a
        # labeled, anti-leakage hint block inside the prompt.
        if "examples" in referencing_col_df.columns:
            self.referencing_col_examples_lst = referencing_col_df["examples"].apply(
                lambda x: x.strip() if isinstance(x, str) and x.strip() else ""
            ).to_list()
        else:
            self.referencing_col_examples_lst = ["" for _ in self.referencing_col_lst]
        self.referencing_col_use_examples_in_llm_lst = referencing_col_df["use_examples_in_llm"]
        # Retrieval query = definition + examples (examples help embedding recall,
        # but will NOT be shown verbatim to the LLM in the generation prompt).
        self.referencing_col_retrieval_query_lst = [
            ctx if not ex else f"{ctx} Examples: {ex}."
            for ctx, ex in zip(self.referencing_col_context_lst, self.referencing_col_examples_lst)
        ]

        # col with choice
        self.referencing_col_with_choice_lst = referencing_col_with_choice_df["column"].to_list()
        self.referencing_col_choice_with_choice_lst = referencing_col_with_choice_df["choice"].apply(lambda x: x.split(",")).to_list()

        # load vector store
        self.vector_store = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=InfinityEmbeddings(model="NeuML/pubmedbert-base-embeddings"),
            collection_name=chroma_db_collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
    
        # load the device
        self.device = device if device is not None else "cpu"
        
        # NOTE: config for search and generate, add it as params later
        self.top_k = top_k
        self.top_k_rerank = top_k_rerank
        self.max_new_tokens = max_new_tokens
        self.similarity_score_threshold = similarity_score_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.detail_choice_similarity_score_threshold = detail_choice_similarity_score_threshold
    
    def make_messages(self, query: str, documents: List[str], examples: str = "", use_examples_in_llm: bool = True) -> List[Dict]:
        document_str = "\n\n".join([f"EXCERPT {i + 1}:\n{d}" for i, d in enumerate(documents)])

        # Examples from the CSV are quarantined in a clearly labeled block and
        # explicitly forbidden unless they also appear in the EXCERPTs. This
        # keeps them available as weak context without encouraging the model
        # to regurgitate them as extractions.
        examples_answer = (
            f"\nRetrieval example answers (DO NOT output any of these unless they appear verbatim in the EXCERPTs above):\n{examples}\n"
            if examples and use_examples_in_llm else ""
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict biomedical information extraction engine. "
                    "Only output values that appear verbatim in the provided EXCERPTs. "
                    "Do not copy any term from the field definition, from retrieval hints, "
                    "or from your own domain knowledge if it is not literally present in the EXCERPTs. "
                    "If the EXCERPTs do not support a value, return an empty list. "
                    "Respond with a single JSON object and nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"""Goal: extract candidate values for a field, grounded strictly in the EXCERPTs.

Field:
{query}
{examples_answer}
EXCERPTs:
{document_str}

Rules:
- Only include items that appear literally in the EXCERPTs (exact casing, exact spelling).
- No paraphrasing, no expansions, no translations.
- If a long name and an abbreviation both appear in the EXCERPTs, include BOTH.
- De-duplicate items.
- If nothing in the EXCERPTs supports the field, return {{"items": []}}.

Respond with a single JSON object only, no prose, no markdown fence:
{{"items": ["<verbatim string from EXCERPT>", ...]}}"""
            },
        ]
        return messages

    # def make_prompt(self, query: str, documents: List[str]) -> str:    
    #     messages = self.make_messages(query, documents)
    #     prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     return prompt
    

    def extract_lst_from_llm_output(self, text: str) -> List[str]:
        text = text.replace("```json", "").replace("```", "").strip()
        # Prefer the structured JSON object produced by the new prompt.
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            import json
            try:
                obj = json.loads(json_match.group(0))
                items = obj.get("items", []) if isinstance(obj, dict) else []
                if isinstance(items, list):
                    lst = [str(x) for x in items if isinstance(x, (str, int, float))]
                    lst = list(set([item.lower() for item in lst]))
                    return lst
            except Exception:
                pass
        # Backward-compatible fallback for bare Python list outputs.
        matches = re.findall(r"\[.*?\]", text, re.DOTALL)
        if not matches:
            return []
        try:
            lst = ast.literal_eval(matches[-1])
            lst = list(set([item.lower() for item in lst]))
            return lst
        except Exception:
            return []

    def extract_possible_info_from_paper(self, pmid: int, pmcid: str) -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category
        """
        res = {}

        ingest_doc_from_pmc(pmid, pmcid)

        for ref_col, ref_col_context, ref_col_examples, ref_col_use_examples_in_llm, ref_col_retrieval_query in zip(
            self.referencing_col_lst,
            self.referencing_col_context_lst,
            self.referencing_col_examples_lst,
            self.referencing_col_use_examples_in_llm_lst,
            self.referencing_col_retrieval_query_lst,
        ):
            # Retrieval uses definition + examples (better recall); the LLM
            # prompt sees only the definition, with examples in a quarantined block.
            query = ref_col_context
            retrieval_query = ref_col_retrieval_query
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query = retrieval_query,
                k = self.top_k,
                filter = {"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
            )
            documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
            # if no docs can found => no useful info 
            if len(documents) == 0:
                res[ref_col] = []
                continue

            # rerank
            scores = rerank(retrieval_query, documents)
            documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            documents = documents[:self.top_k_rerank]

            for doc in documents:
                messages = self.make_messages(query, [doc], examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
                response = LLAMA_CLIENT.chat.completions.create(
                    model="local",
                    messages=messages, max_tokens=self.max_new_tokens,
                    temperature=self.temperature, top_p=self.top_p,
                    response_format={"type": "json_object"},
                )
                response = response.choices[0].message.content
                if ref_col not in res:
                    res[ref_col] = []
                new_info = self.extract_lst_from_llm_output(response)
                new_info = list(map(lambda x: x.lower(), new_info))
                res[ref_col] = list(set(res[ref_col] + new_info))
        
        col_with_category = []
        for ref_col, ref_col_choice in zip(
            self.referencing_col_with_choice_lst, self.referencing_col_choice_with_choice_lst
        ):
            if ref_col in res:
                detail_choice_similarity = calculate_similarity_scores(res[ref_col], ref_col_choice)
                # get the max of each col
                max_by_choice = detail_choice_similarity.max(axis = 0).values
                valid_choice = []
                for i in range(len(ref_col_choice)):
                    if max_by_choice[i] > self.detail_choice_similarity_score_threshold:
                        valid_choice.append(ref_col_choice[i])
                res[f"{ref_col} category"] = deepcopy(valid_choice)
                col_with_category.append(ref_col)
        for ref_col in col_with_category:
            temp, temp_category = res[ref_col], res[f"{ref_col} category"]
            res[ref_col] = deepcopy(temp_category)
            res[f"{ref_col} details"] = deepcopy(temp)
            del res[f"{ref_col} category"]
        return res

    def extract_possible_info_from_pdf_paper(self, pmid: int, filename: str) -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category
        """
        res = {}

        ingest_doc_from_pdf(pmid, filename)

        for ref_col, ref_col_context, ref_col_examples, ref_col_use_examples_in_llm, ref_col_retrieval_query in zip(
            self.referencing_col_lst,
            self.referencing_col_context_lst,
            self.referencing_col_examples_lst,
            self.referencing_col_use_examples_in_llm_lst,
            self.referencing_col_retrieval_query_lst,
        ):
            query = ref_col_context
            retrieval_query = ref_col_retrieval_query
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query = retrieval_query,
                k = self.top_k,
                filter = {"$and": [{"PMID": str(pmid)}]},
            )
            documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
            # if no docs can found => no useful info 
            if len(documents) == 0:
                res[ref_col] = []
                continue

            # rerank
            scores = rerank(retrieval_query, documents)
            documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            documents = documents[:self.top_k_rerank]

            messages = self.make_messages(query, documents, examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
            response = LLAMA_CLIENT.chat.completions.create(
                model="local",
                messages=messages, max_tokens=self.max_new_tokens,
                temperature=self.temperature, top_p=self.top_p,
                response_format={"type": "json_object"},
            )
            response = response.choices[0].message.content
            res[ref_col] = self.extract_lst_from_llm_output(response)

        for ref_col, ref_col_choice in zip(
            self.referencing_col_with_choice_lst, self.referencing_col_choice_with_choice_lst
        ):
            if ref_col in res:
                detail_choice_similarity = calculate_similarity_scores(res[ref_col], ref_col_choice)
                # get the max of each col
                max_by_choice = detail_choice_similarity.max(axis = 0).values
                valid_choice = []
                for i in range(len(ref_col_choice)):
                    if max_by_choice[i] > self.detail_choice_similarity_score_threshold:
                        valid_choice.append(ref_col_choice[i])
                res[f"{ref_col} category"] = deepcopy(valid_choice)

        return res

def match_possible_info_to_df(df: pd.DataFrame, col_to_possible_info: Dict, 
                              threshold: float = 0.6):
    notes_col = [col for col in df.columns if "notes" in col]
    if len(notes_col) == 0:
        for col in col_to_possible_info:
            if len(col_to_possible_info[col]) == 0:
                df[col] = pd.NA
            else:
                df[col] = combine_possible_info(col_to_possible_info[col])
    else:
        # use info in notes to map, for each info, find the best one
        for col in col_to_possible_info:
            if len(col_to_possible_info[col]) == 0:
                df[col] = pd.NA
            else:
                used_col = []
                for n_col in notes_col:
                    # for each note, check if the match is actually related to that note by check the max similarity
                    unique_value = df[[n_col]].dropna()[n_col].unique().tolist()
                    similarity_score = calculate_similarity_scores(col_to_possible_info[col], unique_value) # #possible info * #unique value
                    # if torch.max(similarity_score) < 0.6:
                    # if best match do not have sim score at least 0.4 - 0.6
                    unique_value_to_possible_info = {}
                    if torch.min(torch.max(similarity_score, dim = 0).values) <= threshold:
                        for i, u in enumerate(unique_value):
                            unique_value_to_possible_info[u] = combine_possible_info(col_to_possible_info[col])
                    else:
                        # best_inx = torch.argmax(similarity_score, dim = 0)
                        # unique_value_to_possible_info = {}
                        # for i, u in enumerate(unique_value):
                        #     unique_value_to_possible_info[u] = col_to_possible_info[col][best_inx[i]]
                        for i, u in enumerate(unique_value):
                            valid_info = [col_to_possible_info[col][inx] for inx in range(similarity_score.shape[0]) if similarity_score[inx, i] >= threshold]
                            unique_value_to_possible_info[u] = combine_possible_info(valid_info)
                    df[f"{col} from {n_col}"] = df[n_col].apply(lambda x: unique_value_to_possible_info.get(x, ""))
                    used_col.append(f"{col} from {n_col}")
                df[col] = df[used_col].apply(lambda x: combine_possible_info(x), axis = 1)
                df[col] = df[col].apply(lambda x: pd.NA if len(x) == 0 else x)
                df = df.drop(used_col, axis = 1)
    return df

def match_possible_info_to_df_with_clues(df: pd.DataFrame, pmid: int, pmcid: str, advp_information_retriever: ADVPInformationRetriever):
    notes_col = [col for col in df.columns if "notes" in col]
    if len(notes_col) == 0:
        col_to_possible_info = advp_information_retriever.extract_possible_info_from_paper(pmid, pmcid)
        for col in col_to_possible_info:
            if len(col_to_possible_info[col]) == 0:
                df[col] = pd.NA
            else:
                df[col] = combine_possible_info(col_to_possible_info[col])
    else:
        # use info in notes to map, for each info, find the best one
        col_to_used_col = {}
        for n_col in notes_col:
            clues = df[[n_col]].dropna()[n_col].unique().tolist()
            col_to_clues_to_possible_info = advp_information_retriever.extract_possible_info_from_paper_and_clues(pmid, pmcid, clues)
            for col in col_to_clues_to_possible_info:
                if col not in col_to_used_col:
                    col_to_used_col[col] = []
                df[f"{col} from {n_col}"] = df[n_col].apply(lambda x: col_to_clues_to_possible_info[col].get(x, []))
                col_to_used_col[col].append(f"{col} from {n_col}")
        for col in col_to_clues_to_possible_info:
            df[col] = df[col_to_used_col[col]].apply(lambda x: combine_possible_info_multilist(x), axis = 1)
            df[col] = df[col].apply(lambda x: pd.NA if len(x) == 0 else x)
            df = df.drop(col_to_used_col[col], axis = 1)
        return df