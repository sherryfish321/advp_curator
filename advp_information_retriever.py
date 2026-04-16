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
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

INFINITY_URL = "http://localhost:7997"
LLAMA_CLIENT = OpenAI(base_url="http://localhost:8001/v1", api_key="none")

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
    # Remove special characters and punctuation (keep basic ones if needed)
    text = re.sub(r'[^a-z0-9\s\.\,\-]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_document(document: Document) -> Document:
    cleaned_text = clean_text(document.page_content)
    cleaned_document = Document(page_content=cleaned_text, metadata=document.metadata)
    return cleaned_document

def ingest_doc_from_pmc(pmid: int, pmcid: str,
                        chroma_db_path: str = "./chroma_db", chroma_db_collection_name: str = "advp2", 
                        chunk_size: int = 500, chunk_overlap: int = 50, print_progress: bool = False):
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
                        chroma_db_path: str = "./chroma_db", chroma_db_collection_name: str = "advp2", 
                        chunk_size: int = 500, chunk_overlap: int = 50, print_progress: bool = False):
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
        "model": "NeuML/pubmedbert-base-embeddings"
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
        "model": "BAAI/bge-reranker-base"
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

# class AllowedTokensProcessor(LogitsProcessor):
#     def __init__(self, allowed_token_ids):
#         self.allowed_token_ids = allowed_token_ids

#     def __call__(self, input_ids, scores):
#         mask = torch.full_like(scores, float("-inf"))
#         mask[:, list(self.allowed_token_ids)] = 0
#         return scores + mask

# class AllowedTokensProcessorLlamaCpp:
#     def __init__(self, allowed_token_ids):
#         self.allowed_token_ids = allowed_token_ids

#     def __call__(self, input_ids, scores):
#         import numpy as np
#         mask = np.full_like(scores, float("-inf"))
#         for token_id in self.allowed_token_ids:
#             mask[token_id] = 0
#         return scores + mask
    
class ADVPInformationRetriever:
    def __init__(self, referencing_col_df: pd.DataFrame, chroma_db_path: str = "./chroma_db", chroma_db_collection_name: str = "advp2", 
                 # embeddings_model_name: str = "NeuML/pubmedbert-base-embeddings", reranker_model_name: str = "jinaai/jina-reranker-v1-turbo-en",
                 # llm_model_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct", llm_gguf_path: Optional[str] = "./qwen2.5-7b-instruct-q4/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                 # use_hf: bool = True, 
                 device: Optional[str] = None):
        # load ref col df
        self.referencing_col_lst = referencing_col_df["column"].to_list()
        self.referencing_col_context_lst = referencing_col_df.apply(lambda x: x["column"] if pd.isna(x["description"]) else x["column"] + ": " + x["description"], axis = 1).to_list()
        self.referencing_col_choices_lst = referencing_col_df["choices"].apply(lambda x: x.split(";") if isinstance(x, str) and ";" in x else []).to_list()

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
        self.top_k = 20
        self.top_k_rerank = 5
        self.max_new_tokens = 64
        self.similarity_score_threshold = 0.0
        self.temperature = 0
        self.top_p = 1
    
    def make_messages(self, query: str, documents: List[str]) -> List[Dict]:
        document_str = "\n\n".join([f"EXCERPT {i + 1}:\n{d}" for i, d in enumerate(documents)])
        # Example:
        # Question: What kind of study is in the paper? (Allowed values: "SNP-based", "gene-based")
        # Document: We performed both variant-level and gene-level association analyses. First, SNP-based GWAS summary statistics were computed in each ancestry group (EUR, EAS, AFR) and then combined via fixed-effect meta-analysis. In addition, we conducted a gene-based test aggregating rare variants per gene to prioritize candidate genes. The workflow included three stages: (i) discovery in EUR, (ii) validation in EAS and AFR, and (iii) trans-ancestry meta-analysis.
        # Output: ["gene-based", "SNP-based"]
        # - If the field value applies to the entire paper (not tied to a specific cohort/stage), prefix it with GLOBAL: (e.g., GLOBAL: imputed to 1000 Genomes).

        # Now do the same:

        messages = [
            {
                "role": "system", 
                "content": "You are a strict biomedical information extraction engine. Only use the provided text. Never guess. Output must follow the required format exactly."
            },
            {
                "role": "user",
                "content": f"""
Goal: extract candidate values for a field that will later be matched to table column headers.

Field:
{query}

Important:
- The output will be mapped to rows by matching against column names/headers.
- Therefore, prefer short “header-like” labels exactly as written (e.g., ADNI, IGAP, UK Biobank, discovery, replication).
- If both a long name and an abbreviation/alias appear, include BOTH as separate items (exact casing).

Evidence text:
{document_str}

Rules:
- Return exactly one Python list literal of strings and nothing else.
- Only include items explicitly supported by the excerpts.
- No paraphrasing; copy exact phrases/labels from the excerpts.
- De-duplicate items.
- If insufficient evidence, return [].

Output:"""
            }
        ]
        return messages

    # def make_prompt(self, query: str, documents: List[str]) -> str:    
    #     messages = self.make_messages(query, documents)
    #     prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     return prompt
    

    def extract_lst_from_llm_output(self, text: str) -> List[str]:
        text = text.replace("```", "").strip()
        matches = re.findall(r"\[.*?\]", text, re.DOTALL)
        if not matches or len(matches) < 1:
            return []
        list_str = matches[-1]
        try:
            return ast.literal_eval(list_str)
        except Exception:
            return []

    def extract_possible_info_from_paper(self, pmid: int, pmcid: str) -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category
        """
        res = {}

        ingest_doc_from_pmc(pmid, pmcid)

        for ref_col, ref_col_context in zip(self.referencing_col_lst, self.referencing_col_context_lst):
            # search for related context
            # full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
            query = ref_col_context
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query = query, 
                k = self.top_k,
                filter = {"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
            )
            documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
            # if no docs can found => no useful info 
            if len(documents) == 0:
                res[ref_col] = []
                continue

            # rerank
            scores = rerank(query, documents)
            documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            documents = documents[:self.top_k_rerank]

            # extract a list of possible info from llm
            # full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
            # if self.use_hf:
            #     prompt = self.make_prompt(query, documents)
            #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            #     outputs = self.model.generate(
            #         **inputs,
            #         max_new_tokens=self.max_new_tokens,
            #         do_sample=False,
            #         temperature=self.temperature,
            #         top_p=self.top_p,
            #     )
            #     response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            # else:
            #     messages = self.make_messages(query, documents)
            #     response = self.llm.create_chat_completion(
            #         messages=messages,
            #         max_tokens=self.max_new_tokens,
            #         temperature=self.temperature,
            #         top_p=self.top_p,
            #     )
            #     response = response["choices"][0]["message"]["content"]
            messages = self.make_messages(query, documents)
            response = LLAMA_CLIENT.chat.completions.create(
                model="local",
                messages=messages, max_tokens=self.max_new_tokens,
                temperature=self.temperature, top_p=self.top_p,
            )
            response = response.choices[0].message.content
            res[ref_col] = self.extract_lst_from_llm_output(response)
    
        return res
    
    def extract_possible_info_from_pdf_paper(self, pmid: int, filename: str) -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category
        """
        res = {}

        ingest_doc_from_pdf(pmid, filename)

        for ref_col, ref_col_context in zip(self.referencing_col_lst, self.referencing_col_context_lst):
            # search for related context
            # full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
            query = ref_col_context
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query = query, 
                k = self.top_k,
                filter = {"$and": [{"PMID": str(pmid)}]},
            )
            documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
            # if no docs can found => no useful info 
            if len(documents) == 0:
                res[ref_col] = []
                continue

            # rerank
            scores = rerank(query, documents)
            documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            documents = documents[:self.top_k_rerank]

            # extract a list of possible info from llm
            # full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
            # if self.use_hf:
            #     prompt = self.make_prompt(query, documents)
            #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            #     outputs = self.model.generate(
            #         **inputs,
            #         max_new_tokens=self.max_new_tokens,
            #         do_sample=False,
            #         temperature=self.temperature,
            #         top_p=self.top_p,
            #     )
            #     response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            # else:
            #     messages = self.make_messages(query, documents)
            #     response = self.llm.create_chat_completion(
            #         messages=messages,
            #         max_tokens=self.max_new_tokens,
            #         temperature=self.temperature,
            #         top_p=self.top_p,
            #     )
            #     response = response["choices"][0]["message"]["content"]
            messages = self.make_messages(query, documents)
            response = LLAMA_CLIENT.chat.completions.create(
                model="local",
                messages=messages, max_tokens=self.max_new_tokens,
                temperature=self.temperature, top_p=self.top_p,
            )
            response = response.choices[0].message.content
            res[ref_col] = self.extract_lst_from_llm_output(response)
    
        return res

class ADVPInformationRetrieverKeyword:
    def __init__(self, info_type: str, keyword_dict: Dict[str, List]):
        if keyword_dict is None:
            raise Exception("Please include a keyword dictionary")
        self.info_type = info_type
        self.keyword_dict = keyword_dict
    
    def extract_possible_info_from_paper(self, pmid: int, pmcid: str) -> List[str]:
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
        possible_info = []
        for keyword in self.keyword_dict:
            for keyword_variation in self.keyword_dict[keyword]:
                if keyword_variation in curr_doc:
                    possible_info.append(keyword)
                    break
        return possible_info


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