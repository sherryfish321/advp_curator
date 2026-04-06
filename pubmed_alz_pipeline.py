import argparse
import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_TOOL = "advp_pubmed_alz_pipeline"
DEFAULT_EMAIL = "team@example.org"

STOP_PUBLICATION_TYPES = {
    "review",
    "systematic review",
    "meta-analysis",
    "editorial",
    "comment",
    "letter",
    "news",
}

AD_POSITIVE_MESH = {
    "alzheimer disease",
    "alzheimer's disease",
    "neurodegenerative diseases",
    "mild cognitive impairment",
    "amyloid beta-peptides",
    "tau proteins",
    "apolipoproteins e",
}

GENETICS_HINTS = {
    "gene",
    "genes",
    "genetic",
    "genetics",
    "genome-wide association study",
    "gwas",
    "genome wide association",
    "variant",
    "variants",
    "locus",
    "risk allele",
    "polymorphism",
    "snp",
    "apoe",
    "mutation",
}

AD_TITLE_ABSTRACT_HINTS = {
    "alzheimer",
    "alzheimers",
    "alzheimer's",
    "late-onset alzheimer",
    "dementia",
    "amyloid",
    "tau",
    "cognitive decline",
    "adni",
}


def normalize_term(value: str) -> str:
    value = value or ""
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


class PubMedClient:
    def __init__(self, email: str, api_key: Optional[str] = None, tool: str = DEFAULT_TOOL, delay: float = 0.34):
        self.email = email
        self.api_key = api_key
        self.tool = tool
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{tool} ({email})"})

    def _request(self, endpoint: str, params: Dict[str, str]) -> requests.Response:
        payload = {"tool": self.tool, "email": self.email, **params}
        if self.api_key:
            payload["api_key"] = self.api_key
        response = self.session.get(f"{EUTILS_BASE}/{endpoint}", params=payload, timeout=60)
        response.raise_for_status()
        time.sleep(self.delay)
        return response

    def esearch(self, query: str, retmax: int = 200, mindate: Optional[int] = None, maxdate: Optional[int] = None) -> List[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(retmax),
            "sort": "pub date",
        }
        if mindate is not None and maxdate is not None:
            params["datetype"] = "pdat"
            params["mindate"] = str(mindate)
            params["maxdate"] = str(maxdate)
        data = self._request("esearch.fcgi", params).json()
        return data.get("esearchresult", {}).get("idlist", [])

    def efetch_pubmed_xml(self, pmids: Sequence[str]) -> ET.Element:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        text = self._request("efetch.fcgi", params).text
        return ET.fromstring(text)


def text_or_empty(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def parse_pubmed_article(article: ET.Element) -> Dict[str, object]:
    medline = article.find("MedlineCitation")
    article_node = medline.find("Article") if medline is not None else None
    pmid = text_or_empty(medline.find("PMID")) if medline is not None else ""
    title = text_or_empty(article_node.find("ArticleTitle")) if article_node is not None else ""

    abstract_parts = []
    if article_node is not None:
        abstract = article_node.find("Abstract")
        if abstract is not None:
            for part in abstract.findall("AbstractText"):
                label = part.attrib.get("Label")
                content = text_or_empty(part)
                abstract_parts.append(f"{label}: {content}" if label else content)
    abstract_text = "\n".join(part for part in abstract_parts if part)

    publication_types = []
    if article_node is not None:
        for node in article_node.findall("./PublicationTypeList/PublicationType"):
            value = text_or_empty(node)
            if value:
                publication_types.append(value)

    mesh_terms = []
    if medline is not None:
        for mesh in medline.findall("./MeshHeadingList/MeshHeading"):
            descriptor = text_or_empty(mesh.find("DescriptorName"))
            qualifiers = [text_or_empty(q) for q in mesh.findall("QualifierName") if text_or_empty(q)]
            label = descriptor if not qualifiers else f"{descriptor} / {'; '.join(qualifiers)}"
            if label:
                mesh_terms.append(label)

    pub_year = ""
    if article_node is not None:
        pub_date = article_node.find("./Journal/JournalIssue/PubDate")
        for tag in ("Year", "MedlineDate"):
            value = text_or_empty(pub_date.find(tag)) if pub_date is not None else ""
            if value:
                pub_year = value[:4]
                break

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract_text,
        "publication_types": unique_preserve_order(publication_types),
        "mesh_terms": unique_preserve_order(mesh_terms),
        "year": pub_year,
    }


def fetch_pubmed_records(client: PubMedClient, pmids: Sequence[str], batch_size: int = 100) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not pmids:
        return records
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        root = client.efetch_pubmed_xml(batch)
        for article in root.findall("./PubmedArticle"):
            records.append(parse_pubmed_article(article))
    return records


def build_default_query(year: Optional[int] = None) -> str:
    parts = [
        '(("Alzheimer Disease"[MeSH Terms]) OR (alzheimer*[Title/Abstract]))',
        '("gene"[Title] OR "genes"[Title] OR "genetics"[Title] OR "genetic*"[Title/Abstract])',
        'NOT (Review[Publication Type])',
    ]
    if year is not None:
        parts.append(f'("{year}"[Date - Publication])')
    return " AND ".join(parts)


def load_advp_pmids(path: Path) -> List[str]:
    df = pd.read_csv(path, sep="\t", usecols=["Pubmed PMID"])
    pmids = df["Pubmed PMID"].dropna().astype(int).astype(str).tolist()
    return unique_preserve_order(pmids)


def build_keyword_profile(records: Sequence[Dict[str, object]], min_count: int = 2) -> Dict[str, List[str]]:
    mesh_counter: Counter = Counter()
    publication_type_counter: Counter = Counter()

    for record in records:
        for mesh in record.get("mesh_terms", []):
            mesh_counter[normalize_term(mesh)] += 1
        for publication_type in record.get("publication_types", []):
            publication_type_counter[normalize_term(publication_type)] += 1

    top_mesh = [term for term, count in mesh_counter.most_common() if count >= min_count]
    top_publication_types = [term for term, count in publication_type_counter.most_common() if count >= min_count]
    return {
        "mesh_terms": top_mesh,
        "publication_types": top_publication_types,
    }


def score_record(record: Dict[str, object], profile: Optional[Dict[str, List[str]]] = None) -> Dict[str, object]:
    title = normalize_term(str(record.get("title", "")))
    abstract = normalize_term(str(record.get("abstract", "")))
    publication_types = [normalize_term(v) for v in record.get("publication_types", [])]
    mesh_terms = [normalize_term(v) for v in record.get("mesh_terms", [])]
    haystack = f"{title}\n{abstract}"

    reasons: List[str] = []
    score = 0

    if any(pub_type in STOP_PUBLICATION_TYPES for pub_type in publication_types):
        score -= 4
        reasons.append("excluded publication type")

    if any(mesh in AD_POSITIVE_MESH for mesh in mesh_terms):
        score += 4
        reasons.append("AD-related MeSH")

    if any(keyword in haystack for keyword in AD_TITLE_ABSTRACT_HINTS):
        score += 3
        reasons.append("AD keyword in title/abstract")

    if any(keyword in haystack for keyword in GENETICS_HINTS):
        score += 2
        reasons.append("genetics keyword in title/abstract")

    if any(keyword in title for keyword in {"gene", "genes", "genetic", "genetics", "gwas"}):
        score += 2
        reasons.append("genetics keyword in title")

    if profile:
        profile_mesh = set(profile.get("mesh_terms", []))
        profile_pub_types = set(profile.get("publication_types", []))
        shared_mesh = sorted(set(mesh_terms) & profile_mesh)
        shared_pub_types = sorted(set(publication_types) & profile_pub_types)
        if shared_mesh:
            score += min(3, len(shared_mesh))
            reasons.append(f"profile mesh overlap: {', '.join(shared_mesh[:3])}")
        if shared_pub_types:
            score += 1
            reasons.append(f"profile publication type overlap: {', '.join(shared_pub_types[:3])}")

    is_relevant = score >= 4
    return {
        **record,
        "score": score,
        "is_alzheimers_relevant": is_relevant,
        "reason": " | ".join(reasons),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "pmid",
        "year",
        "title",
        "publication_types",
        "mesh_terms",
        "score",
        "is_alzheimers_relevant",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["publication_types"] = "; ".join(row.get("publication_types", []))
            output["mesh_terms"] = "; ".join(row.get("mesh_terms", []))
            writer.writerow({key: output.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Search PubMed and score whether papers are Alzheimer's-related.")
    parser.add_argument("--query", help="Explicit PubMed query. If omitted, build a default AD + genetics query.")
    parser.add_argument("--year", type=int, help="Filter search to a single publication year.")
    parser.add_argument("--retmax", type=int, default=200, help="Maximum number of PubMed hits to fetch.")
    parser.add_argument("--email", default=DEFAULT_EMAIL, help="Email for NCBI E-utilities.")
    parser.add_argument("--api-key", default=None, help="Optional NCBI API key.")
    parser.add_argument(
        "--advp-tsv",
        default="advp.variant.records.hg38.tsv",
        help="ADVP TSV used to collect seed PMIDs for profile building.",
    )
    parser.add_argument(
        "--profile-json",
        default=None,
        help="Optional precomputed keyword profile JSON. If provided, skip profile building from ADVP PMIDs.",
    )
    parser.add_argument(
        "--profile-sample-size",
        type=int,
        default=50,
        help="How many unique ADVP PMIDs to fetch when building a local profile.",
    )
    parser.add_argument("--out-csv", default="pubmed_alz_candidates.csv", help="Output CSV path.")
    parser.add_argument("--out-profile-json", default="pubmed_alz_profile.json", help="Output keyword profile JSON path.")
    args = parser.parse_args()

    query = args.query or build_default_query(args.year)
    client = PubMedClient(email=args.email, api_key=args.api_key)

    profile: Optional[Dict[str, List[str]]] = None
    if args.profile_json:
        profile = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    else:
        advp_pmids = load_advp_pmids(Path(args.advp_tsv))
        seed_pmids = advp_pmids[: args.profile_sample_size]
        seed_records = fetch_pubmed_records(client, seed_pmids)
        profile = build_keyword_profile(seed_records)
        if profile is not None:
            Path(args.out_profile_json).write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    pmids = client.esearch(query=query, retmax=args.retmax, mindate=args.year, maxdate=args.year)
    records = fetch_pubmed_records(client, pmids)
    scored_rows = [score_record(record, profile=profile) for record in records]
    scored_rows.sort(key=lambda row: (row["score"], row["pmid"]), reverse=True)

    write_csv(Path(args.out_csv), scored_rows)

    summary = {
        "query": query,
        "year": args.year,
        "hits": len(pmids),
        "relevant_hits": sum(1 for row in scored_rows if row["is_alzheimers_relevant"]),
        "out_csv": str(Path(args.out_csv).resolve()),
        "out_profile_json": str(Path(args.out_profile_json).resolve()) if profile else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
