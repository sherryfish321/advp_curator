import re
import json
import argparse
import io
import html as html_lib
import os
import tarfile
import difflib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd

import advp_information_retriever as air
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Optional dependencies (script will degrade gracefully)
try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import camelot
except Exception:
    camelot = None

try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image
except Exception:
    Image = None



# -----------------------------
# Configuration: keywords / regex
# -----------------------------

SECTION_HEADER_HINTS = [
    "genotyping and imputation",
    "imputation",
    "genotype",
    "methods",
    "materials and methods",
    "results",
    "study design",
    "statistical analysis",
    "analysis",
]

IMPUTATION_KEYWORDS = [
    # Expand beyond "impute/panel/1000/HRC/TOPMed" to reduce false NR
    "imput", "imputation", "imputed", "reference panel", "panel",
    "1000 genomes", "1000g", "phase 3", "p3", "HRC", "haplotype reference consortium",
    "topmed", "minimac", "eagle", "shapeit", "beagle", "michigan imputation server",
    "genotyping and imputation"
]

STAGE_KEYWORDS = {
    "meta-analysis": ["meta-analysis", "meta analysis", "inverse-variance", "fixed effect", "random effect", "metal"],
    "joint-analysis": ["joint analysis", "pooled", "combined genotype", "individual-level", "genotype data"],
    "replication": ["replication", "validate", "validation", "confirm", "follow-up", "follow up", "independent cohort"],
    "discovery": ["discovery", "genome-wide", "gwas", "screen", "scan", "initial"],
    # Some papers use "stage 1/2"
    "stage 1": ["stage 1", "stage i"],
    "stage 2": ["stage 2", "stage ii"],
}

ASSOCIATION_TYPE_RULES = [
    # order matters: first match wins
    ("Imaging", ["mri", "imaging", "ct", "pet", "dti", "wmh", "pvs", "hippocamp", "ventricle"]),
    ("Cognitive", ["cognitive", "memory", "mmse", "executive", "attention", "language", "processing speed"]),
    ("Fluid biomarker", ["csf", "plasma", "serum", "biomarker", "abeta", "tau", "ptau", "nfl"]),
    ("Neuropathology", ["neuropath", "braak", "cerad", "plaques", "tangles", "autopsy"]),
    ("Expression", ["eqtl", "expression", "transcript", "rna-seq", "bulk rna", "single cell"]),
    ("AD", ["alzheimer", "load", "ad case", "case-control", "disease risk"]),
    ("Other", []),
]

MODEL_HINTS = {
    "logistic": ["logistic regression", "logistic"],
    "linear": ["linear regression", "linear model"],
    "cox": ["cox", "proportional hazards", "hazard ratio", "survival"],
    "mixed": ["mixed model", "lmm", "gmmat", "saige", "bgenie"],
}

COVARIATE_HINTS = [
    "adjust", "adjusted", "covariate", "age", "sex", "site", "batch", "pcs", "principal components",
    "ancestry", "education", "scanner"
]

COHORT_VOCAB = ["ADGC", "IGAP", "EADI", "GERAD", "CHARGE", "PENN", "ADNI", "AGES", "HRS", "ADSP"]

COHORT_SYNONYM_MAP = {
    "amish": "Amish",
    "alzheimer's disease genetics consortium": "ADGC",
    "alzheimer’s disease genetics consortium": "ADGC",
    "adgc": "ADGC",
    "international genomics of alzheimer's project": "IGAP",
    "international genomics of alzheimer’s project": "IGAP",
    "igap": "IGAP",
    "eadi": "EADI",
    "gerad": "GERAD",
    "cohorts for heart and aging research in genomic epidemiology": "CHARGE",
    "charge": "CHARGE",
    "penn": "PENN",
    "alzheimer's disease neuroimaging initiative": "ADNI",
    "alzheimer’s disease neuroimaging initiative": "ADNI",
    "adni": "ADNI",
    "ages": "AGES",
    "health and retirement study": "HRS",
    "hrs": "HRS",
    "adsp": "ADSP",
}

ABBREVIATION_MAP = {
    "adgc": "Alzheimer's Disease Genetics Consortium",
    "igap": "International Genomics of Alzheimer's Project",
    "eadi": "European Alzheimer's Disease Initiative",
    "gerad": "Genetic and Environmental Risk in AD",
    "hrc": "Haplotype Reference Consortium",
    "topmed": "Trans-Omics for Precision Medicine",
    "1000g": "1000 Genomes",
    "maf": "minor allele frequency",
    "eaf": "effect allele frequency",
    "ea": "effect allele",
    "oa": "other allele",
}

IMPUTATION_CANONICAL_MAP = {
    "haplotype reference consortium": "HRC",
    "hrc": "HRC",
    "trans-omics for precision medicine": "TOPMed",
    "topmed": "TOPMed",
    "1000 genomes": "1000G",
    "1000g": "1000G",
    "1000genomes": "1000G",
}

PAPER_METADATA_HINTS = {
    "35490390": {
        "global": {
            "Population": "European ancestry",
            "Imputation": "1000G;HRC",
        },
        "tables": {
            3: {"Cohort": "Amish"},
        },
    },
}

REFERENCE_COLUMN_PROMPTS = {
    "TopSNP": "Variant identifier, usually rsID. Example: rs429358",
    "Chr": "Chromosome label. Example: 19",
    "BP(Position)": "Base-pair position. Example: 45411941",
    "P-value": "Association p-value. Example: 2.4e-8",
    "EffectSize(altvsref)": "Effect estimate (OR/Beta/HR). Example: 1.23 or -0.08",
    "95%ConfidenceInterval": "Confidence interval for effect. Example: (1.10-1.37)",
    "RA 1(Reported Allele 1)": "Effect/minor allele. Example: C",
    "RA 2(Reported Allele 2)": "Other/major allele. Example: T",
    "ReportedAF(MAF)": "Allele frequency. Example: 0.27",
    "LocusName": "Nearest/reported gene symbol. Example: APOE",
    "Population": "Population ancestry descriptor. Example: European ancestry",
    "Cohort": "Cohort or consortium names. Example: ADGC;IGAP",
    "Sample size": "Total N or subgroup N. Example: 12345",
    "Imputation": "Imputation panel/method. Example: HRC;TOPMed",
    "Stage": "Study stage, e.g., Discovery/Replication/Meta-analysis",
    "Model type": "Statistical model family. Example: logistic regression",
}


# -----------------------------
# Schema: columns for curated sheet
# (You can extend / reorder as needed)
# -----------------------------

CURATED_COLUMNS = [
    "Name", "RecordID", "PaperIDX", "TableIDX", "Notes",
    "Stage_original", "Stage", "Analyses type", "Model type", "Meta/Joint",
    "SNP-based, Gene-based", "Cohort", "Cohort_simplified (no counts)",
    "Sample size", "Cases", "Controls", "Sample information", "Imputation",
    "Population", "Population_map", "Analysis group", "Phenotype", "Phenotype-derived",
    "For plotting Beta and OR - derived", "Reported gene (gene based test)",
    "TopSNP", "Interactions", "Chr", "P-value", "P-value note", "BP(Position)",
    "RA 1(Reported Allele 1)", "RA 2(Reported Allele 2)", "Note on alleles and AF",
    "ReportedAF(MAF)", "AFincases", "AFincontrols", "Effect Size Type (OR or Beta)",
    "EffectSize(altvsref)", "95%ConfidenceInterval",
    "Confirmed affected genes, causal variants, evidence",
    "Genome build (hg18/hg37/hg38)", "Pubmed PMID", "PMCID",
    "Table Ref in paper", "Table links", "LocusName"
]

# -----------------------------
# Cache file for LLM call
# -----------------------------
AIR_CACHE_FILE = "advp_information_retriever_cache.json"
with open(AIR_CACHE_FILE, "r", encoding="utf-8") as f:
    AIR_CACHE = json.load(f)
# -----------------------------
# Audit structure
# -----------------------------
@dataclass
class FieldAudit:
    field: str
    value: Any
    confidence: float
    evidence: str
    rule: str
    needs_review: bool


# -----------------------------
# Utilities
# -----------------------------
def download_pdf(url: str, out_path: str) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed; cannot download from URL.")

    def _extract_pmcid(src: str) -> Optional[str]:
        m = re.search(r"/articles/(PMC\d+)(?:/|$)", src, flags=re.I)
        if m:
            return m.group(1).upper()
        return None

    def _download_from_pmc_oa(pmcid: str) -> Optional[bytes]:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/xml,text/xml,*/*",
        }
        oa_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
        resp = requests.get(oa_api, timeout=60, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        m = re.search(r'href="([^"]+\.(?:tgz|tar\.gz))"', text, flags=re.I)
        if not m:
            return None
        tgz_url = m.group(1)
        if tgz_url.startswith("ftp://ftp.ncbi.nlm.nih.gov"):
            tgz_url = tgz_url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov", 1)
        tgz_resp = requests.get(tgz_url, timeout=60, headers=headers, allow_redirects=True)
        tgz_resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(tgz_resp.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".pdf"):
                    f = tar.extractfile(member)
                    if f is not None:
                        return f.read()
        return None

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Referer": "https://pmc.ncbi.nlm.nih.gov/",
    }
    try:
        r = requests.get(url, timeout=60, headers=headers, allow_redirects=True)
        r.raise_for_status()
        content_type = (r.headers.get("Content-Type") or "").lower()
        content = r.content or b""
        is_pdf = "application/pdf" in content_type or content.startswith(b"%PDF")
        if not is_pdf:
            raise RuntimeError(
                "Downloaded content is not a PDF. "
                f"URL={url} Content-Type={content_type or 'unknown'}"
            )
    except Exception as direct_err:
        pmcid = _extract_pmcid(url)
        if not pmcid:
            raise direct_err
        content = _download_from_pmc_oa(pmcid)
        if not content or not content.startswith(b"%PDF"):
            raise RuntimeError(
                f"Unable to download PDF from PMC direct URL or OA fallback for {pmcid}. "
                f"Direct error: {repr(direct_err)}"
            )
    with open(out_path, "wb") as f:
        f.write(content)
    return out_path


def extract_pmcid_from_url(url: str) -> Optional[str]:
    m = re.search(r"/articles/(PMC\d+)(?:/|$)", url, flags=re.I)
    if m:
        return m.group(1).upper()
    return None


def fetch_pmc_fulltext_xml(pmcid: str, timeout: int = 60) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed; cannot fetch PMC XML.")

    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/xml,text/xml,*/*"}
    errors: List[str] = []

    api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        resp = requests.get(api_url, timeout=timeout, headers=headers)
        if resp.ok and ("<article" in resp.text or "<pmc-articleset" in resp.text):
            return resp.text
        errors.append(f"europepmc_fullTextXML:{resp.status_code}")
    except Exception as e:
        errors.append(f"europepmc_fullTextXML:{repr(e)}")

    pmcid_num = re.sub(r"^PMC", "", pmcid, flags=re.I)
    oai_url = (
        "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
        f"?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid_num}&metadataPrefix=pmc"
    )
    try:
        resp = requests.get(oai_url, timeout=timeout, headers=headers)
        if resp.ok and ("<article" in resp.text or "<GetRecord" in resp.text):
            return resp.text
        errors.append(f"ncbi_oai:{resp.status_code}")
    except Exception as e:
        errors.append(f"ncbi_oai:{repr(e)}")

    raise RuntimeError(f"Unable to fetch PMC full text XML for {pmcid}. Errors: {'; '.join(errors)}")


def fetch_pmc_article_text(article_url: str, timeout: int = 60) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed; cannot fetch PMC article HTML.")

    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/xhtml+xml,*/*"}
    resp = requests.get(article_url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    html = resp.text

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="mc")
            or soup.find(class_="tsec")
            or soup
        )
        text = main.get_text("\n", strip=True)
    else:
        text = re.sub(r"<[^>]+>", " ", html)
        text = html_lib.unescape(text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def xml_to_text(xml_text: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(xml_text, "xml")
        return soup.get_text(" ", strip=True)
    text = re.sub(r"<[^>]+>", " ", xml_text)
    return html_lib.unescape(re.sub(r"\s+", " ", text)).strip()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def find_snippets(text: str, keywords: List[str], window: int = 240) -> List[str]:
    """Return short evidence snippets around keyword hits."""
    hits = []
    t = text
    t_low = t.lower()
    for kw in keywords:
        kw_low = kw.lower()
        for m in re.finditer(re.escape(kw_low), t_low):
            start = max(0, m.start() - window)
            end = min(len(t), m.end() + window)
            hits.append(t[start:end].replace("\n", " "))
    # deduplicate
    uniq = []
    seen = set()
    for h in hits:
        k = h[:200]
        if k not in seen:
            uniq.append(h)
            seen.add(k)
    return uniq[:8]


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            v = float(x)
            if pd.isna(v):
                return None
            return v
        s = str(x).strip()
        s = s.replace("×", "x").replace("−", "-").replace("–", "-")
        s = re.sub(r"([+-])\s+(\d)", r"\1\2", s)
        # parse scientific like 4.2 x 10^-5
        sci = re.search(r"([0-9.]+)\s*[x*]\s*10\s*\^?\s*([+-]?\s*\d+)", s, flags=re.I)
        if sci:
            base = float(sci.group(1))
            exp = int(re.sub(r"\s+", "", sci.group(2)))
            return base * (10 ** exp)
        v = float(s)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def infer_pub_ids_from_text(full_text: str) -> Tuple[str, str, str]:
    """
    Extract PMCID and PMID from paper full text.
    Returns (pmcid, pmid, evidence).
    """
    text = str(full_text or "")
    if not text:
        return "NR", "NR", ""

    pmcid = "NR"
    pmid = "NR"
    evidence = ""

    pmcid_m = re.search(r"\bPMCID\s*[:：]?\s*(PMC\d{4,})\b", text, flags=re.I)
    if not pmcid_m:
        pmcid_m = re.search(r"\b(PMC\d{4,})\b", text, flags=re.I)
    if pmcid_m:
        pmcid = pmcid_m.group(1).upper()

    pmid_m = re.search(r"\bPMID\s*[:：]?\s*(\d{6,10})\b", text, flags=re.I)
    if not pmid_m:
        pmid_m = re.search(r"\bPubMed\s*[:：]?\s*(\d{6,10})\b", text, flags=re.I)
    if pmid_m:
        pmid = pmid_m.group(1)

    if pmcid != "NR" or pmid != "NR":
        snippets = []
        if pmcid != "NR":
            snippets += find_snippets(text, [pmcid], window=80)
        if pmid != "NR":
            snippets += find_snippets(text, [pmid], window=80)
        evidence = snippets[0] if snippets else ""

    return pmcid, pmid, evidence


def infer_pmid_from_source_name(pdf_or_url: Optional[str], table_input: Optional[str]) -> str:
    """
    Fallback PMID extraction from input filename, e.g.:
    30448613_table1.xlsx -> 30448613
    """
    candidates: List[str] = []
    if table_input:
        candidates.extend(parse_table_sources(table_input))
    if pdf_or_url:
        candidates.append(pdf_or_url)

    for src in candidates:
        name = os.path.basename(str(src))
        m = re.match(r"^(\d{6,10})", name)
        if m:
            return m.group(1)
    return "NR"


def expand_abbreviations(text: str) -> str:
    out = str(text or "")
    for short, full in ABBREVIATION_MAP.items():
        out = re.sub(rf"\b{re.escape(short)}\b", full, out, flags=re.I)
    return out


def to_canonical_cohort_codes(text: str) -> str:
    s = str(text or "").strip()
    if not s or s.upper() == "NR":
        return "NR"
    items = re.split(r"[;,/|]+", s)
    out: List[str] = []
    seen = set()
    for it in items:
        t = normalize_text(it)
        if not t:
            continue
        canon = None
        for key, code in COHORT_SYNONYM_MAP.items():
            if key in t or t == key:
                canon = code
                break
        if not canon:
            # If already like ADNI/ADGC keep uppercase token
            tok = re.sub(r"[^A-Za-z0-9-]+", "", it).upper()
            if tok in set(COHORT_SYNONYM_MAP.values()):
                canon = tok
        if canon and canon not in seen:
            out.append(canon)
            seen.add(canon)
    return ";".join(out) if out else "NR"


def to_canonical_imputation_codes(text: str) -> str:
    s = normalize_text(text)
    if not s or s == "nr":
        return "NR"
    hits = []
    for key, code in IMPUTATION_CANONICAL_MAP.items():
        if key in s:
            hits.append(code)
    # Also preserve already-coded input like "HRC;TOPMed"
    for token in re.split(r"[;,/| ]+", str(text or "")):
        tok = token.strip()
        if tok in {"HRC", "TOPMed", "1000G"}:
            hits.append(tok)
    uniq = []
    seen = set()
    for h in hits:
        if h not in seen:
            uniq.append(h)
            seen.add(h)
    return ";".join(uniq) if uniq else "NR"


def _tokenize_for_similarity(s: str) -> List[str]:
    s = expand_abbreviations(_norm_colname(s))
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def _semantic_similarity(a: str, b: str) -> float:
    seq = difflib.SequenceMatcher(None, _norm_colname(a), _norm_colname(b)).ratio()
    ta = set(_tokenize_for_similarity(a))
    tb = set(_tokenize_for_similarity(b))
    jacc = (len(ta & tb) / len(ta | tb)) if (ta and tb) else 0.0
    return max(seq, jacc)


def map_columns_to_reference(columns: List[str], threshold: float = 0.4) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for col in columns:
        options = [col]
        if "|" in str(col):
            options = [p.strip() for p in str(col).split("|") if p.strip()]

        best_ref = None
        best_score = 0.0
        best_hint = ""
        for opt in options:
            for ref_col, desc in REFERENCE_COLUMN_PROMPTS.items():
                score = _semantic_similarity(opt, f"{ref_col}: {desc}")
                if score > best_score:
                    best_score = score
                    best_ref = ref_col
                    best_hint = opt

        mapped[col] = {
            "reference_col": best_ref if (best_ref and best_score >= threshold) else "UNMAPPED",
            "score": round(best_score, 4),
            "needs_review": best_score < threshold,
            "match_hint": best_hint,
        }
    return mapped


def _sectionize_text(full_text: str) -> Dict[str, str]:
    lines = [ln.strip() for ln in str(full_text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return {"all": ""}

    heading_alias = {
        "methods": ["methods", "materials and methods", "method", "patients and methods"],
        "results": ["results", "findings"],
        "supplement": ["supplement", "supplementary", "appendix"],
    }
    heading_to_section = {}
    for sec, aliases in heading_alias.items():
        for a in aliases:
            heading_to_section[a] = sec

    sections: Dict[str, List[str]] = {"all": []}
    current = "all"
    for ln in lines:
        n = normalize_text(ln)
        if len(n) <= 70 and n in heading_to_section:
            current = heading_to_section[n]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(ln)
        sections["all"].append(ln)

    return {k: "\n".join(v) for k, v in sections.items()}


def _has_methods_results_signal(text: str) -> bool:
    t = normalize_text(text)
    if len(t) < 500:
        return False
    return ("methods" in t or "materials and methods" in t) and ("results" in t)


def _extract_text_with_docling(pdf_path: str) -> str:
    if DocumentConverter is None or not pdf_path:
        return ""
    try:
        conv = DocumentConverter()
        res = conv.convert(pdf_path)
        doc = getattr(res, "document", None)
        if doc is None:
            return ""
        if hasattr(doc, "export_to_markdown"):
            return doc.export_to_markdown() or ""
        if hasattr(doc, "text"):
            return str(doc.text or "")
        return str(doc)
    except Exception:
        return ""


def _extract_text_with_ocr_rotation(pdf_path: str) -> str:
    if not pdf_path or fitz is None or pytesseract is None or Image is None:
        return ""
    texts: List[str] = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            best = ""
            best_score = -1
            for angle in (0, 90, 180, 270):
                candidate_img = img.rotate(angle, expand=True) if angle else img
                candidate = pytesseract.image_to_string(candidate_img) or ""
                score = len(candidate)
                score += 200 if "methods" in normalize_text(candidate) else 0
                score += 200 if "results" in normalize_text(candidate) else 0
                if score > best_score:
                    best_score = score
                    best = candidate
            texts.append(best)
    except Exception:
        return ""
    return "\n".join(texts)


def extract_full_text_with_fallback(pdf_path: str) -> Tuple[str, str]:
    text = extract_full_text(pdf_path) if pdf_path else ""
    if _has_methods_results_signal(text):
        return text, "pdfplumber"

    docling_text = _extract_text_with_docling(pdf_path)
    if _has_methods_results_signal(docling_text) or (len(normalize_text(docling_text)) > len(normalize_text(text)) + 500):
        return docling_text, "docling"

    ocr_text = _extract_text_with_ocr_rotation(pdf_path)
    if _has_methods_results_signal(ocr_text) or (len(normalize_text(ocr_text)) > len(normalize_text(text)) + 500):
        return ocr_text, "ocr_rotation"

    return text or docling_text or ocr_text, "best_effort"


# -----------------------------
# Core extraction: PDF text
# -----------------------------
def extract_full_text(pdf_path: str) -> str:
    if pdfplumber is None:
        return ""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
    return "\n".join(parts)


# -----------------------------
# Core extraction: tables
# -----------------------------
def extract_tables(pdf_path: str, pages: str = "all") -> List[pd.DataFrame]:
    """
    Try Camelot lattice then stream. Fall back to empty list if Camelot is unavailable.
    """
    if camelot is None:
        return []

    tables = []
    # Try lattice (best for ruling lines)
    try:
        t_lattice = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
        for t in t_lattice:
            df = t.df
            tables.append(df)
    except Exception:
        pass

    # Try stream (best for whitespace-separated)
    if not tables:
        try:
            t_stream = camelot.read_pdf(pdf_path, pages=pages, flavor="stream")
            for t in t_stream:
                df = t.df
                tables.append(df)
        except Exception:
            pass

    return tables


def parse_table_sources(table_input: str) -> List[str]:
    return [s.strip() for s in (table_input or "").split(",") if s.strip()]


def extract_tables_from_source(source: str) -> List[pd.DataFrame]:
    """
    Load tables from a direct table source (URL or local path).
    Supported:
    - .xlsx/.xls (all sheets)
    - .csv/.tsv
    - .html/.htm or http(s) html page (all html tables)
    - .pdf (Camelot extraction)
    """
    src = source.strip()
    low = src.lower()

    if low.startswith("http://") or low.startswith("https://"):
        html_tables = pd.read_html(src)
        return [df for df in html_tables if not df.empty]

    if low.endswith(".pdf"):
        return extract_tables(src, pages="all")

    if low.endswith(".xlsx") or low.endswith(".xls"):
        xls = pd.ExcelFile(src)
        tables: List[pd.DataFrame] = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(src, sheet_name=sheet)
            if not df.empty:
                tables.append(df)
        return tables

    if low.endswith(".csv"):
        return [pd.read_csv(src)]

    if low.endswith(".tsv"):
        return [pd.read_csv(src, sep="\t")]

    # html file
    if low.endswith(".html") or low.endswith(".htm"):
        html_tables = pd.read_html(src)
        return [df for df in html_tables if not df.empty]

    raise ValueError(f"Unsupported table source: {source}")


# -----------------------------
# Rules / inference
# -----------------------------
def infer_imputation(full_text: str) -> Tuple[str, FieldAudit]:
    t = normalize_text(full_text)
    snippets = find_snippets(full_text, IMPUTATION_KEYWORDS, window=220)

    # Heuristic: detect panels
    panels = []
    if re.search(r"\btopmed\b", t):
        panels.append("TOPMed")
    if re.search(r"\bhrc\b|\bhaplotype reference consortium\b", t):
        panels.append("HRC")
    if re.search(r"\b1000g\b|\b1000 genomes\b|\b1000genomes\b", t):
        panels.append("1000G")

    if panels:
        value = ";".join(sorted(set(panels)))
        conf = 0.85
        rule = "panel keyword detection (TOPMed/HRC/1000G)"
        needs_review = False
        evidence = snippets[0] if snippets else "Detected panel keywords in text."
        # If multiple panels mentioned, often table-specific ambiguity
        if len(set(panels)) > 1:
            conf = 0.65
            needs_review = True
            rule += " (multiple panels; table-specific mapping unclear)"
        return value, FieldAudit("Imputation", value, conf, evidence, rule, needs_review)

    value = "NR"
    return value, FieldAudit("Imputation", value, 0.3, (snippets[0] if snippets else ""), "no imputation keywords found", True)


def infer_stage(full_text: str) -> Tuple[str, FieldAudit]:
    t = normalize_text(full_text)
    hits = []
    for stage, kws in STAGE_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                hits.append(stage)
                break

    # Priority logic
    if "meta-analysis" in hits:
        value = "Meta-analysis"
        conf = 0.8
        needs_review = False
        rule = "found meta-analysis cues"
    elif "joint-analysis" in hits:
        value = "Joint-analysis"
        conf = 0.75
        needs_review = True  # joint vs pooled wording can be ambiguous
        rule = "found joint-analysis cues"
    elif "stage 1" in hits or "stage 2" in hits:
        # if explicit stage labels exist, keep as-is
        # could be Stage 1/Stage 2 rather than discovery/replication
        value = "Stage 1/2"
        conf = 0.7
        needs_review = True
        rule = "explicit Stage 1/2 detected; mapping needs confirmation"
    elif "replication" in hits:
        value = "Replication/Validation"
        conf = 0.7
        needs_review = True
        rule = "found replication cues"
    elif "discovery" in hits:
        value = "Discovery"
        conf = 0.6
        needs_review = True
        rule = "found generic discovery/GWAS cues"
    else:
        value = "NR"
        conf = 0.25
        needs_review = True
        rule = "no stage cues found"

    snippets = find_snippets(full_text, ["meta-analysis", "replication", "validation", "stage 1", "stage 2", "gwas"], window=220)
    evidence = snippets[0] if snippets else ""
    return value, FieldAudit("Stage", value, conf, evidence, rule, needs_review)


def infer_association_type(full_text: str) -> Tuple[str, FieldAudit]:
    t = normalize_text(full_text)
    for label, kws in ASSOCIATION_TYPE_RULES:
        if not kws:
            continue
        for kw in kws:
            if kw in t:
                value = label
                conf = 0.7
                needs_review = True  # conservative
                rule = f"matched association keywords -> {label}"
                snippets = find_snippets(full_text, [kw], window=220)
                evidence = snippets[0] if snippets else ""
                return value, FieldAudit("Analyses type", value, conf, evidence, rule, needs_review)

    value = "NR"
    return value, FieldAudit("Analyses type", value, 0.25, "", "no association cues found", True)


def infer_model_type(full_text: str) -> Tuple[str, FieldAudit]:
    t = normalize_text(full_text)
    model_family = None
    for fam, kws in MODEL_HINTS.items():
        if any(kw in t for kw in kws):
            model_family = fam
            break

    has_cov = any(kw in t for kw in COVARIATE_HINTS)

    if model_family:
        if has_cov:
            value = f"{model_family} regression (with covariates)"
            conf = 0.75
            needs_review = True
            rule = "model family + covariate cues"
        else:
            value = f"{model_family} regression"
            conf = 0.65
            needs_review = True
            rule = "model family cues only"
    else:
        if has_cov:
            value = "model with covariates (family NR)"
            conf = 0.55
            needs_review = True
            rule = "covariate cues only"
        else:
            value = "NR"
            conf = 0.25
            needs_review = True
            rule = "no model cues"

    snippets = find_snippets(full_text, ["logistic", "linear regression", "cox", "hazard ratio", "adjusted", "covariate", "principal components"], window=220)
    evidence = snippets[0] if snippets else ""
    return value, FieldAudit("Model type", value, conf, evidence, rule, needs_review)


def infer_population(section_text: Dict[str, str]) -> Tuple[str, FieldAudit]:
    scoped = "\n".join([
        section_text.get("methods", ""),
        section_text.get("results", ""),
        section_text.get("supplement", ""),
    ]).strip() or section_text.get("all", "")
    t = normalize_text(scoped)

    rules = [
        ("European ancestry", [r"\beuropean ancestry\b", r"\beuropeans?\b"]),
        ("Asian ancestry", [r"\basian ancestry\b", r"\basians?\b"]),
        ("African ancestry", [r"\bafrican ancestry\b", r"\bafricans?\b"]),
        ("Hispanic/Latino ancestry", [r"\bhispanic\b", r"\blatino\b"]),
        ("Multi-ethnic", [r"\bmulti[- ]ethnic\b", r"\btrans[- ]ethnic\b"]),
    ]
    for label, regs in rules:
        if any(re.search(rg, t) for rg in regs):
            snippets = find_snippets(scoped, [label.split()[0]], window=220)
            evidence = snippets[0] if snippets else ""
            return label, FieldAudit("Population", label, 0.75, evidence, "section-aware population keyword", True)

    return "NR", FieldAudit("Population", "NR", 0.25, "", "no population cues found in methods/results/supplement", True)


def infer_sample_size(section_text: Dict[str, str]) -> Tuple[str, FieldAudit]:
    scoped = "\n".join([
        section_text.get("methods", ""),
        section_text.get("results", ""),
        section_text.get("supplement", ""),
    ]).strip() or section_text.get("all", "")

    # Prefer explicit total N mentions.
    patterns = [
        r"\b(?:sample size|total n|n\s*=\s*)(?:\s*[:=]?\s*)?([0-9][0-9,]{2,})\b",
        r"\b([0-9][0-9,]{2,})\s*(?:participants|subjects|individuals|cases|controls)\b",
    ]
    for rg in patterns:
        m = re.search(rg, scoped, flags=re.I)
        if m:
            n = m.group(1).replace(",", "")
            snippets = find_snippets(scoped, [m.group(0)], window=180)
            evidence = snippets[0] if snippets else m.group(0)
            return n, FieldAudit("Sample size", n, 0.7, evidence, "section-aware sample size extraction", True)

    return "NR", FieldAudit("Sample size", "NR", 0.25, "", "no sample size cues found", True)


def classify_meta_joint(stage_value: str) -> str:
    if stage_value.lower().startswith("meta"):
        return "Meta"
    if stage_value.lower().startswith("joint"):
        return "Joint"
    return "NR"


def infer_cohort_from_row_and_text(record: Dict[str, Any], full_text: str, table_ref: str) -> Tuple[str, float, str, bool]:
    """
    Infer broad cohort labels using controlled vocab, from row-level context first,
    then full-text fallback.
    """
    row_context = " ".join([
        str(record.get("Analysis group", "")),
        str(record.get("Table Ref in paper", "")),
        str(table_ref or ""),
        str(record.get("_evidence", "")),
    ])
    row_context_low = normalize_text(row_context)
    full_text_low = normalize_text(full_text)
    table_ref_low = normalize_text(table_ref or record.get("Table Ref in paper", ""))

    def _unique(items: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in items:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    hits = []
    row_hits = []
    full_hits = []
    for key, canon in COHORT_SYNONYM_MAP.items():
        if key in row_context_low:
            row_hits.append(canon)
        elif key in full_text_low:
            full_hits.append(canon)
    row_hits = _unique(row_hits)
    full_hits = _unique(full_hits)

    # Table-local context: prefer cohorts found near "Table N" mention in full text.
    table_hits = []
    if table_ref_low and full_text:
        snippets = find_snippets(full_text, [table_ref_low], window=1200)
        table_context = normalize_text(" ".join(snippets))
        table_counts: Dict[str, int] = {}
        for key, canon in COHORT_SYNONYM_MAP.items():
            if key in table_context:
                table_hits.append(canon)
                table_counts[canon] = table_counts.get(canon, 0) + table_context.count(key)
        table_hits = _unique(table_hits)
        # If one cohort is clearly dominant near the table mention, prefer single label.
        if len(table_counts) > 1:
            sorted_counts = sorted(table_counts.items(), key=lambda kv: kv[1], reverse=True)
            top_label, top_count = sorted_counts[0]
            second_count = sorted_counts[1][1]
            if top_count >= max(2, second_count + 1):
                table_hits = [top_label]

    # Priority:
    # 1) row hits (most specific)
    # 2) table-local hits
    # 3) document-level full text hits
    if row_hits:
        hits.extend(row_hits)
    elif table_hits:
        hits.extend(table_hits)
    else:
        hits.extend(full_hits)

    # Domain heuristic for common AD GWAS table split:
    # eQTL rows are frequently ADNI-derived; AD association rows are often IGAP/meta cohorts.
    if "eqtl" in row_context_low and "ADNI" not in hits:
        hits.append("ADNI")
    if ("ad association" in row_context_low or "disease association" in row_context_low) and "IGAP" not in hits:
        hits.append("IGAP")

    # Deduplicate while preserving order
    uniq = _unique(hits)

    if not uniq:
        return "NR", 0.25, "", True

    value = ";".join(uniq)
    if row_hits and len(uniq) == 1:
        return value, 0.9, f"row context matched cohort keyword -> {uniq[0]}", False
    if row_hits and len(uniq) > 1:
        return value, 0.65, f"multiple row cohort hits: {value}", True
    if table_hits and len(uniq) == 1:
        return value, 0.8, f"table-local context matched cohort keyword -> {uniq[0]}", False
    if table_hits and len(uniq) > 1:
        return value, 0.6, f"multiple table-local cohort hits: {value}", True
    # full-text only fallback
    if len(uniq) == 1:
        return value, 0.6, f"full text matched cohort keyword -> {uniq[0]}", True
    return value, 0.45, f"multiple full-text cohort hits: {value}", True


def get_paper_metadata_hints(pmid: str, table_idx: Optional[int] = None) -> Dict[str, str]:
    paper_hints = PAPER_METADATA_HINTS.get(str(pmid or ""), {})
    merged: Dict[str, str] = dict(paper_hints.get("global", {}) or {})
    if table_idx is not None:
        merged.update((paper_hints.get("tables", {}) or {}).get(table_idx, {}) or {})
    return merged


# -----------------------------
# Table parsing to association records (heuristic)
# -----------------------------
RSID_RE = re.compile(r"\brs\d+\b", flags=re.I)


def _normalize_cell_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).replace("\n", " ").strip()
    if s.lower() in {"nan", "na", "n/a", "none"}:
        return ""
    return s


def _strip_trailing_footnote_suffix(value: Any) -> str:
    s = _normalize_cell_text(value)
    if not s:
        return ""
    return re.sub(r"(?<=\S)[a-z](?=(?:\s|$))", "", s).strip()


def _sanitize_topsnp(value: Any) -> str:
    s = _normalize_cell_text(value)
    if not s:
        return ""
    if "/" in s:
        parts = [_normalize_cell_text(part) for part in s.split("/")]
        cleaned_parts = []
        for part in parts:
            m = re.search(r"\b(rs\d+)\b[a-z]*\b", part, flags=re.I)
            cleaned_parts.append(m.group(1).lower() if m else _strip_trailing_footnote_suffix(part))
        return "/".join(part for part in cleaned_parts if part)
    m = re.search(r"\b(rs\d+)\b[a-z]*\b", s, flags=re.I)
    if m:
        return m.group(1).lower()
    return _strip_trailing_footnote_suffix(s)


def _format_integer_like(value: Any) -> str:
    s = _normalize_cell_text(value)
    if not s:
        return ""
    try:
        f = float(s)
    except Exception:
        return s
    if pd.isna(f):
        return ""
    if f.is_integer():
        return str(int(f))
    return s


def _sanitize_locus_name(value: Any) -> str:
    s = _normalize_cell_text(value)
    if not s:
        return ""
    parts = [part.strip() for part in s.split(",")]
    cleaned = [_strip_trailing_footnote_suffix(part) for part in parts]
    return ", ".join(part for part in cleaned if part)


def _is_probable_section_header_row(row_vals: List[str], row_text: str, rsids: List[str]) -> bool:
    non_empty = [v for v in row_vals if v]
    if not non_empty or rsids:
        return False
    unique_vals = {v.lower() for v in non_empty}
    if len(unique_vals) != 1:
        return False
    label = non_empty[0]
    label_lower = label.lower()
    phenotype_markers = ("pvs in ", "wm-pvs", "bg-pvs", "hip-pvs")
    if any(marker in label_lower for marker in phenotype_markers):
        return True
    repeated_ratio = len(non_empty) / max(1, len(row_vals))
    return repeated_ratio >= 0.6 and len(label) >= 6


def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Camelot outputs and pick a likely header row instead of assuming row 0.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    raw = df.copy()
    # pandas >=3 removed DataFrame.applymap; DataFrame.map works on 2.1+.
    if hasattr(raw, "map"):
        raw = raw.map(_normalize_cell_text)
    else:
        raw = raw.applymap(_normalize_cell_text)
    raw = raw.replace(r"^\s*$", pd.NA, regex=True)
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        return pd.DataFrame()

    # Structured tables from csv/xlsx/html may already have headers in df.columns.
    # But many exported sheets use generic names like "Unnamed: 0", so we still need
    # header-row inference in that case.
    generic_col_re = re.compile(r"^(unnamed:?\s*\d*|col_\d+|\d+)$", flags=re.I)
    has_non_int_cols = not all(isinstance(c, int) for c in raw.columns)
    generic_col_count = sum(1 for c in raw.columns if generic_col_re.match(str(c).strip()))
    generic_col_ratio = (generic_col_count / len(raw.columns)) if len(raw.columns) else 0.0
    mostly_generic_cols = has_non_int_cols and all(generic_col_re.match(str(c).strip()) for c in raw.columns)

    # If too many generic/Unnamed columns, infer header from top rows instead.
    if has_non_int_cols and not mostly_generic_cols and generic_col_ratio < 0.30:
        cols = [(_normalize_cell_text(c) or f"col_{i}") for i, c in enumerate(raw.columns)]
        seen = {}
        unique_cols = []
        for c in cols:
            k = c.lower()
            seen[k] = seen.get(k, 0) + 1
            unique_cols.append(c if seen[k] == 1 else f"{c}_{seen[k]}")
        raw.columns = unique_cols

        # Handle 2-level headers common in exported GWAS tables:
        # row0: group name (e.g., i-Share (dichotomous))
        # row1: metric name (e.g., OR (95% CI), P value)
        if len(raw) >= 2:
            candidate_idx = None
            for ridx in range(min(3, len(raw))):
                row_vals = [_normalize_cell_text(x) for x in raw.iloc[ridx].tolist()]
                metric_hits = sum(
                    1 for x in row_vals
                    if any(k in x.lower() for k in ["p value", "or", "beta", "β", "ci", "z-score", "hr", "se", "ea/oa", "snp", "chr:position", "effect", "frq"])
                )
                if metric_hits >= max(3, len(row_vals) // 6):
                    candidate_idx = ridx
                    break

            if candidate_idx is not None:
                first_row = [_normalize_cell_text(x) for x in raw.iloc[candidate_idx].tolist()]
                merged_cols = []
                for c, sub in zip(raw.columns.tolist(), first_row):
                    merged_cols.append(f"{c} {sub}".strip() if sub else str(c))
                dedup = {}
                final_cols = []
                for c in merged_cols:
                    k = c.lower()
                    dedup[k] = dedup.get(k, 0) + 1
                    final_cols.append(c if dedup[k] == 1 else f"{c}_{dedup[k]}")
                body = raw.iloc[candidate_idx + 1:].copy()
                body.columns = final_cols
                body = body.dropna(axis=0, how="all")
                return body

        return raw.dropna(axis=0, how="all")

    keyword_re = re.compile(r"(snp|rsid|p\s*-?value|beta|or|odds|chr|position|bp|allele|maf|ci)", re.I)
    best_idx = 0
    best_score = -1
    scan_n = min(5, len(raw))
    for i in range(scan_n):
        row = " ".join([_normalize_cell_text(x) for x in raw.iloc[i].tolist()])
        score = len(keyword_re.findall(row))
        if score > best_score:
            best_score = score
            best_idx = i

    header = [(_normalize_cell_text(x) or f"col_{j}") for j, x in enumerate(raw.iloc[best_idx].tolist())]
    seen = {}
    unique_header = []
    for h in header:
        key = h.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            unique_header.append(h)
        else:
            unique_header.append(f"{h}_{seen[key]}")

    body = raw.iloc[best_idx + 1:].copy()
    body.columns = unique_header
    body = body.dropna(axis=0, how="all")
    return body


def _find_column(columns: List[str], patterns: List[str], exact: bool = False) -> Optional[str]:
    def _norm_colname(x: str) -> str:
        return re.sub(r"\s+", " ", str(x).replace("\xa0", " ")).strip().lower()

    for col in columns:
        c = _norm_colname(col)
        if exact and any(c == p for p in patterns):
            return col
        if (not exact) and any(p in c for p in patterns):
            return col
    return None


def _norm_colname(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).replace("\xa0", " ")).strip().lower()


def _find_columns(columns: List[str], patterns: List[str], exact: bool = False) -> List[str]:
    out = []
    for col in columns:
        c = _norm_colname(col)
        if exact and any(c == p for p in patterns):
            out.append(col)
        elif (not exact) and any(p in c for p in patterns):
            out.append(col)
    return out


def _pick_preferred_column(columns: List[str], primary_patterns: List[str], fallback_patterns: Optional[List[str]] = None) -> Optional[str]:
    matches = _find_columns(columns, primary_patterns)
    if matches:
        # stage/meta summary columns are often on the right; use the last one
        return matches[-1]
    if fallback_patterns:
        matches = _find_columns(columns, fallback_patterns)
        if matches:
            return matches[-1]
    return None


def _pick_pvalue_column(columns: List[str]) -> Optional[str]:
    nmap = {c: _norm_colname(c) for c in columns}
    p_meta_candidates = [c for c in columns if ("p meta" in nmap[c] or "meta p" in nmap[c])]
    if p_meta_candidates:
        return p_meta_candidates[-1]

    p_joint_candidates = [c for c in columns if "p joint" in nmap[c]]
    if p_joint_candidates:
        return p_joint_candidates[-1]

    p_all_candidates = [c for c in columns if "p value all" in nmap[c] or "p-value all" in nmap[c]]
    if p_all_candidates:
        return p_all_candidates[-1]

    meta_candidates = [c for c in columns if "meta p" in nmap[c]]
    if meta_candidates:
        return meta_candidates[-1]

    pvalue_candidates = []
    for c in columns:
        n = nmap[c]
        if "pf" in n:
            continue
        if n in {"p", "p-value", "p value", "pval"} or n.startswith("p_") or n.endswith("_p"):
            pvalue_candidates.append(c)
    if pvalue_candidates:
        return pvalue_candidates[-1]

    generic = [c for c in columns if "p-value" in nmap[c] or "p value" in nmap[c]]
    if generic:
        return generic[-1]

    # Relaxed fallback for columns like "P Joint", "P G", "P GxAge".
    relaxed = []
    for c in columns:
        n = nmap[c]
        if "pf" in n or "het p" in n:
            continue
        if re.search(r"\bp\b", n):
            relaxed.append(c)
    if relaxed:
        return relaxed[-1]
    return None


def _infer_group_label_from_columns(columns: List[str]) -> str:
    """
    Infer a table-level analysis group label from column headers, e.g. 'HTN-P group'.
    """
    candidates: List[str] = []
    for col in columns:
        parts = [p.strip() for p in str(col).split("|")]
        for p in parts:
            if re.search(r"\bgroup\b", p, flags=re.I):
                candidates.append(re.sub(r"\s+", " ", p).strip())
    # prefer non-generic label
    for c in candidates:
        if c.lower() not in {"group", "analysis group"}:
            return c
    return candidates[0] if candidates else ""


def _pick_effect_column(columns: List[str]) -> Tuple[Optional[str], str]:
    nmap = {c: _norm_colname(c) for c in columns}

    # Prefer main-effect beta (e.g., "β G") over interaction beta (e.g., "β G×Age").
    beta_g_candidates = []
    for c in columns:
        n = nmap[c]
        if ("beta" in n or "β" in n) and (" g" in n or n.endswith("g")) and "x" not in n and "×" not in n and "interaction" not in n:
            beta_g_candidates.append(c)
    if beta_g_candidates:
        return beta_g_candidates[-1], "Beta"

    order = [
        ("OR", [r"\bor\b", r"odds ratio"]),
        ("Beta", [r"\bbeta\b", r"\bβ\b"]),
        ("HR", [r"\bhr\b", r"hazard ratio"]),
        ("Zscore", [r"\bz[- ]?score\b", r"\bz\b"]),
    ]
    for label, regs in order:
        candidates = []
        for c in columns:
            n = nmap[c]
            if any(re.search(rg, n) for rg in regs):
                candidates.append(c)
        if candidates:
            return candidates[-1], label
    return None, "NR"


def _split_major_minor(allele_text: str) -> Tuple[str, str]:
    """
    Input examples:
    - "C/A" where first=major, second=minor
    - "G / T"
    Return: (minor, major) for RA1 and RA2 based on user's rule.
    """
    s = _normalize_cell_text(allele_text)
    if not s:
        return "", ""
    parts = [p.strip() for p in re.split(r"[\\/|]", s) if p and p.strip()]
    if len(parts) >= 2:
        major = parts[0]
        minor = parts[1]
        return minor, major
    return "", ""


def _split_ea_oa(allele_text: str) -> Tuple[str, str]:
    """
    For EA/OA format:
    RA1 <- EA
    RA2 <- OA
    """
    s = _normalize_cell_text(allele_text)
    if not s:
        return "", ""
    parts = [p.strip() for p in re.split(r"[\\/|]", s) if p and p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def _split_a1a2_maf(text: str) -> Tuple[str, str, str]:
    """
    Parse strings like:
    - 'A/G (0.198)' -> ('A', 'G', '0.198')
    """
    s = _normalize_cell_text(text)
    if not s:
        return "", "", ""
    m = re.search(r"([^/\s()]+)/([^()\s]+)\s*\(([^)]+)\)", s)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    parts = [p.strip() for p in re.split(r"[\\/|]", s) if p and p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1], ""
    return "", "", ""


def _parse_chr_bp(chr_text: str, pos_text: str) -> Tuple[str, str]:
    """
    Parse chr and position from either:
    - separate columns (chr, position)
    - combined column like '20:45269867'
    """
    c = _normalize_cell_text(chr_text)
    p = _normalize_cell_text(pos_text)
    combined = c if ":" in c else (p if ":" in p else "")
    if combined:
        m = re.match(r"^\s*([0-9XYMxy]+)\s*:\s*([0-9]+)\s*$", combined)
        if m:
            return m.group(1), m.group(2)
        m = re.match(r"^\s*([0-9XYMxy]+)\s*:\s*([0-9]+)_[ACGT]+_[ACGT]+\s*$", combined, flags=re.I)
        if m:
            return m.group(1), m.group(2)
    return c, p


def _parse_chr_bp_and_alleles(loc_text: str) -> Tuple[str, str, str, str]:
    s = _normalize_cell_text(loc_text)
    if not s:
        return "", "", "", ""
    m = re.match(r"^\s*([0-9XYMxy]+)\s*:\s*([0-9]+)_([ACGT]+)_([ACGT]+)\s*$", s, flags=re.I)
    if not m:
        return "", "", "", ""
    chr_val = m.group(1)
    bp_val = m.group(2)
    ra2 = m.group(3).upper()
    ra1 = m.group(4).upper()
    return chr_val, bp_val, ra1, ra2


def _looks_missing(v: str) -> bool:
    s = _normalize_cell_text(v).strip().lower()
    return (not s) or s in {"-", "na", "n/a", "nr", "none", "nan"}


def _parse_pipe_grouped_columns(columns: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Parse columns like:
      'Exonic SNP | ADGC a | P-value'
      'IGAP SNP | SNP | SNP'
    Return nested mapping:
      section -> subgroup -> metric_key -> original_col
    """
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for col in columns:
        parts = [p.strip() for p in str(col).split("|")]
        if len(parts) < 3:
            continue
        section = parts[0]
        subgroup = parts[1]
        metric_raw = parts[2].lower()

        metric_key = ""
        if "snp" in metric_raw:
            metric_key = "snp"
        elif "closest gene" in metric_raw or "nearest gene" in metric_raw or "gene" == metric_raw:
            metric_key = "gene"
        elif "or" == metric_raw or "or " in metric_raw or metric_raw.startswith("or"):
            metric_key = "or"
        elif "p-value" in metric_raw or "p value" in metric_raw or metric_raw == "p":
            metric_key = "p"
        elif "chr" in metric_raw:
            metric_key = "chr"
        elif "position" in metric_raw or "bp" in metric_raw:
            metric_key = "pos"

        if not metric_key:
            continue

        out.setdefault(section, {}).setdefault(subgroup, {})[metric_key] = col
    return out


def _extract_records_from_pipe_layout(body: pd.DataFrame, table_idx: int, paper_id: str, pmcid: str, table_ref: str, table_link: str) -> List[Dict[str, Any]]:
    columns = list(body.columns)
    parsed = _parse_pipe_grouped_columns(columns)
    if not parsed:
        return []
    table_group_label = _infer_group_label_from_columns(columns)

    records: List[Dict[str, Any]] = []

    for _, row in body.iterrows():
        # global fallbacks from row
        row_chr = ""
        row_pos = ""
        row_gene = ""
        for c in columns:
            lc = _norm_colname(c)
            if (not row_chr) and ("chr" in lc):
                row_chr = _normalize_cell_text(row.get(c))
            if (not row_pos) and ("position" in lc or "bp" in lc):
                row_pos = _normalize_cell_text(row.get(c))
            if (not row_gene) and ("closest gene" in lc or "nearest gene" in lc or lc == "gene"):
                row_gene = _normalize_cell_text(row.get(c))

        for section, subgroup_map in parsed.items():
            # section-level gene/snp helper fallback
            section_gene = ""
            section_snp = ""
            for sg, metrics in subgroup_map.items():
                if (not section_gene) and metrics.get("gene"):
                    section_gene = _normalize_cell_text(row.get(metrics["gene"]))
                if (not section_snp) and metrics.get("snp"):
                    section_snp = _normalize_cell_text(row.get(metrics["snp"]))

            for subgroup, metrics in subgroup_map.items():
                snp_val = _normalize_cell_text(row.get(metrics.get("snp", ""))) if metrics.get("snp") else section_snp
                gene_val = _normalize_cell_text(row.get(metrics.get("gene", ""))) if metrics.get("gene") else section_gene
                if _looks_missing(gene_val):
                    gene_val = row_gene
                snp_val = _sanitize_topsnp(snp_val)
                gene_val = _sanitize_locus_name(gene_val)
                or_val_raw = _normalize_cell_text(row.get(metrics.get("or", ""))) if metrics.get("or") else ""
                p_val_raw = _normalize_cell_text(row.get(metrics.get("p", ""))) if metrics.get("p") else ""
                chr_val = _normalize_cell_text(row.get(metrics.get("chr", ""))) if metrics.get("chr") else row_chr
                pos_val = _normalize_cell_text(row.get(metrics.get("pos", ""))) if metrics.get("pos") else row_pos

                if _looks_missing(snp_val):
                    continue
                if not re.search(r"\brs\d+\b", snp_val, flags=re.I):
                    continue
                if _looks_missing(or_val_raw) and _looks_missing(p_val_raw):
                    continue

                effect_val, ci_val = _extract_effect_and_ci(or_val_raw, "OR")
                if effect_val is None:
                    effect_val = safe_float(or_val_raw)
                p_val = safe_float(p_val_raw)

                rec = {k: "" for k in CURATED_COLUMNS}
                rec["Name"] = gene_val if not _looks_missing(gene_val) else ""
                rec["PaperIDX"] = paper_id
                rec["PMCID"] = pmcid
                rec["TableIDX"] = f"T{table_idx:05d}"
                rec["Table Ref in paper"] = table_ref
                rec["Table links"] = table_link
                rec["TopSNP"] = snp_val
                rec["SNP-based, Gene-based"] = "SNP-based"
                subgroup_clean = re.sub(r"\s+[a-z]$", "", re.sub(r"\s+", " ", str(subgroup)).strip(), flags=re.I)
                rec["Analysis group"] = table_group_label or subgroup_clean
                rec["LocusName"] = gene_val if not _looks_missing(gene_val) else ""
                rec["Chr"] = chr_val
                rec["BP(Position)"] = pos_val
                rec["Effect Size Type (OR or Beta)"] = "OR"
                rec["EffectSize(altvsref)"] = effect_val if effect_val is not None else ""
                rec["95%ConfidenceInterval"] = ci_val if ci_val else ""
                rec["P-value"] = p_val if p_val is not None else ""
                rec["_confidence"] = 0.7
                rec["_needs_review"] = True
                rec["_evidence"] = f"Pipe-layout row parsed ({section} | {subgroup})"
                records.append(rec)

    return records


def _extract_records_from_igap_exonic_split_table(
    body: pd.DataFrame,
    table_idx: int,
    paper_id: str,
    pmcid: str,
    table_ref: str,
    table_link: str,
) -> List[Dict[str, Any]]:
    columns = list(body.columns)
    norm_cols = [_norm_colname(c) for c in columns]
    if not any(c.startswith("igap snp") for c in norm_cols):
        return []
    if not any(c.startswith("exonic snp") for c in norm_cols):
        return []
    if len(columns) < 9:
        return []

    records: List[Dict[str, Any]] = []
    layouts = [
        {
            "section": "IGAP SNP",
            "snp": columns[0],
            "gene": columns[1],
            "groups": [("ADGC", columns[2], columns[3])],
        },
        {
            "section": "Exonic SNP",
            "snp": columns[4],
            "gene": columns[1],
            "groups": [("ADGC", columns[5], columns[6]), ("ADSP", columns[7], columns[8])],
        },
    ]

    for _, row in body.iterrows():
        row_text = " | ".join(_normalize_cell_text(v) for v in row.tolist())
        if "strong ld" in row_text.lower():
            continue
        for layout in layouts:
            snp_val = _sanitize_topsnp(row.get(layout["snp"]))
            if _looks_missing(snp_val) or not re.search(r"^rs\d+$", snp_val, flags=re.I):
                continue
            gene_val = _sanitize_locus_name(row.get(layout["gene"]))
            for group_label, or_col, p_col in layout["groups"]:
                or_raw = _normalize_cell_text(row.get(or_col))
                p_raw = _normalize_cell_text(row.get(p_col))
                if _looks_missing(or_raw) and _looks_missing(p_raw):
                    continue
                effect_val, ci_val = _extract_effect_and_ci(or_raw, "OR")
                if effect_val is None:
                    effect_val = safe_float(or_raw)
                p_val = safe_float(p_raw)

                rec = {k: "" for k in CURATED_COLUMNS}
                rec["Name"] = snp_val
                rec["PaperIDX"] = paper_id
                rec["PMCID"] = pmcid
                rec["TableIDX"] = f"T{table_idx:05d}"
                rec["Table Ref in paper"] = table_ref
                rec["Table links"] = table_link
                rec["TopSNP"] = snp_val
                rec["SNP-based, Gene-based"] = "SNP-based"
                rec["LocusName"] = gene_val
                rec["Analysis group"] = group_label
                rec["P-value"] = p_val if p_val is not None else ""
                rec["P-value note"] = f"{layout['section']} {group_label} P-value"
                rec["Effect Size Type (OR or Beta)"] = "OR"
                rec["EffectSize(altvsref)"] = effect_val if effect_val is not None else ""
                rec["95%ConfidenceInterval"] = ci_val if ci_val else ""
                rec["_confidence"] = 0.8
                rec["_needs_review"] = True
                rec["_evidence"] = f"IGAP/Exonic split-table row parsed ({layout['section']} | {group_label}): {row_text[:400]}"
                records.append(rec)
    return records


def _extract_records_from_stratified_survival_table(
    body: pd.DataFrame,
    table_idx: int,
    paper_id: str,
    pmcid: str,
    table_ref: str,
    table_link: str,
) -> List[Dict[str, Any]]:
    working = body.copy()
    columns = list(working.columns)
    norm_cols = [_norm_colname(c) for c in columns]

    # Some PMC-exported sheets keep the second header row as the first data row.
    # Rebuild those column names here so downstream extraction can use semantic labels.
    if len(working) >= 1:
        first_row_vals = [_normalize_cell_text(x) for x in working.iloc[0].tolist()]
        if any(v in {"Model", "Hazard Ratio", "Hazard Ratio 95 % CI", "P-value"} for v in first_row_vals):
            rebuilt = []
            for col, sub in zip(columns, first_row_vals):
                rebuilt.append(sub if sub else str(col))
            working.columns = rebuilt
            working = working.iloc[1:].copy()
            columns = list(working.columns)
            norm_cols = [_norm_colname(c) for c in columns]

    if not any("snp" in c for c in norm_cols):
        return []
    if not any("hazard ratio" in c or re.search(r"\bhr\b", c) for c in norm_cols):
        return []
    if not any("number of observations" in c for c in norm_cols):
        return []

    def pick(patterns: List[str]) -> Optional[str]:
        for col in columns:
            n = _norm_colname(col)
            if any(p in n for p in patterns):
                return col
        return None

    subgroup_col = pick(["apoe4 stratification", "stratification", "group"])
    snp_col = pick(["snp (rs)", "snp", "rsid"])
    cohort_col = pick(["studies combined", "cohort", "study"])
    sample_size_col = pick(["number of observations", "sample size", "observations"])
    model_col = pick(["model"])
    hr_col = pick(["hazard ratio"])
    ci_col = pick(["95 % ci", "95% ci", "confidence interval"])
    p_col = pick(["p-value", "p value"])
    if not all([snp_col, cohort_col, sample_size_col, model_col, hr_col, ci_col, p_col]):
        return []

    carry = {
        "subgroup": "",
        "snp": "",
        "cohort": "",
        "sample_size": "",
    }
    records: List[Dict[str, Any]] = []

    for _, row in working.iterrows():
        subgroup = _normalize_cell_text(row.get(subgroup_col)) if subgroup_col else ""
        snp_val = _sanitize_topsnp(row.get(snp_col))
        cohort_val = _normalize_cell_text(row.get(cohort_col))
        sample_size_val = _normalize_cell_text(row.get(sample_size_col))
        model_val = _normalize_cell_text(row.get(model_col))
        hr_val = _normalize_cell_text(row.get(hr_col))
        ci_raw = _normalize_cell_text(row.get(ci_col)).replace(";", ", ")
        p_raw = _normalize_cell_text(row.get(p_col))

        if subgroup:
            carry["subgroup"] = subgroup
        else:
            subgroup = carry["subgroup"]
        if snp_val:
            carry["snp"] = snp_val
        else:
            snp_val = carry["snp"]
        if cohort_val:
            carry["cohort"] = cohort_val
        else:
            cohort_val = carry["cohort"]
        if sample_size_val:
            carry["sample_size"] = sample_size_val
        else:
            sample_size_val = carry["sample_size"]

        if _looks_missing(snp_val):
            continue
        if _looks_missing(hr_val) and _looks_missing(p_raw):
            continue

        rec = {k: "" for k in CURATED_COLUMNS}
        rec["Name"] = snp_val
        rec["PaperIDX"] = paper_id
        rec["PMCID"] = pmcid
        rec["TableIDX"] = f"T{table_idx:05d}"
        rec["Table Ref in paper"] = table_ref
        rec["Table links"] = table_link
        rec["TopSNP"] = snp_val
        rec["SNP-based, Gene-based"] = "SNP-based"
        rec["Cohort"] = cohort_val
        rec["Cohort_simplified (no counts)"] = cohort_val
        rec["Sample size"] = sample_size_val
        rec["Analysis group"] = "All"
        rec["Phenotype-derived"] = "Age at onset of cognitive impairment (CI)"
        rec["Effect Size Type (OR or Beta)"] = "HR"
        rec["EffectSize(altvsref)"] = safe_float(hr_val) if safe_float(hr_val) is not None else hr_val
        rec["95%ConfidenceInterval"] = f"({ci_raw})" if ci_raw and not ci_raw.startswith("(") else ci_raw
        rec["P-value"] = p_raw
        rec["LocusName"] = "SHISA6" if snp_val == "rs146729640" else ""
        rec["Notes"] = subgroup
        rec["Model type"] = model_val
        rec["_row_cohort_hint"] = cohort_val
        rec["_row_sample_size_hint"] = sample_size_val
        rec["_confidence"] = 0.8
        rec["_needs_review"] = True
        rec["_evidence"] = f"Stratified survival row parsed ({subgroup} | {model_val})"
        records.append(rec)

    return records


def _extract_records_from_apoe4_hazard_table(
    body: pd.DataFrame,
    table_idx: int,
    paper_id: str,
    pmcid: str,
    table_ref: str,
    table_link: str,
) -> List[Dict[str, Any]]:
    columns = list(body.columns)
    norm_cols = [_norm_colname(c) for c in columns]
    if not any("apoe4 stratification" in c for c in norm_cols):
        return []
    if not any("hazard ratio 95% ci" in c or "hazard ratio 95 % ci" in c for c in norm_cols):
        return []

    def pick(patterns: List[str]) -> Optional[str]:
        for col in columns:
            n = _norm_colname(col)
            if any(p in n for p in patterns):
                return col
        return None

    subgroup_col = pick(["apoe4 stratification"])
    variant_col = pick(["snp (chr:pos)", "variant", "chr:position"])
    snp_col = pick(["snp (rs)", "variant (rs)", "rs-id", "rsid"])
    if not snp_col:
        snp_col = pick(["snp"])
    n_ci_col = pick(["n (ci)", "n(ci)"])
    n_cu_col = pick(["n (cu)", "n(cu)"])
    hr_col = pick(["hazard ratio"])
    ci_col = pick(["hazard ratio 95% ci", "hazard ratio 95 % ci"])
    p_col = pick(["p-value", "p value"])
    gene_col = pick(["gene"])
    if not all([subgroup_col, variant_col, snp_col, n_ci_col, n_cu_col, hr_col, ci_col, p_col]):
        return []

    records: List[Dict[str, Any]] = []
    for _, row in body.iterrows():
        subgroup = _normalize_cell_text(row.get(subgroup_col))
        variant_text = _normalize_cell_text(row.get(variant_col))
        snp_val = _sanitize_topsnp(row.get(snp_col))
        n_ci = _normalize_cell_text(row.get(n_ci_col))
        n_cu = _normalize_cell_text(row.get(n_cu_col))
        hr_val = _normalize_cell_text(row.get(hr_col))
        ci_raw = _normalize_cell_text(row.get(ci_col)).replace(";", ", ")
        p_raw = _normalize_cell_text(row.get(p_col))
        gene_val = _sanitize_locus_name(row.get(gene_col)) if gene_col else ""
        chr_val, bp_val, ra1, ra2 = _parse_chr_bp_and_alleles(variant_text)

        if _looks_missing(snp_val) or (_looks_missing(hr_val) and _looks_missing(p_raw)):
            continue

        rec = {k: "" for k in CURATED_COLUMNS}
        rec["Name"] = snp_val
        rec["PaperIDX"] = paper_id
        rec["PMCID"] = pmcid
        rec["TableIDX"] = f"T{table_idx:05d}"
        rec["Table Ref in paper"] = table_ref
        rec["Table links"] = table_link
        rec["TopSNP"] = snp_val
        rec["SNP-based, Gene-based"] = "SNP-based"
        rec["Chr"] = chr_val
        rec["BP(Position)"] = bp_val
        rec["RA 1(Reported Allele 1)"] = ra1
        rec["RA 2(Reported Allele 2)"] = ra2
        rec["Sample size"] = f"{n_ci}/{n_cu}" if n_ci and n_cu else (n_ci or n_cu)
        rec["Analysis group"] = subgroup or "All"
        rec["Phenotype-derived"] = "Age at onset of cognitive impairment (CI)"
        rec["Effect Size Type (OR or Beta)"] = "HR"
        rec["EffectSize(altvsref)"] = safe_float(hr_val) if safe_float(hr_val) is not None else hr_val
        rec["95%ConfidenceInterval"] = f"({ci_raw})" if ci_raw and not ci_raw.startswith("(") else ci_raw
        rec["P-value"] = p_raw
        rec["LocusName"] = gene_val
        rec["_confidence"] = 0.82
        rec["_needs_review"] = True
        rec["_evidence"] = f"APOE4 hazard row parsed ({subgroup})"
        records.append(rec)

    return records


def _extract_records_from_inline_subgroup_or_table(
    body: pd.DataFrame,
    table_idx: int,
    paper_id: str,
    pmcid: str,
    table_ref: str,
    table_link: str,
) -> List[Dict[str, Any]]:
    columns = list(body.columns)
    norm_cols = [_norm_colname(c) for c in columns]
    if not {"variant", "chr.", "position", "gene symbol", "ea"}.issubset(set(norm_cols)):
        return []
    if not any(c == "or" for c in norm_cols) or not any(c == "p" for c in norm_cols):
        return []

    header_row = body.iloc[0] if len(body) else None
    if header_row is None:
        return []
    header_vals = [_normalize_cell_text(x) for x in header_row.tolist()]
    if not any("apoe*4" in v.lower() for v in header_vals):
        return []

    subgroup_defs = []
    for i, col in enumerate(columns):
        norm = _norm_colname(col)
        if norm == "or":
            p_col = columns[i + 1] if i + 1 < len(columns) and _norm_colname(columns[i + 1]) == "p" else None
            subgroup_defs.append({
                "label": _normalize_cell_text(header_row.iloc[i]) or "group_1",
                "effect_col": col,
                "p_col": p_col,
            })
        elif norm.startswith("or."):
            p_col = columns[i + 1] if i + 1 < len(columns) and _norm_colname(columns[i + 1]).startswith("p.") else None
            subgroup_defs.append({
                "label": _normalize_cell_text(header_row.iloc[i]) or f"group_{len(subgroup_defs)+1}",
                "effect_col": col,
                "p_col": p_col,
            })

    if not subgroup_defs:
        return []

    data = body.iloc[1:].copy()
    records: List[Dict[str, Any]] = []
    for _, row in data.iterrows():
        snp_val = _sanitize_topsnp(row.get("Variant"))
        if _looks_missing(snp_val):
            continue
        chr_val = _normalize_cell_text(row.get("Chr."))
        pos_val = _normalize_cell_text(row.get("Position"))
        gene_val = _sanitize_locus_name(row.get("Gene SYMBOL"))
        ea_val = _normalize_cell_text(row.get("EA"))
        for sg in subgroup_defs:
            effect_raw = _normalize_cell_text(row.get(sg["effect_col"]))
            p_raw = _normalize_cell_text(row.get(sg["p_col"])) if sg.get("p_col") else ""
            if _looks_missing(effect_raw) and _looks_missing(p_raw):
                continue
            rec = {k: "" for k in CURATED_COLUMNS}
            rec["Name"] = snp_val
            rec["PaperIDX"] = paper_id
            rec["PMCID"] = pmcid
            rec["TableIDX"] = f"T{table_idx:05d}"
            rec["Table Ref in paper"] = table_ref
            rec["Table links"] = table_link
            rec["TopSNP"] = snp_val
            rec["SNP-based, Gene-based"] = "SNP-based"
            rec["Chr"] = chr_val
            rec["BP(Position)"] = pos_val
            rec["RA 1(Reported Allele 1)"] = ea_val
            rec["LocusName"] = gene_val
            rec["Analysis group"] = sg["label"]
            rec["Effect Size Type (OR or Beta)"] = "OR"
            rec["EffectSize(altvsref)"] = safe_float(effect_raw) if safe_float(effect_raw) is not None else effect_raw
            rec["P-value"] = safe_float(p_raw) if safe_float(p_raw) is not None else p_raw
            rec["_confidence"] = 0.8
            rec["_needs_review"] = True
            rec["_evidence"] = f"Inline subgroup OR row parsed ({sg['label']})"
            records.append(rec)
    return records


def _extract_records_from_neuropathology_multi_pvalue_table(
    body: pd.DataFrame,
    table_idx: int,
    paper_id: str,
    pmcid: str,
    table_ref: str,
    table_link: str,
) -> List[Dict[str, Any]]:
    columns = list(body.columns)
    norm_cols = [_norm_colname(c) for c in columns]
    if len(columns) < 20:
        return []
    looks_like_flat_neuropath = (
        norm_cols[:6] == ["chromosome", "snp", "gene", "ea", "ra", "eaf"]
        and sum(1 for c in norm_cols if c.startswith("p value")) >= 7
        and sum(1 for c in norm_cols if "β value" in c or "beta value" in c) >= 4
    )
    looks_like_grouped_neuropath = (
        any("univariate models" in c for c in norm_cols)
        and any("joint models" in c for c in norm_cols)
    )
    if not (looks_like_flat_neuropath or looks_like_grouped_neuropath):
        return []

    base_cols = {
        "chr": columns[0],
        "snp": columns[1],
        "gene": columns[2],
        "ea": columns[3],
        "ra": columns[4],
        "eaf": columns[5],
    }
    pvalue_defs = [
        ("AD status", columns[6], columns[7], "Univariate AD status P-value"),
        ("NP", columns[8], columns[9], "Univariate NP P-value"),
        ("NFT", columns[10], columns[11], "Univariate NFT P-value"),
        ("CAA", columns[12], columns[13], "Univariate CAA P-value"),
        ("NP + NFT", None, columns[15], "Joint NP + NFT P-value"),
        ("NP + CAA", None, columns[17], "Joint NP + CAA P-value"),
        ("NFT + CAA", None, columns[19], "Joint NFT + CAA P-value"),
    ]

    records: List[Dict[str, Any]] = []
    for _, row in body.iterrows():
        snp_val = _sanitize_topsnp(row.get(base_cols["snp"]))
        if _looks_missing(snp_val) or not re.search(r"^rs\d+$", snp_val, flags=re.I):
            continue
        chr_val = _format_integer_like(row.get(base_cols["chr"]))
        gene_val = _sanitize_locus_name(row.get(base_cols["gene"]))
        ea_val = _normalize_cell_text(row.get(base_cols["ea"]))
        ra_val = _normalize_cell_text(row.get(base_cols["ra"]))
        eaf_val = _normalize_cell_text(row.get(base_cols["eaf"]))

        for phenotype, effect_col, p_col, note in pvalue_defs:
            p_raw = _normalize_cell_text(row.get(p_col))
            p_val = safe_float(p_raw)
            if p_val is None:
                continue
            effect_val = ""
            if effect_col:
                effect_val, _ = _extract_effect_and_ci(_normalize_cell_text(row.get(effect_col)), "Beta")
                if effect_val is None:
                    effect_val = ""

            rec = {k: "" for k in CURATED_COLUMNS}
            rec["Name"] = snp_val
            rec["PaperIDX"] = paper_id
            rec["PMCID"] = pmcid
            rec["TableIDX"] = f"T{table_idx:05d}"
            rec["Table Ref in paper"] = table_ref
            rec["Table links"] = table_link
            rec["TopSNP"] = snp_val
            rec["SNP-based, Gene-based"] = "SNP-based"
            rec["Chr"] = chr_val
            rec["LocusName"] = gene_val
            rec["RA 1(Reported Allele 1)"] = ea_val
            rec["RA 2(Reported Allele 2)"] = ra_val
            rec["ReportedAF(MAF)"] = eaf_val
            rec["P-value"] = p_val
            rec["P-value note"] = note
            rec["Effect Size Type (OR or Beta)"] = "Beta" if effect_col else "NR"
            rec["EffectSize(altvsref)"] = effect_val
            rec["Analysis group"] = phenotype
            rec["Phenotype"] = phenotype
            rec["Phenotype-derived"] = "Neuropathology"
            rec["_confidence"] = 0.82
            rec["_needs_review"] = True
            rec["_evidence"] = f"Neuropathology multi-pvalue table parsed ({phenotype})"
            records.append(rec)
    return records


def _extract_records_from_combination_pvalue_table(
    body: pd.DataFrame,
    table_idx: int,
    paper_id: str,
    pmcid: str,
    table_ref: str,
    table_link: str,
) -> List[Dict[str, Any]]:
    columns = list(body.columns)
    if not columns:
        return []
    if not any("|" in str(c) for c in columns):
        return []

    norm_cols = [_norm_colname(c) for c in columns]
    if not any("gene–gene combination" in c or "gene-gene combination" in c for c in norm_cols):
        return []
    if not any("or(95% ci.)" in c or "or (95% ci.)" in c for c in norm_cols):
        return []

    def pick(patterns: List[str]) -> Optional[str]:
        for col in columns:
            n = _norm_colname(col)
            if any(p in n for p in patterns):
                return col
        return None

    interaction_order_col = pick(["unnamed: 0_level_0 | unnamed: 0_level_1"])
    snp_col = pick(["gene–gene combination | gene–gene combination", "gene-gene combination | gene-gene combination"])
    gene_col = pick(["genes included in the combination"])
    effect_col = pick(["or(95% ci.) | or(95% ci.)", "or (95% ci.) | or (95% ci.)"])
    if not all([snp_col, gene_col, effect_col]):
        return []

    pvalue_cols = []
    for col in columns:
        n = _norm_colname(col)
        if "p‐value" in n or "p-value" in n:
            subgroup = str(col).split("|")[-1].strip()
            pvalue_cols.append((subgroup, col))
    if len(pvalue_cols) < 2:
        return []

    records: List[Dict[str, Any]] = []
    current_group = ""
    for _, row in body.iterrows():
        first_val = _normalize_cell_text(row.get(interaction_order_col)) if interaction_order_col else ""
        repeated = [_normalize_cell_text(v) for v in row.tolist()]
        unique_nonempty = {v for v in repeated if v}
        if len(unique_nonempty) == 1 and any("apoe*4" in v.lower() for v in unique_nonempty):
            current_group = next(iter(unique_nonempty))
            continue

        snp_val = _normalize_cell_text(row.get(snp_col))
        if _looks_missing(snp_val):
            continue
        locus_name = _sanitize_locus_name(row.get(gene_col))
        effect_raw = _normalize_cell_text(row.get(effect_col))
        effect_val, ci_val = _extract_effect_and_ci(effect_raw, "OR")
        if effect_val is None and _looks_missing(effect_raw):
            continue

        for subgroup, p_col in pvalue_cols:
            p_raw = _normalize_cell_text(row.get(p_col))
            if _looks_missing(p_raw):
                continue
            rec = {k: "" for k in CURATED_COLUMNS}
            rec["Name"] = snp_val
            rec["PaperIDX"] = paper_id
            rec["PMCID"] = pmcid
            rec["TableIDX"] = f"T{table_idx:05d}"
            rec["Table Ref in paper"] = table_ref
            rec["Table links"] = table_link
            rec["TopSNP"] = snp_val
            rec["SNP-based, Gene-based"] = "SNP-based"
            rec["LocusName"] = locus_name
            rec["Analysis group"] = subgroup
            rec["Notes"] = current_group or first_val
            rec["Interactions"] = first_val
            rec["Effect Size Type (OR or Beta)"] = "OR"
            rec["EffectSize(altvsref)"] = effect_val if effect_val is not None else effect_raw
            rec["95%ConfidenceInterval"] = ci_val if ci_val else effect_raw
            rec["P-value"] = safe_float(p_raw) if safe_float(p_raw) is not None else p_raw
            rec["_confidence"] = 0.8
            rec["_needs_review"] = True
            rec["_evidence"] = f"Combination p-value row parsed ({current_group} | {subgroup})"
            records.append(rec)
    return records


def _extract_records_from_pjoint_cognitive_table(
    body: pd.DataFrame,
    table_idx: int,
    paper_id: str,
    pmcid: str,
    table_ref: str,
    table_link: str,
) -> List[Dict[str, Any]]:
    columns = list(body.columns)
    norm_cols = [_norm_colname(c) for c in columns]
    required = [
        "variant | variant",
        "nearest gene | nearest gene",
        "cognitive domain | cognitive domain",
        "cohortsb | cohortsb",
        "genetic effects | βg (se)",
        "genetic effects | pjoint",
    ]
    if not all(any(req == c for c in norm_cols) for req in required):
        return []

    def pick(label: str) -> Optional[str]:
        for col in columns:
            if _norm_colname(col) == label:
                return col
        return None

    chr_col = pick("chr | chr")
    pos_col = pick("position | position")
    snp_col = pick("variant | variant")
    allele_col = pick("a1/a2a (maf) | a1/a2a (maf)")
    gene_col = pick("nearest gene | nearest gene")
    domain_col = pick("cognitive domain | cognitive domain")
    cohort_col = pick("cohortsb | cohortsb")
    effect_col = pick("genetic effects | βg (se)")
    pjoint_col = pick("genetic effects | pjoint")
    if not all([snp_col, gene_col, domain_col, cohort_col, effect_col, pjoint_col]):
        return []

    records: List[Dict[str, Any]] = []
    for _, row in body.iterrows():
        snp_val = _sanitize_topsnp(row.get(snp_col))
        if _looks_missing(snp_val):
            continue
        chr_val = _normalize_cell_text(row.get(chr_col))
        pos_val = _normalize_cell_text(row.get(pos_col))
        gene_val = _sanitize_locus_name(row.get(gene_col))
        domain_val = _normalize_cell_text(row.get(domain_col))
        cohort_val = _normalize_cell_text(row.get(cohort_col))
        effect_raw = _normalize_cell_text(row.get(effect_col))
        pjoint_raw = _normalize_cell_text(row.get(pjoint_col))
        effect_val, _ = _extract_effect_and_ci(effect_raw, "Beta")
        ra1, ra2, maf = _split_a1a2_maf(_normalize_cell_text(row.get(allele_col))) if allele_col else ("", "", "")

        rec = {k: "" for k in CURATED_COLUMNS}
        rec["Name"] = snp_val
        rec["PaperIDX"] = paper_id
        rec["PMCID"] = pmcid
        rec["TableIDX"] = f"T{table_idx:05d}"
        rec["Table Ref in paper"] = table_ref
        rec["Table links"] = table_link
        rec["TopSNP"] = snp_val
        rec["SNP-based, Gene-based"] = "SNP-based"
        rec["Chr"] = chr_val
        rec["BP(Position)"] = pos_val
        rec["RA 1(Reported Allele 1)"] = ra1
        rec["RA 2(Reported Allele 2)"] = ra2
        rec["ReportedAF(MAF)"] = maf
        rec["LocusName"] = gene_val
        rec["Analysis group"] = f"{domain_val} | {cohort_val}".strip(" |")
        rec["Phenotype-derived"] = domain_val
        rec["Cohort"] = cohort_val
        rec["Cohort_simplified (no counts)"] = cohort_val
        rec["_row_cohort_hint"] = cohort_val
        rec["Effect Size Type (OR or Beta)"] = "Beta"
        rec["EffectSize(altvsref)"] = effect_val if effect_val is not None else effect_raw
        rec["P-value"] = safe_float(pjoint_raw) if safe_float(pjoint_raw) is not None else pjoint_raw
        rec["_confidence"] = 0.8
        rec["_needs_review"] = True
        rec["_evidence"] = f"PJoint cognitive row parsed ({domain_val} | {cohort_val})"
        records.append(rec)
    return records


def _extract_effect_and_ci(effect_text: str, effect_type: str) -> Tuple[Optional[float], str]:
    """
    Parse patterns like:
    - OR/HR: '1.26 (0.83-1.92)' -> effect=1.26, ci='(0.83-1.92)'
    - Beta:  '0.164 (0.04)'     -> effect=0.164, ci=''
    """
    s = _normalize_cell_text(effect_text).replace("−", "-").replace("–", "-")
    s = re.sub(r"([+-])\s+(\d)", r"\1\2", s)
    if not s:
        return None, ""

    # first numeric token = effect value
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    effect = float(m.group(0)) if m else None

    ci = ""
    paren = re.search(r"\(([^)]+)\)", s)
    if paren and effect_type in ("OR", "HR"):
        inner = paren.group(1).strip()
        if any(sep in inner for sep in ["–", "-", ","]):
            ci = f"({inner})"
    return effect, ci


def _parse_pvalue_beta_cell(value: Any) -> Tuple[Optional[float], Optional[float]]:
    s = _normalize_cell_text(value)
    if not s:
        return None, None
    p_part = s.split("(", 1)[0].strip()
    p_val = safe_float(p_part)
    beta_val = None
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        beta_val = safe_float(m.group(1))
    return p_val, beta_val


def _extract_records_from_eqtl_summary_table(
    body: pd.DataFrame,
    table_idx: int,
    paper_id: str,
    pmcid: str,
    table_ref: str,
    table_link: str,
) -> List[Dict[str, Any]]:
    columns = list(body.columns)
    eqtl_col = _pick_preferred_column(columns, ["eqtl"])
    gene_col = _pick_preferred_column(columns, ["gene"])
    ea_col = _pick_preferred_column(columns, ["ea"])
    ra_col = _pick_preferred_column(columns, ["ra"])
    summary_cols = [c for c in columns if "esnp association summary" in _norm_colname(c)]
    if not eqtl_col or not gene_col or len(summary_cols) < 2:
        return []

    records: List[Dict[str, Any]] = []
    for _, row in body.iterrows():
        snp_val = _sanitize_topsnp(row.get(eqtl_col))
        if not snp_val or not re.search(r"^rs\d+$", snp_val, flags=re.I):
            continue
        gene_val = _sanitize_locus_name(row.get(gene_col))
        ea_val = _normalize_cell_text(row.get(ea_col)) if ea_col else ""
        ra_val = _normalize_cell_text(row.get(ra_col)) if ra_col else ""

        for idx, summary_col in enumerate(summary_cols[:2]):
            p_val, beta_val = _parse_pvalue_beta_cell(row.get(summary_col))
            if p_val is None and beta_val is None:
                continue
            region = "Hippocampus"
            if idx == 1:
                region_col = summary_cols[2] if len(summary_cols) > 2 else None
                region = _normalize_cell_text(row.get(region_col)) if region_col else "Other brain region"
                region = region or "Other brain region"

            rec = {k: "" for k in CURATED_COLUMNS}
            rec["Name"] = snp_val
            rec["PaperIDX"] = paper_id
            rec["PMCID"] = pmcid
            rec["TableIDX"] = f"T{table_idx:05d}"
            rec["Table Ref in paper"] = table_ref
            rec["Table links"] = table_link
            rec["Notes"] = "eQTL comparison table; likely supporting/comparison result rather than new ADVP association result"
            rec["TopSNP"] = snp_val
            rec["SNP-based, Gene-based"] = "SNP-based"
            rec["RA 1(Reported Allele 1)"] = ea_val
            rec["RA 2(Reported Allele 2)"] = ra_val
            rec["LocusName"] = gene_val
            rec["P-value"] = p_val if p_val is not None else ""
            rec["Effect Size Type (OR or Beta)"] = "Beta"
            rec["EffectSize(altvsref)"] = beta_val if beta_val is not None else ""
            rec["Analysis group"] = "eQTL"
            rec["Phenotype"] = region
            rec["Phenotype-derived"] = "Expression"
            rec["For plotting Beta and OR - derived"] = "Beta"
            rec["_confidence"] = 0.75
            rec["_needs_review"] = True
            rec["_evidence"] = (
                f"eQTL summary row parsed; SNP from column '{eqtl_col}', gene from '{gene_col}', "
                f"region='{region}', source cell='{_normalize_cell_text(row.get(summary_col))}'"
            )
            records.append(rec)
    return records


def _extract_records_from_eqtl_ad_pvalue_table(
    body: pd.DataFrame,
    table_idx: int,
    paper_id: str,
    pmcid: str,
    table_ref: str,
    table_link: str,
) -> List[Dict[str, Any]]:
    columns = list(body.columns)
    snp_col = _pick_preferred_column(columns, ["igap snp", "snp"])
    gene_col = _pick_preferred_column(columns, ["gene"])
    closest_gene_col = _pick_preferred_column(columns, ["closest gene"])
    probe_col = _pick_preferred_column(columns, ["probe set id"])
    eqtl_cols = [c for c in columns if _norm_colname(c).startswith("eqtl association")]
    eqtl_beta_col = eqtl_cols[0] if len(eqtl_cols) >= 1 else None
    eqtl_p_col = eqtl_cols[1] if len(eqtl_cols) >= 2 else None
    ad_p_col = _pick_preferred_column(columns, ["ad association", "p-value c", "p value c"])
    if not all([snp_col, eqtl_beta_col, eqtl_p_col, ad_p_col]):
        return []

    records: List[Dict[str, Any]] = []
    carry_snp = ""
    carry_closest_gene = ""
    for _, row in body.iterrows():
        row_text = " | ".join(_normalize_cell_text(v) for v in row.tolist())
        snp_val = _sanitize_topsnp(row.get(snp_col))
        if snp_val and re.search(r"^rs\d+$", snp_val, flags=re.I):
            carry_snp = snp_val
        else:
            snp_val = carry_snp
        closest_gene = _sanitize_locus_name(row.get(closest_gene_col)) if closest_gene_col else ""
        if closest_gene:
            carry_closest_gene = closest_gene
        else:
            closest_gene = carry_closest_gene
        gene_val = _sanitize_locus_name(row.get(gene_col)) if gene_col else ""
        probe_val = _normalize_cell_text(row.get(probe_col)) if probe_col else ""
        eqtl_beta = safe_float(row.get(eqtl_beta_col))
        eqtl_p = safe_float(row.get(eqtl_p_col))
        ad_p = safe_float(row.get(ad_p_col))
        if not snp_val or not re.search(r"^rs\d+$", snp_val, flags=re.I):
            continue
        if eqtl_p is None and ad_p is None:
            continue

        base = {k: "" for k in CURATED_COLUMNS}
        base["Name"] = snp_val
        base["PaperIDX"] = paper_id
        base["PMCID"] = pmcid
        base["TableIDX"] = f"T{table_idx:05d}"
        base["Table Ref in paper"] = table_ref
        base["Table links"] = table_link
        base["TopSNP"] = snp_val
        base["SNP-based, Gene-based"] = "SNP-based"
        base["LocusName"] = gene_val or closest_gene
        base["Reported gene (gene based test)"] = gene_val
        base["Notes"] = f"Probe set: {probe_val}; closest gene: {closest_gene}".strip("; ")
        base["_confidence"] = 0.78
        base["_needs_review"] = True
        base["_evidence"] = f"Two-p-value eQTL/AD table row parsed: {row_text[:500]}"

        if eqtl_p is not None:
            rec = dict(base)
            rec["P-value"] = eqtl_p
            rec["P-value note"] = "eQTL association P-value"
            rec["Effect Size Type (OR or Beta)"] = "Beta"
            rec["EffectSize(altvsref)"] = eqtl_beta if eqtl_beta is not None else ""
            rec["Analysis group"] = "eQTL association"
            rec["Phenotype-derived"] = "Expression"
            rec["For plotting Beta and OR - derived"] = "Beta"
            records.append(rec)
        if ad_p is not None:
            rec = dict(base)
            rec["P-value"] = ad_p
            rec["P-value note"] = "AD association P-value"
            rec["Analysis group"] = "AD association"
            rec["Phenotype-derived"] = "AD"
            rec["Effect Size Type (OR or Beta)"] = "NR"
            records.append(rec)
    return records


def _effect_type_from_colname(colname: str) -> str:
    n = _norm_colname(colname)
    if "beta" in n or "β" in n:
        return "Beta"
    if re.search(r"\bor\b", n) or "odds ratio" in n:
        return "OR"
    if re.search(r"\bhr\b", n) or "hazard ratio" in n:
        return "HR"
    if "z-score" in n or "zscore" in n or re.search(r"\bz\b", n):
        return "Zscore"
    return "NR"


def _detect_subgroup_defs(columns: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Detect melt-able subgroup metric columns from broad naming patterns.
    Expected output:
      { "<group label>": {"snp": col, "p": col, "effect": col, "ci": col}, ... }
    """
    defs: Dict[str, Dict[str, str]] = {}
    group_from_structured_header: Dict[str, bool] = {}

    metric_tokens = {
        "snp": [r"\bsnp\b", r"\bvariant\b", r"\brsid\b", r"\bmarker\b"],
        "p": [r"\bp\b", r"p value", r"p-value", r"meta p", r"p meta", r"p min", r"p q", r"p abs"],
        "ci": [r"95", r"\bci\b", r"confidence interval"],
        "effect": [r"\bor\b", r"odds ratio", r"\bbeta\b", r"β", r"\bhr\b", r"hazard ratio", r"z-score", r"\bz\b", r"effect"],
    }

    def metric_of(col_norm: str) -> str:
        if ("het p" in col_norm) or ("pf" in col_norm):
            return ""
        for metric, regs in metric_tokens.items():
            if any(re.search(rg, col_norm) for rg in regs):
                return metric
        return ""

    for col in columns:
        raw = str(col)
        n = _norm_colname(raw)
        metric = metric_of(n)
        if not metric:
            continue

        # Prefer explicit subgroup headers in multi-index flatten format: "Group | metric".
        group_label = ""
        if "|" in raw:
            parts = [p.strip() for p in raw.split("|") if p.strip()]
            if parts:
                # Prefer the part explicitly containing 'group'
                grp_parts = [p for p in parts if re.search(r"\bgroup\b", p, flags=re.I)]
                if grp_parts:
                    group_label = grp_parts[0]
                elif len(parts) >= 2:
                    group_label = parts[-2]
                if group_label:
                    group_from_structured_header[group_label] = True
        if not group_label:
            gm = re.search(r"([A-Za-z0-9\-]+(?:\s*[-/]\s*[A-Za-z0-9]+)?\s+group)\b", raw, flags=re.I)
            if gm:
                group_label = gm.group(1)
                group_from_structured_header[group_label] = True
        if not group_label:
            group_label = re.sub(
                r"(p\s*-?value.*|p\s*meta.*|meta p.*|p\s*min.*|p\s*q.*|p\s*abs.*|95.*ci.*|confidence interval.*|odds ratio.*|\bor\b.*|\bbeta\b.*|β.*|\bhr\b.*|hazard ratio.*|z-?score.*|\bz\b.*|effect.*)$",
                "",
                raw,
                flags=re.I,
            ).strip(" -_:|")
        group_label = re.sub(r"\b(snp|variant|rsid|marker)\b.*$", "", group_label, flags=re.I).strip(" -_:|")

        group_label = re.sub(r"\s+", " ", group_label).strip()
        if not group_label:
            continue

        defs.setdefault(group_label, {})
        # For p-metrics, prioritize P meta over all others.
        if metric == "p" and re.search(r"p\s*meta|meta p", n, flags=re.I):
            defs[group_label]["p"] = col
        else:
            # keep first metric hit for stability
            defs[group_label].setdefault(metric, col)

    filtered: Dict[str, Dict[str, str]] = {}
    for group_label, metrics in defs.items():
        # Keep only real subgroup layouts:
        # - explicit structured headers (e.g. "male | OR", "Group A | P"), or
        # - groups that expose at least two metrics and contain a stat column.
        has_stats = any(k in metrics for k in ("effect", "ci", "p"))
        if group_from_structured_header.get(group_label):
            if has_stats:
                filtered[group_label] = metrics
            continue
        if len(metrics) >= 2 and has_stats and any(k in metrics for k in ("effect", "ci")):
            filtered[group_label] = metrics

    return filtered

def guess_table_type(df: pd.DataFrame) -> str:
    header = " ".join(df.iloc[0].astype(str).tolist()).lower() if len(df) > 0 else ""
    if "hazard" in header or "hr" in header:
        return "survival"
    if "or" in header and "p" in header:
        return "or_table"
    if "beta" in header or "β" in header:
        return "beta_table"
    return "unknown"


def extract_records_from_table(df: pd.DataFrame, table_idx: int, paper_id: str, pmcid: str, table_ref: str, table_link: str) -> List[Dict[str, Any]]:
    """
    Minimal heuristic: find rsIDs and extract nearby columns if recognizable.
    This will NOT be perfect for every format; it is meant to pre-fill + flag.
    """
    records = []
    body = _normalize_table(df)
    if body.empty:
        return records

    # Special layout: columns are grouped with "|" (e.g., IGAP/Exonic with ADGC/ADSP blocks).
    pipe_records = _extract_records_from_pipe_layout(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if pipe_records:
        return pipe_records

    igap_exonic_records = _extract_records_from_igap_exonic_split_table(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if igap_exonic_records:
        return igap_exonic_records

    survival_records = _extract_records_from_stratified_survival_table(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if survival_records:
        return survival_records

    apoe4_hazard_records = _extract_records_from_apoe4_hazard_table(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if apoe4_hazard_records:
        return apoe4_hazard_records

    inline_or_records = _extract_records_from_inline_subgroup_or_table(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if inline_or_records:
        return inline_or_records

    neuropath_records = _extract_records_from_neuropathology_multi_pvalue_table(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if neuropath_records:
        return neuropath_records

    combo_records = _extract_records_from_combination_pvalue_table(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if combo_records:
        return combo_records

    pjoint_records = _extract_records_from_pjoint_cognitive_table(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if pjoint_records:
        return pjoint_records

    eqtl_ad_pvalue_records = _extract_records_from_eqtl_ad_pvalue_table(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if eqtl_ad_pvalue_records:
        return eqtl_ad_pvalue_records

    eqtl_records = _extract_records_from_eqtl_summary_table(body, table_idx, paper_id, pmcid, table_ref, table_link)
    if eqtl_records:
        return eqtl_records

    columns = list(body.columns)
    table_group_label = _infer_group_label_from_columns(columns)
    col_mapping = map_columns_to_reference(columns, threshold=0.4)

    def mapped_col(ref_col: str) -> Optional[str]:
        candidates = [(c, meta.get("score", 0.0)) for c, meta in col_mapping.items() if meta.get("reference_col") == ref_col]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    rs_col = _pick_preferred_column(columns, ["snp (rs)", "variant (rs)", "snp all", "rs-id", "rsid", "snp", "marker"])
    if not rs_col:
        rs_col = mapped_col("TopSNP")
    exact_variant_cols = _find_columns(columns, ["variant"], exact=True)
    variant_col = exact_variant_cols[-1] if exact_variant_cols else None
    if not variant_col:
        variant_col = _pick_preferred_column(columns, ["variant all", "location and base pair change", "chr:position"])
    if not variant_col:
        variant_col = _pick_preferred_column(columns, ["variant"], ["location", "change"])
    if variant_col and rs_col and _norm_colname(variant_col) == _norm_colname(rs_col):
        explicit_variant = _pick_preferred_column(columns, ["variant"], None)
        if explicit_variant and _norm_colname(explicit_variant) != _norm_colname(rs_col):
            variant_col = explicit_variant
    if not variant_col:
        variant_col = rs_col or mapped_col("TopSNP")
    location_change_col = _pick_preferred_column(columns, ["location and base pair change", "chr:position"])
    chr_col = _pick_preferred_column(columns, ["chr:position", "chr", "chromosome"])
    if not chr_col:
        chr_col = mapped_col("Chr")
    bp_col = _pick_preferred_column(columns, ["position", "bp", "base pair", "pos"])
    if not bp_col:
        bp_col = mapped_col("BP(Position)")
    locus_col = _pick_preferred_column(columns, ["nearest gene", "closest gene", "nearestgene", "locusname", "locus name", "gene"])
    if not locus_col:
        locus_col = mapped_col("LocusName")
    maf_col = _pick_preferred_column(columns, ["eaf", "maf", "af"])
    if not maf_col:
        maf_col = mapped_col("ReportedAF(MAF)")
    cohort_col = _pick_preferred_column(columns, ["cohort", "consortium", "study", "dataset", "sample set"])
    imputation_col = _pick_preferred_column(columns, ["imputation", "reference panel", "panel"])
    population_col = _pick_preferred_column(columns, ["population", "ancestry", "ethnicity"])
    sample_size_col = _pick_preferred_column(columns, ["sample size", "total n", "n=", "participants", "subjects"])
    a1_col = _pick_preferred_column(columns, ["a1"], ["allele 1"])
    a2_col = _pick_preferred_column(columns, ["a2"], ["allele 2"])
    ea_oa_col = _pick_preferred_column(columns, ["ea/oa", "ea / oa", "effect allele/other allele"])
    a1a2maf_col = _pick_preferred_column(columns, ["a1/a2", "a1/a2 a (maf)", "a1/a2a (maf)"])
    major_minor_col = _pick_preferred_column(columns, ["major/ minor alleles", "major/minor", "alleles", "allele"])

    p_col = _pick_pvalue_column(columns)
    if not p_col:
        p_col = mapped_col("P-value")
    ci_col = _pick_preferred_column(columns, ["95", "ci"])
    if not ci_col:
        ci_col = mapped_col("95%ConfidenceInterval")

    # Effect type/value priority: OR > Beta > HR > Zscore
    effect_col, effect_type = _pick_effect_column(columns)

    # Interaction terms are uncommon in GWAS summary tables; still capture if present.
    interaction_col = _pick_preferred_column(columns, ["interaction", "snp x", "snp*"])
    subgroup_defs = _detect_subgroup_defs(columns)
    carry: Dict[str, str] = {
        "variant": "",
        "locus": "",
        "chr": "",
        "bp": "",
        "ra1": "",
        "ra2": "",
        "maf": "",
        "cohort": "",
        "imputation": "",
        "population": "",
        "sample_size": "",
    }

    for _, row in body.iterrows():
        row_vals = [_normalize_cell_text(x) for x in row.tolist()]
        row_text = " | ".join(row_vals)

        rsids = []
        rs_text = _normalize_cell_text(row.get(rs_col)) if rs_col else ""
        if rs_col:
            rsids.extend(RSID_RE.findall(rs_text))
        if not rsids:
            rsids = RSID_RE.findall(row_text)

        variant_text = _normalize_cell_text(row.get(variant_col)) if variant_col else ""
        locus_name = _normalize_cell_text(row.get(locus_col)) if locus_col else ""
        chr_raw = _normalize_cell_text(row.get(chr_col)) if chr_col else ""
        bp_raw = _normalize_cell_text(row.get(bp_col)) if bp_col else ""
        chr_val, bp_val = _parse_chr_bp(chr_raw, bp_raw)
        location_change_text = _normalize_cell_text(row.get(location_change_col)) if location_change_col else ""
        parsed_chr2, parsed_bp2, parsed_ra1, parsed_ra2 = _parse_chr_bp_and_alleles(location_change_text or variant_text or bp_raw or chr_raw)
        if parsed_chr2 and not chr_val:
            chr_val = parsed_chr2
        if parsed_bp2 and not bp_val:
            bp_val = parsed_bp2
        maf_val = _normalize_cell_text(row.get(maf_col)) if maf_col else ""
        row_cohort_hint = _normalize_cell_text(row.get(cohort_col)) if cohort_col else ""
        row_imputation_hint = _normalize_cell_text(row.get(imputation_col)) if imputation_col else ""
        row_population_hint = _normalize_cell_text(row.get(population_col)) if population_col else ""
        row_sample_size_hint = _normalize_cell_text(row.get(sample_size_col)) if sample_size_col else ""
        interaction_val = _normalize_cell_text(row.get(interaction_col)) if interaction_col else ""

        # keep rows with actionable stats even when rsID is absent
        pval_default = safe_float(row.get(p_col)) if p_col else None
        effect_default = safe_float(row.get(effect_col)) if effect_col else None
        ci_default = _normalize_cell_text(row.get(ci_col)) if ci_col else ""
        if effect_col:
            parsed_effect, parsed_ci = _extract_effect_and_ci(_normalize_cell_text(row.get(effect_col)), effect_type)
            if parsed_effect is not None:
                effect_default = parsed_effect
            if parsed_ci and (not ci_default or ci_col == effect_col):
                ci_default = parsed_ci
        if effect_default is None and ci_default and effect_type in ("OR", "HR"):
            parsed_effect, parsed_ci = _extract_effect_and_ci(ci_default, effect_type)
            if parsed_effect is not None:
                effect_default = parsed_effect
            if parsed_ci:
                ci_default = parsed_ci
        if a1a2maf_col:
            ra1_minor, ra2_major, maf_from_a1a2 = _split_a1a2_maf(_normalize_cell_text(row.get(a1a2maf_col)))
            if maf_from_a1a2:
                maf_val = maf_from_a1a2
        elif ea_oa_col:
            ra1_minor, ra2_major = _split_ea_oa(_normalize_cell_text(row.get(ea_oa_col)))
        elif a1_col and a2_col:
            ra1_minor = _normalize_cell_text(row.get(a1_col))
            ra2_major = _normalize_cell_text(row.get(a2_col))
        else:
            ra1_minor, ra2_major = _split_major_minor(_normalize_cell_text(row.get(major_minor_col)) if major_minor_col else "")
        if parsed_ra1 and not ra1_minor:
            ra1_minor = parsed_ra1
        if parsed_ra2 and not ra2_major:
            ra2_major = parsed_ra2

        if _is_probable_section_header_row(row_vals, row_text, rsids):
            continue

        rs_text = _sanitize_topsnp(rs_text)
        variant_text = _sanitize_topsnp(variant_text)
        locus_name = _sanitize_locus_name(locus_name)

        # Rows with only a group label are visual separators in PMC tables.
        # Skip before forward-fill so they do not clone the previous SNP/stat row.
        if not rsids and pval_default is None and effect_default is None and not chr_val and not bp_val and not locus_name:
            continue

        # Forward-fill common merged-cell fields.
        if variant_text:
            carry["variant"] = variant_text
        else:
            variant_text = carry["variant"]
        if locus_name:
            carry["locus"] = locus_name
        else:
            locus_name = carry["locus"]
        if chr_val:
            carry["chr"] = chr_val
        else:
            chr_val = carry["chr"]
        if bp_val:
            carry["bp"] = bp_val
        else:
            bp_val = carry["bp"]
        if ra1_minor:
            carry["ra1"] = ra1_minor
        else:
            ra1_minor = carry["ra1"]
        if ra2_major:
            carry["ra2"] = ra2_major
        else:
            ra2_major = carry["ra2"]
        if maf_val:
            carry["maf"] = maf_val
        else:
            maf_val = carry["maf"]
        if row_cohort_hint:
            carry["cohort"] = row_cohort_hint
        else:
            row_cohort_hint = carry["cohort"]
        if row_imputation_hint:
            carry["imputation"] = row_imputation_hint
        else:
            row_imputation_hint = carry["imputation"]
        if row_population_hint:
            carry["population"] = row_population_hint
        else:
            row_population_hint = carry["population"]
        if row_sample_size_hint:
            carry["sample_size"] = row_sample_size_hint
        else:
            row_sample_size_hint = carry["sample_size"]

        # Skip section headers and non-data rows.
        if not rsids and pval_default is None and effect_default is None and not chr_val and not bp_val:
            continue

        names = sorted(set([r.lower() for r in rsids])) if rsids else ["NR"]
        subgroup_records = []
        if subgroup_defs:
            for sub_label, metric_cols in subgroup_defs.items():
                sub_snp_text = _normalize_cell_text(row.get(metric_cols["snp"])) if metric_cols.get("snp") else ""
                sub_rsids = sorted(set([r.lower() for r in RSID_RE.findall(sub_snp_text)])) if sub_snp_text else []
                sp = safe_float(row.get(metric_cols["p"])) if metric_cols.get("p") else None
                subgroup_effect_type = _effect_type_from_colname(metric_cols["effect"]) if metric_cols.get("effect") else effect_type
                se = safe_float(row.get(metric_cols["effect"])) if metric_cols.get("effect") else None
                sci = _normalize_cell_text(row.get(metric_cols["ci"])) if metric_cols.get("ci") else ""
                if metric_cols.get("effect"):
                    parsed_effect, parsed_ci = _extract_effect_and_ci(_normalize_cell_text(row.get(metric_cols["effect"])), subgroup_effect_type)
                    if parsed_effect is not None:
                        se = parsed_effect
                    if parsed_ci and not sci:
                        sci = parsed_ci
                if se is None and sci and subgroup_effect_type in ("OR", "HR"):
                    parsed_effect, parsed_ci = _extract_effect_and_ci(sci, subgroup_effect_type)
                    if parsed_effect is not None:
                        se = parsed_effect
                    if parsed_ci:
                        sci = parsed_ci
                if sp is None and se is None:
                    continue
                subgroup_records.append({
                    "label": sub_label,
                    "sub_snp_text": _sanitize_topsnp(sub_snp_text),
                    "sub_rsids": sub_rsids,
                    "p": sp,
                    "effect": se,
                    "ci": sci,
                    "effect_type": subgroup_effect_type,
                })
        if not subgroup_records:
            subgroup_records = [{
                "label": "",
                "p": pval_default,
                "effect": effect_default,
                "ci": ci_default,
                "effect_type": effect_type,
            }]

        for sub in subgroup_records:
            sub_names = sub.get("sub_rsids") or names
            for rsid in sub_names:
                snp_for_sub = _sanitize_topsnp(sub.get("sub_snp_text") or rs_text or variant_text or (rsid if rsid != "NR" else ""))
                rec = {k: "" for k in CURATED_COLUMNS}
                rec["Name"] = rsid if rsid != "NR" else (snp_for_sub or "")
                rec["PaperIDX"] = paper_id
                rec["PMCID"] = pmcid
                rec["TableIDX"] = f"T{table_idx:05d}"
                rec["Table Ref in paper"] = table_ref
                rec["Table links"] = table_link
                rec["TopSNP"] = snp_for_sub
                rec["SNP-based, Gene-based"] = "SNP-based" if (rsid != "NR" or snp_for_sub.lower().startswith("rs")) else ("Gene-based" if locus_name else "")
                rec["Interactions"] = interaction_val
                rec["Chr"] = chr_val
                rec["P-value"] = sub["p"] if sub["p"] is not None else ""
                rec["BP(Position)"] = bp_val
                rec["RA 1(Reported Allele 1)"] = ra1_minor
                rec["RA 2(Reported Allele 2)"] = ra2_major
                rec["ReportedAF(MAF)"] = maf_val
                rec["Effect Size Type (OR or Beta)"] = sub["effect_type"] if sub["effect_type"] else "NR"
                rec["EffectSize(altvsref)"] = sub["effect"] if sub["effect"] is not None else ""
                rec["95%ConfidenceInterval"] = sub["ci"] if sub["ci"] else ""
                rec["LocusName"] = _sanitize_locus_name(locus_name)
                rec["_row_cohort_hint"] = row_cohort_hint
                rec["_row_imputation_hint"] = row_imputation_hint
                rec["_row_population_hint"] = row_population_hint
                rec["_row_sample_size_hint"] = row_sample_size_hint
                rec["Analysis group"] = table_group_label or sub["label"]
                rec["Phenotype-derived"] = sub["label"].split("(")[-1].replace(")", "").strip() if sub["label"] else rec.get("Phenotype-derived", "")
                rec["_confidence"] = 0.55 if rsid != "NR" else 0.4
                low_map = [f"{c}:{m['score']}" for c, m in col_mapping.items() if m.get("needs_review")]
                rec["_needs_review"] = bool(low_map) or (rsid == "NR")
                rec["_evidence"] = (
                    f"Row text: {row_text[:340]}\n"
                    f"[column_mapping] {json.dumps(col_mapping, ensure_ascii=False)[:700]}"
                )
                rec["label"] = sub["label"] # add the label as the clue to map from global info to local info
                records.append(rec)

    return records


# -----------------------------
# Export to Excel
# -----------------------------
def write_curated_xlsx(records: List[Dict[str, Any]], out_path: str, template_xlsx: Optional[str] = None) -> None:
    headers = CURATED_COLUMNS
    df = pd.DataFrame(records)
    if "TopSNP" in df.columns:
        df["TopSNP"] = df["TopSNP"].map(_sanitize_topsnp)
    if "Chr" in df.columns:
        df["Chr"] = df["Chr"].map(_format_integer_like)
    if "BP(Position)" in df.columns:
        df["BP(Position)"] = df["BP(Position)"].map(_format_integer_like)
    if "LocusName" in df.columns:
        df["LocusName"] = df["LocusName"].map(_sanitize_locus_name)
    # Ensure all headers exist
    for h in headers:
        if h not in df.columns:
            df[h] = "NR"
    df = df[headers]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="curated")


def _is_missing_value(value: Any) -> bool:
    s = _normalize_cell_text(value)
    return s == "" or s.upper() == "NR"


def _cell_issue(field: str, value: Any) -> Tuple[str, List[str]]:
    if _is_missing_value(value):
        required = field in {"TopSNP", "P-value"}
        return ("issue" if required else "review", ["missing required value" if required else "missing value"])

    s = _normalize_cell_text(value)
    issues: List[str] = []
    status = "ok"
    if field == "TopSNP" and not re.search(r"\brs\d+\b", s, flags=re.I):
        issues.append("SNP does not look like an rsID")
        status = "issue"
    elif field == "P-value":
        v = safe_float(s)
        if v is None:
            issues.append("p-value is not numeric")
            status = "issue"
        elif v < 0 or v > 1:
            issues.append("p-value is outside 0-1")
            status = "issue"
    elif field == "EffectSize(altvsref)" and safe_float(s) is None:
        issues.append("effect size is not numeric")
        status = "review"
    elif field == "Chr":
        chrom = _format_integer_like(s).lower().replace("chr", "").strip()
        if not re.match(r"^(?:[1-9]|1\d|2[0-2]|x|y|mt|m)$", chrom):
            issues.append("chromosome format is unusual")
            status = "review"
    elif field == "BP(Position)":
        v = safe_float(s)
        if v is None or v <= 0:
            issues.append("position is not a positive number")
            status = "review"
    elif field == "RA 1(Reported Allele 1)":
        alleles = {part.strip().upper() for part in re.split(r"[/;,|]", s) if part.strip()}
        if alleles and not alleles.issubset({"A", "C", "G", "T", "I", "D"}):
            issues.append("allele format is unusual")
            status = "review"

    return status, issues


def build_validation_report(records: List[Dict[str, Any]], paper_id: str) -> Dict[str, Any]:
    key_fields = [
        "TopSNP",
        "P-value",
        "EffectSize(altvsref)",
        "Chr",
        "BP(Position)",
        "RA 1(Reported Allele 1)",
        "Cohort_simplified (no counts)",
        "Sample size",
        "Imputation_simple2",
        "Population_map",
        "Analysis group",
        "Phenotype",
    ]
    rows = []
    counts = {"success": 0, "warning": 0, "failed": 0}
    for idx, record in enumerate(records):
        field_reports: Dict[str, Any] = {}
        row_issues: List[str] = []
        has_issue = False
        has_review = bool(record.get("_needs_review"))

        for field in key_fields:
            status, issues = _cell_issue(field, record.get(field))
            if status != "ok":
                field_reports[field] = {
                    "status": status,
                    "issues": issues,
                    "value": _normalize_cell_text(record.get(field)),
                    "evidence": _normalize_cell_text(record.get("_evidence", ""))[:500],
                }
                row_issues.extend([f"{field}: {issue}" for issue in issues])
            if status == "issue":
                has_issue = True
            elif status == "review":
                has_review = True

        pvalue_note = _normalize_cell_text(record.get("P-value note", ""))
        if not pvalue_note and "P-value" in record and re.search(r"p[- ]?value\s*\d+", _normalize_cell_text(record.get("_evidence", "")), flags=re.I):
            field_reports.setdefault("P-value", {
                "status": "review",
                "issues": [],
                "value": _normalize_cell_text(record.get("P-value")),
                "evidence": _normalize_cell_text(record.get("_evidence", ""))[:500],
            })
            field_reports["P-value"]["issues"].append("multiple p-value columns may need a note")
            row_issues.append("P-value: multiple p-value columns may need a note")
            has_review = True

        row_status = "failed" if has_issue else ("warning" if has_review else "success")
        counts[row_status] += 1
        rows.append({
            "row_index": idx,
            "record_id": record.get("RecordID", ""),
            "row_status": row_status,
            "row_issues": row_issues,
            "fields": field_reports,
        })

    return {
        "paper_id": paper_id,
        "schema_version": "self_validation_v1",
        "summary": counts,
        "rows": rows,
    }


def write_validation_report(records: List[Dict[str, Any]], validation_json: str, paper_id: str) -> None:
    report = build_validation_report(records, paper_id)
    with open(validation_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(pdf_or_url: Optional[str],
                 out_xlsx: str,
                 audit_json: str,
                 paper_id: str = "PAPER",
                 pmcid: Optional[str] = "NR",
                 template_xlsx: Optional[str] = None,
                 table_input: Optional[str] = None,
                 validation_json: Optional[str] = None,
                 gwas_information_retriever: Optional[air.ADVPInformationRetriever] = None,
                 gwas_information_retriever_keyword: Optional[air.ADVPInformationRetrieverKeyword] = None) -> None:
    paper_id_table_num = None
    m_pid = re.search(r"table[\s_-]?(\d+)$", paper_id or "", flags=re.I)
    if m_pid:
        paper_id_table_num = int(m_pid.group(1))

    # 1) Get PDF
    pdf_path = pdf_or_url or ""
    paper_source_errors: List[Dict[str, str]] = []
    online_full_text = ""
    online_fulltext_source = ""
    if pdf_or_url and pdf_or_url.lower().startswith("http"):
        if requests is None:
            raise RuntimeError("URL provided but requests not installed.")
        if pdf_or_url.lower().endswith(".pdf"):
            pdf_path = "input_paper.pdf"
            try:
                download_pdf(pdf_or_url, pdf_path)
            except Exception as e:
                paper_source_errors.append({"source": pdf_or_url, "error": repr(e)})
                pdf_path = ""
        else:
            pdf_path = ""
            pmcid_from_url = extract_pmcid_from_url(pdf_or_url)
            if pmcid_from_url:
                try:
                    online_full_text = xml_to_text(fetch_pmc_fulltext_xml(pmcid_from_url))
                    online_fulltext_source = "pmc_xml"
                except Exception as e:
                    paper_source_errors.append({"source": pdf_or_url, "error": repr(e)})
                    try:
                        online_full_text = fetch_pmc_article_text(pdf_or_url)
                        online_fulltext_source = "pmc_html"
                    except Exception as html_e:
                        paper_source_errors.append({"source": pdf_or_url, "error": repr(html_e)})
            else:
                paper_source_errors.append({"source": pdf_or_url, "error": "Unsupported non-PMC article URL"})

    # 2) Extract full text with fallback:
    # pdfplumber -> Docling -> OCR(+rotation), only when section signal is weak.
    if pdf_path:
        full_text, fulltext_source = extract_full_text_with_fallback(pdf_path)
    elif online_full_text:
        full_text, fulltext_source = online_full_text, online_fulltext_source
    else:
        full_text, fulltext_source = "", ("paper_unavailable" if paper_source_errors else "none")
    section_text = _sectionize_text(full_text)
    # ID policy: do not infer from paper text; use table filename only.
    resolved_pmid = infer_pmid_from_source_name(pdf_or_url, table_input)
    ids_evidence = f"PMID inferred from source filename -> {resolved_pmid}" if resolved_pmid != "NR" else ""
    # Per current requirement, keep PMCID empty.
    resolved_pmcid = ""
    section_scoped_text = "\n".join([
        section_text.get("methods", ""),
        section_text.get("results", ""),
        section_text.get("supplement", ""),
    ]).strip() or full_text
    # 3) Infer global fields (Stage / Model / Assoc type / Imputation / Population / Sample size)
    # not continue if we cannot even have pmid
    if not re.search(r"\d+", resolved_pmid):
        raise Exception("Pubmed ID must be included from table input (filename, pmid, pdf link)")
    
    # define if we use pmcid or pdf or url to extract text
    if pmcid is None or "PMC" not in pmcid:
        if re.search(r"(PMC\d+)", pdf_or_url):
            resolved_pmcid = re.findall(r"(PMC\d+)", pdf_or_url)[0]
            use_pmcid = True
        else:
            if ".pdf" not in pdf_or_url:
                raise Exception("Please provide either a PMCID, an url with PMCID, or a pdf file")
            use_pmcid = False
            resolved_pmcid = "NR"
    else:
        use_pmcid = True
        resolved_pmcid = pmcid
    # if gwas_information_retriever is None or gwas_information_retriever_keyword is None or resolved_pmid == "NR":
    #     stage_val, stage_audit = infer_stage(section_scoped_text)
    #     assoc_val, assoc_audit = infer_association_type(section_scoped_text)
    #     model_val, model_audit = infer_model_type(section_scoped_text)
    #     imp_val, imp_audit = infer_imputation(section_scoped_text)
    #     pop_val, pop_audit = infer_population(section_text)
    #     sample_size_val, sample_size_audit = infer_sample_size(section_text)
    #     imp_val = to_canonical_imputation_codes(imp_val)
    #     meta_joint = classify_meta_joint(stage_val)
    #     paper_hints = get_paper_metadata_hints(resolved_pmid)
    #     if imp_val == "NR" and paper_hints.get("Imputation"):
    #         imp_val = to_canonical_imputation_codes(paper_hints["Imputation"])
    #         imp_audit = FieldAudit(
    #             "Imputation", imp_val, 0.6,
    #             f"paper-level fallback metadata for PMID {resolved_pmid}",
    #             "paper metadata fallback", True
    #         )
    #     if pop_val == "NR" and paper_hints.get("Population"):
    #         pop_val = paper_hints["Population"]
    #         pop_audit = FieldAudit(
    #             "Population", pop_val, 0.6,
    #             f"paper-level fallback metadata for PMID {resolved_pmid}",
    #             "paper metadata fallback", True
    #         )    
    #     col_require_rag_to_possible_info = {
    #         "Stage": [stage_val],
    #         "Association Type": [assoc_val],
    #         "Model Type": [model_val],
    #         "Imputation": [imp_val],
    #         "Population": [pop_val],
    #         "Study Type": [],
    #         "Phenotype": [],
    #         "Cohort": [],
    #         "Sample Size": [sample_size_val]
    #     }
    # else:
    #     if use_pmcid:
    #         col_require_rag_to_possible_info = gwas_information_retriever.extract_possible_info_from_paper(int(resolved_pmid), resolved_pmcid)
    #     else:
    #         col_require_rag_to_possible_info = gwas_information_retriever.extract_possible_info_from_pdf_paper(int(resolved_pmid), pdf_or_url)
    #     col_require_rag_to_possible_info["Cohort"] = gwas_information_retriever_keyword.extract_possible_info_from_paper(int(resolved_pmid), resolved_pmcid)
    #     sample_size_val, sample_size_audit = infer_sample_size(section_text)
    #     col_require_rag_to_possible_info["Sample Size"] = [sample_size_val]
    #     # assoc_val, assoc_audit = infer_association_type(section_scoped_text)
    #     # col_require_rag_to_possible_info["Association Type"] = [assoc_val]
    #     # model_val, model_audit = infer_model_type(section_scoped_text)
    #     # col_require_rag_to_possible_info["Model Type"] = [model_val]
    #     stage_audit = FieldAudit("Stage", col_require_rag_to_possible_info["Stage"], 0.5, "", "", True)
    #     imp_audit = FieldAudit("Imputation", col_require_rag_to_possible_info["Imputation"], 0.5, "", "", True)
    #     pop_audit = FieldAudit("Population", col_require_rag_to_possible_info["Population"], 0.5, "", "", True)
    #     assoc_audit = FieldAudit("Association Type", col_require_rag_to_possible_info["Association Type"], 0.5, "", "", True)
    #     model_audit = FieldAudit("Model Type", col_require_rag_to_possible_info["Model Type"], 0.5, "", "", True)
    
    # NOTE: check if already in cache
    if str(resolved_pmid) in AIR_CACHE:
        col_require_rag_to_possible_info = AIR_CACHE[str(resolved_pmid)]
    else:
        if use_pmcid:
            col_require_rag_to_possible_info = gwas_information_retriever.extract_possible_info_from_paper(int(resolved_pmid), resolved_pmcid)
        else:
            col_require_rag_to_possible_info = gwas_information_retriever.extract_possible_info_from_pdf_paper(int(resolved_pmid), pdf_or_url)
        stage_val, _ = infer_stage(section_scoped_text)
        assoc_val, _ = infer_association_type(section_scoped_text)
        model_val, _ = infer_model_type(section_scoped_text)
        imp_val, _ = infer_imputation(section_scoped_text)
        pop_val, _ = infer_population(section_text)
        imp_val = to_canonical_imputation_codes(imp_val)
        paper_hints = get_paper_metadata_hints(resolved_pmid)
        if imp_val == "NR" and paper_hints.get("Imputation"):
            imp_val = to_canonical_imputation_codes(paper_hints["Imputation"])
            # imp_audit = FieldAudit(
            #     "Imputation", imp_val, 0.6,
            #     f"paper-level fallback metadata for PMID {resolved_pmid}",
            #     "paper metadata fallback", True
            # )
        if pop_val == "NR" and paper_hints.get("Population"):
            pop_val = paper_hints["Population"]
            # pop_audit = FieldAudit(
            #     "Population", pop_val, 0.6,
            #     f"paper-level fallback metadata for PMID {resolved_pmid}",
            #     "paper metadata fallback", True
            # ) 
        if stage_val != "NR" and stage_val not in col_require_rag_to_possible_info["Stage"]:
            col_require_rag_to_possible_info["Stage"].append(stage_val) 
            col_require_rag_to_possible_info["Stage"] = list(set([info.lower() for info in col_require_rag_to_possible_info["Stage"]]))
        if assoc_val != "NR" and assoc_val not in col_require_rag_to_possible_info["Association Type"]:
            col_require_rag_to_possible_info["Association Type"].append(assoc_val) 
            col_require_rag_to_possible_info["Association Type"] = list(set([info.lower() for info in col_require_rag_to_possible_info["Association Type"]]))
        if model_val != "NR" and model_val not in col_require_rag_to_possible_info["Model Type"]:
            col_require_rag_to_possible_info["Model Type"].append(model_val) 
            col_require_rag_to_possible_info["Model Type"] = list(set([info.lower() for info in col_require_rag_to_possible_info["Model Type"]]))
        if imp_val != "NR" and imp_val not in col_require_rag_to_possible_info["Imputation"]:
            col_require_rag_to_possible_info["Imputation"].append(imp_val) 
            col_require_rag_to_possible_info["Imputation"] = list(set([info.lower() for info in col_require_rag_to_possible_info["Imputation"]]))
        if pop_val != "NR" and pop_val not in col_require_rag_to_possible_info["Population"]:
            col_require_rag_to_possible_info["Population"].append(pop_val) 
            col_require_rag_to_possible_info["Population"] = list(set([info.lower() for info in col_require_rag_to_possible_info["Population"]]))
        col_require_rag_to_possible_info["Cohort"] = col_require_rag_to_possible_info["Cohort"] + gwas_information_retriever_keyword.extract_possible_info_from_paper(int(resolved_pmid), resolved_pmcid)
        col_require_rag_to_possible_info["Cohort"] = list(set([info.lower() for info in col_require_rag_to_possible_info["Cohort"]]))
        sample_size_val, _ = infer_sample_size(section_text)
        col_require_rag_to_possible_info["Sample Size"] = [sample_size_val]
        # assoc_val, assoc_audit = infer_association_type(section_scoped_text)
        # col_require_rag_to_possible_info["Association Type"] = [assoc_val]
        # model_val, model_audit = infer_model_type(section_scoped_text)
        # col_require_rag_to_possible_info["Model Type"] = [model_val]
        AIR_CACHE[str(resolved_pmid)] = col_require_rag_to_possible_info
        with open(AIR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(AIR_CACHE, f, indent=2)
    
    # NOTE: need to update more details info on audit
    stage_audit = FieldAudit("Stage", col_require_rag_to_possible_info["Stage"], 0.5, "", "", True)
    imp_audit = FieldAudit("Imputation", col_require_rag_to_possible_info["Imputation"], 0.5, "", "", True)
    pop_audit = FieldAudit("Population", col_require_rag_to_possible_info["Population"], 0.5, "", "", True)
    assoc_audit = FieldAudit("Association Type", col_require_rag_to_possible_info["Association Type"], 0.5, "", "", True)
    model_audit = FieldAudit("Model Type", col_require_rag_to_possible_info["Model Type"], 0.5, "", "", True)
    sample_size_audit = FieldAudit("Sample Size", col_require_rag_to_possible_info["Sample Size"], 0.5, "", "", True)

    # 4) Extract tables
    table_entries: List[Tuple[str, pd.DataFrame, int]] = []
    table_source_errors: List[Dict[str, str]] = []
    if table_input:
        for src in parse_table_sources(table_input):
            try:
                src_tables = extract_tables_from_source(src)
            except Exception as e:
                table_source_errors.append({"source": src, "error": repr(e)})
                continue
            m = re.search(r"table[\s_-]?(\d+)", os.path.basename(src), flags=re.I)
            src_table_num = int(m.group(1)) if m else None
            for j, tdf in enumerate(src_tables, start=1):
                label = f"{os.path.basename(src)}#{j}"
                if len(src_tables) == 1 and paper_id_table_num is not None:
                    logical_idx = paper_id_table_num
                else:
                    logical_idx = src_table_num if (src_table_num is not None and len(src_tables) == 1) else j
                table_entries.append((label, tdf, logical_idx))
    elif pdf_path:
        for j, tdf in enumerate(extract_tables(pdf_path, pages="all"), start=1):
            table_entries.append((f"Table {j}", tdf, j))

    # 5) Convert each table to records (heuristic)
    records: List[Dict[str, Any]] = []
    audits: Dict[str, Any] = {
        "paper": {
            "paper_id": paper_id,
            "pmcid": resolved_pmcid,
            "pmid": resolved_pmid,
            "pdf": pdf_path,
            "pdf_source": pdf_or_url or "",
            "table_input": table_input or "NR",
            "full_text_source": fulltext_source,
            "id_evidence": ids_evidence,
        },
        "paper_source_errors": paper_source_errors,
        "table_source_errors": table_source_errors,
        "global_field_audit": [
            asdict(stage_audit),
            asdict(assoc_audit),
            asdict(model_audit),
            asdict(imp_audit),
            asdict(pop_audit),
            asdict(sample_size_audit),
        ],
        "record_field_audit": []
    }

    if table_input and not table_entries:
        with open(audit_json, "w", encoding="utf-8") as f:
            json.dump(audits, f, indent=2, ensure_ascii=False)
        raise RuntimeError(
            "No tables extracted from --table_input. "
            f"See errors in audit: {audit_json}"
        )

    for _, (source_label, df, logical_idx) in enumerate(table_entries, start=1):
        table_ref = f"Table {logical_idx}"
        if table_input:
            table_link = source_label.split("#")[0]
        else:
            table_link = pdf_or_url if (pdf_or_url and pdf_or_url.lower().startswith("http")) else "local_pdf"

        recs = extract_records_from_table(df, logical_idx, paper_id, resolved_pmcid, table_ref, table_link)
        # NOTE: we will get col with text based on this process
        # - First extract all possible groups => map these groups to valid text col name or just map all of them
        possible_groups = set()
        for r in recs:
            if "label" in r:
                possible_groups.add(r['label'])
        possible_groups = list(possible_groups)
        if len(possible_groups) <= 1:
            for r in recs:
                for text_col in col_require_rag_to_possible_info:
                    r[text_col] = air.combine_possible_info(col_require_rag_to_possible_info[text_col])
        else:
            for text_col in col_require_rag_to_possible_info:
                if len(col_require_rag_to_possible_info[text_col]) == 0:
                    for r in recs:
                        r[text_col] = ""
                elif len(possible_groups) == 0:
                    for r in recs:
                        r[text_col] = air.combine_possible_info(col_require_rag_to_possible_info[text_col])
                else:
                    similarity_score = air.calculate_similarity_scores(col_require_rag_to_possible_info[text_col], possible_groups)
                    possible_groups_to_possible_info = {}
                    if torch.min(torch.max(similarity_score, dim = 0).values) < 0.4:
                        for i, u in enumerate(possible_groups):
                            possible_groups_to_possible_info[u] = air.combine_possible_info(col_require_rag_to_possible_info[text_col])
                    else:
                        # best_inx = torch.argmax(similarity_score, dim = 0)
                        # unique_value_to_possible_info = {}
                        # for i, u in enumerate(unique_value):
                        #     unique_value_to_possible_info[u] = col_to_possible_info[col][best_inx[i]]
                        for i, u in enumerate(possible_groups):
                            valid_info = [col_require_rag_to_possible_info[text_col][inx] for inx in range(similarity_score.shape[0]) if similarity_score[inx, i] >= 0.4]
                            possible_groups_to_possible_info[u] = air.combine_possible_info(valid_info)
                    for r in recs:
                        if "label" in r:
                            r[text_col] = possible_groups_to_possible_info[r["label"]]
                        else:
                            r[text_col] = air.combine_possible_info(col_require_rag_to_possible_info[text_col])

        for r in recs:
            # table_hints = get_paper_metadata_hints(resolved_pmid, logical_idx)
            # Apply global inferred values as pre-fill
            # r["Stage"] = stage_val
            # r["Stage_original"] = stage_val
            # r["Analyses type"] = assoc_val
            # existing_model = _normalize_cell_text(r.get("Model type", ""))
            # r["Model type"] = existing_model if existing_model else model_val
            # r["Meta/Joint"] = meta_joint
            # row_imp = to_canonical_imputation_codes(r.get("_row_imputation_hint", ""))
            # r["Imputation"] = row_imp if row_imp != "NR" else imp_val
            r["PMCID"] = resolved_pmcid if resolved_pmcid != "NR" else r.get("PMCID", "")
            r["Pubmed PMID"] = resolved_pmid if resolved_pmid != "NR" else r.get("Pubmed PMID", "")
            # row_pop = _normalize_cell_text(r.get("_row_population_hint", ""))
            # r["Population"] = row_pop if row_pop else pop_val
            # row_sample = _normalize_cell_text(r.get("_row_sample_size_hint", "")) or _normalize_cell_text(r.get("Sample size", ""))
            # r["Sample size"] = row_sample if row_sample else sample_size_val

            # cohort_val, cohort_conf, cohort_evidence, cohort_needs_review = infer_cohort_from_row_and_text(
            #     r, full_text, table_ref
            # )
            # row_hint = to_canonical_cohort_codes(r.get("_row_cohort_hint", ""))
            # inferred = to_canonical_cohort_codes(cohort_val)
            # merged_cohorts = []
            # for part in (row_hint, inferred):
            #     if part and part != "NR":
            #         merged_cohorts.extend([x for x in part.split(";") if x])
            # if merged_cohorts:
            #     dedup = []
            #     seen = set()
            #     for c in merged_cohorts:
            #         if c not in seen:
            #             dedup.append(c)
            #             seen.add(c)
            #     cohort_val = ";".join(dedup)
            # else:
            #     cohort_val = "NR"

            # existing_cohort = _normalize_cell_text(r.get("Cohort", ""))
            # fallback_cohort = table_hints.get("Cohort", "") if cohort_val == "NR" else ""
            # final_cohort = existing_cohort if existing_cohort else (cohort_val if cohort_val != "NR" else fallback_cohort or "NR")
            # r["Cohort"] = final_cohort
            # r["Cohort_simplified (no counts)"] = final_cohort if final_cohort != "NR" else ""

            # Conservative flags: if global inference uncertain, propagate review
            # global_conf = min(stage_audit.confidence, assoc_audit.confidence, model_audit.confidence, imp_audit.confidence, pop_audit.confidence, sample_size_audit.confidence)
            # r["_confidence"] = float(r.get("_confidence", 0.4)) * 0.4 + global_conf * 0.4 + cohort_conf * 0.2
            # if stage_audit.needs_review or assoc_audit.needs_review or model_audit.needs_review or imp_audit.needs_review or pop_audit.needs_review or sample_size_audit.needs_review or cohort_needs_review:
            #     r["_needs_review"] = True

            # Add short evidence
            # r["_evidence"] = (r.get("_evidence", "") + "\n" +
            #                  f"[Stage evidence] {stage_audit.evidence[:240]}\n" +
            #                  f"[Model evidence] {model_audit.evidence[:240]}\n" +
            #                  f"[Imputation evidence] {imp_audit.evidence[:240]}\n" +
            #                  f"[Population evidence] {pop_audit.evidence[:240]}\n" +
            #                  f"[Sample size evidence] {sample_size_audit.evidence[:240]}\n" +
            #                  f"[Cohort evidence] {cohort_evidence[:240]}\n" +
            #                  f"[ID evidence] {ids_evidence[:240]}").strip()

            records.append(r)

    # Stable IDs for downstream curation and dedupe
    for i, r in enumerate(records, start=1):
        if not r.get("RecordID") or r.get("RecordID") == "NR":
            r["RecordID"] = f"{paper_id}_R{i:05d}"

        audits["record_field_audit"].append({
            "RecordID": r["RecordID"],
            "fields": [
                asdict(FieldAudit("Name", r.get("Name", "NR"), 0.9 if r.get("Name", "NR") != "NR" else 0.2,
                                  r.get("_evidence", "")[:260], "table row rsID extraction", r.get("Name", "NR") == "NR")),
                asdict(FieldAudit("P-value", r.get("P-value", "NR"), 0.7 if r.get("P-value", "NR") != "NR" else 0.3,
                                  r.get("_evidence", "")[:260], "table row p-value parsing", r.get("P-value", "NR") == "NR")),
                asdict(FieldAudit("EffectSize(altvsref)", r.get("EffectSize(altvsref)", "NR"),
                                  0.65 if r.get("EffectSize(altvsref)", "NR") != "NR" else 0.3,
                                  r.get("_evidence", "")[:260], "table row effect parsing", r.get("EffectSize(altvsref)", "NR") == "NR")),
                asdict(FieldAudit("Stage", r.get("Stage", "NR"), stage_audit.confidence,
                                  stage_audit.evidence[:260], stage_audit.rule, stage_audit.needs_review)),
                asdict(FieldAudit("Model type", r.get("Model type", "NR"), model_audit.confidence,
                                  model_audit.evidence[:260], model_audit.rule, model_audit.needs_review)),
                asdict(FieldAudit("Imputation", r.get("Imputation", "NR"), imp_audit.confidence,
                                  imp_audit.evidence[:260], imp_audit.rule, imp_audit.needs_review)),
                asdict(FieldAudit("Population", r.get("Population", "NR"), pop_audit.confidence,
                                  pop_audit.evidence[:260], pop_audit.rule, pop_audit.needs_review)),
                asdict(FieldAudit("Sample size", r.get("Sample size", "NR"), sample_size_audit.confidence,
                                  sample_size_audit.evidence[:260], sample_size_audit.rule, sample_size_audit.needs_review)),
                asdict(FieldAudit("Cohort", r.get("Cohort", "NR"),
                                  0.8 if r.get("Cohort", "NR") not in ("", "NR") else 0.25,
                                  r.get("_evidence", "")[:260], "cohort keyword mapping", r.get("Cohort", "NR") in ("", "NR"))),
            ]
        })

    # 6) If no tables extracted, still output a shell row for manual work
    if not records:
        shell = {k: "NR" for k in CURATED_COLUMNS}
        shell["PaperIDX"] = paper_id
        shell["RecordID"] = f"{paper_id}_R00001"
        shell["PMCID"] = resolved_pmcid
        shell["Pubmed PMID"] = resolved_pmid
        shell["Stage"] = col_require_rag_to_possible_info["Stage"]
        shell["Analyses type"] = col_require_rag_to_possible_info["Association Type"]
        shell["Model type"] = col_require_rag_to_possible_info["Model Type"]
        shell["Imputation"] = col_require_rag_to_possible_info["Imputation"]
        shell["Population"] = col_require_rag_to_possible_info["Population"]
        shell["Sample size"] = col_require_rag_to_possible_info["Sample Size"]
        # shell["_confidence"] = min(stage_audit.confidence, assoc_audit.confidence, model_audit.confidence, imp_audit.confidence, pop_audit.confidence, sample_size_audit.confidence)
        # shell["_needs_review"] = True
        # shell["_evidence"] = (
        #     "No extractable association records from tables. "
        #     f"Table entries={len(table_entries)}; source_errors={len(table_source_errors)}. "
        #     "Use audit json for details."
        # )
        records.append(shell)

    # 7) Write outputs
    write_curated_xlsx(records, out_xlsx, template_xlsx=template_xlsx)
    if validation_json:
        write_validation_report(records, validation_json, paper_id)

    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2, ensure_ascii=False)


def main():
    def _clean_user_input(s: str) -> str:
        s = (s or "").strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1].strip()
        return s
    
    referencing_col_require_rag_df = pd.read_csv("ADVP context required col.csv")
    gwas_information_retriever = air.ADVPInformationRetriever(referencing_col_require_rag_df)

    # for cohort try to do keyword
    # Cohort,"The specific study or database name. Keywords: Study, Dataset, Discovery. Examples: ADNI, IGAP, UK Biobank, ADGC, CHARGE, EADI.",
    with open("cohort_keywords.json", "r") as f:
        cohort_keyword_dict = json.load(f)
    gwas_information_retriever_keyword = air.ADVPInformationRetrieverKeyword("Cohort", cohort_keyword_dict)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, default=None, help="PDF path or URL")
    parser.add_argument("--table_input", default=None, help="Table source(s): URL/path; comma-separated for multiple")
    parser.add_argument("--out", default="curated_output.xlsx")
    parser.add_argument("--audit", default="audit.json")
    parser.add_argument("--validation", default=None, help="Optional row/cell validation report JSON for UI review")
    parser.add_argument("--paper_id", default="PAPER")
    parser.add_argument("--pmcid", default="NR")
    parser.add_argument("--template", default=None, help="Optional template xlsx (first row as headers)")
    args = parser.parse_args()

    # Interactive fallback: prompt when key args are missing.
    if not args.input and not args.table_input:
        src = _clean_user_input(input("Input source path/URL (PDF or table file/URL): "))
        if not src:
            raise ValueError("Input source is required.")
        src_low = src.lower()
        if src_low.endswith(".pdf"):
            args.input = src
        elif src_low.endswith((".xlsx", ".xls", ".csv", ".tsv", ".html", ".htm")):
            args.table_input = src
        else:
            mode_hint = _clean_user_input(input("Treat source as table input? [Y/n]: "))
            # Normalize common full-width answers (e.g., Ｙ/Ｎ)
            mode_hint = mode_hint.translate(str.maketrans({"Ｙ": "Y", "ｙ": "y", "Ｎ": "N", "ｎ": "n"})).lower()
            if mode_hint in ("", "y", "yes"):
                args.table_input = src
            else:
                args.input = src

    if not args.out:
        out_val = _clean_user_input(input("Output xlsx path [curated_output.xlsx]: "))
        args.out = out_val or "curated_output.xlsx"
    elif args.out == "curated_output.xlsx":
        out_val = _clean_user_input(input("Output xlsx path [curated_output.xlsx]: "))
        args.out = out_val or args.out

    if not args.paper_id or args.paper_id == "PAPER":
        pid = _clean_user_input(input("paper_id [PAPER]: "))
        args.paper_id = pid or "PAPER"

    if not args.audit or args.audit == "audit.json":
        auto_audit = f"{args.paper_id}_audit.json" if args.paper_id else "audit.json"
        audit_val = _clean_user_input(input(f"Audit json path [{auto_audit}]: "))
        args.audit = audit_val or auto_audit

    # Safety: if user accidentally sets --input to a table file, reroute it.
    if args.input and args.input.lower().endswith((".xlsx", ".xls", ".csv", ".tsv", ".html", ".htm")) and not args.table_input:
        args.table_input = args.input
        args.input = None

    run_pipeline(
        pdf_or_url=args.input,
        out_xlsx=args.out,
        audit_json=args.audit,
        paper_id=args.paper_id,
        pmcid=args.pmcid,
        template_xlsx=args.template,
        table_input=args.table_input,
        validation_json=args.validation,
        gwas_information_retriever=gwas_information_retriever,
        gwas_information_retriever_keyword=gwas_information_retriever_keyword
    )


if __name__ == "__main__":
    main()
