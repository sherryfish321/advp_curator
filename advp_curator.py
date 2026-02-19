import re
import json
import argparse
import os
import difflib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd

# Optional dependencies (script will degrade gracefully)
try:
    import requests
except Exception:
    requests = None

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
    "Imputation_simple2": "Imputation panel/method. Example: HRC;TOPMed",
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
    "Sample size", "Cases", "Controls", "Sample information", "Imputation_simple2",
    "Population", "Population_map", "Analysis group", "Phenotype", "Phenotype-derived",
    "For plotting Beta and OR - derived", "Reported gene (gene based test)",
    "TopSNP", "Interactions", "Chr", "P-value", "BP(Position)",
    "RA 1(Reported Allele 1)", "RA 2(Reported Allele 2)", "Note on alleles and AF",
    "ReportedAF(MAF)", "AFincases", "AFincontrols", "Effect Size Type (OR or Beta)",
    "EffectSize(altvsref)", "95%ConfidenceInterval",
    "Confirmed affected genes, causal variants, evidence",
    "Genome build (hg18/hg37/hg38)", "Pubmed PMID", "PMCID",
    "Table Ref in paper", "Table links", "LocusName"
]


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
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


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
        return value, FieldAudit("Imputation_simple2", value, conf, evidence, rule, needs_review)

    value = "NR"
    return value, FieldAudit("Imputation_simple2", value, 0.3, (snippets[0] if snippets else ""), "no imputation keywords found", True)


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
    return c, p


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
            if (not row_gene) and ("closest gene" in lc or "nearest gene" in lc):
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
                rec["Analysis group"] = re.sub(r"\s+[a-z]$", "", re.sub(r"\s+", " ", str(subgroup)).strip(), flags=re.I)
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


def _extract_effect_and_ci(effect_text: str, effect_type: str) -> Tuple[Optional[float], str]:
    """
    Parse patterns like:
    - OR/HR: '1.26 (0.83-1.92)' -> effect=1.26, ci='(0.83-1.92)'
    - Beta:  '0.164 (0.04)'     -> effect=0.164, ci=''
    """
    s = _normalize_cell_text(effect_text).replace("−", "-")
    if not s:
        return None, ""

    # first numeric token = effect value
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    effect = float(m.group(0)) if m else None

    ci = ""
    paren = re.search(r"\(([^)]+)\)", s)
    if paren and effect_type in ("OR", "HR"):
        inner = paren.group(1).strip()
        if "–" in inner or "-" in inner:
            ci = f"({inner})"
    return effect, ci


def _effect_type_from_colname(colname: str) -> str:
    n = _norm_colname(colname)
    if "beta" in n or "β" in n:
        return "Beta"
    if re.search(r"\bor\b", n) or "odds ratio" in n:
        return "OR"
    if re.search(r"\bhr\b", n) or "hazard ratio" in n:
        return "HR"
    if "z-score" in n or "zscore" in n:
        return "Zscore"
    return "NR"


def _detect_subgroup_defs(columns: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Detect melt-able subgroup metric columns from broad naming patterns.
    Expected output:
      { "<group label>": {"p": col, "effect": col, "ci": col}, ... }
    """
    defs: Dict[str, Dict[str, str]] = {}

    metric_tokens = {
        "p": [r"\bp\b", r"p value", r"p-value", r"meta p"],
        "ci": [r"95", r"\bci\b", r"confidence interval"],
        "effect": [r"\bor\b", r"odds ratio", r"\bbeta\b", r"β", r"\bhr\b", r"hazard ratio", r"z-score", r"effect"],
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
            if len(parts) >= 2:
                group_label = parts[-2]
        if not group_label:
            group_label = re.sub(
                r"(p\s*-?value.*|meta p.*|95.*ci.*|confidence interval.*|odds ratio.*|\bor\b.*|\bbeta\b.*|β.*|\bhr\b.*|hazard ratio.*|z-?score.*|effect.*)$",
                "",
                raw,
                flags=re.I,
            ).strip(" -_:|")

        group_label = re.sub(r"\s+", " ", group_label).strip()
        if not group_label:
            continue

        defs.setdefault(group_label, {})
        # keep first metric hit for stability
        defs[group_label].setdefault(metric, col)

    return defs

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

    columns = list(body.columns)
    col_mapping = map_columns_to_reference(columns, threshold=0.4)

    def mapped_col(ref_col: str) -> Optional[str]:
        candidates = [(c, meta.get("score", 0.0)) for c, meta in col_mapping.items() if meta.get("reference_col") == ref_col]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    variant_col = _pick_preferred_column(columns, ["snp all", "variant all", "variant", "rsid", "snp", "marker"])
    if not variant_col:
        variant_col = mapped_col("TopSNP")
    rs_col = variant_col
    chr_col = _pick_preferred_column(columns, ["chr:position", "chr", "chromosome"])
    if not chr_col:
        chr_col = mapped_col("Chr")
    bp_col = _pick_preferred_column(columns, ["position", "bp", "base pair", "pos"])
    if not bp_col:
        bp_col = mapped_col("BP(Position)")
    locus_col = _pick_preferred_column(columns, ["nearest gene", "closest gene", "nearestgene", "locusname", "locus name"])
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
        if rs_col:
            rsids.extend(RSID_RE.findall(_normalize_cell_text(row.get(rs_col))))
        if not rsids:
            rsids = RSID_RE.findall(row_text)

        variant_text = _normalize_cell_text(row.get(variant_col)) if variant_col else ""
        locus_name = _normalize_cell_text(row.get(locus_col)) if locus_col else ""
        chr_raw = _normalize_cell_text(row.get(chr_col)) if chr_col else ""
        bp_raw = _normalize_cell_text(row.get(bp_col)) if bp_col else ""
        chr_val, bp_val = _parse_chr_bp(chr_raw, bp_raw)
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
            if parsed_ci and not ci_default:
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

        for rsid in names:
            for sub in subgroup_records:
                rec = {k: "" for k in CURATED_COLUMNS}
                rec["Name"] = rsid if rsid != "NR" else (variant_text or "")
                rec["PaperIDX"] = paper_id
                rec["PMCID"] = pmcid
                rec["TableIDX"] = f"T{table_idx:05d}"
                rec["Table Ref in paper"] = table_ref
                rec["Table links"] = table_link
                rec["TopSNP"] = variant_text if variant_text else (rsid if rsid != "NR" else "")
                rec["SNP-based, Gene-based"] = "SNP-based" if (rsid != "NR" or variant_text.lower().startswith("rs")) else ("Gene-based" if locus_name else "")
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
                rec["LocusName"] = locus_name
                rec["_row_cohort_hint"] = row_cohort_hint
                rec["_row_imputation_hint"] = row_imputation_hint
                rec["_row_population_hint"] = row_population_hint
                rec["_row_sample_size_hint"] = row_sample_size_hint
                rec["Analysis group"] = sub["label"]
                rec["Phenotype-derived"] = sub["label"].split("(")[-1].replace(")", "").strip() if sub["label"] else rec.get("Phenotype-derived", "")
                rec["_confidence"] = 0.55 if rsid != "NR" else 0.4
                low_map = [f"{c}:{m['score']}" for c, m in col_mapping.items() if m.get("needs_review")]
                rec["_needs_review"] = bool(low_map) or (rsid == "NR")
                rec["_evidence"] = (
                    f"Row text: {row_text[:340]}\n"
                    f"[column_mapping] {json.dumps(col_mapping, ensure_ascii=False)[:700]}"
                )
                records.append(rec)

    return records


# -----------------------------
# Export to Excel
# -----------------------------
def write_curated_xlsx(records: List[Dict[str, Any]], out_path: str, template_xlsx: Optional[str] = None) -> None:
    headers = CURATED_COLUMNS
    df = pd.DataFrame(records)
    # Ensure all headers exist
    for h in headers:
        if h not in df.columns:
            df[h] = "NR"
    df = df[headers]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="curated")


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(pdf_or_url: Optional[str],
                 out_xlsx: str,
                 audit_json: str,
                 paper_id: str = "PAPER",
                 pmcid: str = "NR",
                 template_xlsx: Optional[str] = None,
                 table_input: Optional[str] = None) -> None:

    # 1) Get PDF
    pdf_path = pdf_or_url or ""
    if pdf_or_url and pdf_or_url.lower().startswith("http"):
        if requests is None:
            raise RuntimeError("URL provided but requests not installed.")
        pdf_path = "input_paper.pdf"
        download_pdf(pdf_or_url, pdf_path)

    # 2) Extract full text with fallback:
    # pdfplumber -> Docling -> OCR(+rotation), only when section signal is weak.
    if pdf_path:
        full_text, fulltext_source = extract_full_text_with_fallback(pdf_path)
    else:
        full_text, fulltext_source = "", "none"
    section_text = _sectionize_text(full_text)
    section_scoped_text = "\n".join([
        section_text.get("methods", ""),
        section_text.get("results", ""),
        section_text.get("supplement", ""),
    ]).strip() or full_text

    # 3) Infer global fields (Stage / Model / Assoc type / Imputation / Population / Sample size)
    stage_val, stage_audit = infer_stage(section_scoped_text)
    assoc_val, assoc_audit = infer_association_type(section_scoped_text)
    model_val, model_audit = infer_model_type(section_scoped_text)
    imp_val, imp_audit = infer_imputation(section_scoped_text)
    pop_val, pop_audit = infer_population(section_text)
    sample_size_val, sample_size_audit = infer_sample_size(section_text)
    imp_val = to_canonical_imputation_codes(imp_val)
    meta_joint = classify_meta_joint(stage_val)

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
            "pmcid": pmcid,
            "pdf": pdf_path,
            "table_input": table_input or "NR",
            "full_text_source": fulltext_source,
        },
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

        recs = extract_records_from_table(df, logical_idx, paper_id, pmcid, table_ref, table_link)
        for r in recs:
            # Apply global inferred values as pre-fill
            r["Stage"] = stage_val
            r["Stage_original"] = stage_val
            r["Analyses type"] = assoc_val
            r["Model type"] = model_val
            r["Meta/Joint"] = meta_joint
            row_imp = to_canonical_imputation_codes(r.get("_row_imputation_hint", ""))
            r["Imputation_simple2"] = row_imp if row_imp != "NR" else imp_val
            row_pop = _normalize_cell_text(r.get("_row_population_hint", ""))
            r["Population"] = row_pop if row_pop else pop_val
            row_sample = _normalize_cell_text(r.get("_row_sample_size_hint", ""))
            r["Sample size"] = row_sample if row_sample else sample_size_val

            cohort_val, cohort_conf, cohort_evidence, cohort_needs_review = infer_cohort_from_row_and_text(
                r, full_text, table_ref
            )
            row_hint = to_canonical_cohort_codes(r.get("_row_cohort_hint", ""))
            inferred = to_canonical_cohort_codes(cohort_val)
            merged_cohorts = []
            for part in (row_hint, inferred):
                if part and part != "NR":
                    merged_cohorts.extend([x for x in part.split(";") if x])
            if merged_cohorts:
                dedup = []
                seen = set()
                for c in merged_cohorts:
                    if c not in seen:
                        dedup.append(c)
                        seen.add(c)
                cohort_val = ";".join(dedup)
            else:
                cohort_val = "NR"

            r["Cohort"] = cohort_val
            r["Cohort_simplified (no counts)"] = cohort_val if cohort_val != "NR" else ""

            # Conservative flags: if global inference uncertain, propagate review
            global_conf = min(stage_audit.confidence, assoc_audit.confidence, model_audit.confidence, imp_audit.confidence, pop_audit.confidence, sample_size_audit.confidence)
            r["_confidence"] = float(r.get("_confidence", 0.4)) * 0.4 + global_conf * 0.4 + cohort_conf * 0.2
            if stage_audit.needs_review or assoc_audit.needs_review or model_audit.needs_review or imp_audit.needs_review or pop_audit.needs_review or sample_size_audit.needs_review or cohort_needs_review:
                r["_needs_review"] = True

            # Add short evidence
            r["_evidence"] = (r.get("_evidence", "") + "\n" +
                             f"[Stage evidence] {stage_audit.evidence[:240]}\n" +
                             f"[Model evidence] {model_audit.evidence[:240]}\n" +
                             f"[Imputation evidence] {imp_audit.evidence[:240]}\n" +
                             f"[Population evidence] {pop_audit.evidence[:240]}\n" +
                             f"[Sample size evidence] {sample_size_audit.evidence[:240]}\n" +
                             f"[Cohort evidence] {cohort_evidence[:240]}").strip()

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
                asdict(FieldAudit("Imputation_simple2", r.get("Imputation_simple2", "NR"), imp_audit.confidence,
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
        shell["PMCID"] = pmcid
        shell["Stage"] = stage_val
        shell["Analyses type"] = assoc_val
        shell["Model type"] = model_val
        shell["Imputation_simple2"] = imp_val
        shell["Population"] = pop_val
        shell["Sample size"] = sample_size_val
        shell["_confidence"] = min(stage_audit.confidence, assoc_audit.confidence, model_audit.confidence, imp_audit.confidence, pop_audit.confidence, sample_size_audit.confidence)
        shell["_needs_review"] = True
        shell["_evidence"] = (
            "No extractable association records from tables. "
            f"Table entries={len(table_entries)}; source_errors={len(table_source_errors)}. "
            "Use audit json for details."
        )
        records.append(shell)

    # 7) Write outputs
    write_curated_xlsx(records, out_xlsx, template_xlsx=template_xlsx)

    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2, ensure_ascii=False)


def main():
    def _clean_user_input(s: str) -> str:
        s = (s or "").strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1].strip()
        return s

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, default=None, help="PDF path or URL")
    parser.add_argument("--table_input", default=None, help="Table source(s): URL/path; comma-separated for multiple")
    parser.add_argument("--out", default="curated_output.xlsx")
    parser.add_argument("--audit", default="audit.json")
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
    )


if __name__ == "__main__":
    main()
