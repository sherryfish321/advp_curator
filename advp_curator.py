import re
import json
import argparse
import os
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
        s = s.replace("×", "x").replace("−", "-")
        # parse scientific like 4.2 x 10^-5
        sci = re.search(r"([0-9.]+)\s*x\s*10\^?(-?\d+)", s, flags=re.I)
        if sci:
            base = float(sci.group(1))
            exp = int(sci.group(2))
            return base * (10 ** exp)
        v = float(s)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


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


def classify_meta_joint(stage_value: str) -> str:
    if stage_value.lower().startswith("meta"):
        return "Meta"
    if stage_value.lower().startswith("joint"):
        return "Joint"
    return "NR"


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
    return None


def _pick_effect_column(columns: List[str]) -> Tuple[Optional[str], str]:
    nmap = {c: _norm_colname(c) for c in columns}
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
    Detect per-subgroup metric columns, e.g.:
    - i-Share (dichotomous)
    - i-Share (continuous)
    - Nagahama (dichotomous)
    - Nagahama (continuous)
    """
    defs: Dict[str, Dict[str, str]] = {}
    for col in columns:
        n = _norm_colname(col)
        if "i-share" not in n and "nagahama" not in n:
            continue
        if "dichotomous" not in n and "continuous" not in n:
            continue

        cohort = "i-Share" if "i-share" in n else "Nagahama"
        mode = "dichotomous" if "dichotomous" in n else "continuous"
        label = f"{cohort} ({mode})"

        defs.setdefault(label, {})
        if "p value all" in n or "p-value all" in n:
            defs[label]["p"] = col
        elif ("p value" in n or "p-value" in n or re.search(r"\bp\b", n)) and "het p" not in n and "pf" not in n:
            defs[label].setdefault("p", col)
        elif "95" in n and "ci" in n:
            defs[label]["ci"] = col
        elif any(k in n for k in ["beta", "β", "odds ratio", " hr", "hazard ratio", "z-score", "zscore"]):
            defs[label]["effect"] = col

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

    columns = list(body.columns)
    variant_col = _pick_preferred_column(columns, ["snp all", "variant all", "variant", "rsid", "snp", "marker"])
    rs_col = variant_col
    chr_col = _pick_preferred_column(columns, ["chr:position", "chr", "chromosome"])
    bp_col = _pick_preferred_column(columns, ["position", "bp", "base pair", "pos"])
    locus_col = _pick_preferred_column(columns, ["nearest gene", "closest gene", "gene"])
    maf_col = _pick_preferred_column(columns, ["eaf", "maf", "af"])
    ea_oa_col = _pick_preferred_column(columns, ["ea/oa", "ea / oa", "effect allele/other allele"])
    major_minor_col = _pick_preferred_column(columns, ["major/ minor alleles", "major/minor", "alleles", "allele"])

    p_col = _pick_pvalue_column(columns)
    ci_col = _pick_preferred_column(columns, ["95", "ci"])

    # Effect type/value priority: OR > Beta > HR > Zscore
    effect_col, effect_type = _pick_effect_column(columns)

    # Interaction terms are uncommon in GWAS summary tables; still capture if present.
    interaction_col = _pick_preferred_column(columns, ["interaction", "snp x", "snp*"])
    subgroup_defs = _detect_subgroup_defs(columns)

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
        interaction_val = _normalize_cell_text(row.get(interaction_col)) if interaction_col else ""

        # keep rows with actionable stats even when rsID is absent
        pval_default = safe_float(row.get(p_col)) if p_col else None
        effect_default = safe_float(row.get(effect_col)) if effect_col else None
        ci_default = _normalize_cell_text(row.get(ci_col)) if ci_col else ""
        if ea_oa_col:
            ra1_minor, ra2_major = _split_ea_oa(_normalize_cell_text(row.get(ea_oa_col)))
        else:
            ra1_minor, ra2_major = _split_major_minor(_normalize_cell_text(row.get(major_minor_col)) if major_minor_col else "")

        # Skip section headers and non-data rows.
        if not rsids and pval_default is None and effect_default is None and not chr_val and not bp_val:
            continue

        names = sorted(set([r.lower() for r in rsids])) if rsids else ["NR"]
        subgroup_records = []
        if subgroup_defs:
            for sub_label, metric_cols in subgroup_defs.items():
                sp = safe_float(row.get(metric_cols["p"])) if metric_cols.get("p") else None
                se = safe_float(row.get(metric_cols["effect"])) if metric_cols.get("effect") else None
                sci = _normalize_cell_text(row.get(metric_cols["ci"])) if metric_cols.get("ci") else ""
                if sp is None and se is None:
                    continue
                subgroup_records.append({
                    "label": sub_label,
                    "p": sp,
                    "effect": se,
                    "ci": sci,
                    "effect_type": _effect_type_from_colname(metric_cols["effect"]) if metric_cols.get("effect") else effect_type,
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
                rec["Analysis group"] = sub["label"]
                rec["Phenotype-derived"] = sub["label"].split("(")[-1].replace(")", "").strip() if sub["label"] else rec.get("Phenotype-derived", "")
                rec["_confidence"] = 0.55 if rsid != "NR" else 0.4
                rec["_needs_review"] = True
                rec["_evidence"] = f"Row text: {row_text[:340]}"
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

    # 2) Extract full text
    full_text = extract_full_text(pdf_path) if pdf_path else ""

    # 3) Infer global fields (Stage / Model / Assoc type / Imputation)
    stage_val, stage_audit = infer_stage(full_text)
    assoc_val, assoc_audit = infer_association_type(full_text)
    model_val, model_audit = infer_model_type(full_text)
    imp_val, imp_audit = infer_imputation(full_text)
    meta_joint = classify_meta_joint(stage_val)

    # 4) Extract tables
    table_entries: List[Tuple[str, pd.DataFrame]] = []
    table_source_errors: List[Dict[str, str]] = []
    if table_input:
        for src in parse_table_sources(table_input):
            try:
                src_tables = extract_tables_from_source(src)
            except Exception as e:
                table_source_errors.append({"source": src, "error": repr(e)})
                continue
            for j, tdf in enumerate(src_tables, start=1):
                label = f"{os.path.basename(src)}#{j}"
                table_entries.append((label, tdf))
    elif pdf_path:
        for j, tdf in enumerate(extract_tables(pdf_path, pages="all"), start=1):
            table_entries.append((f"Table {j}", tdf))

    # 5) Convert each table to records (heuristic)
    records: List[Dict[str, Any]] = []
    audits: Dict[str, Any] = {
        "paper": {
            "paper_id": paper_id,
            "pmcid": pmcid,
            "pdf": pdf_path,
            "table_input": table_input or "NR",
        },
        "table_source_errors": table_source_errors,
        "global_field_audit": [
            asdict(stage_audit),
            asdict(assoc_audit),
            asdict(model_audit),
            asdict(imp_audit),
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

    for idx, (source_label, df) in enumerate(table_entries, start=1):
        table_ref = f"Table {idx}"
        if table_input:
            table_link = source_label.split("#")[0]
        else:
            table_link = pdf_or_url if (pdf_or_url and pdf_or_url.lower().startswith("http")) else "local_pdf"

        recs = extract_records_from_table(df, idx, paper_id, pmcid, table_ref, table_link)
        for r in recs:
            # Apply global inferred values as pre-fill
            r["Stage"] = stage_val
            r["Stage_original"] = stage_val
            r["Analyses type"] = assoc_val
            r["Model type"] = model_val
            r["Meta/Joint"] = meta_joint
            r["Imputation_simple2"] = imp_val

            # Conservative flags: if global inference uncertain, propagate review
            global_conf = min(stage_audit.confidence, assoc_audit.confidence, model_audit.confidence, imp_audit.confidence)
            r["_confidence"] = float(r.get("_confidence", 0.4)) * 0.5 + global_conf * 0.5
            if stage_audit.needs_review or assoc_audit.needs_review or model_audit.needs_review or imp_audit.needs_review:
                r["_needs_review"] = True

            # Add short evidence
            r["_evidence"] = (r.get("_evidence", "") + "\n" +
                             f"[Stage evidence] {stage_audit.evidence[:240]}\n" +
                             f"[Model evidence] {model_audit.evidence[:240]}\n" +
                             f"[Imputation evidence] {imp_audit.evidence[:240]}").strip()

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
        shell["_confidence"] = min(stage_audit.confidence, assoc_audit.confidence, model_audit.confidence, imp_audit.confidence)
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
        src = input("Input source path/URL (PDF or table file/URL): ").strip()
        if not src:
            raise ValueError("Input source is required.")
        src_low = src.lower()
        if src_low.endswith(".pdf"):
            args.input = src
        elif src_low.endswith((".xlsx", ".xls", ".csv", ".tsv", ".html", ".htm")):
            args.table_input = src
        else:
            mode_hint = input("Treat source as table input? [Y/n]: ").strip()
            # Normalize common full-width answers (e.g., Ｙ/Ｎ)
            mode_hint = mode_hint.translate(str.maketrans({"Ｙ": "Y", "ｙ": "y", "Ｎ": "N", "ｎ": "n"})).lower()
            if mode_hint in ("", "y", "yes"):
                args.table_input = src
            else:
                args.input = src

    if not args.out:
        out_val = input("Output xlsx path [curated_output.xlsx]: ").strip()
        args.out = out_val or "curated_output.xlsx"
    elif args.out == "curated_output.xlsx":
        out_val = input("Output xlsx path [curated_output.xlsx]: ").strip()
        args.out = out_val or args.out

    if not args.paper_id or args.paper_id == "PAPER":
        pid = input("paper_id [PAPER]: ").strip()
        args.paper_id = pid or "PAPER"

    if not args.audit or args.audit == "audit.json":
        auto_audit = f"{args.paper_id}_audit.json" if args.paper_id else "audit.json"
        audit_val = input(f"Audit json path [{auto_audit}]: ").strip()
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
