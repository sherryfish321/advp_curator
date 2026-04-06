import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd


def to_str(v) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def norm_snp(v) -> Optional[str]:
    s = to_str(v)
    if not s:
        return None
    s = s.lower()
    # Normalize rs IDs like "rs7920721g" -> "rs7920721".
    m = re.match(r"^(rs\d+)[a-z]+$", s)
    if m:
        return m.group(1)
    return s


def norm_chr(v) -> Optional[str]:
    s = to_str(v)
    if not s:
        return None
    s = s.lower().replace("chr", "").strip()
    return s if s else None


def norm_num(v) -> Optional[float]:
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return None
        # Stabilize floating representation for equality-based matching.
        return float(f"{f:.12g}")
    s = str(v).strip().replace("×", "x").replace("−", "-").replace("–", "-")
    s = s.replace("X", "x")
    try:
        return float(f"{float(s):.12g}")
    except Exception:
        pass
    m = re.match(r"^\s*([+-]?\d*\.?\d+)\s*[x]\s*10\s*\^?\s*([+-]?\d+)\s*$", s)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return float(f"{(base * (10 ** exp)):.12g}")
    m = re.match(r"^\s*([+-]?\d*\.?\d+)\s*e\s*([+-]?\d+)\s*$", s, re.IGNORECASE)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return float(f"{(base * (10 ** exp)):.12g}")
    return None


def norm_pvalue(v) -> Optional[float]:
    return norm_num(v)


def norm_token_set(v) -> Optional[str]:
    s = to_str(v)
    if not s:
        return None
    raw = s.replace("|", ";").replace(",", ";").replace("/", ";")
    toks = [t.strip().lower() for t in raw.split(";") if t.strip()]
    if not toks:
        return None
    return ";".join(sorted(set(toks)))


def safe_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def load_gold(advp_tsv: str, pmid: int) -> Optional[pd.DataFrame]:
    g = pd.read_csv(advp_tsv, sep="\t")
    g = g[g["Pubmed PMID"] == pmid].copy()
    if g.empty:
        return None

    out = pd.DataFrame()
    out["pmid"] = g["Pubmed PMID"].map(norm_num)
    out["snp"] = g["Top SNP"].map(norm_snp)
    out["chr"] = g["#dbSNP_hg38_chr"].map(norm_chr)
    out["bp"] = g["dbSNP_hg38_position"].map(norm_num)
    out["ra1"] = g["RA 1(Reported Allele 1)"].map(lambda x: to_str(x).upper() if to_str(x) else None)
    out["pvalue"] = g["P-value"].map(norm_pvalue)
    out["effect"] = g["OR_nonref"].map(norm_num)
    out["cohort"] = g["Cohort_simple3"].map(norm_token_set)
    out["population"] = g["Population_map"].map(norm_token_set)
    out["sample_size"] = g["Sample size"].map(norm_num)
    out["analysis_group"] = g["Analysis group"].map(norm_token_set)
    out["locus_name"] = g["LocusName"].map(to_str) if "LocusName" in g.columns else None
    out["phenotype"] = g["Phenotype"].map(norm_token_set) if "Phenotype" in g.columns else None
    out["phenotype_derived"] = g["Phenotype-derived"].map(norm_token_set) if "Phenotype-derived" in g.columns else None
    return out


def export_pred_table(pred_df: pd.DataFrame, out_dir: str, pmid: int, label: str, scope: str) -> str:
    export_df = pred_df.copy()
    export_path = os.path.join(
        out_dir,
        f"pmid_{pmid}_pred_{sanitize_filename(label)}_{scope}.csv",
    )
    export_df.to_csv(export_path, index=False)
    return export_path


def load_pred(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        d = pd.read_excel(path)
    elif path.lower().endswith(".csv"):
        d = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    out = pd.DataFrame()
    out["pmid"] = d["Pubmed PMID"].map(norm_num) if "Pubmed PMID" in d.columns else None
    out["snp"] = d["TopSNP"].map(norm_snp) if "TopSNP" in d.columns else None
    out["chr"] = d["Chr"].map(norm_chr) if "Chr" in d.columns else None
    out["bp"] = d["BP(Position)"].map(norm_num) if "BP(Position)" in d.columns else None
    if "RA 1(Reported Allele 1)" in d.columns:
        out["ra1"] = d["RA 1(Reported Allele 1)"].map(lambda x: to_str(x).upper() if to_str(x) else None)
    else:
        out["ra1"] = None
    out["pvalue"] = d["P-value"].map(norm_pvalue) if "P-value" in d.columns else None
    out["effect"] = d["EffectSize(altvsref)"].map(norm_num) if "EffectSize(altvsref)" in d.columns else None
    if "Cohort_simplified (no counts)" in d.columns:
        out["cohort"] = d["Cohort_simplified (no counts)"].map(norm_token_set)
    elif "Cohort" in d.columns:
        out["cohort"] = d["Cohort"].map(norm_token_set)
    else:
        out["cohort"] = None
    out["population"] = d["Population_map"].map(norm_token_set) if "Population_map" in d.columns else None
    out["sample_size"] = d["Sample size"].map(norm_num) if "Sample size" in d.columns else None
    out["analysis_group"] = d["Analysis group"].map(norm_token_set) if "Analysis group" in d.columns else None
    out["locus_name"] = d["LocusName"].map(to_str) if "LocusName" in d.columns else None
    out["phenotype"] = d["Phenotype"].map(norm_token_set) if "Phenotype" in d.columns else None
    out["phenotype_derived"] = d["Phenotype-derived"].map(norm_token_set) if "Phenotype-derived" in d.columns else None
    return out


def aggregate_field_metrics(
    pred_rows: pd.DataFrame,
    gold_rows: pd.DataFrame,
    row_key_cols: List[str],
    fields: List[str],
) -> Dict[str, float]:
    pred_pairs = Counter()
    gold_pairs = Counter()
    for field in fields:
        if field not in pred_rows.columns or field not in gold_rows.columns:
            continue
        if not pred_rows[field].notna().any() and not gold_rows[field].notna().any():
            continue
        pred_df = pred_rows[pred_rows[field].notna()].copy()
        gold_df = gold_rows[gold_rows[field].notna()].copy()
        for key, value in zip(build_keys(pred_df, row_key_cols), pred_df[field].tolist()):
            pred_pairs[tuple(list(key) + [field, value])] += 1
        for key, value in zip(build_keys(gold_df, row_key_cols), gold_df[field].tolist()):
            gold_pairs[tuple(list(key) + [field, value])] += 1
    inter = pred_pairs & gold_pairs
    tp = sum(inter.values())
    fp = sum(pred_pairs.values()) - tp
    fn = sum(gold_pairs.values()) - tp
    return safe_f1(tp, fp, fn)


def apply_pred_scope(pred_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "raw":
        return pred_df
    if scope == "advp_like":
        # ADVP-like inclusion for this project: SNP + P-value required.
        # Effect is validated when available, but not required to keep the row.
        keep = pred_df["snp"].notna() & pred_df["pvalue"].notna()
        return pred_df[keep].copy()
    raise ValueError(f"Unsupported scope: {scope}")


def build_keys(df: pd.DataFrame, key_cols: List[str]) -> List[tuple]:
    keys = []
    for _, row in df.iterrows():
        keys.append(tuple(row[c] for c in key_cols))
    return keys


def evaluate_one(
    pred_df: pd.DataFrame,
    gold_df: Optional[pd.DataFrame],
    path: str,
    key_mode: str = "auto",
    ignore_bp: bool = False,
    ignore_ra1: bool = False,
) -> Dict:
    pred_rows = pred_df.dropna(subset=["snp"]).copy()
    if gold_df is None:
        return {
            "file": path,
            "status": "not_in_advp",
            "row_key_cols": [],
            "n_pred_rows": int(len(pred_rows)),
            "n_gold_rows": 0,
            "row_metrics": {"precision": None, "recall": None, "f1": None},
            "field_metrics": {},
            "aggregate_metrics": {},
            "unmatched_pred_examples": [],
            "unmatched_gold_examples": [],
        }

    if key_mode == "pmid_snp_pvalue":
        key_cols = [c for c in ["pmid", "snp", "pvalue"] if c in pred_df.columns and c in gold_df.columns]
        if "snp" not in key_cols:
            key_cols.insert(0, "snp")
    else:
        key_candidates = ["pmid", "snp", "chr", "bp", "ra1", "pvalue"]
        if ignore_bp:
            key_candidates = [c for c in key_candidates if c != "bp"]
        if ignore_ra1:
            key_candidates = [c for c in key_candidates if c != "ra1"]
        key_cols = []
        for c in key_candidates:
            if c in pred_df.columns and c in gold_df.columns:
                if pred_df[c].notna().any() and gold_df[c].notna().any():
                    key_cols.append(c)
        if "snp" not in key_cols:
            key_cols.insert(0, "snp")

    gold_rows = gold_df.dropna(subset=["snp"]).copy()

    pred_key_counter = Counter(build_keys(pred_rows, key_cols))
    gold_key_counter = Counter(build_keys(gold_rows, key_cols))

    inter = pred_key_counter & gold_key_counter
    tp = sum(inter.values())
    fp = sum(pred_key_counter.values()) - tp
    fn = sum(gold_key_counter.values()) - tp
    row_metrics = safe_f1(tp, fp, fn)

    field_metrics = {}
    compare_fields = ["pvalue", "effect", "cohort", "sample_size", "analysis_group", "population", "chr", "bp", "ra1"]
    if ignore_bp:
        compare_fields = [f for f in compare_fields if f != "bp"]
    if ignore_ra1:
        compare_fields = [f for f in compare_fields if f != "ra1"]
    row_key_cols = key_cols
    for field in compare_fields:
        if field not in pred_rows.columns or field not in gold_rows.columns:
            continue
        if not pred_rows[field].notna().any():
            continue
        if not gold_rows[field].notna().any():
            continue
        pred_pairs = Counter(
            tuple(list(k) + [v]) for k, v in zip(build_keys(pred_rows, row_key_cols), pred_rows[field].tolist())
        )
        gold_pairs = Counter(
            tuple(list(k) + [v]) for k, v in zip(build_keys(gold_rows, row_key_cols), gold_rows[field].tolist())
        )
        inter_f = pred_pairs & gold_pairs
        tp_f = sum(inter_f.values())
        fp_f = sum(pred_pairs.values()) - tp_f
        fn_f = sum(gold_pairs.values()) - tp_f
        field_metrics[field] = safe_f1(tp_f, fp_f, fn_f)

    easy_fields = ["pvalue", "effect", "cohort", "sample_size", "analysis_group", "population", "chr"]
    all_advp_fields = ["pmid", "snp", "chr", "bp", "pvalue", "effect", "cohort", "sample_size", "analysis_group", "population", "locus_name", "phenotype", "phenotype_derived", "ra1"]
    if ignore_bp:
        all_advp_fields = [f for f in all_advp_fields if f != "bp"]
    if ignore_ra1:
        all_advp_fields = [f for f in all_advp_fields if f != "ra1"]
    aggregate_metrics = {
        "easy_fields": {
            "fields": easy_fields,
            "metrics": aggregate_field_metrics(pred_rows, gold_rows, row_key_cols, easy_fields),
        },
        "all_advp_fields": {
            "fields": all_advp_fields,
            "metrics": aggregate_field_metrics(pred_rows, gold_rows, row_key_cols, all_advp_fields),
        },
    }

    unmatched_pred = list((pred_key_counter - gold_key_counter).elements())[:10]
    unmatched_gold = list((gold_key_counter - pred_key_counter).elements())[:10]

    return {
        "file": path,
        "status": "evaluated",
        "row_key_cols": key_cols,
        "n_pred_rows": int(sum(pred_key_counter.values())),
        "n_gold_rows": int(sum(gold_key_counter.values())),
        "row_metrics": row_metrics,
        "field_metrics": field_metrics,
        "aggregate_metrics": aggregate_metrics,
        "unmatched_pred_examples": unmatched_pred,
        "unmatched_gold_examples": unmatched_gold,
    }


def diagnose_unmatched(pred_df: pd.DataFrame, gold_df: Optional[pd.DataFrame], key_cols: List[str]) -> pd.DataFrame:
    if gold_df is None:
        return pd.DataFrame(
            [{"type": "info", "reason": "pmid_not_in_advp", "snp": None, "pmid": None, "chr": None, "bp": None, "ra1": None, "pvalue": None, "effect": None}]
        )
    pred_rows = pred_df.dropna(subset=["snp"]).copy()
    gold_rows = gold_df.dropna(subset=["snp"]).copy()
    pred_keys = build_keys(pred_rows, key_cols)
    gold_key_counter = Counter(build_keys(gold_rows, key_cols))

    records = []
    for i, key in enumerate(pred_keys):
        if gold_key_counter[key] > 0:
            gold_key_counter[key] -= 1
            continue
        prow = pred_rows.iloc[i]
        snp = prow.get("snp")
        g_same_snp = gold_rows[gold_rows["snp"] == snp]
        if g_same_snp.empty:
            reason = "snp_not_in_gold"
        else:
            mismatches = []
            for c in key_cols:
                if c == "snp":
                    continue
                pv = prow.get(c)
                gvals = set(g_same_snp[c].tolist()) if c in g_same_snp.columns else set()
                if pv not in gvals:
                    mismatches.append(f"{c}_mismatch")
            reason = "|".join(mismatches) if mismatches else "key_mismatch"

        records.append(
            {
                "type": "pred_unmatched",
                "reason": reason,
                "snp": prow.get("snp"),
                "pmid": prow.get("pmid"),
                "chr": prow.get("chr"),
                "bp": prow.get("bp"),
                "ra1": prow.get("ra1"),
                "pvalue": prow.get("pvalue"),
                "effect": prow.get("effect"),
            }
        )

    # Add gold rows missing from prediction for completeness.
    pred_key_counter = Counter(build_keys(pred_rows, key_cols))
    gold_keys = build_keys(gold_rows, key_cols)
    for i, key in enumerate(gold_keys):
        if pred_key_counter[key] > 0:
            pred_key_counter[key] -= 1
            continue
        grow = gold_rows.iloc[i]
        reason = "missing_in_prediction"
        records.append(
            {
                "type": "gold_unmatched",
                "reason": reason,
                "snp": grow.get("snp"),
                "pmid": grow.get("pmid"),
                "chr": grow.get("chr"),
                "bp": grow.get("bp"),
                "ra1": grow.get("ra1"),
                "pvalue": grow.get("pvalue"),
                "effect": grow.get("effect"),
            }
        )
    return pd.DataFrame(records)


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def main():
    ap = argparse.ArgumentParser(description="Evaluate extracted ADVP tables against ADVP TSV gold data.")
    ap.add_argument("--advp_tsv", required=True, help="Path to advp.variant.records.hg38.tsv")
    ap.add_argument("--pmid", required=True, type=int, help="PMID to evaluate")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more extracted table files (.xlsx/.csv)")
    ap.add_argument("--out_dir", default="eval_reports", help="Output folder for metrics and details")
    ap.add_argument(
        "--pred_scope",
        default="raw",
        choices=["raw", "advp_like"],
        help="raw: compare all predicted rows; advp_like: keep rows more likely to be included in ADVP",
    )
    ap.add_argument(
        "--key_mode",
        default="auto",
        choices=["auto", "pmid_snp_pvalue"],
        help="auto: use richer composite key; pmid_snp_pvalue: match only on PMID+SNP+P-value",
    )
    ap.add_argument(
        "--ignore_bp",
        action="store_true",
        help="Ignore BP(Position) in key matching and field metrics (useful for hg37/hg38 coordinate differences).",
    )
    ap.add_argument(
        "--ignore_ra1",
        action="store_true",
        help="Ignore RA1 allele in key matching and field metrics.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gold = load_gold(args.advp_tsv, args.pmid)

    reports = []
    preds = []
    exported_pred_paths = []
    for p in args.inputs:
        pred = load_pred(p)
        pred = apply_pred_scope(pred, args.pred_scope)
        preds.append(pred)
        exported_pred_paths.append(export_pred_table(pred, args.out_dir, args.pmid, os.path.basename(p), args.pred_scope))
        reports.append(
            evaluate_one(
                pred,
                gold,
                p,
                key_mode=args.key_mode,
                ignore_bp=args.ignore_bp,
                ignore_ra1=args.ignore_ra1,
            )
        )

    if len(preds) > 1:
        combined = pd.concat(preds, ignore_index=True)
        combined_export_path = export_pred_table(combined, args.out_dir, args.pmid, "combined_inputs", args.pred_scope)
        reports.append(
            evaluate_one(
                combined,
                gold,
                "combined_inputs",
                key_mode=args.key_mode,
                ignore_bp=args.ignore_bp,
                ignore_ra1=args.ignore_ra1,
            )
        )

    summary_rows = []
    for r in reports:
        summary_rows.append(
            {
                "file": r["file"],
                "status": r.get("status", "evaluated"),
                "key_cols": ",".join(r["row_key_cols"]),
                "n_pred_rows": r["n_pred_rows"],
                "n_gold_rows": r["n_gold_rows"],
                "row_precision": r["row_metrics"]["precision"],
                "row_recall": r["row_metrics"]["recall"],
                "row_f1": r["row_metrics"]["f1"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, f"pmid_{args.pmid}_summary_{args.pred_scope}.csv")
    summary_df.to_csv(summary_csv, index=False)

    details_json = os.path.join(args.out_dir, f"pmid_{args.pmid}_details_{args.pred_scope}.json")
    with open(details_json, "w") as f:
        json.dump(reports, f, indent=2, default=str)

    # Per-file mismatch diagnostics for debugging recall/precision failures.
    for p, pred in zip(args.inputs, preds):
        rep = next((r for r in reports if r["file"] == p), None)
        if not rep:
            continue
        diag = diagnose_unmatched(pred, gold, rep["row_key_cols"])
        diag_path = os.path.join(
            args.out_dir, f"pmid_{args.pmid}_mismatch_{sanitize_filename(os.path.basename(p))}_{args.pred_scope}.csv"
        )
        diag.to_csv(diag_path, index=False)
        print(f"Saved mismatch details: {diag_path}")

    if len(preds) > 1:
        combined_rep = next((r for r in reports if r["file"] == "combined_inputs"), None)
        if combined_rep:
            combined_diag = diagnose_unmatched(combined, gold, combined_rep["row_key_cols"])
            combined_diag_path = os.path.join(
                args.out_dir, f"pmid_{args.pmid}_mismatch_combined_inputs_{args.pred_scope}.csv"
            )
            combined_diag.to_csv(combined_diag_path, index=False)
            print(f"Saved mismatch details: {combined_diag_path}")

    print(f"Saved summary: {summary_csv}")
    print(f"Saved details: {details_json}")
    for export_path in exported_pred_paths:
        print(f"Saved predicted ADVP-style table: {export_path}")
    if len(preds) > 1:
        print(f"Saved predicted ADVP-style table: {combined_export_path}")
    if gold is None:
        print(f"PMID {args.pmid} not found in ADVP gold data. Skipped precision/recall/F1 scoring.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
