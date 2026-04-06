#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import email
import html
import json
import os
import re
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd

from table_link_to_excel import discover_relevant_pmc_tables


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_links(raw: str) -> List[str]:
    links = []
    for part in raw.replace(",", "\n").splitlines():
        u = part.strip()
        if u:
            links.append(u)
    return links


def parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def maybe_auto_discover_links(paper_input: str, links: List[str], auto_discover: bool) -> tuple[List[str], List[Dict[str, object]]]:
    if links:
        return links, []
    if not auto_discover:
        return links, []
    discovered = discover_relevant_pmc_tables(paper_input)
    selected = [item["link"] for item in discovered if item.get("selected")]
    if not selected:
        raise ValueError("Auto-discovery found PMC tables, but none matched the current gene + p-value rule")
    return selected, discovered


def collect_mismatch_paths(out_dir: Path, pmid: int, pred_scope: str, harmonized_paths: List[Path]) -> List[Path]:
    paths = []
    for hp in harmonized_paths:
        mismatch = out_dir / f"pmid_{pmid}_mismatch_{hp.name}_{pred_scope}.csv"
        if mismatch.exists():
            paths.append(mismatch)
    return paths


def collect_issue_paths(out_dir: Path, pmid: int, pred_scope: str, harmonized_paths: List[Path]) -> List[Path]:
    combined = out_dir / f"pmid_{pmid}_mismatch_combined_inputs_{pred_scope}.csv"
    if combined.exists():
        return [combined]
    return collect_mismatch_paths(out_dir, pmid, pred_scope, harmonized_paths)


def build_fix_file(mismatch_paths: List[Path], out_dir: Path, pmid: int) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fix_path = out_dir / f"pmid_{pmid}_fix_ready_{ts}.csv"
    rows = []
    for mp in mismatch_paths:
        with mp.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("type") == "pred_unmatched":
                    r["source_mismatch_file"] = str(mp)
                    rows.append(r)

    fieldnames = [
        "type",
        "reason",
        "snp",
        "pmid",
        "chr",
        "bp",
        "ra1",
        "pvalue",
        "effect",
        "source_mismatch_file",
    ]

    with fix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return fix_path


def append_run_log(log_dir: Path, payload: Dict[str, object]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pmid_{payload['pmid']}_run_{ts}.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return log_path


def summarize_mismatch_reasons(mismatch_paths: List[Path], mismatch_type: str) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    for mp in mismatch_paths:
        with mp.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("type") or "").strip() != mismatch_type:
                    continue
                reason = (row.get("reason") or "").strip()
                if not reason:
                    continue
                counts[reason] = counts.get(reason, 0) + 1
    return [{"reason": reason, "count": count} for reason, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]


def summarize_missing_fields(harmonized_paths: List[Path]) -> List[Dict[str, object]]:
    target_fields = [
        "TopSNP",
        "Chr",
        "BP(Position)",
        "P-value",
        "EffectSize(altvsref)",
        "Cohort_simplified (no counts)",
        "Sample size",
        "Population_map",
        "Analysis group",
    ]
    counts: Dict[str, int] = {field: 0 for field in target_fields}
    for path in harmonized_paths:
        df = pd.read_excel(path)
        for field in target_fields:
            if field not in df.columns:
                continue
            series = df[field]
            missing_mask = series.isna() | series.astype(str).str.strip().isin(["", "NR", "nan", "None"])
            counts[field] += int(missing_mask.sum())
    return [{"field": field, "missing_rows": count} for field, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if count > 0]


def read_details_aggregate_metrics(details_json: Path) -> Dict[str, object]:
    if not details_json.exists():
        return {}
    with details_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    combined = next((item for item in data if str(item.get("file", "")).endswith("combined_inputs") or item.get("file") == "combined_inputs"), None)
    if combined:
        return combined.get("aggregate_metrics", {})
    if len(data) == 1:
        return data[0].get("aggregate_metrics", {})
    return {}


def metric_pct(v: object) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return "NA"


def is_http_url(value: str) -> bool:
    return value.lower().startswith("http://") or value.lower().startswith("https://")


def parse_request_data(handler: BaseHTTPRequestHandler) -> Dict[str, List[str]]:
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length)
    if "multipart/form-data" not in content_type:
        return parse_qs(raw.decode("utf-8"))

    message = email.message_from_bytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + raw
    )
    data: Dict[str, List[str]] = {}
    for part in message.walk():
        if part.get_content_maintype() == "multipart":
            continue
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        filename = part.get_param("filename", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if not name:
            continue
        if filename:
            upload_dir = Path(os.getcwd()) / "uploaded_papers"
            upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
            save_path = upload_dir / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
            save_path.write_bytes(payload)
            data.setdefault(name, []).append(str(save_path))
        else:
            data.setdefault(name, []).append(payload.decode("utf-8", errors="ignore"))
    return data


def infer_pmid_from_sheet_path(path: Path) -> str:
    match = re.search(r"from_(\d+)_table\d+_v\d+\.xlsx$", path.name)
    return match.group(1) if match else ""


def infer_table_number_from_sheet_path(path: Path) -> int:
    match = re.search(r"_table(\d+)_v\d+\.xlsx$", path.name)
    return int(match.group(1)) if match else 1


def build_table_link_map(table_links: List[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for idx, link in enumerate(table_links, start=1):
        mapping[infer_table_number_from_link(link, idx)] = link
    return mapping


def infer_table_number_from_link(link: str, fallback: int) -> int:
    match = re.search(r"/table/T(\d+)/?$", link, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"/table/Tab(\d+)/?$", link, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"/table/[^/?#]*tbl[-_]?0*(\d+)/?$", link, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"/table/[^/?#]*table[-_]?0*(\d+)/?$", link, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"/table/[^/?#]*-T(\d+)/?$", link, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"[?&]table=T(\d+)\b", link, flags=re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"[?&]table_id=T(\d+)\b", link, flags=re.I)
    if match:
        return int(match.group(1))
    return fallback


def find_related_tables(path: Path) -> List[Path]:
    pmid = infer_pmid_from_sheet_path(path)
    if not pmid:
        return [path]
    pattern = f"from_{pmid}_table*_v*.xlsx"
    tables = sorted(path.parent.glob(pattern))
    return tables or [path]


def load_run_context_for_sheet(path: Path) -> Dict[str, object]:
    log_dir = path.parent.parent / "ui_run_logs"
    if not log_dir.exists():
        return {}
    latest_match: Dict[str, object] = {}
    for log_path in sorted(log_dir.glob("*.json"), reverse=True):
        with log_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        generated = data.get("generated_harmonized", [])
        if str(path) in generated:
            latest_match = data
            break
    return latest_match


def annotation_store_path(path: Path) -> Path:
    ann_dir = path.parent.parent / "ui_annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    return ann_dir / f"{path.stem}.json"


def load_annotations(path: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    ann_path = annotation_store_path(path)
    if not ann_path.exists():
        return {}
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(path: Path, annotations: Dict[str, Dict[str, Dict[str, str]]]) -> Path:
    ann_path = annotation_store_path(path)
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    return ann_path


def curated_editable_columns() -> List[str]:
    return [
        "RecordID",
        "TopSNP",
        "Chr",
        "BP(Position)",
        "P-value",
        "EffectSize(altvsref)",
        "Cohort_simplified (no counts)",
        "Sample size",
        "Population_map",
        "Analysis group",
        "LocusName",
    ]


def load_curated_rows(path: Path, limit: int = None) -> Dict[str, object]:
    df = pd.read_excel(path)
    annotations = load_annotations(path)
    cols = [c for c in curated_editable_columns() if c in df.columns]
    rows = []
    view = df if limit is None else df.head(limit)
    for idx, row in view.iterrows():
        row_dict = {"_row_index": int(idx)}
        for col in cols:
            val = row.get(col)
            row_dict[col] = "" if pd.isna(val) else str(val)
            row_ann = annotations.get(str(idx), {}).get(col, {})
            row_dict[f"{col}__status"] = row_ann.get("status", "")
            row_dict[f"{col}__comment"] = row_ann.get("comment", "")
        rows.append(row_dict)
    return {"columns": cols, "rows": rows, "total_rows": int(len(df))}


def save_curated_edits(path: Path, form_data: Dict[str, List[str]]) -> Dict[str, object]:
    df = pd.read_excel(path)
    annotations = load_annotations(path)
    row_indexes = form_data.get("row_index", [])
    updated = 0
    annotated = 0
    editable = [c for c in curated_editable_columns() if c in df.columns]
    for pos, row_idx_raw in enumerate(row_indexes):
        if not row_idx_raw.strip():
            continue
        row_idx = int(row_idx_raw)
        if row_idx not in df.index:
            continue
        changed = False
        for col in editable:
            values = form_data.get(f"field__{col}", [])
            if pos >= len(values):
                continue
            new_val = values[pos]
            old_val = df.at[row_idx, col]
            old_norm = "" if pd.isna(old_val) else str(old_val)
            if old_norm != new_val:
                df.at[row_idx, col] = new_val if new_val != "" else None
                changed = True
            status_values = form_data.get(f"status__{col}", [])
            comment_values = form_data.get(f"comment__{col}", [])
            status = status_values[pos].strip() if pos < len(status_values) else ""
            comment = comment_values[pos].strip() if pos < len(comment_values) else ""
            if status or comment:
                annotations.setdefault(str(row_idx), {})[col] = {"status": status, "comment": comment}
                annotated += 1
            elif str(row_idx) in annotations and col in annotations[str(row_idx)]:
                del annotations[str(row_idx)][col]
                if not annotations[str(row_idx)]:
                    del annotations[str(row_idx)]
        if changed:
            updated += 1
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="curated")
    ann_path = save_annotations(path, annotations)
    return {"updated_rows": updated, "annotated_cells": annotated, "annotation_path": str(ann_path)}


def summarize_counts(mismatch_paths: List[Path]) -> Dict[str, int]:
    success = 0
    failed = 0
    for mp in mismatch_paths:
        with mp.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("type") == "pred_unmatched":
                    failed += 1
    # Success here means all extracted prediction rows that are not unmatched by evaluator.
    # We estimate this from mismatch files: total pred rows = pred_unmatched + matched.
    # Matched count isn't printed directly in mismatch files, so this value is computed from per-file summary later.
    return {"success_rows": success, "failed_rows": failed}


def count_harmonized_rows(harmonized_paths: List[Path]) -> int:
    total = 0
    for path in harmonized_paths:
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        total += int(len(df))
    return total


def read_summary_metrics(summary_csv: Path, harmonized_paths: List[Path]) -> Dict[str, int]:
    pred_total = 0
    for_name = {p.name for p in harmonized_paths}
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = Path(r.get("file", "")).name
            if name in for_name:
                try:
                    pred_total += int(float(r.get("n_pred_rows", "0") or 0))
                except ValueError:
                    pass
    return {
        "pred_total": pred_total,
        "actual_total": count_harmonized_rows(harmonized_paths),
    }


def read_row_match_metrics(summary_csv: Path) -> Dict[str, object]:
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    combined = next((row for row in rows if row.get("file") == "combined_inputs"), None)
    target = combined or (rows[0] if rows else None)
    if not target:
        return {}
    return {
        "fields": ["pmid", "snp", "chr", "pvalue"],
        "metrics": {
            "precision": target.get("row_precision"),
            "recall": target.get("row_recall"),
            "f1": target.get("row_f1"),
        },
    }


def run_pipeline(
    pmid: int,
    paper_input: str,
    table_links: List[str],
    owner_name: str,
    source_type: str,
    advp_tsv: str,
    pred_scope: str,
    key_mode: str,
    ignore_bp: bool,
    ignore_ra1: bool,
    base_dir: str,
    auto_discovered_tables: List[Dict[str, object]] = None,
) -> Dict[str, object]:
    if source_type != "pmc_table_link":
        raise ValueError("Only source_type=pmc_table_link is supported in current MVP")

    root = Path(base_dir).resolve()
    table_dir = root / "table_from_link"
    harmonized_dir = root / "harmonized_tables_advp"
    audit_dir = root / "audit"
    eval_dir = root / "eval_reports"
    run_log_dir = root / "ui_run_logs"
    for d in (table_dir, harmonized_dir, audit_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    generated_tables = []
    generated_harmonized = []

    for idx, link in enumerate(table_links, start=1):
        table_num = infer_table_number_from_link(link, idx)
        tag = f"{pmid}_table{table_num}"
        table_xlsx = table_dir / f"{tag}.xlsx"
        harmonized_xlsx = harmonized_dir / f"from_{tag}_v1.xlsx"
        audit_json = audit_dir / f"from_{tag}_audit.json"
        ensure_parent(table_xlsx)
        ensure_parent(harmonized_xlsx)
        ensure_parent(audit_json)

        run_cmd([
            "python3",
            "table_link_to_excel.py",
            "--url",
            link,
            "--out",
            str(table_xlsx),
        ])

        run_cmd([
            "python3",
            "advp_curator.py",
            "--input",
            paper_input,
            "--table_input",
            str(table_xlsx),
            "--out",
            str(harmonized_xlsx),
            "--audit",
            str(audit_json),
            "--paper_id",
            tag,
        ])

        generated_tables.append(table_xlsx)
        generated_harmonized.append(harmonized_xlsx)

    eval_cmd = [
        "python3",
        "evaluate_advp_alignment.py",
        "--advp_tsv",
        str((Path(base_dir) / advp_tsv).resolve()),
        "--pmid",
        str(pmid),
        "--inputs",
    ]
    eval_cmd.extend(str(p) for p in generated_harmonized)
    eval_cmd.extend([
        "--pred_scope",
        pred_scope,
        "--key_mode",
        key_mode,
        "--out_dir",
        str(eval_dir),
    ])
    if ignore_bp:
        eval_cmd.append("--ignore_bp")
    if ignore_ra1:
        eval_cmd.append("--ignore_ra1")

    run_cmd(eval_cmd)

    summary_csv = eval_dir / f"pmid_{pmid}_summary_{pred_scope}.csv"
    details_json = eval_dir / f"pmid_{pmid}_details_{pred_scope}.json"
    mismatch_paths = collect_mismatch_paths(eval_dir, pmid, pred_scope, generated_harmonized)
    issue_paths = collect_issue_paths(eval_dir, pmid, pred_scope, generated_harmonized)
    fix_file = build_fix_file(mismatch_paths, eval_dir, pmid)
    predicted_issue_summary = summarize_mismatch_reasons(issue_paths, "pred_unmatched")
    missing_prediction_summary = summarize_mismatch_reasons(issue_paths, "gold_unmatched")
    missing_summary = summarize_missing_fields(generated_harmonized)
    aggregate_metrics = read_details_aggregate_metrics(details_json)
    row_match_metrics = read_row_match_metrics(summary_csv)

    mismatch_counts = summarize_counts(mismatch_paths)
    summary_counts = read_summary_metrics(summary_csv, generated_harmonized)
    pred_total = summary_counts.get("actual_total") or summary_counts.get("pred_total", 0)
    success_rows = max(0, pred_total - mismatch_counts["failed_rows"])
    run_log = append_run_log(
        run_log_dir,
        {
            "pmid": pmid,
            "owner_name": owner_name,
            "paper_input": paper_input,
            "table_links": table_links,
            "auto_discovered_tables": auto_discovered_tables or [],
            "generated_harmonized": [str(p) for p in generated_harmonized],
            "summary_csv": str(summary_csv),
            "details_json": str(details_json),
            "fix_file": str(fix_file),
            "created_at": dt.datetime.now().isoformat(),
        },
    )

    return {
        "pmid": pmid,
        "owner_name": owner_name,
        "generated_tables": [str(p) for p in generated_tables],
        "generated_harmonized": [str(p) for p in generated_harmonized],
        "summary_csv": str(summary_csv),
        "details_json": str(details_json),
        "fix_file": str(fix_file),
        "run_log": str(run_log),
        "success_rows": success_rows,
        "failed_rows": mismatch_counts["failed_rows"],
        "table_count": len(generated_harmonized),
        "auto_discovered_tables": auto_discovered_tables or [],
        "predicted_issue_summary": predicted_issue_summary,
        "missing_prediction_summary": missing_prediction_summary,
        "missing_summary": missing_summary,
        "aggregate_metrics": aggregate_metrics,
        "row_match_metrics": row_match_metrics,
    }


def render_summary_table(title: str, rows: List[Dict[str, object]], key_label: str, value_label: str) -> str:
    if not rows:
        return (
            f"<section class='subcard'><h3>{html.escape(title)}</h3>"
            "<p class='muted'>No issues found in this category.</p></section>"
        )
    body = []
    for row in rows[:12]:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row[key_label]))}</td>"
            f"<td>{html.escape(str(row[value_label]))}</td>"
            "</tr>"
        )
    return (
        f"<section class='subcard'><h3>{html.escape(title)}</h3>"
        "<details class='toggle-box' open><summary>Show Details</summary>"
        "<table class='summary-table'><thead><tr>"
        f"<th>{html.escape(key_label.replace('_', ' ').title())}</th>"
        f"<th>{html.escape(value_label.replace('_', ' ').title())}</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></details></section>"
    )


def render_metric_panel(title: str, payload: Dict[str, object], description: str) -> str:
    if not payload:
        return (
            f"<section class='subcard'><h3>{html.escape(title)}</h3>"
            "<p class='muted'>No metric data available.</p></section>"
        )
    metrics = payload.get("metrics", {})
    fields = payload.get("fields", [])
    chips = "".join(f"<li class='check-item'><span class='check-icon'>+</span>{html.escape(str(field))}</li>" for field in fields)
    icon = "Row" if "Row Match" in title else "Field"
    return (
        f"<section class='subcard metric-card'><div class='card-title'><span class='title-icon'>{html.escape(icon)}</span><h3>{html.escape(title)}</h3></div>"
        f"<p class='muted'>{html.escape(description)}</p>"
        "<div class='metric-row'>"
        f"<div class='mini-stat'><span class='k'>Precision</span><span class='v'>{metric_pct(metrics.get('precision'))}</span></div>"
        f"<div class='mini-stat'><span class='k'>Recall</span><span class='v'>{metric_pct(metrics.get('recall'))}</span></div>"
        f"<div class='mini-stat'><span class='k'>F1</span><span class='v'>{metric_pct(metrics.get('f1'))}</span></div>"
        "</div>"
        "<details class='toggle-box' open><summary>Fields Used</summary>"
        f"<ul class='check-list'>{chips}</ul></details></section>"
    )


def render_edit_table(file_path: str) -> str:
    path = Path(file_path).resolve()
    run_context = load_run_context_for_sheet(path)
    if isinstance(run_context.get("generated_harmonized"), list) and run_context.get("generated_harmonized"):
        related_tables = [Path(p) for p in run_context["generated_harmonized"]]
    else:
        related_tables = find_related_tables(path)
    table_num = infer_table_number_from_sheet_path(path)
    paper_input = str(run_context.get("paper_input", ""))
    table_links = run_context.get("table_links", []) if isinstance(run_context.get("table_links", []), list) else []
    table_link_map = build_table_link_map(table_links)
    current_table_link = table_link_map.get(table_num, "")
    data = load_curated_rows(path)
    header = "".join(f"<th>{html.escape(col)}</th>" for col in data["columns"])
    rows_html = []
    for row in data["rows"]:
        cells = [f"<td><input type='hidden' name='row_index' value='{row['_row_index']}' />{row['_row_index']}</td>"]
        for col in data["columns"]:
            val = html.escape(row[col])
            status = row.get(f"{col}__status", "")
            comment = html.escape(row.get(f"{col}__comment", ""))
            status_class = f"mark-{status}" if status else ""
            cells.append(
                f"<td class='edit-cell {status_class}'>"
                "<div class='cell-stack'>"
                f"<input name='field__{html.escape(col)}' value='{val}' />"
                f"<select name='status__{html.escape(col)}' class='status-select'>"
                f"<option value='' {'selected' if status == '' else ''}>No Mark</option>"
                f"<option value='review' {'selected' if status == 'review' else ''}>Needs Review</option>"
                f"<option value='issue' {'selected' if status == 'issue' else ''}>Issue</option>"
                f"<option value='resolved' {'selected' if status == 'resolved' else ''}>Resolved</option>"
                "</select>"
                f"<input name='comment__{html.escape(col)}' value='{comment}' placeholder='Comment' class='comment-input' />"
                "</div>"
                "</td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    ref_actions = []
    if current_table_link:
        ref_actions.append(f"<a class='btn-link accent' href='{html.escape(current_table_link)}' target='_blank' rel='noopener noreferrer'>Open Current Table Link</a>")
    if paper_input:
        if is_http_url(paper_input):
            ref_actions.append(f"<a class='btn-link' href='{html.escape(paper_input)}' target='_blank' rel='noopener noreferrer'>Open Paper File</a>")
        else:
            ref_actions.append(f"<a class='btn-link' href='/download?{urlencode({'path': paper_input})}'>Open Paper File</a>")
    table_reference_rows = []
    for idx, table_path in enumerate(related_tables, start=1):
        related_table_num = infer_table_number_from_sheet_path(table_path)
        linked_url = table_link_map.get(related_table_num, "")
        is_current = "Current" if table_path == path else f"Table {related_table_num}"
        row_class = "ref-row current" if table_path == path else "ref-row"
        open_btn = (
            f"<a class='btn-link accent' href='{html.escape(linked_url)}' target='_blank' rel='noopener noreferrer' onclick='event.stopPropagation()'>Open Link</a>"
            if linked_url else
            "<span class='muted'>No link available</span>"
        )
        table_reference_rows.append(
            f"<a class='{row_class}' href='/edit?{urlencode({'path': str(table_path)})}'>"
            f"<div class='ref-title'><span class='table-badge'>{html.escape(is_current)}</span><span class='table-name'>{html.escape(table_path.name)}</span></div>"
            f"<div class='muted ref-url'>{html.escape(linked_url or 'No link available')}</div>"
            f"<div class='ref-actions'>{open_btn}</div>"
            "</a>"
        )
    return (
        "<h2>Edit Curated Sheet</h2>"
        f"<p class='muted'>Editing <code>{html.escape(str(path))}</code>. "
        f"Showing first {len(data['rows'])} rows out of {data['total_rows']}.</p>"
        "<div class='subcard'>"
        "<h3>Reference Links</h3>"
        f"<p><strong>Paper</strong><br><code>{html.escape(paper_input or 'Not available')}</code></p>"
        f"<p><strong>Current Table Link</strong><br><span class='link-pill'>{html.escape(current_table_link or 'Not available')}</span></p>"
        "<div class='ref-grid'>"
        + "".join(table_reference_rows)
        + "</div>"
        "<div class='actions'>"
        + "".join(ref_actions)
        + "</div></div>"
        "<form method='post' action='/save-edits'>"
        f"<input type='hidden' name='path' value='{html.escape(str(path))}' />"
        "<div class='table-wrap'><table class='editor-table'><thead><tr><th>Row</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
        "<div class='actions'>"
        "<button type='submit'>Save Changes</button>"
        f"<a class='btn-link' href='/'>Back</a>"
        "</div></form>"
    )


def render_tabbed_sections(result: Dict[str, object], download_link: str, edit_link: str) -> str:
    sections = []
    sections.append(
        "<div class='tab-shell'>"
        "<div class='tab-bar'>"
        "<button type='button' class='tab-btn active' data-tab='overview'>Overview</button>"
        "<button type='button' class='tab-btn' data-tab='accuracy'>Accuracy</button>"
        "<button type='button' class='tab-btn' data-tab='issues'>Issues</button>"
        "<button type='button' class='tab-btn' data-tab='files'>Files</button>"
        "</div>"
        "<section class='tab-panel active' data-panel='overview'>"
        "<div class='stats'>"
        f"<div class='stat ok'><span class='k'>Success Rows</span><span class='v'>{result['success_rows']}</span></div>"
        f"<div class='stat bad'><span class='k'>Failed Rows</span><span class='v'>{result['failed_rows']}</span></div>"
        "</div>"
        "<div class='actions'>"
        f"<a class='btn-link primary' href='{download_link}'>Download {html.escape(Path(str(result['fix_file'])).name)}</a>"
        + (f"<a class='btn-link' href='{edit_link}'>Edit Curated Sheet</a>" if edit_link else "")
        + "<a class='btn-link' href='/'>Run Another</a>"
        "</div>"
        "</section>"
        "<section class='tab-panel' data-panel='accuracy'>"
        "<div class='panel-grid'>"
        + render_metric_panel(
            "Row Match Accuracy",
            result.get("row_match_metrics", {}),
            "Uses PMID, SNP, chromosome, and p-value to determine whether a predicted row matches an ADVP row.",
        )
        + render_metric_panel(
            "Field Accuracy: Easy Fields",
            result["aggregate_metrics"].get("easy_fields", {}),
            "Uses pvalue, effect, cohort, sample_size, analysis_group, population, and chr.",
        )
        + render_metric_panel(
            "Field Accuracy: All Mapped ADVP Fields",
            result["aggregate_metrics"].get("all_advp_fields", {}),
            "Uses all currently mapped ADVP-comparable fields while keeping row matching on PMID, SNP, chromosome, and p-value.",
        )
        + "</div></section>"
        "<section class='tab-panel' data-panel='issues'>"
        "<div class='panel-grid'>"
        + render_summary_table("Predicted But Not In ADVP", result["predicted_issue_summary"], "reason", "count")
        + render_summary_table("In ADVP But Missing From Prediction", result["missing_prediction_summary"], "reason", "count")
        + render_summary_table("Missing ADVP Fields", result["missing_summary"], "field", "missing_rows")
        + "</div></section>"
        "<section class='tab-panel' data-panel='files'>"
        "<div class='subcard'>"
        f"<p><strong>Summary CSV</strong><br><code>{html.escape(str(result['summary_csv']))}</code></p>"
        f"<p><strong>Details JSON</strong><br><code>{html.escape(str(result['details_json']))}</code></p>"
        f"<p><strong>Fix File</strong><br><code>{html.escape(str(result['fix_file']))}</code></p>"
        f"<p><strong>Run Log</strong><br><code>{html.escape(str(result['run_log']))}</code></p>"
        "</div></section>"
        "</div>"
    )
    return "".join(sections)


def html_page(body: str) -> bytes:
    page = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>ADVP Curator Runner</title>
  <style>
    :root {{
      --bg: #edf2f7;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #5f6b7a;
      --line: #d9e1ea;
      --brand-900: #24364a;
      --brand-700: #3f5465;
      --brand-600: #52658e;
      --accent-100: #edf6ff;
      --accent-200: #d9ebff;
      --accent-300: #bdd9ff;
      --contrast-100: #fff3e8;
      --contrast-200: #ffe1c5;
      --contrast-500: #c96f2d;
      --gold-100: #fff8df;
      --rose-100: #fff0f0;
      --mint-100: #e9f8ee;
      --ok-bg: #ebf8f1;
      --ok-ink: #25603b;
      --bad-bg: #fff2f2;
      --bad-ink: #8f2c2c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at 0% 0%, #f8fbff 0, transparent 35%),
        radial-gradient(circle at 100% 100%, #e7eef8 0, transparent 35%),
        var(--bg);
      line-height: 1.45;
    }}
    .topbar {{
      background: linear-gradient(90deg, var(--brand-900), var(--brand-700));
      color: #fff;
      border-bottom: 2px solid #d2dbe5;
    }}
    .topbar-inner {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 12px 18px;
      display: flex;
      gap: 18px;
      align-items: center;
      justify-content: space-between;
    }}
    .brand {{ font-weight: 700; letter-spacing: .2px; }}
    .nav {{ color: #c9d6e4; font-size: 14px; }}
    .container {{ max-width: 1040px; margin: 28px auto; padding: 0 18px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 25px rgba(24, 45, 77, 0.08);
      padding: 24px;
    }}
    h2 {{ margin: 0 0 14px; font-size: 24px; }}
    p {{ margin: 8px 0; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    label {{
      display: block;
      margin: 14px 0 6px;
      font-size: 13px;
      font-weight: 700;
      color: #32485f;
      text-transform: uppercase;
      letter-spacing: .3px;
    }}
    input, textarea, select {{
      width: 100%;
      border: 1px solid #cfd8e3;
      background: #fbfdff;
      color: var(--ink);
      border-radius: 10px;
      padding: 11px 12px;
      font-size: 14px;
      outline: none;
    }}
    textarea {{ min-height: 130px; resize: vertical; }}
    input:focus, textarea:focus, select:focus {{ border-color: var(--brand-600); box-shadow: 0 0 0 3px rgba(82, 101, 142, .16); }}
    button {{
      margin-top: 16px;
      border: 0;
      border-radius: 12px;
      background: linear-gradient(90deg, var(--brand-700), var(--brand-600));
      color: #fff;
      font-weight: 700;
      font-size: 14px;
      padding: 0 18px;
      min-height: 46px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      line-height: 1;
    }}
    button:hover {{ filter: brightness(.97); }}
    .stats {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin: 14px 0 6px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
      background: #f9fbfd;
    }}
    .stat .k {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; text-transform: uppercase; }}
    .stat .v {{ font-size: 24px; font-weight: 700; }}
    .ok {{ background: var(--ok-bg); border-color: #c9e8d6; }}
    .ok .v {{ color: var(--ok-ink); }}
    .bad {{ background: var(--bad-bg); border-color: #f1d0d0; }}
    .bad .v {{ color: var(--bad-ink); }}
    .actions {{ margin-top: 12px; display: flex; gap: 10px; flex-wrap: wrap; }}
    .tab-shell {{ margin-top: 18px; }}
    .tab-bar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
      padding-bottom: 8px;
      border-bottom: 1px solid #dde5ee;
    }}
    .tab-btn {{
      margin: 0;
      padding: 9px 14px;
      background: #eef3f8;
      color: #36516c;
      border: 1px solid #d8e1ea;
      border-radius: 999px;
      font-weight: 700;
    }}
    .tab-btn.active {{
      background: linear-gradient(90deg, var(--brand-700), var(--brand-600));
      color: #fff;
      border-color: #425a77;
    }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .panel-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }}
    .subcard {{
      background: linear-gradient(180deg, #fcfdff 0%, #f7fbff 100%);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
    }}
    .metric-card {{
      background: linear-gradient(180deg, #fcfdff 0%, #f7fbff 100%);
    }}
    .card-title {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }}
    .title-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 44px;
      height: 28px;
      border-radius: 999px;
      background: #e9f0f7;
      color: #2b4966;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .3px;
      padding: 0 10px;
    }}
    h3 {{ margin: 0 0 10px; font-size: 16px; color: #2c425d; }}
    .metric-row {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 10px 0 12px; }}
    .mini-stat {{ background: #f5f8fb; border: 1px solid #dfe7ef; border-radius: 10px; padding: 10px; }}
    .mini-stat .k {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }}
    .mini-stat .v {{ font-size: 20px; font-weight: 700; color: #29415b; }}
    .toggle-box {{
      border-top: 1px solid #e1e8ef;
      margin-top: 12px;
      padding-top: 10px;
    }}
    .toggle-box summary {{
      cursor: pointer;
      font-weight: 700;
      color: #37506a;
      list-style: none;
    }}
    .toggle-box summary::-webkit-details-marker {{ display: none; }}
    .toggle-box summary::before {{
      content: ">";
      display: inline-block;
      margin-right: 8px;
      color: #5a7088;
      transform: rotate(0deg);
      transition: transform .15s ease;
    }}
    .toggle-box[open] summary::before {{
      transform: rotate(90deg);
    }}
    .check-list {{
      list-style: none;
      padding: 0;
      margin: 12px 0 0;
      display: grid;
      gap: 8px;
    }}
    .check-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid #e1e8ef;
      border-radius: 10px;
      background: #fff;
      color: #334a63;
      font-size: 13px;
    }}
    .option-toggle {{
      display: block;
      margin: 8px 0 2px;
      cursor: pointer;
    }}
    .option-toggle input {{
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }}
    .option-card {{
      display: flex;
      align-items: center;
      gap: 12px;
      border: 1px solid #d7e2ee;
      border-radius: 14px;
      padding: 12px 14px;
      background: linear-gradient(180deg, #fcfdff 0%, #f6faff 100%);
      transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease, background .15s ease;
    }}
    .option-toggle:hover .option-card {{
      border-color: #bdd0e5;
      box-shadow: 0 8px 20px rgba(58, 88, 126, 0.08);
      transform: translateY(-1px);
    }}
    .option-indicator {{
      width: 22px;
      height: 22px;
      border-radius: 999px;
      border: 2px solid #9db3ca;
      background: #fff;
      box-shadow: inset 0 0 0 4px #fff;
      flex: 0 0 auto;
      transition: border-color .15s ease, background .15s ease, box-shadow .15s ease;
    }}
    .option-copy {{
      display: grid;
      gap: 2px;
      min-width: 0;
    }}
    .option-title {{
      font-weight: 700;
      color: #2a4763;
      font-size: 14px;
    }}
    .option-help {{
      color: #607286;
      font-size: 12px;
      line-height: 1.4;
    }}
    .option-toggle input:checked + .option-card {{
      border-color: #7eaee6;
      background: linear-gradient(135deg, #dcecff 0%, #f3f8ff 48%, #ffffff 100%);
      box-shadow:
        inset 0 0 0 2px rgba(64, 106, 165, 0.16),
        0 10px 24px rgba(69, 112, 171, 0.12);
    }}
    .option-toggle input:checked + .option-card .option-indicator {{
      border-color: #2f5f95;
      background: #2f5f95;
      box-shadow: inset 0 0 0 4px #dcecff;
    }}
    .check-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      border-radius: 999px;
      background: #e8f6ee;
      color: #26764a;
      font-size: 12px;
      font-weight: 700;
      flex: 0 0 auto;
    }}
    .summary-table, .editor-table {{ width: 100%; border-collapse: collapse; }}
    .summary-table th, .summary-table td, .editor-table th, .editor-table td {{
      border-top: 1px solid #e4eaf1;
      padding: 14px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    .summary-table th, .editor-table th {{ color: #516377; font-size: 12px; text-transform: uppercase; letter-spacing: .25px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }}
    .editor-table {{ border-collapse: separate; border-spacing: 0 10px; }}
    .editor-table tbody tr td {{ background: #fff; }}
    .editor-table input {{ min-width: 120px; margin: 0; padding: 10px 11px; background: #fff; }}
    .cell-stack {{ display: grid; gap: 8px; min-width: 190px; }}
    .status-select, .comment-input {{ margin: 0; }}
    .edit-cell {{ border-radius: 14px; }}
    .edit-cell.mark-review {{ background: #fff6d8; }}
    .edit-cell.mark-issue {{ background: #ffe4e4; }}
    .edit-cell.mark-resolved {{ background: #e3f6e9; }}
    .edit-cell.mark-review .cell-stack {{
      background: #fff6d8;
      border: 1px solid #f0cf67;
      border-radius: 12px;
      padding: 10px;
    }}
    .edit-cell.mark-issue .cell-stack {{
      background: #ffe4e4;
      border: 1px solid #e89292;
      border-radius: 12px;
      padding: 10px;
    }}
    .edit-cell.mark-resolved .cell-stack {{
      background: #e3f6e9;
      border: 1px solid #86c89b;
      border-radius: 12px;
      padding: 10px;
    }}
    .ref-grid {{ display: grid; gap: 12px; margin-top: 12px; }}
    .ref-row {{
      text-decoration: none;
      color: inherit;
      border: 1px solid #dce5ef;
      border-radius: 12px;
      padding: 14px;
      background: linear-gradient(135deg, #fbfdff 0%, #f6faff 70%, #fff8ef 100%);
      display: grid;
      gap: 8px;
      transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
    }}
    .ref-row:hover {{
      transform: translateY(-1px);
      box-shadow: 0 8px 18px rgba(48, 79, 122, 0.08);
      border-color: #c7d9ee;
    }}
    .ref-row.current {{
      background: linear-gradient(135deg, #dcecff 0%, #f3f8ff 45%, #ffffff 100%);
      border-color: #7eaee6;
      box-shadow:
        inset 0 0 0 2px rgba(64, 106, 165, 0.18),
        0 10px 22px rgba(69, 112, 171, 0.12);
    }}
    .ref-title {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .table-badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      background: linear-gradient(90deg, #334c68, #4a6784);
      color: #fff;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .2px;
    }}
    .ref-row.current .table-badge {{
      background: linear-gradient(90deg, #244568, #3e72a6);
      box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.65);
    }}
    .ref-row.current .table-name {{
      color: #1f4f80;
    }}
    .table-name {{ font-weight: 700; color: #2b4560; }}
    .ref-url {{ word-break: break-all; }}
    .ref-actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .link-pill {{
      display: inline-block;
      background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
      border: 1px solid #d3deee;
      border-radius: 999px;
      padding: 8px 12px;
      color: #45617d;
      word-break: break-all;
    }}
    .btn-link {{
      display: inline-block;
      text-decoration: none;
      border: 1px solid #cfd8e3;
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
      color: #25425f;
      border-radius: 12px;
      padding: 0 16px;
      min-height: 46px;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
      font-size: 14px;
    }}
    .btn-link:hover {{ border-color: #adc4da; background: linear-gradient(180deg, #ffffff 0%, #eef5fc 100%); }}
    .btn-link.primary {{ background: linear-gradient(90deg, #2f4d68, #496788); border-color: #2f4d68; color: #fff; }}
    .btn-link.accent {{
      background: linear-gradient(180deg, #fff8f1 0%, var(--contrast-100) 100%);
      border-color: var(--contrast-200);
      color: #9a5523;
    }}
    .btn-link.accent:hover {{
      background: linear-gradient(180deg, #fffdf9 0%, #ffe9d3 100%);
      border-color: #f2cda9;
      color: #86471c;
    }}
    pre {{
      background: #f4f7fb;
      border: 1px solid #d9e1ea;
      border-radius: 10px;
      padding: 12px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    code {{
      background: #f1f5f9;
      border: 1px solid #d9e1ea;
      border-radius: 6px;
      padding: 1px 6px;
      font-family: "SFMono-Regular", Menlo, monospace;
      font-size: 12px;
    }}
    @media (max-width: 700px) {{
      .stats {{ grid-template-columns: 1fr; }}
      .panel-grid {{ grid-template-columns: 1fr; }}
      .metric-row {{ grid-template-columns: 1fr; }}
      .card {{ padding: 18px; }}
      .topbar-inner {{ padding: 10px 14px; }}
      h2 {{ font-size: 21px; }}
    }}
  </style>
</head>
<body>
<header class=\"topbar\">
  <div class=\"topbar-inner\">
    <div class=\"brand\">ADVP Curator</div>
    <div class=\"nav\">Publications · Variants · Association Records</div>
  </div>
</header>
<main class=\"container\">
  <div class=\"card\">{body}</div>
</main>
<script>
document.addEventListener("click", function (event) {{
  const btn = event.target.closest(".tab-btn");
  if (!btn) return;
  const shell = btn.closest(".tab-shell");
  if (!shell) return;
  const tabName = btn.getAttribute("data-tab");
  shell.querySelectorAll(".tab-btn").forEach((node) => node.classList.remove("active"));
  shell.querySelectorAll(".tab-panel").forEach((node) => node.classList.remove("active"));
  btn.classList.add("active");
  const panel = shell.querySelector(`.tab-panel[data-panel="${{tabName}}"]`);
  if (panel) panel.classList.add("active");
}});
</script>
</body>
</html>"""
    return page.encode("utf-8")


class AdvpUIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            form = """
<h2>Study Curation Runner</h2>
<p class="muted">Paste table links, select source type, then run end-to-end conversion and ADVP alignment.</p>
<details class="toggle-box" open>
<summary>What This Run Will Generate</summary>
<ul class="check-list">
  <li class="check-item"><span class="check-icon">1</span>Curated ADVP sheet(s)</li>
  <li class="check-item"><span class="check-icon">2</span>Row match accuracy and field accuracy summaries</li>
  <li class="check-item"><span class="check-icon">3</span>Mismatch report, fix-ready CSV, and missing-field summary</li>
</ul>
</details>
<form method="post" action="/run" enctype="multipart/form-data">
<label>Owner</label>
<input name="owner_name" value="" placeholder="Your name" />
<label>PMID</label>
<input name="pmid" value="30448613" required />
<label>Paper URL</label>
<input name="paper_input" value="https://pmc.ncbi.nlm.nih.gov/articles/PMC6331247/" placeholder="PMC article page or PDF link" />
<label>Or upload paper PDF</label>
<input type="file" name="paper_upload" accept=".pdf,application/pdf" />
<label>Source type</label>
<select name="source_type">
  <option value="pmc_table_link" selected>pmc_table_link</option>
</select>
<label>Table links (one per line)</label>
<textarea name="table_links" rows="6">https://pmc.ncbi.nlm.nih.gov/articles/PMC6331247/table/T1/
https://pmc.ncbi.nlm.nih.gov/articles/PMC6331247/table/T2/
https://pmc.ncbi.nlm.nih.gov/articles/PMC6331247/table/T3/</textarea>
<label class="option-toggle">
  <input type="checkbox" name="auto_discover_tables" value="1" />
  <span class="option-card">
    <span class="option-indicator"></span>
    <span class="option-copy">
      <span class="option-title">Auto-discover PMC tables when links are empty</span>
      <span class="option-help">Find candidate tables directly from the article page and select tables with genetics and association signals.</span>
    </span>
  </span>
</label>
<label>ADVP TSV path (relative to project root)</label>
<input name="advp_tsv" value="advp.variant.records.hg38.tsv" required />
<button type="submit">Run</button>
</form>
"""
            self.wfile.write(html_page(form))
            return

        if parsed.path == "/edit":
            q = parse_qs(parsed.query)
            path = q.get("path", [""])[0]
            p = Path(path).resolve()
            root = Path(os.getcwd()).resolve()
            if not str(p).startswith(str(root)) or not p.exists() or not p.is_file():
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"File not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_page(render_edit_table(str(p))))
            return

        if parsed.path == "/download":
            q = parse_qs(parsed.query)
            path = q.get("path", [""])[0]
            p = Path(path).resolve()
            root = Path(os.getcwd()).resolve()
            if not str(p).startswith(str(root)) or not p.exists() or not p.is_file():
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"File not found")
                return
            content_type = "application/octet-stream"
            if p.suffix.lower() == ".csv":
                content_type = "text/csv; charset=utf-8"
            elif p.suffix.lower() == ".json":
                content_type = "application/json; charset=utf-8"
            elif p.suffix.lower() == ".pdf":
                content_type = "application/pdf"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f"attachment; filename={p.name}")
            self.end_headers()
            self.wfile.write(p.read_bytes())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path not in ("/run", "/save-edits"):
            self.send_response(404)
            self.end_headers()
            return

        data = parse_request_data(self)

        try:
            if self.path == "/save-edits":
                path = Path(data.get("path", [""])[0]).resolve()
                root = Path(os.getcwd()).resolve()
                if not str(path).startswith(str(root)) or not path.exists() or not path.is_file():
                    raise ValueError("Invalid curated sheet path")
                save_result = save_curated_edits(path, data)
                body = (
                    "<h2>Curated Sheet Saved</h2>"
                    f"<p class='muted'>{save_result['updated_rows']} row(s) updated in <code>{html.escape(str(path))}</code>.</p>"
                    f"<p class='muted'>{save_result['annotated_cells']} annotated cell(s) saved. "
                    f"Annotations file: <code>{html.escape(save_result['annotation_path'])}</code></p>"
                    "<div class='actions'>"
                    f"<a class='btn-link primary' href='/edit?{urlencode({'path': str(path)})}'>Continue Editing</a>"
                    "<a class='btn-link' href='/'>Back</a>"
                    "</div>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_page(body))
                return

            pmid = int(data.get("pmid", [""])[0])
            owner_name = data.get("owner_name", [""])[0].strip()
            paper_input = data.get("paper_input", [""])[0].strip()
            paper_upload = data.get("paper_upload", [""])[0].strip() if data.get("paper_upload") else ""
            paper_input = paper_upload or paper_input
            source_type = data.get("source_type", ["pmc_table_link"])[0].strip()
            links_raw = data.get("table_links", [""])[0]
            auto_discover_tables = parse_bool(data.get("auto_discover_tables", [""])[0]) if data.get("auto_discover_tables") else False
            advp_tsv = data.get("advp_tsv", ["advp.variant.records.hg38.tsv"])[0].strip()
            links = parse_links(links_raw)
            if not paper_input:
                raise ValueError("Provide a paper URL or upload a PDF file")
            links, discovered = maybe_auto_discover_links(paper_input, links, auto_discover_tables)
            if not links:
                raise ValueError("No table links provided")

            result = run_pipeline(
                pmid=pmid,
                paper_input=paper_input,
                table_links=links,
                owner_name=owner_name,
                source_type=source_type,
                advp_tsv=advp_tsv,
                pred_scope="advp_like",
                key_mode="auto",
                ignore_bp=True,
                ignore_ra1=True,
                base_dir=os.getcwd(),
                auto_discovered_tables=discovered,
            )

            dl = "/download?" + urlencode({"path": result["fix_file"]})
            first_edit = ""
            if result["generated_harmonized"]:
                first_edit = "/edit?" + urlencode({"path": result["generated_harmonized"][0]})
            lines = [
                "<h2>Run Result</h2>",
                f"<p class=\"muted\">PMID {result['pmid']} · {result['table_count']} tables processed"
                + (f" · Owner {html.escape(result['owner_name'])}" if result["owner_name"] else "")
                + "</p>",
                render_tabbed_sections(result, dl, first_edit),
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_page("\n".join(lines)))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = (
                "<h2>Run Failed</h2>"
                "<p class=\"muted\">Pipeline encountered an error. Inspect the message below.</p>"
                f"<pre>{html.escape(str(e))}</pre>"
                "<div class=\"actions\"><a class=\"btn-link\" href='/'>Back</a></div>"
            )
            self.wfile.write(html_page(msg))


def cli_mode(args: argparse.Namespace) -> None:
    links = parse_links(args.table_links)
    links, discovered = maybe_auto_discover_links(args.paper_input, links, args.auto_discover_tables)
    result = run_pipeline(
        pmid=args.pmid,
        paper_input=args.paper_input,
        table_links=links,
        owner_name=args.owner_name,
        source_type=args.source_type,
        advp_tsv=args.advp_tsv,
        pred_scope=args.pred_scope,
        key_mode=args.key_mode,
        ignore_bp=args.ignore_bp,
        ignore_ra1=args.ignore_ra1,
        base_dir=args.base_dir,
        auto_discovered_tables=discovered,
    )
    print(f"PMID: {result['pmid']}")
    print(f"Tables processed: {result['table_count']}")
    print(f"Success rows: {result['success_rows']}")
    print(f"Failed rows: {result['failed_rows']}")
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Fix file: {result['fix_file']}")


def web_mode(args: argparse.Namespace) -> None:
    server = HTTPServer((args.host, args.port), AdvpUIHandler)
    print(f"Serving ADVP UI at http://{args.host}:{args.port}")
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ADVP curator workflow runner (CLI + minimal web UI)")
    sub = p.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run", help="Run pipeline from CLI")
    run.add_argument("--owner_name", default="", help="Person responsible for this run")
    run.add_argument("--pmid", type=int, required=True)
    run.add_argument("--paper_input", required=True, help="Paper PDF path or URL")
    run.add_argument("--table_links", default="", help="Comma or newline-separated links")
    run.add_argument("--auto_discover_tables", action="store_true", help="Auto-discover relevant PMC tables when links are omitted")
    run.add_argument("--source_type", default="pmc_table_link", choices=["pmc_table_link"])
    run.add_argument("--advp_tsv", default="advp.variant.records.hg38.tsv")
    run.add_argument("--pred_scope", default="advp_like", choices=["raw", "advp_like"])
    run.add_argument("--key_mode", default="auto", choices=["auto", "pmid_snp_pvalue"])
    run.add_argument("--ignore_bp", action="store_true", help="Ignore BP during evaluation key matching")
    run.add_argument("--no-ignore_bp", dest="ignore_bp", action="store_false", help="Use BP during evaluation")
    run.add_argument("--ignore_ra1", action="store_true", help="Ignore RA1 during evaluation key matching")
    run.add_argument("--no-ignore_ra1", dest="ignore_ra1", action="store_false", help="Use RA1 during evaluation")
    run.set_defaults(ignore_bp=True)
    run.set_defaults(ignore_ra1=True)
    run.add_argument("--base_dir", default=os.getcwd())

    web = sub.add_parser("web", help="Start minimal local web form")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8899)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "run":
        cli_mode(args)
    elif args.mode == "web":
        web_mode(args)


if __name__ == "__main__":
    main()
