#!/usr/bin/env python3
"""MCP-friendly callable service layer for the ADVP curator workflow.

This module keeps UI state out of the core curation calls.  It intentionally
delegates table extraction and ADVP mapping to the existing implementation so
the current web behavior and extraction heuristics stay unchanged.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from table_link_to_excel import discover_relevant_pmc_tables


@dataclass(frozen=True)
class AdvpServiceConfig:
    project_root: Path
    table_dir: Path
    harmonized_dir: Path
    audit_dir: Path
    eval_dir: Path
    run_log_dir: Path
    advp_tsv: Path

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path,
        advp_tsv: str | Path = "advp.variant.records.hg38.tsv",
    ) -> "AdvpServiceConfig":
        root = Path(project_root).resolve()
        advp_path = Path(advp_tsv)
        if not advp_path.is_absolute():
            advp_path = root / advp_path
        return cls(
            project_root=root,
            table_dir=root / "table_from_link",
            harmonized_dir=root / "harmonized_tables_advp",
            audit_dir=root / "audit",
            eval_dir=root / "eval_reports",
            run_log_dir=root / "ui_run_logs",
            advp_tsv=advp_path.resolve(),
        )

    def ensure_output_dirs(self) -> None:
        for path in (self.table_dir, self.harmonized_dir, self.audit_dir, self.eval_dir, self.run_log_dir):
            path.mkdir(parents=True, exist_ok=True)

    def to_json_dict(self) -> Dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _run_cmd(cmd: List[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def _append_run_log(log_dir: Path, payload: Dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pmid_{payload['pmid']}_run_{ts}.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return log_path


def _excel_preview(path: Path, limit: int = 5) -> Dict[str, Any]:
    if not path.exists():
        return {"columns": [], "row_count": 0, "rows": []}
    df = pd.read_excel(path)
    row_count = int(len(df))
    df = df.head(limit).where(pd.notna(df.head(limit)), "")
    return {
        "columns": list(df.columns),
        "row_count": row_count,
        "rows": df.to_dict(orient="records"),
    }


def _audit_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"paper_source_errors": [], "table_source_errors": [], "needs_review_count": 0}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    needs_review_count = 0
    for record in data.get("record_field_audit", []) or []:
        for field in record.get("fields", []) or []:
            if field.get("needs_review"):
                needs_review_count += 1
    return {
        "paper_source_errors": data.get("paper_source_errors", []) or [],
        "table_source_errors": data.get("table_source_errors", []) or [],
        "needs_review_count": needs_review_count,
        "paper": data.get("paper", {}) or {},
    }


def _eval_workbook_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"sheets": [], "summary": [], "field_accuracy": []}
    workbook = pd.ExcelFile(path)
    payload: Dict[str, Any] = {"sheets": workbook.sheet_names}
    for sheet_name in ("summary", "field_accuracy"):
        if sheet_name not in workbook.sheet_names:
            payload[sheet_name] = []
            continue
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.where(pd.notna(df), "")
        payload[sheet_name] = df.to_dict(orient="records")
    return payload


def _safe_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("_")
    return cleaned or "advp_run"


def discover_pmc_tables(article_url: str, min_score: int = 5) -> Dict[str, Any]:
    """MCP tool candidate: discover relevant PMC tables for a paper URL."""
    tables = discover_relevant_pmc_tables(article_url, min_score=min_score)
    return {
        "schema_version": "advp.mcp.v1",
        "tool": "discover_pmc_tables",
        "article_url": article_url,
        "min_score": min_score,
        "tables": tables,
        "selected_links": [item["link"] for item in tables if item.get("selected")],
    }


def extract_table(table_url: str, output_path: str | Path, config: AdvpServiceConfig) -> Dict[str, Any]:
    """MCP tool candidate: save one PMC/web table URL as an Excel file."""
    config.ensure_output_dirs()
    out = Path(output_path)
    if not out.is_absolute():
        out = config.project_root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    stdout = _run_cmd(
        [
            sys.executable,
            str(config.project_root / "table_link_to_excel.py"),
            "--url",
            table_url,
            "--out",
            str(out),
        ],
        cwd=config.project_root,
    )
    return {
        "schema_version": "advp.mcp.v1",
        "tool": "extract_table",
        "source": table_url,
        "output_path": str(out),
        "table_preview": _excel_preview(out, limit=5),
        "stdout": stdout,
    }


def map_to_advp(
    paper_input: str,
    table_input: str | Path,
    output_path: str | Path,
    audit_path: str | Path,
    paper_id: str,
    pmcid: str,
    config: AdvpServiceConfig,
    validation_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """MCP tool candidate: run existing ADVP mapping on an extracted table."""
    config.ensure_output_dirs()
    out = Path(output_path)
    audit = Path(audit_path)
    validation = Path(validation_path) if validation_path else None
    if not out.is_absolute():
        out = config.project_root / out
    if not audit.is_absolute():
        audit = config.project_root / audit
    if validation is not None and not validation.is_absolute():
        validation = config.project_root / validation
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    if validation is not None:
        validation.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(config.project_root / "advp_curator.py"),
        "--input",
        paper_input,
        "--table_input",
        str(table_input),
        "--out",
        str(out),
        "--audit",
        str(audit),
        "--paper_id",
        paper_id,
        "--pmcid",
        pmcid or "NR",
    ]
    if validation is not None:
        cmd.extend(["--validation", str(validation)])
    stdout = _run_cmd(cmd, cwd=config.project_root)
    return {
        "schema_version": "advp.mcp.v1",
        "tool": "map_to_advp",
        "paper_input": paper_input,
        "table_input": str(table_input),
        "output_path": str(out),
        "audit_path": str(audit),
        "validation_path": str(validation) if validation is not None else "",
        "records_preview": _excel_preview(out, limit=5),
        "audit_summary": _audit_summary(audit),
        "stdout": stdout,
    }


def evaluate_against_advp(
    pmid: int,
    input_paths: List[str | Path],
    config: AdvpServiceConfig,
    pred_scope: str = "advp_like",
    key_mode: str = "auto",
    ignore_bp: bool = True,
    ignore_ra1: bool = True,
) -> Dict[str, Any]:
    """MCP tool candidate: evaluate mapped files against the ADVP TSV."""
    config.ensure_output_dirs()
    cmd = [
        sys.executable,
        str(config.project_root / "evaluate_advp_alignment.py"),
        "--advp_tsv",
        str(config.advp_tsv),
        "--pmid",
        str(pmid),
        "--inputs",
    ]
    cmd.extend(str(Path(p)) for p in input_paths)
    cmd.extend([
        "--pred_scope",
        pred_scope,
        "--key_mode",
        key_mode,
        "--out_dir",
        str(config.eval_dir),
    ])
    if ignore_bp:
        cmd.append("--ignore_bp")
    if ignore_ra1:
        cmd.append("--ignore_ra1")
    stdout = _run_cmd(cmd, cwd=config.project_root)
    report_path = config.eval_dir / f"pmid_{pmid}_eval_report_{pred_scope}.xlsx"
    return {
        "schema_version": "advp.mcp.v1",
        "tool": "evaluate_against_advp",
        "pmid": pmid,
        "inputs": [str(Path(p)) for p in input_paths],
        "eval_report": str(report_path),
        "eval_summary": _eval_workbook_summary(report_path),
        "stdout": stdout,
    }


def open_curated_sheet(path: str | Path, limit: Optional[int] = None) -> Dict[str, Any]:
    """MCP tool candidate: read a curated sheet as machine-readable JSON."""
    sheet_path = Path(path).resolve()
    if not sheet_path.exists() or not sheet_path.is_file():
        raise FileNotFoundError(f"Curated sheet not found: {sheet_path}")
    if sheet_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(sheet_path)
    elif sheet_path.suffix.lower() == ".csv":
        df = pd.read_csv(sheet_path)
    else:
        raise ValueError(f"Unsupported curated sheet type: {sheet_path.suffix}")
    if limit is not None:
        df = df.head(limit)
    df = df.where(pd.notna(df), "")
    return {
        "schema_version": "advp.mcp.v1",
        "tool": "open_curated_sheet",
        "path": str(sheet_path),
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "rows": df.to_dict(orient="records"),
    }


def run_curation_workflow(
    pmid: int,
    pmcid: str,
    paper_input: str,
    table_links: List[str],
    owner_name: str,
    source_type: str,
    config: AdvpServiceConfig,
    pred_scope: str = "advp_like",
    key_mode: str = "auto",
    ignore_bp: bool = True,
    ignore_ra1: bool = True,
    auto_discovered_tables: Optional[List[Dict[str, Any]]] = None,
    table_number_resolver=None,
) -> Dict[str, Any]:
    """Callable end-to-end workflow for UI, CLI, or MCP wrappers."""
    if source_type != "pmc_table_link":
        raise ValueError("Only source_type=pmc_table_link is supported in current MVP")
    config.ensure_output_dirs()

    def default_table_num(link: str, fallback: int) -> int:
        import re

        patterns = [
            r"/table/T(\d+)/?$",
            r"/table/Tab(\d+)/?$",
            r"/table/[^/?#]*tbl[-_]?0*(\d+)/?$",
            r"/table/[^/?#]*table[-_]?0*(\d+)/?$",
            r"/table/[^/?#]*-T(\d+)/?$",
            r"[?&]table=T(\d+)\b",
            r"[?&]table_id=T(\d+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, link, flags=re.I)
            if match:
                return int(match.group(1))
        return fallback

    resolve_table_num = table_number_resolver or default_table_num
    generated_tables: List[Path] = []
    generated_harmonized: List[Path] = []

    for idx, link in enumerate(table_links, start=1):
        table_num = resolve_table_num(link, idx)
        tag = f"{pmid}_table{table_num}"
        table_xlsx = config.table_dir / f"{tag}.xlsx"
        harmonized_xlsx = config.harmonized_dir / f"from_{tag}_v1.xlsx"
        audit_json = config.audit_dir / f"from_{tag}_audit.json"

        extract_table(link, table_xlsx, config)
        map_to_advp(
            paper_input=paper_input,
            table_input=table_xlsx,
            output_path=harmonized_xlsx,
            audit_path=audit_json,
            paper_id=tag,
            pmcid=pmcid,
            config=config,
        )
        generated_tables.append(table_xlsx)
        generated_harmonized.append(harmonized_xlsx)

    evaluation = evaluate_against_advp(
        pmid=pmid,
        input_paths=generated_harmonized,
        config=config,
        pred_scope=pred_scope,
        key_mode=key_mode,
        ignore_bp=ignore_bp,
        ignore_ra1=ignore_ra1,
    )
    run_log = _append_run_log(
        config.run_log_dir,
        {
            "schema_version": "advp.mcp.v1",
            "pmid": pmid,
            "pmcid": pmcid,
            "owner_name": owner_name,
            "paper_input": paper_input,
            "table_links": table_links,
            "auto_discovered_tables": auto_discovered_tables or [],
            "generated_tables": [str(p) for p in generated_tables],
            "generated_harmonized": [str(p) for p in generated_harmonized],
            "eval_report": evaluation["eval_report"],
            "config": config.to_json_dict(),
            "created_at": dt.datetime.now().isoformat(),
        },
    )
    return {
        "schema_version": "advp.mcp.v1",
        "tool": "run_full_curation",
        "pmid": pmid,
        "pmcid": pmcid,
        "owner_name": owner_name,
        "generated_tables": [str(p) for p in generated_tables],
        "generated_harmonized": [str(p) for p in generated_harmonized],
        "eval_report": evaluation["eval_report"],
        "run_log": str(run_log),
        "table_count": len(generated_harmonized),
        "auto_discovered_tables": auto_discovered_tables or [],
        "records_preview": [
            {"path": str(path), **_excel_preview(path, limit=5)}
            for path in generated_harmonized
        ],
        "eval_summary": evaluation.get("eval_summary", {}),
    }


def run_full_curation(
    pmid: int,
    paper_input: str,
    table_links: Optional[List[str]] = None,
    config: Optional[AdvpServiceConfig] = None,
    pmcid: str = "",
    owner_name: str = "",
    auto_discover_tables: bool = False,
    pred_scope: str = "advp_like",
    key_mode: str = "auto",
    ignore_bp: bool = True,
    ignore_ra1: bool = True,
) -> Dict[str, Any]:
    """MCP convenience tool: discover optional tables and run the full workflow."""
    if config is None:
        config = AdvpServiceConfig.from_project_root(Path.cwd())
    links = list(table_links or [])
    discovered: List[Dict[str, Any]] = []
    if not links and auto_discover_tables:
        discovery = discover_pmc_tables(paper_input)
        discovered = discovery["tables"]
        links = discovery["selected_links"]
    if not links:
        raise ValueError("No table links provided or discovered.")
    return run_curation_workflow(
        pmid=pmid,
        pmcid=pmcid,
        paper_input=paper_input,
        table_links=links,
        owner_name=owner_name,
        source_type="pmc_table_link",
        config=config,
        pred_scope=pred_scope,
        key_mode=key_mode,
        ignore_bp=ignore_bp,
        ignore_ra1=ignore_ra1,
        auto_discovered_tables=discovered,
    )
