#!/usr/bin/env python3
"""MCP server wrapper for the ADVP curator service layer.

Run with:
    python3 advp_mcp_server.py

The server uses stdio transport through the official Python MCP SDK.  Set
ADVP_PROJECT_ROOT and ADVP_TSV to override the default project paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from advp_services import (
    AdvpServiceConfig,
    discover_pmc_tables as service_discover_pmc_tables,
    evaluate_against_advp as service_evaluate_against_advp,
    extract_table as service_extract_table,
    map_to_advp as service_map_to_advp,
    open_curated_sheet as service_open_curated_sheet,
    run_full_curation as service_run_full_curation,
)

try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    FastMCP = None


def _config() -> AdvpServiceConfig:
    root = Path(os.environ.get("ADVP_PROJECT_ROOT", Path.cwd())).resolve()
    advp_tsv = os.environ.get("ADVP_TSV", "advp.variant.records.hg38.tsv")
    return AdvpServiceConfig.from_project_root(root, advp_tsv=advp_tsv)


def _project_path(path: str, *, default_dir: Optional[Path] = None, must_exist: bool = False) -> Path:
    config = _config()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (default_dir or config.project_root) / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(config.project_root)
    except ValueError as exc:
        raise ValueError(f"Path is outside ADVP_PROJECT_ROOT: {candidate}") from exc
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Path not found: {candidate}")
    return candidate


def create_server():
    if FastMCP is None:
        raise RuntimeError(
            "The Python MCP SDK is not installed. Install it with `python3 -m pip install mcp`."
        )

    mcp = FastMCP("advp-curator")

    @mcp.tool()
    def discover_pmc_tables(article_url: str, min_score: int = 5) -> Dict[str, Any]:
        """Discover candidate association tables from a PMC article URL."""
        return service_discover_pmc_tables(article_url=article_url, min_score=min_score)

    @mcp.tool()
    def extract_table(table_url: str, output_path: str) -> Dict[str, Any]:
        """Extract a PMC/web table to an Excel file under ADVP_PROJECT_ROOT."""
        config = _config()
        safe_output = _project_path(output_path, default_dir=config.table_dir)
        return service_extract_table(table_url=table_url, output_path=safe_output, config=config)

    @mcp.tool()
    def map_to_advp(
        paper_input: str,
        table_input: str,
        output_path: str,
        audit_path: str,
        paper_id: str,
        pmcid: str = "NR",
        validation_path: str = "",
    ) -> Dict[str, Any]:
        """Map an extracted table into the ADVP curated schema."""
        config = _config()
        safe_table_input = _project_path(table_input, must_exist=True)
        safe_output = _project_path(output_path, default_dir=config.harmonized_dir)
        safe_audit = _project_path(audit_path, default_dir=config.audit_dir)
        safe_validation = _project_path(validation_path, default_dir=config.audit_dir) if validation_path else None
        return service_map_to_advp(
            paper_input=paper_input,
            table_input=safe_table_input,
            output_path=safe_output,
            audit_path=safe_audit,
            paper_id=paper_id,
            pmcid=pmcid,
            config=config,
            validation_path=safe_validation,
        )

    @mcp.tool()
    def evaluate_against_advp(
        pmid: int,
        inputs: List[str],
        pred_scope: str = "advp_like",
        key_mode: str = "auto",
        ignore_bp: bool = True,
        ignore_ra1: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate mapped ADVP output against the ADV-P TSV."""
        config = _config()
        safe_inputs = [_project_path(path, must_exist=True) for path in inputs]
        return service_evaluate_against_advp(
            pmid=pmid,
            input_paths=safe_inputs,
            config=config,
            pred_scope=pred_scope,
            key_mode=key_mode,
            ignore_bp=ignore_bp,
            ignore_ra1=ignore_ra1,
        )

    @mcp.tool()
    def open_curated_sheet(path: str, limit: Optional[int] = 25) -> Dict[str, Any]:
        """Open a curated Excel/CSV sheet under ADVP_PROJECT_ROOT and return rows as JSON."""
        safe_path = _project_path(path, must_exist=True)
        return service_open_curated_sheet(path=safe_path, limit=limit)

    @mcp.tool()
    def run_full_curation(
        pmid: int,
        paper_input: str,
        table_links: Optional[List[str]] = None,
        pmcid: str = "",
        owner_name: str = "",
        auto_discover_tables: bool = False,
        pred_scope: str = "advp_like",
        key_mode: str = "auto",
        ignore_bp: bool = True,
        ignore_ra1: bool = True,
    ) -> Dict[str, Any]:
        """Run discovery/extraction/mapping/evaluation as one agent-friendly workflow."""
        return service_run_full_curation(
            pmid=pmid,
            paper_input=paper_input,
            table_links=table_links or [],
            config=_config(),
            pmcid=pmcid,
            owner_name=owner_name,
            auto_discover_tables=auto_discover_tables,
            pred_scope=pred_scope,
            key_mode=key_mode,
            ignore_bp=ignore_bp,
            ignore_ra1=ignore_ra1,
        )

    return mcp


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
