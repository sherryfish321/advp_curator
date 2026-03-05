#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import html
import os
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlencode, urlparse


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


def collect_mismatch_paths(out_dir: Path, pmid: int, pred_scope: str, harmonized_paths: List[Path]) -> List[Path]:
    paths = []
    for hp in harmonized_paths:
        mismatch = out_dir / f"pmid_{pmid}_mismatch_{hp.name}_{pred_scope}.csv"
        if mismatch.exists():
            paths.append(mismatch)
    return paths


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
    return {"pred_total": pred_total}


def run_pipeline(
    pmid: int,
    paper_input: str,
    table_links: List[str],
    source_type: str,
    advp_tsv: str,
    pred_scope: str,
    key_mode: str,
    ignore_bp: bool,
    base_dir: str,
) -> Dict[str, object]:
    if source_type != "pmc_table_link":
        raise ValueError("Only source_type=pmc_table_link is supported in current MVP")

    root = Path(base_dir).resolve()
    table_dir = root / "table_from_link"
    harmonized_dir = root / "harmonized_tables_advp"
    audit_dir = root / "audit"
    eval_dir = root / "eval_reports"
    for d in (table_dir, harmonized_dir, audit_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    generated_tables = []
    generated_harmonized = []

    for idx, link in enumerate(table_links, start=1):
        tag = f"{pmid}_table{idx}"
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

    run_cmd(eval_cmd)

    summary_csv = eval_dir / f"pmid_{pmid}_summary_{pred_scope}.csv"
    mismatch_paths = collect_mismatch_paths(eval_dir, pmid, pred_scope, generated_harmonized)
    fix_file = build_fix_file(mismatch_paths, eval_dir, pmid)

    mismatch_counts = summarize_counts(mismatch_paths)
    summary_counts = read_summary_metrics(summary_csv, generated_harmonized)
    success_rows = max(0, summary_counts["pred_total"] - mismatch_counts["failed_rows"])

    return {
        "pmid": pmid,
        "generated_tables": [str(p) for p in generated_tables],
        "generated_harmonized": [str(p) for p in generated_harmonized],
        "summary_csv": str(summary_csv),
        "fix_file": str(fix_file),
        "success_rows": success_rows,
        "failed_rows": mismatch_counts["failed_rows"],
        "table_count": len(generated_harmonized),
    }


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
      border-radius: 10px;
      background: linear-gradient(90deg, var(--brand-700), var(--brand-600));
      color: #fff;
      font-weight: 700;
      font-size: 14px;
      padding: 11px 16px;
      cursor: pointer;
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
    .btn-link {{
      display: inline-block;
      text-decoration: none;
      border: 1px solid #cfd8e3;
      background: #fff;
      color: #25425f;
      border-radius: 10px;
      padding: 9px 12px;
      font-weight: 600;
      font-size: 14px;
    }}
    .btn-link.primary {{ background: #2f4d68; border-color: #2f4d68; color: #fff; }}
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
<form method="post" action="/run">
<label>PMID</label>
<input name="pmid" value="30448613" required />
<label>Paper path (PDF)</label>
<input name="paper_input" value="/Users/sherryhuang/advp_curator/paper/nihms-1510343.pdf" required />
<label>Source type</label>
<select name="source_type">
  <option value="pmc_table_link" selected>pmc_table_link</option>
</select>
<label>Table links (one per line)</label>
<textarea name="table_links" rows="6" required>https://pmc.ncbi.nlm.nih.gov/articles/PMC6331247/table/T1/
https://pmc.ncbi.nlm.nih.gov/articles/PMC6331247/table/T2/
https://pmc.ncbi.nlm.nih.gov/articles/PMC6331247/table/T3/</textarea>
<label>ADVP TSV path (relative to project root)</label>
<input name="advp_tsv" value="advp.variant.records.hg38.tsv" required />
<button type="submit">Run</button>
</form>
"""
            self.wfile.write(html_page(form))
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
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f"attachment; filename={p.name}")
            self.end_headers()
            self.wfile.write(p.read_bytes())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        data = parse_qs(raw)

        try:
            pmid = int(data.get("pmid", [""])[0])
            paper_input = data.get("paper_input", [""])[0].strip()
            source_type = data.get("source_type", ["pmc_table_link"])[0].strip()
            links_raw = data.get("table_links", [""])[0]
            advp_tsv = data.get("advp_tsv", ["advp.variant.records.hg38.tsv"])[0].strip()
            links = parse_links(links_raw)
            if not links:
                raise ValueError("No table links provided")

            result = run_pipeline(
                pmid=pmid,
                paper_input=paper_input,
                table_links=links,
                source_type=source_type,
                advp_tsv=advp_tsv,
                pred_scope="advp_like",
                key_mode="auto",
                ignore_bp=True,
                base_dir=os.getcwd(),
            )

            dl = "/download?" + urlencode({"path": result["fix_file"]})
            lines = [
                "<h2>Run Result</h2>",
                f"<p class=\"muted\">PMID {result['pmid']} · {result['table_count']} tables processed</p>",
                "<div class=\"stats\">",
                f"<div class=\"stat ok\"><span class=\"k\">Success Rows</span><span class=\"v\">{result['success_rows']}</span></div>",
                f"<div class=\"stat bad\"><span class=\"k\">Failed Rows</span><span class=\"v\">{result['failed_rows']}</span></div>",
                "</div>",
                f"<p>Summary CSV: <code>{html.escape(result['summary_csv'])}</code></p>",
                f"<p>Fix file: <code>{html.escape(result['fix_file'])}</code></p>",
                "<div class=\"actions\">",
                f"<a class=\"btn-link primary\" href=\"{dl}\">Download {html.escape(Path(result['fix_file']).name)}</a>",
                "<a class=\"btn-link\" href=\"/\">Run Another</a>",
                "</div>",
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
    result = run_pipeline(
        pmid=args.pmid,
        paper_input=args.paper_input,
        table_links=links,
        source_type=args.source_type,
        advp_tsv=args.advp_tsv,
        pred_scope=args.pred_scope,
        key_mode=args.key_mode,
        ignore_bp=args.ignore_bp,
        base_dir=args.base_dir,
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
    run.add_argument("--pmid", type=int, required=True)
    run.add_argument("--paper_input", required=True, help="Paper PDF path")
    run.add_argument("--table_links", required=True, help="Comma or newline-separated links")
    run.add_argument("--source_type", default="pmc_table_link", choices=["pmc_table_link"])
    run.add_argument("--advp_tsv", default="advp.variant.records.hg38.tsv")
    run.add_argument("--pred_scope", default="advp_like", choices=["raw", "advp_like"])
    run.add_argument("--key_mode", default="auto", choices=["auto", "pmid_snp_pvalue"])
    run.add_argument("--ignore_bp", action="store_true", help="Ignore BP during evaluation key matching")
    run.add_argument("--no-ignore_bp", dest="ignore_bp", action="store_false", help="Use BP during evaluation")
    run.set_defaults(ignore_bp=True)
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
