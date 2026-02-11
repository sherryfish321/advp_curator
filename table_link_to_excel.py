import argparse
import re
from typing import List, Optional, Tuple

import pandas as pd

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def fetch_html(url: str, timeout: int = 60) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed. Run: python3 -m pip install requests")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.text


def _extract_pmc_info(url: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"/articles/(PMC\d+)(?:/table/([^/?#]+)/?)?", url, flags=re.I)
    if not m:
        return None, None
    pmcid = m.group(1).upper()
    table_id = m.group(2)
    return pmcid, table_id


def fetch_pmc_fulltext_xml(pmcid: str, timeout: int = 60) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed. Run: python3 -m pip install requests")
    api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/xml,text/xml,*/*"}
    resp = requests.get(api_url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.text


def parse_html_table(table_tag) -> List[List[str]]:
    """
    Parse a table with rowspan/colspan into a fully expanded 2D matrix.
    """
    rows_out: List[List[Optional[str]]] = []
    trs = table_tag.find_all("tr")

    for r_idx, tr in enumerate(trs):
        while len(rows_out) <= r_idx:
            rows_out.append([])
        row = rows_out[r_idx]

        cells = tr.find_all(["th", "td"])
        c_idx = 0
        for cell in cells:
            while c_idx < len(row) and row[c_idx] is not None:
                c_idx += 1

            txt = _clean_text(cell.get_text(" ", strip=True))
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            for rr in range(r_idx, r_idx + rowspan):
                while len(rows_out) <= rr:
                    rows_out.append([])
                target_row = rows_out[rr]
                while len(target_row) < c_idx + colspan:
                    target_row.append(None)
                for cc in range(c_idx, c_idx + colspan):
                    if target_row[cc] is None:
                        target_row[cc] = txt

            c_idx += colspan

    max_len = max((len(r) for r in rows_out), default=0)
    final_rows: List[List[str]] = []
    for r in rows_out:
        padded = r + [None] * (max_len - len(r))
        final_rows.append([x if x is not None else "" for x in padded])
    return final_rows


def table_to_dataframe(table_tag) -> pd.DataFrame:
    matrix = parse_html_table(table_tag)
    if not matrix:
        return pd.DataFrame()
    return pd.DataFrame(matrix)


def sanitize_sheet_name(name: str, fallback: str) -> str:
    name = _clean_text(name)
    if not name:
        name = fallback
    name = re.sub(r"[\\/*?:\[\]]", "_", name)
    return name[:31]


def pick_table(soup, table_id: Optional[str], table_selector: Optional[str], table_index: int):
    if table_id:
        t = soup.find("table", {"id": table_id})
        if t is None:
            wrap = soup.find("table-wrap", {"id": table_id})
            if wrap is not None:
                t = wrap.find("table")
        if t is None:
            raise ValueError(f"Cannot find table with id={table_id}")
        return [t]

    if table_selector:
        tables = soup.select(table_selector)
    else:
        tables = soup.find_all("table")

    if not tables:
        raise ValueError("No <table> found on page.")

    if table_index < 0 or table_index >= len(tables):
        raise ValueError(f"table_index {table_index} is out of range. Found {len(tables)} tables.")

    return [tables[table_index]]


def main():
    parser = argparse.ArgumentParser(description="Download table(s) from a URL and save as Excel.")
    parser.add_argument("--url", required=True, help="Web page URL that contains table(s)")
    parser.add_argument("--out", default="table_from_url.xlsx", help="Output xlsx path")
    parser.add_argument("--table_id", default=None, help="Specific table id attribute to target")
    parser.add_argument("--table_selector", default=None, help="CSS selector for table(s), e.g. '#Tab2 table'")
    parser.add_argument("--table_index", type=int, default=0, help="0-based table index if multiple tables exist")
    parser.add_argument("--all_tables", action="store_true", help="Export all matched tables into separate sheets")
    args = parser.parse_args()

    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is not installed. Run: python3 -m pip install beautifulsoup4")

    pmcid, url_table_id = _extract_pmc_info(args.url)
    resolved_table_id = args.table_id or url_table_id

    html = ""
    soup = None
    parse_with_xml = False
    try:
        html = fetch_html(args.url)
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:
        # PMC pages may return 403 to scripts; fallback to Europe PMC XML.
        if pmcid:
            xml = fetch_pmc_fulltext_xml(pmcid)
            soup = BeautifulSoup(xml, "xml")
            parse_with_xml = True
        else:
            raise e

    if parse_with_xml and not resolved_table_id and url_table_id:
        resolved_table_id = url_table_id

    if resolved_table_id:
        tables = pick_table(soup, table_id=resolved_table_id, table_selector=None, table_index=args.table_index)
    elif args.all_tables:
        if parse_with_xml:
            wraps = soup.find_all("table-wrap")
            tables = [w.find("table") for w in wraps if w.find("table") is not None]
        else:
            tables = soup.select(args.table_selector) if args.table_selector else soup.find_all("table")
        if not tables:
            raise ValueError("No matched tables found.")
    else:
        tables = pick_table(
            soup,
            table_id=None,
            table_selector=args.table_selector,
            table_index=args.table_index,
        )

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        for i, table in enumerate(tables, start=1):
            df = table_to_dataframe(table)
            caption = ""
            cap = table.find("caption")
            if cap:
                caption = _clean_text(cap.get_text(" ", strip=True))
            sheet = sanitize_sheet_name(caption, fallback=f"table_{i}")
            df.to_excel(writer, index=False, header=False, sheet_name=sheet)

    print(f"Saved {len(tables)} table(s) to {args.out}")


if __name__ == "__main__":
    main()
