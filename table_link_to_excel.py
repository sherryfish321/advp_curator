import argparse
import io
import re
import subprocess
import tarfile
from typing import Dict, List, Optional, Tuple

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
    if resp.ok and resp.text:
        return resp.text
    # Fallback: some PMC pages block Python requests but allow curl.
    if resp.status_code == 403:
        curl_text = _curl_get_text(url, timeout=timeout)
        if curl_text:
            return curl_text
    resp.raise_for_status()
    return resp.text


def _curl_get_text(url: str, timeout: int = 60) -> Optional[str]:
    try:
        cmd = [
            "curl",
            "-L",
            "--max-time",
            str(timeout),
            "-A",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "-H",
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            url,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout
    except Exception:
        pass
    return None


def _extract_pmc_info(url: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"/articles/(PMC\d+)(?:/table/([^/?#]+)/?)?", url, flags=re.I)
    if not m:
        m2 = re.search(r"/articles/(PMC\d+)/?", url, flags=re.I)
        if not m2:
            return None, None
        pmcid = m2.group(1).upper()
        q = re.search(r"[?&]table_id=([^&#]+)", url, flags=re.I)
        table_id = q.group(1) if q else None
        return pmcid, table_id
    pmcid = m.group(1).upper()
    table_id = m.group(2)
    return pmcid, table_id


def _try_pmc_direct_table_download(pmcid: str, table_id: str, timeout: int = 60) -> Optional[List[pd.DataFrame]]:
    if requests is None:
        return None
    pmcid_num = re.sub(r"^PMC", "", pmcid, flags=re.I)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    candidates = [
        f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{pmcid_num}/bin/{table_id}.csv",
        f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{pmcid_num}/bin/{table_id}.htm",
        f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{pmcid_num}/bin/{table_id}.html",
        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/table/{table_id}/?download=1",
        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/table/{table_id}/?download=csv",
    ]

    for url in candidates:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            ctype = (r.headers.get("Content-Type") or "").lower()
            text = r.text if (r.ok and r.text.strip()) else ""
            if (not text) and r.status_code == 403:
                text = _curl_get_text(url, timeout=timeout) or ""
                ctype = "text/html"
            if not text:
                continue
            if "csv" in ctype or re.search(r",", text.splitlines()[0] if text.splitlines() else ""):
                try:
                    return [pd.read_csv(io.StringIO(text))]
                except Exception:
                    pass
            # try parse as HTML table(s)
            try:
                dfs = pd.read_html(io.StringIO(text))
                dfs = [d for d in dfs if not d.empty]
                if dfs:
                    return dfs
            except Exception:
                pass
        except Exception:
            continue
    return None


def fetch_pmc_fulltext_xml(pmcid: str, timeout: int = 60) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed. Run: python3 -m pip install requests")
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/xml,text/xml,*/*"}

    errors = []

    # 1) Europe PMC full text XML
    api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        resp = requests.get(api_url, timeout=timeout, headers=headers)
        if resp.ok and resp.text.strip():
            return resp.text
        if resp.status_code == 403:
            curl_text = _curl_get_text(api_url, timeout=timeout)
            if curl_text:
                return curl_text
        errors.append(f"europepmc_fullTextXML:{resp.status_code}")
    except Exception as e:
        errors.append(f"europepmc_fullTextXML:{repr(e)}")

    # 2) NCBI OAI-PMH fallback
    pmcid_num = re.sub(r"^PMC", "", pmcid, flags=re.I)
    oai_url = (
        "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
        f"?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid_num}&metadataPrefix=pmc"
    )
    try:
        resp2 = requests.get(oai_url, timeout=timeout, headers=headers)
        if resp2.ok and resp2.text.strip():
            return resp2.text
        if resp2.status_code == 403:
            curl_text = _curl_get_text(oai_url, timeout=timeout)
            if curl_text:
                return curl_text
        errors.append(f"ncbi_oai:{resp2.status_code}")
    except Exception as e:
        errors.append(f"ncbi_oai:{repr(e)}")

    # 3) PMC Open Access API fallback: download OA package and extract .nxml
    oa_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    try:
        oa_resp = requests.get(oa_api, timeout=timeout, headers=headers)
        oa_text = oa_resp.text if (oa_resp.ok and oa_resp.text) else ""
        if (not oa_text) and oa_resp.status_code == 403:
            oa_text = _curl_get_text(oa_api, timeout=timeout) or ""
        if oa_text:
            oa_soup = BeautifulSoup(oa_text, "xml") if BeautifulSoup is not None else None
            tgz_url = None
            if oa_soup is not None:
                # Try strict format first.
                link = oa_soup.find("link", {"format": "tgz"})
                if link and link.get("href"):
                    tgz_url = link.get("href")
                # Then relaxed search by href suffix.
                if not tgz_url:
                    for lk in oa_soup.find_all("link"):
                        href = lk.get("href")
                        if href and re.search(r"\.(?:tgz|tar\.gz)$", href, flags=re.I):
                            tgz_url = href
                            break

            # Final fallback: regex over raw XML.
            if not tgz_url:
                m = re.search(r'href="([^"]+\.(?:tgz|tar\.gz))"', oa_text, flags=re.I)
                if m:
                    tgz_url = m.group(1)

            if tgz_url:
                    # OA API often returns ftp URLs; convert to https mirror.
                    if tgz_url.startswith("ftp://ftp.ncbi.nlm.nih.gov"):
                        tgz_url = tgz_url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov", 1)
                    tgz_resp = requests.get(tgz_url, timeout=timeout, headers=headers)
                    tgz_resp.raise_for_status()
                    with tarfile.open(fileobj=io.BytesIO(tgz_resp.content), mode="r:gz") as tar:
                        for member in tar.getmembers():
                            if member.isfile() and member.name.lower().endswith(".nxml"):
                                f = tar.extractfile(member)
                                if f is not None:
                                    xml_bytes = f.read()
                                    return xml_bytes.decode("utf-8", errors="ignore")
        errors.append(f"pmc_oa_api:{oa_resp.status_code if 'oa_resp' in locals() else 'unknown'}")
    except Exception as e:
        errors.append(f"pmc_oa_api:{repr(e)}")

    raise RuntimeError(f"Unable to fetch full text XML for {pmcid}. Fallback errors: {'; '.join(errors)}")


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


def _table_wrap_to_link(pmcid: str, table_wrap) -> str:
    table_id = (table_wrap.get("id") or "").strip()
    if table_id:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/table/{table_id}/"
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"


def _normalize_pmc_article_url(article_url: str, pmcid: str) -> str:
    if re.search(rf"/articles/{pmcid}/?$", article_url, flags=re.I):
        return article_url
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"


def _normalize_table_link(href: str, pmcid: str) -> Optional[str]:
    href = (href or "").strip()
    if not href:
        return None
    absolute_patterns = [
        rf"https?://pmc\.ncbi\.nlm\.nih\.gov/articles/{pmcid}/table/[^/?#]+/?",
        rf"/articles/{pmcid}/table/[^/?#]+/?",
        r"^/table/[^/?#]+/?",
        r"^table/[^/?#]+/?",
    ]
    for pattern in absolute_patterns:
        m = re.search(pattern, href, flags=re.I)
        if not m:
            continue
        link = m.group(0)
        if link.startswith("http://") or link.startswith("https://"):
            full = link
        elif link.startswith("/articles/"):
            full = "https://pmc.ncbi.nlm.nih.gov" + link
        elif link.startswith("/table/"):
            full = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}" + link
        else:
            full = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" + link
        if not full.endswith("/"):
            full += "/"
        return full
    return None


def _table_wrap_caption(table_wrap) -> str:
    for tag_name in ("label", "caption", "title"):
        tag = table_wrap.find(tag_name)
        if tag is not None:
            text = _clean_text(tag.get_text(" ", strip=True))
            if text:
                return text
    table = table_wrap.find("table")
    if table is not None:
        cap = table.find("caption")
        if cap is not None:
            text = _clean_text(cap.get_text(" ", strip=True))
            if text:
                return text
    return ""


def _table_keyword_score(text: str) -> Tuple[int, List[str]]:
    norm = normalize_for_match(text)
    reasons: List[str] = []
    score = 0

    gene_patterns = [
        r"\bgene\b",
        r"\bgenes\b",
        r"\bgene symbol\b",
        r"\bnearest gene\b",
        r"\bmapped gene\b",
        r"\blocus\b",
        r"\brsid\b",
        r"\brs\d+\b",
    ]
    pvalue_patterns = [
        r"\bp[\s\-_]*value\b",
        r"\bpvalue\b",
        r"\bp-val\b",
        r"\bmeta[- ]?p\b",
        r"\bnominal p\b",
        r"\badjusted p\b",
        r"\bp for\b",
        r"(^|[\s(])p([\s)_:/-]|$)",
    ]
    effect_patterns = [r"\bor\b", r"\bbeta\b", r"\beffect\b", r"\bhazard ratio\b", r"\b95% ci\b", r"\bci\b"]
    variant_patterns = [r"\bsnp\b", r"\bvariant\b", r"\bchr\b", r"\bchromosome\b", r"\bposition\b"]

    if any(re.search(pattern, norm) for pattern in gene_patterns):
        score += 3
        reasons.append("gene-like keyword")
    if any(re.search(pattern, norm) for pattern in pvalue_patterns):
        score += 3
        reasons.append("p-value-like keyword")
    if any(re.search(pattern, norm) for pattern in effect_patterns):
        score += 1
        reasons.append("association-statistic keyword")
    if any(re.search(pattern, norm) for pattern in variant_patterns):
        score += 1
        reasons.append("variant-like keyword")
    return score, reasons


def _looks_like_gene_count_table(text: str) -> bool:
    norm = normalize_for_match(text)
    bad_patterns = [
        r"\bnumber of genes\b",
        r"\bno\.?\s+of genes\b",
        r"\bcount of genes\b",
        r"\bgenes? overlapping\b",
        r"\bgenes? enriched\b",
        r"\bgenes? identified\b",
        r"\bset of genes\b",
    ]
    return any(re.search(pattern, norm) for pattern in bad_patterns)


def normalize_for_match(text: str) -> str:
    text = _clean_text(text).lower()
    text = text.replace("−", "-").replace("–", "-")
    return text


def score_table_wrap(table_wrap) -> Dict[str, object]:
    table = table_wrap.find("table")
    if table is None:
        return {"score": 0, "reasons": ["missing table"], "caption": "", "preview": ""}

    caption = _table_wrap_caption(table_wrap)
    df = _flatten_columns(table_to_dataframe(table))
    label = _clean_text(text_or_empty_bs4(table_wrap.find("label")))
    parts = [label, caption]
    if not df.empty:
        header_text = " ".join(str(c) for c in df.columns if str(c).strip())
        parts.append(header_text)
        sample_df = df.head(15).astype(str).fillna("")
        preview_values = sample_df.values.flatten().tolist()
        parts.append(" ".join(preview_values[:180]))
    combined_text = " ".join(p for p in parts if p)
    score, reasons = _table_keyword_score(combined_text)

    if _looks_like_gene_count_table(combined_text):
        score -= 4
        reasons.append("gene-count summary table")

    if "gene-like keyword" in reasons and "p-value-like keyword" in reasons:
        score += 2
        reasons.append("gene+p-value rule matched")

    # Accept classic association tables even when "gene" itself is sparse,
    # as long as we see variants plus p-values/effect statistics.
    if "p-value-like keyword" in reasons and ("variant-like keyword" in reasons or "association-statistic keyword" in reasons):
        score += 1
        reasons.append("association-table support")

    return {
        "score": score,
        "reasons": reasons,
        "caption": f"{label} {caption}".strip(),
        "preview": _clean_text(combined_text)[:400],
    }


def _discover_relevant_pmc_tables_from_xml(pmcid: str, min_score: int) -> List[Dict[str, object]]:
    xml = fetch_pmc_fulltext_xml(pmcid)
    soup = BeautifulSoup(xml, "xml")
    wraps = soup.find_all("table-wrap")

    discovered: List[Dict[str, object]] = []
    for idx, wrap in enumerate(wraps, start=1):
        table = wrap.find("table")
        if table is None:
            continue
        score_info = score_table_wrap(wrap)
        item = {
            "pmcid": pmcid,
            "table_index": idx,
            "table_id": (wrap.get("id") or "").strip(),
            "caption": score_info["caption"],
            "score": score_info["score"],
            "reasons": score_info["reasons"],
            "preview": score_info["preview"],
            "link": _table_wrap_to_link(pmcid, wrap),
            "selected": bool(score_info["score"] >= min_score),
            "source": "pmc_xml",
        }
        discovered.append(item)
    discovered.sort(key=lambda row: row["table_index"])
    return discovered


def _discover_relevant_pmc_tables_from_html(article_url: str, pmcid: str, min_score: int) -> List[Dict[str, object]]:
    article_html = fetch_html(_normalize_pmc_article_url(article_url, pmcid))
    soup = BeautifulSoup(article_html, "lxml")

    hrefs: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href") or ""
        normalized = _normalize_table_link(href, pmcid)
        if normalized:
            hrefs.append(normalized)
    raw_patterns = [
        rf"https?://pmc\.ncbi\.nlm\.nih\.gov/articles/{pmcid}/table/[^\"'?#\s>]+/?",
        rf"/articles/{pmcid}/table/[^\"'?#\s>]+/?",
        r"/table/[^\"'?#\s>]+/?",
        r"table/[^\"'?#\s>]+/?",
    ]
    for pattern in raw_patterns:
        for match in re.finditer(pattern, article_html, flags=re.I):
            normalized = _normalize_table_link(match.group(0), pmcid)
            if normalized:
                hrefs.append(normalized)
    table_links = []
    seen = set()
    for link in hrefs:
        if link not in seen:
            seen.add(link)
            table_links.append(link)

    discovered: List[Dict[str, object]] = []
    for idx, link in enumerate(table_links, start=1):
        table_html = fetch_html(link)
        table_soup = BeautifulSoup(table_html, "lxml")
        table = table_soup.find("table")
        if table is None:
            continue
        pseudo_wrap = table_soup
        score_info = score_table_wrap(pseudo_wrap)
        _, table_id = _extract_pmc_info(link)
        discovered.append(
            {
                "pmcid": pmcid,
                "table_index": _extract_table_number_from_id(table_id or "") or idx,
                "table_id": table_id or "",
                "caption": score_info["caption"],
                "score": score_info["score"],
                "reasons": score_info["reasons"],
                "preview": score_info["preview"],
                "link": link,
                "selected": bool(score_info["score"] >= min_score),
                "source": "pmc_html",
            }
        )

    if not discovered:
        inline_tables = _extract_tables_from_html_text(article_html)
        for idx, df in enumerate(inline_tables, start=1):
            html_table_id = f"T{idx}"
            preview_df = _flatten_columns(df)
            combined_text = " ".join(str(x) for x in preview_df.head(15).astype(str).fillna("").values.flatten().tolist()[:180])
            score, reasons = _table_keyword_score(combined_text)
            if _looks_like_gene_count_table(combined_text):
                score -= 4
                reasons.append("gene-count summary table")
            if "gene-like keyword" in reasons and "p-value-like keyword" in reasons:
                score += 2
                reasons.append("gene+p-value rule matched")
            if "p-value-like keyword" in reasons and ("variant-like keyword" in reasons or "association-statistic keyword" in reasons):
                score += 1
                reasons.append("association-table support")
            discovered.append(
                {
                    "pmcid": pmcid,
                    "table_index": idx,
                    "table_id": html_table_id,
                    "caption": f"Table {idx}",
                    "score": score,
                    "reasons": reasons,
                    "preview": _clean_text(combined_text)[:400],
                    "link": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/?table_id={html_table_id}",
                    "selected": bool(score >= min_score),
                    "source": "pmc_article_html",
                }
            )

    if not discovered:
        raise RuntimeError(f"No PMC table links found on article HTML for {pmcid}, and no inline HTML tables were parseable")

    discovered.sort(key=lambda row: row["table_index"])
    return discovered


def discover_relevant_pmc_tables(article_url: str, min_score: int = 5) -> List[Dict[str, object]]:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is not installed. Run: python3 -m pip install beautifulsoup4")

    pmcid, _ = _extract_pmc_info(article_url)
    if not pmcid:
        m = re.search(r"/articles/(PMC\d+)/?", article_url, flags=re.I)
        pmcid = m.group(1).upper() if m else None
    if not pmcid:
        raise ValueError("Auto-discovery currently works only with PMC article URLs. For PDF-only papers, paste table links manually or use manual table upload / table_input instead.")
    try:
        return _discover_relevant_pmc_tables_from_xml(pmcid, min_score=min_score)
    except Exception as xml_error:
        try:
            discovered = _discover_relevant_pmc_tables_from_html(article_url, pmcid, min_score=min_score)
            if discovered:
                return discovered
            raise RuntimeError(f"HTML fallback returned 0 discovered tables for {pmcid}")
        except Exception as html_error:
            raise RuntimeError(
                f"Unable to auto-discover PMC tables for {pmcid}. XML error: {xml_error}. HTML fallback error: {html_error}"
            ) from html_error


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(x).strip() for x in col if str(x).strip() and str(x).strip().lower() != "nan"]
            flat_cols.append(" | ".join(parts) if parts else "")
        out = df.copy()
        out.columns = flat_cols
        return out
    return df


def _extract_tables_from_html_text(html_text: str) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    try:
        parsed = pd.read_html(io.StringIO(html_text))
        dfs = [d for d in parsed if not d.empty]
    except Exception:
        return []
    return dfs


def _is_special_table_page_id(table_id: Optional[str]) -> bool:
    if not table_id:
        return False
    tid = table_id.strip()
    return bool(
        re.match(r"^.+-T\d+$", tid, flags=re.I)
        or re.match(r"^[A-Za-z].*t\d+$", tid, flags=re.I)
    )


def _try_extract_from_article_html(pmcid: str, table_id: Optional[str]) -> Optional[List[pd.DataFrame]]:
    article_url = _normalize_pmc_article_url("", pmcid)
    article_html = fetch_html(article_url)
    soup = BeautifulSoup(article_html, "lxml")
    wanted_num = _extract_table_number_from_id(table_id or "") if table_id else None

    if table_id:
        try:
            tables = pick_table(soup, table_id=table_id, table_selector=None, table_index=0)
            dfs = [table_to_dataframe(table) for table in tables]
            dfs = [df for df in dfs if not df.empty]
            if dfs:
                return dfs
        except Exception:
            pass

    if wanted_num is not None:
        # Some PMC article pages render "Table N" as a heading/label near the inline table
        # without exposing a stable table id. Search for that label and then grab the nearest table.
        label_re = re.compile(rf"^\s*table\s*{wanted_num}\b", flags=re.I)
        for node in soup.find_all(string=label_re):
            parent = node.parent
            if parent is None:
                continue
            candidate = parent.find_next("table")
            if candidate is not None:
                df = table_to_dataframe(candidate)
                if not df.empty:
                    return [df]

    html_dfs = _extract_tables_from_html_text(article_html)
    if wanted_num is not None and 1 <= wanted_num <= len(html_dfs):
        return [html_dfs[wanted_num - 1]]
    return html_dfs or None


def _extract_table_number_from_id(table_id: str) -> Optional[int]:
    if not table_id:
        return None
    patterns = [
        r"^t0*(\d+)$",
        r"^tbl0*(\d+)$",
        r"^table0*(\d+)$",
        r"^tab0*(\d+)$",
        r"^.+-t0*(\d+)$",
    ]
    tid = table_id.strip().lower()
    for pattern in patterns:
        m = re.match(pattern, tid)
        if m:
            return int(m.group(1))
    m = re.search(r"(\d+)$", tid)
    if m:
        return int(m.group(1))
    return None


def _find_table_by_label_or_number(soup, table_id: str):
    wanted_num = _extract_table_number_from_id(table_id)
    if wanted_num is None:
        return None

    wraps = soup.find_all("table-wrap")
    if wraps:
        for idx, wrap in enumerate(wraps, start=1):
            label = _clean_text(text_or_empty_bs4(wrap.find("label"))).lower()
            caption = _table_wrap_caption(wrap).lower()
            if label in {f"table {wanted_num}", f"t{wanted_num}"}:
                table = wrap.find("table")
                if table is not None:
                    return table
            if caption.startswith(f"table {wanted_num}"):
                table = wrap.find("table")
                if table is not None:
                    return table
            if idx == wanted_num:
                fallback_table = wrap.find("table")
                if fallback_table is not None:
                    sequential_candidate = fallback_table
                else:
                    sequential_candidate = None
        if "sequential_candidate" in locals() and sequential_candidate is not None:
            return sequential_candidate

    tables = soup.find_all("table")
    if wanted_num >= 1 and wanted_num <= len(tables):
        return tables[wanted_num - 1]
    return None


def text_or_empty_bs4(node) -> str:
    if node is None:
        return ""
    return node.get_text(" ", strip=True)


def _describe_available_tables(soup) -> str:
    entries: List[str] = []
    wraps = soup.find_all("table-wrap")
    if wraps:
        for idx, wrap in enumerate(wraps, start=1):
            wrap_id = (wrap.get("id") or "").strip() or "-"
            label = _clean_text(text_or_empty_bs4(wrap.find("label"))) or "-"
            caption = _table_wrap_caption(wrap) or "-"
            entries.append(f"{idx}: id={wrap_id}; label={label}; caption={caption[:120]}")
    else:
        tables = soup.find_all("table")
        for idx, table in enumerate(tables[:20], start=1):
            table_id = (table.get("id") or "").strip() or "-"
            caption_tag = table.find("caption")
            caption = _clean_text(text_or_empty_bs4(caption_tag)) if caption_tag is not None else "-"
            entries.append(f"{idx}: id={table_id}; caption={caption[:120]}")
    return " | ".join(entries) if entries else "no tables found"


def pick_table(soup, table_id: Optional[str], table_selector: Optional[str], table_index: int):
    if table_id:
        t = soup.find("table", {"id": table_id})
        if t is None:
            wrap = soup.find("table-wrap", {"id": table_id})
            if wrap is not None:
                t = wrap.find("table")
        if t is None:
            # case-insensitive id fallback
            tid = table_id.lower()
            t = soup.find("table", id=lambda x: isinstance(x, str) and x.lower() == tid)
            if t is None:
                wrap = soup.find("table-wrap", id=lambda x: isinstance(x, str) and x.lower() == tid)
                if wrap is not None:
                    t = wrap.find("table")
        if t is None:
            t = _find_table_by_label_or_number(soup, table_id)
        if t is None:
            available = _describe_available_tables(soup)
            raise ValueError(f"Cannot find table with id={table_id}. Available tables: {available}")
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

    # Fast-path fallback for PMC direct table assets (often bypasses page-level 403).
    if pmcid and resolved_table_id:
        direct_tables = _try_pmc_direct_table_download(pmcid, resolved_table_id)
        if direct_tables:
            with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
                for i, df in enumerate(direct_tables, start=1):
                    df2 = _flatten_columns(df)
                    df2.to_excel(writer, index=False, sheet_name=sanitize_sheet_name("", f"table_{i}"))
            print(f"Saved {len(direct_tables)} table(s) to {args.out}")
            return

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

    # Some PMC table URLs are already single-table pages whose DOM does not preserve
    # the original XML table id. If we can parse exactly one HTML table, use it directly.
    if resolved_table_id and html:
        html_dfs = _extract_tables_from_html_text(html)
        if len(html_dfs) == 1:
            with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
                df2 = _flatten_columns(html_dfs[0])
                df2.to_excel(writer, index=False, sheet_name=sanitize_sheet_name(resolved_table_id, "table_1"))
            print(f"Saved 1 table(s) to {args.out}")
            return

    if parse_with_xml and not resolved_table_id and url_table_id:
        resolved_table_id = url_table_id

    try:
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
    except ValueError:
        # Some PMC table pages return HTML shells/check pages to scripts. Retry against full-text XML.
        if pmcid and not parse_with_xml:
            if html:
                html_dfs = _extract_tables_from_html_text(html)
                if html_dfs:
                    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
                        for i, df in enumerate(html_dfs, start=1):
                            df2 = _flatten_columns(df)
                            df2.to_excel(writer, index=False, sheet_name=sanitize_sheet_name(resolved_table_id or "", f"table_{i}"))
                    print(f"Saved {len(html_dfs)} table(s) to {args.out}")
                    return
            # For PMC table URLs like awz206-T1/T2, the dedicated table page may not expose
            # a parseable DOM table to scripts, but the article page often still contains the
            # inline Table 1 / Table 2 content. Prefer that before attempting XML.
            article_dfs = _try_extract_from_article_html(pmcid, resolved_table_id)
            if article_dfs:
                with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
                    for i, df in enumerate(article_dfs, start=1):
                        df2 = _flatten_columns(df)
                        df2.to_excel(writer, index=False, header=False, sheet_name=sanitize_sheet_name(resolved_table_id or "", f"table_{i}"))
                print(f"Saved {len(article_dfs)} table(s) to {args.out}")
                return
            wanted_num = _extract_table_number_from_id(resolved_table_id or "") if resolved_table_id else None
            if wanted_num is not None and _is_special_table_page_id(resolved_table_id):
                raise RuntimeError(
                    f"Could not parse table page for {resolved_table_id}, and article HTML did not expose inline Table {wanted_num}."
                )
            xml = fetch_pmc_fulltext_xml(pmcid)
            soup = BeautifulSoup(xml, "xml")
            parse_with_xml = True
            if resolved_table_id:
                tables = pick_table(soup, table_id=resolved_table_id, table_selector=None, table_index=args.table_index)
            elif args.all_tables:
                wraps = soup.find_all("table-wrap")
                tables = [w.find("table") for w in wraps if w.find("table") is not None]
                if not tables:
                    raise ValueError("No matched tables found in PMC XML.")
            else:
                tables = pick_table(soup, table_id=None, table_selector=None, table_index=args.table_index)
        else:
            raise

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        for i, table in enumerate(tables, start=1):
            df = table_to_dataframe(table)
            caption = ""
            cap = table.find("caption")
            if cap:
                caption = _clean_text(cap.get_text(" ", strip=True))
            sheet = sanitize_sheet_name(caption, fallback=f"table_{i}")
            df2 = _flatten_columns(df)
            df2.to_excel(writer, index=False, header=False, sheet_name=sheet)

    print(f"Saved {len(tables)} table(s) to {args.out}")


if __name__ == "__main__":
    main()
