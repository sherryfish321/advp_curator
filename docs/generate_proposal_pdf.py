from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMAGES = DOCS / "images"
OUT = DOCS / "ADVP_Curator_Grant_Proposal_20260408.pdf"
RUN_LOG = ROOT / "ui_run_logs" / "pmid_29274321_run_20260408_232432.json"

PAGE_W = 1654
PAGE_H = 2339
MARGIN = 92
BG = "#f5f8fc"
CARD = "#ffffff"
NAVY = "#24384f"
BLUE = "#4f6786"
TEXT = "#2b3b4f"
MUTED = "#64768a"
LINE = "#d8e2ee"
TAG_BG = "#eaf1f8"
TAG_TEXT = "#375476"


def load_font(size: int, bold: bool = False):
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(48, True)
FONT_H2 = load_font(34, True)
FONT_H3 = load_font(28, True)
FONT_BODY = load_font(24, False)
FONT_SMALL = load_font(20, False)
FONT_BADGE = load_font(22, True)


def rounded(draw, box, fill=CARD, outline=LINE, width=2, radius=26):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(draw, text: str, font, width_px: int):
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        cur = ""
        for word in words:
            trial = f"{cur} {word}".strip()
            if not cur or draw.textlength(trial, font=font) <= width_px:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
    return lines


def draw_wrapped(draw, text, x, y, font, fill, width_px, gap=10):
    line_h = font.size + gap
    for line in wrap_text(draw, text, font, width_px):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def draw_bullets(draw, bullets, x, y, width_px):
    for bullet in bullets:
        draw.ellipse((x, y + 8, x + 12, y + 20), fill=BLUE)
        y = draw_wrapped(draw, bullet, x + 28, y, FONT_BODY, TEXT, width_px - 28, gap=8) + 12
    return y


def fit_image(path: Path, max_w: int, max_h: int):
    img = Image.open(path).convert("RGB")
    scale = min(max_w / img.width, max_h / img.height)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(size, Image.LANCZOS)


def new_page():
    return Image.new("RGB", (PAGE_W, PAGE_H), BG)


def header(draw, page_title: str, right_text: str):
    draw.text((MARGIN, 54), "ADVP Curator", font=FONT_H2, fill=NAVY)
    right_w = draw.textlength(right_text, font=FONT_SMALL)
    draw.text((PAGE_W - MARGIN - right_w, 62), right_text, font=FONT_SMALL, fill=MUTED)
    draw.line((MARGIN, 116, PAGE_W - MARGIN, 116), fill=LINE, width=3)
    return draw_wrapped(
        draw,
        page_title,
        MARGIN,
        152,
        FONT_TITLE,
        TEXT,
        PAGE_W - 2 * MARGIN,
        gap=8,
    )


def figure_page(figure_no: int, title: str, image_name: str, overview: str, bullets):
    page = new_page()
    draw = ImageDraw.Draw(page)
    title_bottom = header(draw, f"Figure {figure_no}. {title}", "Grant Proposal Support Material")

    image_top = max(300, int(title_bottom + 36))
    image_box_bottom = 1380
    rounded(draw, (MARGIN, image_top, PAGE_W - MARGIN, image_box_bottom))
    rounded(draw, (MARGIN + 28, image_top + 28, MARGIN + 350, image_top + 84), fill=TAG_BG, outline=TAG_BG, radius=22)
    draw.text((MARGIN + 54, image_top + 44), "Current Website Screenshot", font=FONT_BADGE, fill=TAG_TEXT)
    image = fit_image(IMAGES / image_name, PAGE_W - 2 * MARGIN - 56, image_box_bottom - image_top - 140)
    ix = (PAGE_W - image.width) // 2
    page.paste(image, (ix, image_top + 108))

    text_top = image_box_bottom + 36
    rounded(draw, (MARGIN, text_top, PAGE_W - MARGIN, PAGE_H - 110))
    draw.text((MARGIN + 34, text_top + 42), "What We Have Done So Far", font=FONT_H2, fill=NAVY)
    y = draw_wrapped(draw, overview, MARGIN + 34, text_top + 106, FONT_BODY, TEXT, PAGE_W - 2 * MARGIN - 68, gap=10)
    y += 16
    draw.text((MARGIN + 34, y), "How This Figure Supports the Proposal", font=FONT_H3, fill=NAVY)
    y += 50
    draw_bullets(draw, bullets, MARGIN + 34, y, PAGE_W - 2 * MARGIN - 68)
    return page


def title_page(log):
    page = new_page()
    draw = ImageDraw.Draw(page)
    header(draw, "Grant Proposal Summary", "Generated from current ADVP Curator UI")

    rounded(draw, (MARGIN, 276, PAGE_W - MARGIN, 850), fill="#edf3fa", outline="#d2deeb")
    draw.text((MARGIN + 34, 322), "Project Framing", font=FONT_H2, fill=NAVY)
    intro = (
        "We have built an end-to-end curation and extraction workflow for Alzheimer's disease genetics papers. "
        "The platform takes heterogeneous paper inputs, identifies relevant association tables, converts them into "
        "an ADVP-compatible structure, evaluates extracted records against reference ADVP data, and produces review-ready outputs for manual quality control."
    )
    y = draw_wrapped(draw, intro, MARGIN + 34, 384, FONT_BODY, TEXT, PAGE_W - 2 * MARGIN - 68, gap=10)
    y += 20
    draw.text((MARGIN + 34, y), "Current demonstrated capabilities include:", font=FONT_H3, fill=NAVY)
    y += 52
    bullets = [
        "Multi-format ingestion from PMC table links, article pages, PDFs, Excel files, CSV files, TSV files, and HTML tables.",
        "Automatic discovery and ranking of PMC candidate tables using genetics- and association-oriented heuristics.",
        "Structured mapping of extracted table content into curated ADVP-compatible fields such as SNP, chromosome, base-pair position, allele fields, p-value, effect size, confidence interval, cohort, population, and sample-size metadata.",
        "Integrated evaluation outputs including row-match accuracy, field-level accuracy, mismatch reports, missing-field summaries, and fix-ready CSVs for downstream review.",
        "Browser-based editing and annotation so curators can correct extracted sheets while keeping links back to the paper and source table.",
    ]
    draw_bullets(draw, bullets, MARGIN + 34, y, PAGE_W - 2 * MARGIN - 68)

    rounded(draw, (MARGIN, 900, PAGE_W - MARGIN, PAGE_H - 110))
    draw.text((MARGIN + 34, 944), "Example run reflected in the UI screenshots", font=FONT_H2, fill=NAVY)
    facts = [
        f"PMID: {log['pmid']}",
        f"Paper source: {log['paper_input']}",
        f"Owner: {log['owner_name']}",
        f"Tables selected for processing: {len(log['table_links'])}",
        f"Curated spreadsheets generated: {len(log['generated_harmonized'])}",
        f"Review artifact generated: {Path(log['fix_file']).name}",
    ]
    y = 1008
    for fact in facts:
        y = draw_wrapped(draw, fact, MARGIN + 50, y, FONT_BODY, TEXT, PAGE_W - 2 * MARGIN - 100, gap=8) + 8

    closing = (
        "The following figures document how the current website already embodies the major workflow components that can be described in the grant proposal as completed work."
    )
    draw_wrapped(draw, closing, MARGIN + 34, y + 20, FONT_BODY, TEXT, PAGE_W - 2 * MARGIN - 68, gap=10)
    return page


def main():
    log = json.loads(RUN_LOG.read_text())

    pages = [
        title_page(log),
        figure_page(
            1,
            "Study Intake, Multi-Format Input, And PMC Table Discovery",
            "runner.jpeg",
            "We built a web-based study curation runner that serves as the entry point for the full workflow. "
            "At this stage, a curator can provide a PMID, paper URL, uploaded PDF, source type, and table links, or allow the system to automatically discover candidate PMC tables when explicit links are not provided. "
            "This reflects substantial progress beyond a one-off extraction script because the system now supports standardized run setup, curator ownership, and reproducible execution across heterogeneous source formats.",
            [
                "The interface operationalizes multi-format ingestion, which is a core completed contribution of the project.",
                "The PMC auto-discovery option demonstrates that we have already implemented logic to scan article content and identify candidate association tables without requiring full manual table selection.",
                "The generated-output checklist shows that each run already produces curated ADVP sheets plus downstream evaluation and review artifacts, not just raw extracted tables.",
            ],
        ),
        figure_page(
            2,
            "Row-Level And Field-Level Evaluation Against ADVP",
            "accuracy.jpeg",
            "We built an evaluation layer that compares predicted structured outputs against ADVP reference records at both the row and field levels. "
            "The dashboard reports precision, recall, and F1 for row matching as well as field accuracy summaries for easier mapped fields and for the broader set of ADVP-comparable fields. "
            "This is important because it turns the system into a measurable curation platform, allowing us to quantify extraction quality, diagnose weaknesses, and prioritize refinement work.",
            [
                "The row-match panel supports the claim that the workflow can benchmark whether extracted records correspond to expected ADVP association rows.",
                "The field-accuracy panels support the proposal's quality-control component by showing that harmonized attributes such as p-value, effect, cohort, sample size, analysis group, population, and chromosome are explicitly evaluated.",
                "The visible list of evaluated fields makes the system interpretable and reviewer-friendly, which is useful in a grant narrative focused on rigor and reproducibility.",
            ],
        ),
        figure_page(
            3,
            "Run-Level Review Outputs And Fix-Ready Deliverables",
            "result.jpeg",
            "We built the workflow so that every completed run produces curator-facing review outputs rather than stopping at model predictions. "
            "The result page summarizes successful versus failed rows, exposes a downloadable fix-ready CSV, and provides direct transition points into additional review or correction steps. "
            "This completed functionality reduces manual curation effort because it packages discrepancies and incomplete outputs into artifacts that are immediately actionable.",
            [
                "The success-versus-failed summary supports the claim that the platform provides run-level quality monitoring for practical curation work.",
                "The downloadable fix-ready CSV shows that we have already implemented downstream deliverables specifically designed for manual reconciliation and correction.",
                "The navigation from result summary to editing workflow demonstrates a closed-loop curation design rather than isolated extraction components.",
            ],
        ),
        figure_page(
            4,
            "Browser-Based Editing, Annotation, And Traceable Review",
            "edit.jpeg",
            "We also built a browser-based editing interface so curators can inspect extracted ADVP-compatible sheets, preserve links back to the paper and source table, and annotate or correct fields directly within the workflow. "
            "This is a key completed component because large-scale literature harmonization still requires human oversight, especially for ambiguous fields and heterogeneous table structures. "
            "By integrating editing into the same platform, we have created a practical human-in-the-loop environment for quality control and iterative refinement.",
            [
                "The preserved paper and table links support traceability, allowing curators to verify each extracted record against its original evidence source.",
                "The row-level editing layout supports manual correction of harmonized values while keeping the ADVP-style structure intact.",
                "The mark-and-comment workflow supports annotation, error tracking, and future rule improvement, which strengthens the proposal's story around scalable semi-automated curation.",
            ],
        ),
    ]

    pages = [p.convert("RGB") for p in pages]
    pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=200)
    print(OUT)


if __name__ == "__main__":
    main()
