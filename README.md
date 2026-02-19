# advp_curator

`advp_curator.py` converts paper tables (PDF/URL/Excel/CSV/HTML) into a fixed ADVP curated schema and exports an Excel file.

## Features
- Outputs a fixed set of 46 curated columns (independent of raw table layout)
- Text extraction fallback chain for PDF context fields:
  - `pdfplumber` (default)
  - `Docling` fallback only when Methods/Results signals are weak
  - OCR + page-rotation retry (0/90/180/270) as final rescue
- Section-aware context extraction (`Methods` / `Results` / `Supplement`) for:
  - `Population`, `Cohort`, `Sample size`, `Imputation_simple2`, `Stage`, `Model type`
- Abbreviation expansion and semantic column mapping to curated references
  - e.g. `ADGC/IGAP/HRC/TOPMed` normalization
  - low-confidence mapping score (`< 0.4`) flagged as `needs_review`
- Generic subgroup melt/group logic for multi-header GWAS tables
- Supports input sources:
  - PDF path / PDF URL
  - table URL
  - `.xlsx` / `.xls` / `.csv` / `.tsv` / `.html`
- Automatic header detection (including `Unnamed:*` columns)
- SNP/gene table mapping rules
- `chr:position` parsing (for example `20:45269867`)
- `EA/OA` and `Major/minor` allele parsing
- Prioritizes `P value All` when multiple p-value columns exist
- Subgroup expansion (one SNP row can expand into multiple records)
  - For example `i-Share (dichotomous)` / `i-Share (continuous)` / `Nagahama (...)`
- Generates `RecordID` and `TableIDX` (format `T00001`)
- Generates audit JSON (field evidence and parsing errors)

## Installation
```bash
python3 -m pip install pandas openpyxl requests pdfplumber camelot-py lxml
```

Optional (for fallback rescue quality):
```bash
python3 -m pip install docling pymupdf pytesseract pillow
```

## Usage

### 1) Interactive mode (recommended)
```bash
python3 advp_curator.py
```
The script will prompt for:
- input source path/URL
- output xlsx path
- paper_id
- audit json path

### 2) CLI mode
#### Table file
```bash
python3 advp_curator.py \
  --table_input "/path/to/table.xlsx" \
  --out "/path/to/curated.xlsx" \
  --paper_id "37069360_table1"
```

#### PDF
```bash
python3 advp_curator.py \
  --input "/path/to/paper.pdf" \
  --out "/path/to/curated.xlsx" \
  --paper_id "paper_001"
```

#### Table URL
```bash
python3 advp_curator.py \
  --table_input "https://example.com/table/1" \
  --out "/path/to/curated.xlsx" \
  --paper_id "paper_table1"
```

## Current Field Mapping Rules
- `TopSNP`: `SNP ALL` / `Variant` / `rsID`-like columns
- `SNP-based, Gene-based`: `SNP-based` if rsID exists; otherwise `Gene-based` if a gene column exists
- `Chr`, `BP(Position)`:
  - Prefer parsing from `chr:position`
  - Otherwise use separate `Chr` / `Position` columns
- `RA 1(Reported Allele 1)`, `RA 2(Reported Allele 2)`:
  - `EA/OA` -> `RA1=EA`, `RA2=OA`
  - `Major/minor` -> `RA1=minor`, `RA2=major`
- `ReportedAF(MAF)`: `EAF` / `MAF`
- `LocusName`: `Nearest gene` / `Closest gene`
- `Effect Size Type (OR or Beta)`: inferred from column names (`OR/Beta/HR/Zscore`)
- `EffectSize(altvsref)`: numeric value from the selected effect column
- `P-value`: prefer `P value All`, then `Meta P`, then generic `P value`
- `95%ConfidenceInterval`: `95% CI`-like columns
- `Table Ref in paper`: `Table 1`, `Table 2`, ...
- `TableIDX`: `T00001`, `T00002`, ...

## Outputs
- Curated Excel：`--out` 
- Audit JSON: path from `--audit` (in interactive mode, default is `${paper_id}_audit.json`)

## Troubleshooting
- `No /Root object! - Is this really a PDF?`
  - You likely passed an Excel/CSV file as PDF input. Use `--table_input` or interactive mode with a table file.
- `ImportError: Import lxml failed`
  - Install `lxml`: `python3 -m pip install lxml`
- Table URL only outputs one `NR` row
  - The site may be dynamic or protected. Save the table as `.xlsx/.csv` first, then run with `--table_input`.
