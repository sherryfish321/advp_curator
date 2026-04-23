# PMID 29458411 B Comparison

This folder contains a runnable B-version snapshot based on `origin/extract-text-col` for quick A/B comparison on PMID 29458411 table 1 and table 3.

Contents:
- `advp_curator.py`: B-version test copy adapted only so it can run locally without the branch's external retriever stack.
- `input/`: table 1 and table 3 inputs.
- `output/`: B-version harmonized outputs.
- `audit/`: B-version audit files.
- `eval/pmid_29458411_eval_report_advp_like.xlsx`: unified evaluation report.

Current result:
- `from_29458411_table1_B.xlsx`: 14 rows
- `from_29458411_table3_B.xlsx`: 4 rows, 0 matched SNP rows in ADVP comparison
- Combined eval: precision 1.0, recall 1.0, F1 1.0 on the matched SNP+p-value set
