---
name: extract-ci-dump-serving
description: Extracts serving benchmark rows from ci_dump summaries using CI run IDs and writes sheet-friendly TSV/CSV output files. Use when the user asks to fetch results from ci_dump, compare specific GitHub Actions run IDs, or export benchmark data for spreadsheets.
---

# Extract CI Dump Serving Data

Use this skill to export serving benchmark rows from `ci_dump` for one or more run IDs.

## Preconditions

1. Confirm required tools are available:
   - `command -v python3`
   - `command -v git`
2. Ensure the `ci_dump` branch is available locally:
   - `git fetch origin ci_dump`

## Command

Use `tools/cohere/extract_ci_dump_serving.py` with one `--run-id` per run:

```bash
python3 tools/cohere/extract_ci_dump_serving.py \
  --run-id <RUN_ID_1> \
  --run-id <RUN_ID_2> \
  > ci_dump/serving_extract_<RUN_ID_1>_<RUN_ID_2>.tsv
```

For CSV output:

```bash
python3 tools/cohere/extract_ci_dump_serving.py \
  --output-format csv \
  --run-id <RUN_ID_1> \
  --run-id <RUN_ID_2> \
  > ci_dump/serving_extract_<RUN_ID_1>_<RUN_ID_2>.csv
```

## Output Behavior

- Default source: `origin/ci_dump:data/summary_gb200.json`.
- Default output format: TSV (spreadsheet-friendly).
- Columns already include native fields from the summary (`tp_size`, `input_len`, `output_len`, `max_concurrency`) plus selected performance metrics.
- Metadata columns (`ci_run_id`, `ci_run_url`, `timestamp`) appear at the end.

## Response Pattern

After writing the file:

1. Confirm success.
2. Return only the output file path(s).
3. Do not paste the full table unless explicitly requested.
