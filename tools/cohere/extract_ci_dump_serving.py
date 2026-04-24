#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Iterable

OUTPUT_COLUMNS = [
    "test_index",
    "test_name",
    "tp_size",
    "input_len",
    "output_len",
    "max_concurrency",
    "num_requests",
    "tput_req_s",
    "total_token_tput_tok_s",
    "output_tput_tok_s",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_itl_ms",
    "ci_run_id",
    "ci_run_url",
    "timestamp",
]


def _run_git_show(ref: str, path: str) -> str:
    try:
        return subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to read {path} from {ref}. Did you fetch the branch?"
        ) from exc


def _get_value(serving: dict, key: str, index: str) -> str:
    value_map = serving.get(key, {})
    if not isinstance(value_map, dict):
        return ""
    value = value_map.get(index, "")
    if isinstance(value, str):
        return value.replace("\n", " | ").replace("\t", " ")
    return str(value)


def _parse_required_int(
    serving: dict,
    key: str,
    index: str,
    run_id: str,
) -> int:
    value = _get_value(serving, key, index).strip()
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Expected integer for {key!r} at test_index={index} in run_id={run_id},"
            f" got {value!r}."
        ) from None


def extract_rows(
    entries: list[dict],
    run_ids: Iterable[str],
) -> list[dict[str, str]]:
    by_run_id = {str(entry.get("ci_run_id")): entry for entry in entries}
    rows: list[dict[str, str]] = []
    for run_id in run_ids:
        entry = by_run_id.get(run_id)
        if entry is None:
            continue
        serving = entry.get("serving", {})
        test_name_map = serving.get("Test name", {})
        if not isinstance(test_name_map, dict):
            continue

        sorted_indices = sorted(
            test_name_map.keys(),
            key=lambda idx: (
                _parse_required_int(serving, "Input Len", idx, run_id),
                _parse_required_int(serving, "Output Len", idx, run_id),
                _parse_required_int(serving, "# of max concurrency.", idx, run_id),
                int(idx),
            ),
        )

        for index in sorted_indices:
            test_name = _get_value(serving, "Test name", index)
            rows.append(
                {
                    "ci_run_id": str(entry.get("ci_run_id", "")),
                    "ci_run_url": str(entry.get("ci_run_url", "")),
                    "timestamp": str(entry.get("timestamp", "")),
                    "test_index": index,
                    "test_name": test_name,
                    "tp_size": _get_value(serving, "TP Size", index),
                    "input_len": _get_value(serving, "Input Len", index),
                    "output_len": _get_value(serving, "Output Len", index),
                    "max_concurrency": _get_value(
                        serving, "# of max concurrency.", index
                    ),
                    "num_requests": _get_value(serving, "# of req.", index),
                    "tput_req_s": _get_value(serving, "Tput (req/s)", index),
                    "total_token_tput_tok_s": _get_value(
                        serving, "Total Token Tput (tok/s)", index
                    ),
                    "output_tput_tok_s": _get_value(
                        serving, "Output Tput (tok/s)", index
                    ),
                    "mean_ttft_ms": _get_value(serving, "Mean TTFT (ms)", index),
                    "p99_ttft_ms": _get_value(serving, "P99 TTFT (ms)", index),
                    "mean_tpot_ms": _get_value(serving, "Mean TPOT (ms)", index),
                    "p99_itl_ms": _get_value(serving, "P99 ITL (ms)", index),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract serving rows for specific ci_run_ids from ci_dump summary JSON."
        )
    )
    parser.add_argument(
        "--ref",
        default="origin/ci_dump",
        help="Git ref that contains the summary JSON (default: origin/ci_dump).",
    )
    parser.add_argument(
        "--summary-path",
        default="data/summary_gb200.json",
        help="Path to summary JSON inside the git ref.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="CI run ID to extract. Pass multiple times.",
    )
    parser.add_argument(
        "--output-format",
        choices=["tsv", "csv"],
        default="tsv",
        help="Output format (default: tsv).",
    )
    args = parser.parse_args()

    raw = _run_git_show(args.ref, args.summary_path)
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise RuntimeError("Expected summary JSON to contain a list of run entries.")

    rows = extract_rows(entries, args.run_id)
    if not rows:
        raise RuntimeError("No matching rows found for provided run IDs.")

    fieldnames = OUTPUT_COLUMNS
    delimiter = "\t" if args.output_format == "tsv" else ","
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=fieldnames,
        delimiter=delimiter,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
