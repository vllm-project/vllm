#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare deterministic GSM8K records from two fixed-weight evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INVALID = -9999999


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def comparable(record: dict) -> dict:
    return {
        key: record.get(key)
        for key in (
            "index",
            "label",
            "prediction",
            "content",
            "finish_reason",
            "completion_tokens",
            "error",
        )
    }


def summary(records: list[dict]) -> dict:
    valid = [record for record in records if "error" not in record]
    correct = sum(record.get("prediction") == record.get("label") for record in valid)
    return {
        "accuracy": correct / len(records) if records else 0.0,
        "correct_count": correct,
        "invalid_rate": sum(record.get("prediction") == INVALID for record in records)
        / len(records)
        if records
        else 0.0,
        "request_error_count": len(records) - len(valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    before_records = load(args.before)
    after_records = load(args.after)
    before = [comparable(item) for item in before_records]
    after = [comparable(item) for item in after_records]
    result = {
        "same_records": before == after,
        "before_count": len(before),
        "after_count": len(after),
        "different_indices": [
            index
            for index, (left, right) in enumerate(zip(before, after))
            if left != right
        ],
        "before_summary": {
            **summary(before_records),
        },
        "after_summary": {
            **summary(after_records),
        },
    }
    result["same_count"] = len(before) == len(after)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["same_records"] and result["same_count"] else 1)


if __name__ == "__main__":
    main()
