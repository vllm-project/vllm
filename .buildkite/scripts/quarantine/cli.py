# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI entry point for flaky test quarantine management.

Reads test results from a JSONL file, updates the quarantine list,
and writes the result to a JSON file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# This script lives in .buildkite/scripts/quarantine/ and is run
# standalone, not as an installed package.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from detect import (  # noqa: E402
    QuarantineEntry,
    parse_test_result,
    update_quarantine_list,
)


def load_results(path: Path) -> list:
    results = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSON: {exc}") from exc
            try:
                results.append(parse_test_result(raw))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_num}: {exc}") from exc
    return results


def load_quarantine_list(path: Path) -> list[QuarantineEntry]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [QuarantineEntry(**entry) for entry in data.get("entries", [])]


def save_quarantine_list(
    path: Path,
    entries: list[QuarantineEntry],
    now: datetime,
) -> None:
    data = {
        "generated_at": now.isoformat(),
        "entries": [dataclasses.asdict(e) for e in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect flaky tests and update quarantine list",
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="JSONL file with test results",
    )
    parser.add_argument(
        "--quarantine-file",
        type=Path,
        default=Path(".buildkite/quarantined_tests.json"),
        help="quarantine list JSON (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        results = load_results(args.results_file)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No test results to process.", file=sys.stderr)
        return 0

    current = load_quarantine_list(args.quarantine_file)
    now = datetime.now(timezone.utc)
    updated = update_quarantine_list(current, results, now)
    save_quarantine_list(args.quarantine_file, updated, now)

    prev_keys = {(e.test_id, e.backend) for e in current}
    new_keys = {(e.test_id, e.backend) for e in updated}
    added = new_keys - prev_keys
    removed = prev_keys - new_keys

    print(f"Quarantine list: {len(updated)} entries")
    if added:
        print(f"  Added: {len(added)}")
    if removed:
        print(f"  Reinstated: {len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
