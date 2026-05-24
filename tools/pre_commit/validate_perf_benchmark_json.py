#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate performance benchmark JSON config files parse cleanly."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_BENCHMARK_TESTS_DIR = (
    REPO_ROOT / ".buildkite" / "performance-benchmarks" / "tests"
)


def validate_file(file_path: Path) -> bool:
    try:
        with file_path.open(encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as exc:
        print(
            f"❌ {file_path}: invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        )
        return False

    print(f"✅ {file_path}")
    return True


def iter_target_files(argv: list[str]) -> list[Path]:
    if argv:
        files = [Path(filename).resolve() for filename in argv]
    else:
        files = sorted(PERF_BENCHMARK_TESTS_DIR.glob("*.json"))

    return [
        file_path
        for file_path in files
        if PERF_BENCHMARK_TESTS_DIR in file_path.parents
    ]


def main() -> int:
    target_files = iter_target_files(sys.argv[1:])
    if not target_files:
        return 0

    all_valid = True
    for file_path in target_files:
        all_valid &= validate_file(file_path)

    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
