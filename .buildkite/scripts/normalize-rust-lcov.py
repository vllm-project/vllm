#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def normalize_source_path(source: str, repo_root: Path) -> str | None:
    source = source.replace("\\", "/")
    root = repo_root.resolve()
    candidates: list[PurePosixPath] = []

    if source.startswith("rust/"):
        candidates.append(PurePosixPath(source))

    for prefix in ("/workspace/rust/", f"{root.as_posix()}/rust/"):
        if source.startswith(prefix):
            candidates.append(PurePosixPath("rust") / source.removeprefix(prefix))

    marker = "/rust/"
    if marker in source:
        candidates.append(PurePosixPath("rust") / source.rsplit(marker, maxsplit=1)[1])

    rust_root = (root / "rust").resolve()
    for candidate in candidates:
        if candidate.is_absolute() or ".." in candidate.parts:
            continue
        if not candidate.parts or candidate.parts[0] != "rust":
            continue

        resolved = (root / Path(*candidate.parts)).resolve()
        if not resolved.is_relative_to(rust_root) or not resolved.is_file():
            continue
        return candidate.as_posix()

    return None


def normalize_record(record: list[str], repo_root: Path) -> list[str] | None:
    source_indexes = [
        index for index, line in enumerate(record) if line.startswith("SF:")
    ]
    if len(source_indexes) != 1:
        return None

    source_index = source_indexes[0]
    normalized = normalize_source_path(record[source_index][3:], repo_root)
    if normalized is None:
        return None

    record[source_index] = f"SF:{normalized}"
    return record


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    records: list[list[str]] = []
    record: list[str] = []

    for line in args.input.read_text(encoding="utf-8").splitlines():
        record.append(line)
        if line == "end_of_record":
            normalized = normalize_record(record, repo_root)
            if normalized is not None:
                records.append(normalized)
            record = []

    if record:
        normalized = normalize_record(record, repo_root)
        if normalized is not None:
            records.append(normalized)

    if not records:
        raise RuntimeError("LCOV contains no Rust workspace source records")
    if not any(line.startswith("DA:") for item in records for line in item):
        raise RuntimeError("LCOV contains no Rust line coverage records")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = "\n".join(line for item in records for line in item) + "\n"
    args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
