# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    return filename.replace("/", "_").replace("..", "__").strip("'").strip('"')


def validate_result_paths(paths: Iterable[Path]) -> None:
    """Require distinct result directories for selected parameter combinations."""
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            raise ValueError(
                f"Parameter combinations map to the same result directory: {path}. "
                "Choose distinct `_benchmark_name` values that do not collide "
                "after filename sanitization."
            )
        seen.add(path)
