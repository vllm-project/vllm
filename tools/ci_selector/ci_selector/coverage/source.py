# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the coverage table comes from.

The only module that knows. Callers take a table and never a path, so moving
the table somewhere else means changing `fetch_table` and nothing else.
"""

from __future__ import annotations

import os
from pathlib import Path

from .table import Table, load

# Data rather than code, so it sits outside the importable package and is
# gitignored.
# TODO: read the table from the database instead of a file on disk. Only
# `fetch_table` should need to change.
COVERAGE_DIR = Path(__file__).resolve().parents[2] / "coverage-data"
TABLE_NAME = "table.json.gz"


def table_path() -> Path:
    """Where a local table is expected. `CI_SELECTOR_TABLE` overrides it."""
    override = os.environ.get("CI_SELECTOR_TABLE")
    return Path(override) if override else COVERAGE_DIR / TABLE_NAME


def fetch_table(path: Path | None = None) -> Table:
    """The coverage table, or an empty one that changes nothing.

    A missing table is not an error. The reason rides along on the table, so
    callers need no special case and the selector falls back to the code map.
    """
    target = path or table_path()
    if not target.is_file():
        return Table(
            None,
            unavailable=(
                f"no coverage table at {target}. Put one there (see the README) "
                f"or set CI_SELECTOR_TABLE. Running on the code map alone."
            ),
        )
    return load(target)
