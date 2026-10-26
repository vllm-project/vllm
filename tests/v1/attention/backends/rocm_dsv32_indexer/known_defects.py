# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA indexer (ROCm) defects this suite found that are not fixed yet.

Every test matching a defect's patterns is marked ``xfail(strict=True)``: it
fails today, and the PR that fixes the defect deletes the entry, so the same
tests must then pass. Patterns are pytest node ids in which ``*`` matches any
text; everything else, brackets included, is literal.
"""

import re
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class Defect:
    reason: str
    tests: tuple[str, ...]


_INDEXER = "tests/v1/attention/backends/rocm_dsv32_indexer/"
_BUILDER = _INDEXER + "test_metadata_builder.py::"

KNOWN_DEFECTS: dict[str, Defect] = {}

KNOWN_DEFECTS["queryless-k-gather"] = Defect(
    "DeepseekV32IndexerMetadataBuilder puts prefill rows with no query "
    "tokens but seq_len > 0 (a request whose KV an async connector is "
    "loading, vllm-project/vllm#51486) into the prefill chunks, so their K "
    "is gathered and dequantized although no query reads it.",
    (
        _BUILDER + "test_prefill_chunks_skip_rows_without_queries[one-chunk-kv-load]",
        _BUILDER + "test_prefill_chunks_skip_rows_without_queries[extend-kv-load]",
    ),
)

_MARKED = pytest.StashKey[bool]()


def _matches(nodeid: str, pattern: str) -> bool:
    regex = ".*".join(re.escape(part) for part in pattern.split("*"))
    return re.fullmatch(regex, nodeid) is not None


def mark_known_defects(items: list[pytest.Item]) -> None:
    # Called from more than one conftest, and pytest passes every collected
    # item to each of them, so mark an item at most once.
    for item in items:
        if item.stash.get(_MARKED, False):
            continue
        for defect in KNOWN_DEFECTS.values():
            if any(_matches(item.nodeid, pattern) for pattern in defect.tests):
                item.add_marker(pytest.mark.xfail(strict=True, reason=defect.reason))
                item.stash[_MARKED] = True
                break
