# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA_SPARSE defects this suite found that are not fixed yet.

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


_SPARSE = "tests/v1/attention/backends/rocm_aiter_mla_sparse/"
_DISPATCH = _SPARSE + "test_dispatch.py::"
_BUILDER = _SPARSE + "test_metadata_builder.py::"
_ZERO_QUERY = _DISPATCH + "test_zero_query_rows_after_real_rows"

KNOWN_DEFECTS: dict[str, Defect] = {}

KNOWN_DEFECTS["queryless-context"] = Defect(
    "The dense-MHA context chunk planner (build_mla_chunked_context_metadata, "
    "shared with every MLA backend) gives context to a row with no query "
    "tokens but seq_len > 0 (a request whose KV an async connector is "
    "loading, vllm-project/vllm#51486). A chunk of such rows gathers KV "
    "nobody attends to, and AITER's flash_attn_varlen aborts the process on "
    "it.",
    (
        _BUILDER + "test_context_chunks_skip_rows_without_queries[*-kv-load]",
        _ZERO_QUERY + "[dense-*-kv-load-bf16]",
        _ZERO_QUERY + "[dense-*-kv-load-fp8]",
        _ZERO_QUERY + "[mixed-kv-load-bf16]",
        _ZERO_QUERY + "[mixed-kv-load-fp8]",
        _DISPATCH + "test_zero_query_rows_nope_glm_flash[mixed-kv-load]",
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
