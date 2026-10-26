# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA defects this suite found that are not fixed yet.

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


_FA = "tests/v1/attention/backends/rocm_aiter_fa/"
_CAP = _FA + "test_capabilities.py::"
_DISPATCH = _FA + "test_dispatch.py::"
_BUILDER = _FA + "test_metadata_builder.py::"
_KV_CACHE = _FA + "test_kv_cache.py::"
_FEATURE = _DISPATCH + "test_feature_in_every_phase"
_ENGINE = "tests/v1/e2e/attention/test_rocm_aiter_fa_runtime.py::"

KNOWN_DEFECTS: dict[str, Defect] = {}

KNOWN_DEFECTS["mrv2-batch-order"] = Defect(
    "Model runner v2 orders non-decodes by token count, so a prefill can "
    "precede an extend; split_decodes_prefills_and_extends then runs that "
    "extend without its context.",
    (
        _DISPATCH + "test_batch_order_from_each_model_runner[runner-v2-order]",
        # Every engine run with model runner v2 (the flag before the model).
        _ENGINE + "test_engine_modes[*-True-qwen3]",
        _ENGINE + "test_engine_modes[*-True-qwen3_8]",
        _ENGINE + "test_engine_modes[*-True-muse_spark]",
        _ENGINE + "test_engine_modes[*-True-gpt_oss]",
        _ENGINE + "test_engine_modes_full[*-True]",
        _ENGINE + "test_ngram_spec_decode[*-True]",
    ),
)

KNOWN_DEFECTS["prefill-features"] = Defect(
    "AITER varlen prefill/extend drops the logits soft cap, the head-128 "
    "asm kernel ignores sinks (and extend double counts them across "
    "context chunks), and ALiBi positions restart in each context chunk.",
    (
        _FEATURE + "[*softcap-prefill]",
        _FEATURE + "[*softcap-extend]",
        _FEATURE + "[*softcap-mixed]",
        _FEATURE + "[*-alibi-extend]",
        _FEATURE + "[*-alibi-mixed]",
        _FEATURE + "[d128-*-sinks-prefill]",
        _FEATURE + "[d128-*-sinks-extend]",
        _FEATURE + "[d128-*-sinks-mixed]",
        _FEATURE + "[d256-auto-sinks-extend]",
        _FEATURE + "[d64-auto-sinks-extend]",
        _DISPATCH + "test_logits_soft_cap_every_phase[prefill]",
        _DISPATCH + "test_logits_soft_cap_every_phase[extend]",
        _DISPATCH + "test_logits_soft_cap_every_phase[extend-swa]",
    ),
)

KNOWN_DEFECTS["non-causal"] = Defect(
    "Non-causal extend runs with causal=True, and non-causal decode with an "
    "fp8 cache passes a bf16 query to mha_v3 (Mixed FP8 and non-FP8 tensors).",
    (
        _DISPATCH + "test_dispatch_row[fp8-D6-decode-noncausal]",
        _DISPATCH + "test_non_causal_in_every_phase[auto-extend]",
        _DISPATCH + "test_non_causal_in_every_phase[auto-mixed]",
        _DISPATCH + "test_non_causal_in_every_phase[fp8-decode]",
        _DISPATCH + "test_non_causal_in_every_phase[fp8-extend]",
        _DISPATCH + "test_non_causal_in_every_phase[fp8-mixed]",
    ),
)


KNOWN_DEFECTS["shuffle-lbhnc-layout"] = Defect(
    "Under shuffle the backend lists LBHNC, but shuffle decode needs K and V "
    "as dense planes (LHBNC).",
    (
        _CAP + "test_kv_cache_layouts[shuffle]",
        _KV_CACHE + "test_every_listed_layout_gives_identical_attention[shuffle]",
    ),
)

KNOWN_DEFECTS["fp8-e5m2-declared"] = Defect(
    "fp8_e5m2 is declared, but ROCm reshape_and_cache_flash rejects it at "
    "the first KV write.",
    (
        _CAP + "test_declarations_match_flagship_model_needs",
        _CAP + "test_selector_rejects_undeclared[overrides4-kv_cache_dtype]",
    ),
)

KNOWN_DEFECTS["kv-update-bypass-frozen"] = Defect(
    "supports_kv_cache_update_bypass is computed once at import and goes "
    "stale when the shuffle toggle changes.",
    (_CAP + "test_kv_update_bypass_follows_the_shuffle_toggle",),
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
