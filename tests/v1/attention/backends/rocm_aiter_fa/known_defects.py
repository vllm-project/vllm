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

KNOWN_DEFECTS["shuffle-unsupported-features"] = Defect(
    "The shuffle KV layout accepts sliding window, sinks, soft cap, ALiBi "
    "and multi-token decode, then asserts or computes wrong results at the "
    "first decode instead of refusing the config.",
    (
        _CAP + "test_shuffle_layout_rejects_window_and_sinks[shuffle-*]",
        _DISPATCH + "test_shuffle_refuses_features_it_cannot_decode[*]",
        _BUILDER + "test_shuffle_layout_refuses_multi_token_decode",
        _ENGINE + "test_shuffle_layout_refuses_sliding_window_model",
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


# Defects outside this backend.

KNOWN_DEFECTS["aiter-gluon-buffer-offset"] = Defect(
    "AITER's gfx950 Gluon unified_attention enables 32-bit buffer loads when "
    "k.nelement() * element_size fits in int32, but vLLM's K and V are "
    "interleaved views of one cache, so the K view spans twice that. With a "
    "1-2 GiB K view, blocks past the 2 GiB offset are read wrong.",
    (
        _FA + "test_correctness.py::test_full_matrix"
        "[gpt-oss-120b-tp1-h64kv8d64-sinks-extend-8192-128-*-auto]",
    ),
)
