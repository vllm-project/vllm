# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducer / regression test for
https://github.com/vllm-project/vllm/issues/49649

AITER has no non-persistent sparse-MLA decode kernel for ``gqa_ratio == 64``
fp8/fp8 (``asm_mla.cu:949``: "fp8/fp8 with gqa_ratio=64 only supports persistent
mode"). A persistent-kernel gate that falls back to the non-persistent split-KV
path for such groupings (e.g. for chunked-prefill continuations) therefore
crashes the prefill worker deterministically.

``sparse_mla_requires_persistent`` encodes that invariant so any such gate can
skip it (keep ``use_persistent=True``) for gqa_ratio=64 fp8. This is a pure
decision function, so the test runs on CPU CI (no ROCm/AITER/GPU required).
"""

from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    sparse_mla_requires_persistent,
)


def test_gqa64_fp8_is_persistent_only():
    # The exact GLM-5.1-FP8 DSA disagg config (DP=8/TP=1 => gqa_ratio=64) with
    # fp8 KV: no non-persistent kernel exists -> persistent is mandatory.
    for kv_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        assert sparse_mla_requires_persistent(64, kv_dtype) is True


def test_non_fp8_gqa64_has_non_persistent():
    # bf16 / auto KV at gqa_ratio=64 is not persistent-only.
    for kv_dtype in ("bfloat16", "float16", "auto"):
        assert sparse_mla_requires_persistent(64, kv_dtype) is False


def test_other_gqa_ratios_have_non_persistent():
    # Smaller head groupings (e.g. gqa_ratio=16 at TP=8) are not persistent-only.
    for gqa in (8, 16, 32, 128):
        assert sparse_mla_requires_persistent(gqa, "fp8") is False
