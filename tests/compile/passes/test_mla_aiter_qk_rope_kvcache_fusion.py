# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AITER MLA QK-RoPE+KVCache fusion.

This file covers three things:

1. **INVARIANT 1 (no shape lying)** — :py:func:`test_mla_decode_q_prep_invariant_1`
   asserts that ``mla_decode_q_prep_impl`` produces an output whose batch
   dimension is exactly ``q.size(0)`` for varying ``T``. Guards against any
   future regression where someone slices internally.

2. **Numerical parity** — :py:func:`test_mla_aiter_qk_rope_kvcache_fusion`
   compiles a small DSV3-shaped MLA layer with the lift + AITER fusion passes
   enabled and compares the output against the unfused eager path.

3. **CUDA-graph capture/replay regression** —
   :py:func:`test_mla_aiter_fusion_cuda_graph_capture` runs
   ``cudagraph_mode=FULL`` capture/replay at both boundaries of the active
   decode compile range and asserts no shape-mismatch warning, no GPU fault,
   and numerical parity against the eager path. This is the explicit
   regression test for the bug class the discarded PR hit on the
   (4682, 16384) compile range.
"""
from __future__ import annotations

import pytest
import torch

import vllm.config
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.platforms import current_platform


# --------------------------------------------------------------------------- #
# 1. INVARIANT 1: no shape lying inside mla_decode_q_prep                     #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (current_platform.is_rocm() and is_aiter_found_and_supported()),
    reason="AITER MLA fusion requires ROCm + AITER.",
)
@pytest.mark.parametrize("T", [1, 16, 64, 256])
def test_mla_decode_q_prep_invariant_1(T: int) -> None:
    """``mla_decode_q_prep_impl`` must produce ``out.shape[0] == q.shape[0]``
    for every input batch size.

    Violating this rule is what caused the discarded PR's CUDA-graph capture
    fault on the (4682, 16384) range: that PR's impl sliced internally to
    ``q[:num_decode_tokens]``, producing ``[num_decode, ...]`` while the
    fake_impl declared ``[s59, ...]``. Inductor planned downstream ops to
    ``s59=16384``, runtime delivered ``num_decode=0``, kernel fault.

    This test ensures any future regression where someone re-introduces the
    slice fails immediately.
    """
    from tests.compile.passes._mla_aiter_test_helpers import (
        build_test_layer_and_inputs,
    )

    layer, q = build_test_layer_and_inputs(T)
    out = layer.do_decode_q_prep_concat(q)
    assert out.shape[0] == q.shape[0], (
        f"INVARIANT 1 violation: do_decode_q_prep_concat returned "
        f"out.shape[0]={out.shape[0]} for q.shape[0]={q.shape[0]}. "
        "Some implementer must have re-introduced an internal slice. See "
        "discarded-PR fault on (4682, 16384) compile range."
    )
    assert out.shape[1] == layer.num_heads
    assert out.shape[2] == layer.kv_lora_rank + layer.qk_rope_head_dim


# --------------------------------------------------------------------------- #
# 2. Numerical parity                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (current_platform.is_rocm() and is_aiter_found_and_supported()),
    reason="AITER MLA fusion requires ROCm + AITER.",
)
@pytest.mark.parametrize("T", [4, 32])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("is_neox", [True, False])
def test_mla_aiter_qk_rope_kvcache_fusion(
    T: int,
    kv_cache_dtype: str,
    is_neox: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare unfused eager vs. lift+AITER-fusion compiled output.

    The fused path must be numerically equivalent to the unfused path within
    the AITER op-test FP8 tolerances (atol=0.01, rtol=0.01).
    """
    from tests.compile.passes._mla_aiter_test_helpers import run_parity_check

    run_parity_check(
        T=T,
        kv_cache_dtype=kv_cache_dtype,
        is_neox=is_neox,
        monkeypatch=monkeypatch,
    )


# --------------------------------------------------------------------------- #
# 3. CUDA-graph capture/replay regression                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (current_platform.is_rocm() and is_aiter_found_and_supported()),
    reason="AITER MLA fusion requires ROCm + AITER.",
)
@pytest.mark.parametrize("compile_size", [1, 256])
def test_mla_aiter_fusion_cuda_graph_capture(
    compile_size: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Capture + replay a graph with ``mla_decode_q_prep`` lifted in.

    Asserts:
    - capture succeeds (no ``RuntimeError`` from inductor about shape
      mismatches),
    - replay produces a finite tensor (no GPU fault from a null-pointer
      kernel launch),
    - no ``"shape mismatch"`` / ``"re-record"`` substring in inductor logs.

    This is the explicit regression test for the discarded PR's failure mode
    on the (4682, 16384) compile range: a fake_impl that declared a wider
    shape than the real impl produced caused inductor to plan downstream ops
    against the wrong size, then launch them on an empty buffer at warmup.
    Our fake_impl is shape-honest (Section 1 of the plan, INVARIANT 1) AND
    the lift pass is gated by ``is_applicable_for_range`` so the op is
    literally absent from large-batch graphs (Section 2.5). This test
    exercises both invariants by capturing at the smallest and largest
    decode-bucket sizes.
    """
    from tests.compile.passes._mla_aiter_test_helpers import (
        run_cuda_graph_capture_replay,
    )

    run_cuda_graph_capture_replay(
        compile_size=compile_size,
        monkeypatch=monkeypatch,
        caplog=caplog,
    )
