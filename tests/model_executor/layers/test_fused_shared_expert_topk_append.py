# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused-shared-experts in-kernel top-k append.

FusedTopKBiasRouter appends `num_fused_shared_experts` constant (id, weight)
slots to every token's routing row. The cat-based append costs four tiny
launch-latency-bound kernels per layer per step (+606 us/step measured on
DSV4.1-Flash TP4 batch-1 decode). The in-kernel path hands the top-k kernel
a persistent (capacity, topk + n) buffer and lets it write the shared
columns directly, exactly like SGLang's hash_topk.

These tests prove the in-kernel append is BIT-EXACT with the cat-based
append at the op level and at the router level, that the persistent buffers
grow safely, and that the capture guard falls back to the cat path.
"""

from unittest.mock import patch

import pytest
import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    _FUSED_SHARED_TOPK_BUF_KEEPALIVE,
    _FUSED_SHARED_TOPK_BUFS,
    FusedTopKBiasRouter,
    _get_fused_shared_topk_buffers,
    vllm_topk_softplus_sqrt,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the softplus-sqrt top-k kernel only runs on GPU",
)


@pytest.fixture(autouse=True)
def _reset_buffer_cache():
    _FUSED_SHARED_TOPK_BUFS.clear()
    _FUSED_SHARED_TOPK_BUF_KEEPALIVE.clear()
    yield
    _FUSED_SHARED_TOPK_BUFS.clear()
    _FUSED_SHARED_TOPK_BUF_KEEPALIVE.clear()


def _reference_cat_append(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    num_fused_shared: int,
    shared_expert_base: int,
    shared_expert_weight: float,
    indices_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: top-k into fresh (M, k) tensors, then arange/full/cat
    append of the shared slots."""
    m = gating_output.shape[0]
    topk_weights = torch.empty(
        m, topk, dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(m, topk, dtype=indices_dtype, device=gating_output.device)
    token_expert_indices = torch.empty(
        m, topk, dtype=torch.int32, device=gating_output.device
    )
    with patch.object(envs, "VLLM_MOE_SKIP_PADDING", False):
        vllm_topk_softplus_sqrt(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias=bias,
            routed_scaling_factor=routed_scaling_factor,
        )
    shared_ids = torch.arange(
        shared_expert_base,
        shared_expert_base + num_fused_shared,
        dtype=indices_dtype,
        device=gating_output.device,
    ).expand(m, num_fused_shared)
    shared_w = torch.full(
        (m, num_fused_shared),
        shared_expert_weight,
        dtype=torch.float32,
        device=gating_output.device,
    )
    return (
        torch.cat([topk_weights, shared_w], dim=-1),
        torch.cat([topk_ids, shared_ids], dim=-1),
    )


@pytest.mark.parametrize(
    "num_experts,topk,m,n",
    [
        (384, 6, 6, 1),  # DSV4.1-Flash target layers, decode with spec k=5
        (384, 6, 2048, 1),  # prefill-sized batch (forces buffer growth)
        (256, 4, 33, 2),  # draft-ish shape, two shared slots
    ],
)
def test_in_kernel_append_bit_exactly_matches_cat_append(
    num_experts: int,
    topk: int,
    m: int,
    n: int,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    gating = torch.randn(m, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)
    base = num_experts
    weight = 1.0 / 1.5  # non-trivial shared weight (routed_scaling 1.5)
    rsf = 2.5

    ref_w, ref_ids = _reference_cat_append(
        gating, bias, topk, True, rsf, n, base, weight, torch.int32
    )

    bufs = _get_fused_shared_topk_buffers(m, topk, n, base, device, torch.int32)
    assert bufs is not None
    ids_buf, w_buf = bufs
    assert ids_buf.shape[1] == topk + n
    fast_ids = ids_buf[:m]
    fast_w = w_buf[:m]
    tei = torch.empty(m, topk, dtype=torch.int32, device=device)
    with patch.object(envs, "VLLM_MOE_SKIP_PADDING", False):
        vllm_topk_softplus_sqrt(
            fast_w,
            fast_ids,
            tei,
            gating,
            True,
            e_score_correction_bias=bias,
            routed_scaling_factor=rsf,
            num_fused_shared_experts=n,
            shared_expert_base=base,
            shared_expert_weight=weight,
        )

    torch.testing.assert_close(fast_w, ref_w)
    torch.testing.assert_close(fast_ids, ref_ids)
    # The shared slots specifically: constant id and weight on every row.
    assert torch.equal(
        fast_ids[:, topk:],
        torch.arange(base, base + n, device=device, dtype=torch.int32)
        .expand(m, n)
        .contiguous(),
    )
    assert torch.all(fast_w[:, topk:] == weight)


def test_buffer_growth_keeps_old_storage_alive_and_stays_correct():
    device = torch.device("cuda")
    bufs = _get_fused_shared_topk_buffers(8, 6, 1, 384, device, torch.int32)
    assert bufs is not None
    assert bufs[0].shape[0] >= 8
    grown = _get_fused_shared_topk_buffers(4096, 6, 1, 384, device, torch.int32)
    assert grown is not None
    assert grown[0].shape[0] >= 4096
    assert grown is not bufs
    # Superseded buffers stay alive for captured graphs that reference them.
    assert bufs in _FUSED_SHARED_TOPK_BUF_KEEPALIVE
    # Same key now returns the grown buffer.
    again = _get_fused_shared_topk_buffers(16, 6, 1, 384, device, torch.int32)
    assert again is grown


def test_capture_guard_returns_none_and_router_falls_back():
    torch.manual_seed(0)
    device = torch.device("cuda")
    m, num_experts, topk, n = 6, 384, 6, 1
    gating = torch.randn(m, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)
    hidden = torch.randn(m, 64, device=device, dtype=torch.bfloat16)

    def make_router() -> FusedTopKBiasRouter:
        return FusedTopKBiasRouter(
            top_k=topk,
            global_num_experts=num_experts,
            e_score_correction_bias=bias.clone(),
            renormalize=True,
            routed_scaling_factor=2.5,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=n,
            shared_expert_weight=1.0 / 1.5,
        )

    # Arm 1: fast path (aiter enabled, not capturing).
    with (
        patch.object(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True),
        patch.object(envs, "VLLM_MOE_SKIP_PADDING", False),
    ):
        fast_w, fast_ids = make_router()._select_experts(hidden, gating)

    # Arm 2: capture in progress with insufficient capacity -> None -> cat.
    with (
        patch.object(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True),
        patch.object(torch.cuda, "is_current_stream_capturing", lambda: True),
        patch.object(envs, "VLLM_MOE_SKIP_PADDING", False),
    ):
        cap_w, cap_ids = make_router()._select_experts(hidden, gating)

    # Arm 3: aiter disabled -> cat fallback.
    with (
        patch.object(rocm_aiter_ops, "is_fused_moe_enabled", lambda: False),
        patch.object(envs, "VLLM_MOE_SKIP_PADDING", False),
    ):
        cat_w, cat_ids = make_router()._select_experts(hidden, gating)

    torch.testing.assert_close(fast_w, cat_w)
    torch.testing.assert_close(fast_ids, cat_ids)
    torch.testing.assert_close(cap_w, cat_w)
    torch.testing.assert_close(cap_ids, cat_ids)
    # All arms produced the fused width (topk + n).
    assert fast_ids.shape == (m, topk + n)
    assert cap_ids.shape == (m, topk + n)
    assert cat_ids.shape == (m, topk + n)


def test_router_fast_path_writes_into_persistent_buffer():
    torch.manual_seed(0)
    device = torch.device("cuda")
    m, num_experts, topk, n = 6, 384, 6, 1
    gating = torch.randn(m, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)
    hidden = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
    router = FusedTopKBiasRouter(
        top_k=topk,
        global_num_experts=num_experts,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=2.5,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=n,
        shared_expert_weight=1.0 / 1.5,
    )
    with (
        patch.object(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True),
        patch.object(envs, "VLLM_MOE_SKIP_PADDING", False),
    ):
        w1, ids1 = router._select_experts(hidden, gating)
        bufs = _FUSED_SHARED_TOPK_BUFS.get((device.index or 0, topk + n, torch.int32))
        assert bufs is not None, "fast path must have allocated the buffer"
        # Returned tensors are views of the persistent storage.
        assert ids1.data_ptr() == bufs[0][:m].data_ptr()
        assert w1.data_ptr() == bufs[1][:m].data_ptr()
        # A second call reuses the same storage (no reallocation).
        w2, ids2 = router._select_experts(hidden, gating)
        assert ids2.data_ptr() == ids1.data_ptr()
        torch.testing.assert_close(w2, w1)
        torch.testing.assert_close(ids2, ids1)


def test_shared_expert_append_min_tokens_gates_both_paths():
    """Prefill-only gating of the shared-slot append.

    Models whose fused-shared-expert implementation engages only for
    prefill-sized batches (the DeepSeek-V4 heterogeneous FHMoE path) set
    ``shared_expert_append_min_tokens`` on the router. Below the threshold
    the router must emit plain routed-width top-k on BOTH the in-kernel
    fast path and the cat fallback -- bit-exact with a
    num_fused_shared_experts=0 router, so decode-sized batches keep the
    stock overlapped separate shared-expert execution. At/above the
    threshold (and with the default None) the append must keep firing.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_experts, topk, n, threshold = 384, 6, 1, 16
    gating = torch.randn(64, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)

    def make_router(min_tokens):
        router = FusedTopKBiasRouter(
            top_k=topk,
            global_num_experts=num_experts,
            e_score_correction_bias=bias.clone(),
            renormalize=True,
            routed_scaling_factor=2.5,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=n,
            shared_expert_weight=1.0 / 1.5,
        )
        router.shared_expert_append_min_tokens = min_tokens
        return router

    plain = FusedTopKBiasRouter(
        top_k=topk,
        global_num_experts=num_experts,
        e_score_correction_bias=bias.clone(),
        renormalize=True,
        routed_scaling_factor=2.5,
        scoring_func="sqrtsoftplus",
    )

    for m in (6, threshold - 1, threshold):
        hidden = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
        g = gating[:m]
        for aiter_fast in (True, False):
            with (
                patch.object(
                    rocm_aiter_ops,
                    "is_fused_moe_enabled",
                    lambda fast=aiter_fast: fast,
                ),
                patch.object(envs, "VLLM_MOE_SKIP_PADDING", False),
            ):
                w, ids = make_router(threshold)._select_experts(hidden, g)
                pw, pids = plain._select_experts(hidden, g)
            if m < threshold:
                # Below the threshold: routed width, no append, bit-exact
                # with the never-append router (both fast and cat paths).
                assert ids.shape == (m, topk), (m, aiter_fast, ids.shape)
                torch.testing.assert_close(w, pw)
                torch.testing.assert_close(ids, pids)
            else:
                # At/above the threshold: the append fires with the
                # constant shared slot.
                assert ids.shape == (m, topk + n), (m, aiter_fast, ids.shape)
                assert torch.equal(
                    ids[:, topk:],
                    torch.full((m, n), num_experts, device=device, dtype=ids.dtype),
                )
                assert torch.all(w[:, topk:] == 1.0 / 1.5)

    # Default None keeps appending at every batch size (existing callers,
    # e.g. the FP8->MXFP4 requant path, are unchanged).
    with (
        patch.object(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True),
        patch.object(envs, "VLLM_MOE_SKIP_PADDING", False),
    ):
        w, ids = make_router(None)._select_experts(
            torch.randn(6, 64, device=device, dtype=torch.bfloat16),
            gating[:6],
        )
    assert ids.shape == (6, topk + n)


def test_op_signature_backward_compat():
    """Old callers (no fused args) keep working through the new schema."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    m, num_experts, topk = 6, 384, 6
    gating = torch.randn(m, num_experts, device=device, dtype=torch.float32)
    w = torch.empty(m, topk, dtype=torch.float32, device=device)
    ids = torch.empty(m, topk, dtype=torch.int32, device=device)
    tei = torch.empty(m, topk, dtype=torch.int32, device=device)
    with patch.object(envs, "VLLM_MOE_SKIP_PADDING", False):
        ops.topk_hash_softplus_sqrt(
            w,
            ids,
            tei,
            gating,
            True,  # renormalize
            2.5,  # routed_scaling_factor
        )
    assert ids.shape == (m, topk)
    assert (ids >= 0).all() and (ids < num_experts).all()
    assert torch.isfinite(w).all()


def test_fused_topk_bias_validates_out_buffers():
    device = torch.device("cuda")
    m, num_experts, topk, n = 6, 384, 6, 1
    gating = torch.randn(m, num_experts, device=device, dtype=torch.float32)
    hidden = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
    bad = torch.empty(m, topk, dtype=torch.float32, device=device)
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    with (
        patch.object(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True),
        pytest.raises(AssertionError),
    ):
        fused_topk_bias(
            hidden_states=hidden,
            gating_output=gating,
            scoring_func="sqrtsoftplus",
            e_score_correction_bias=None,
            topk=topk,
            renormalize=True,
            out_topk_weights=bad,  # wrong width (topk instead of topk + n)
            out_topk_ids=bad.to(torch.int32),
            num_fused_shared_experts=n,
        )
