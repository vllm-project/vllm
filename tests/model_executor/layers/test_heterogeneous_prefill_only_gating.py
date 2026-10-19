# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill-only gating of the heterogeneous fused shared expert.

The fused hetero FHMoE only wins on prefill: at decode the baseline
already hides the separate shared expert under the routed chain, and its
grouped-GEMM numerics measurably shift gsm8k greedy outcomes (-0.61 pp)
on the large concurrent decode batches that clear the min-tokens
threshold. The append -- and the fused path -- is therefore restricted
to non-decode forwards:

* ``FusedTopKBiasRouter.shared_expert_append_prefill_only`` (set only by
  the DeepSeek-V4 heterogeneous model) suppresses the shared-slot append
  during unambiguous decode forwards, regardless of batch size;
* ``DeepseekV4HeterogeneousSharedRoutedExperts.forward_modular`` redirects
  any width-(K+1) route that still arrives during a decode forward to the
  stock overlapped separate layout (belt-and-suspenders for router paths
  that append unconditionally).

These tests pin the helper's truth table, the router behavior under
decode/prefill/no forward contexts on both append paths, and the
forward_modular redirect.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    _DECODE_MAX_QUERY_LEN,
    FusedTopKBiasRouter,
    _forward_allows_fused_shared_append,
)

_DECODE_CTX = 1 + 5  # speculative decode with k=5: 6-token verify queries
_PREFILL_CTX = 2048  # chunked-prefill-sized query


def _meta(max_query_len: int | None) -> Any:
    # A duck-typed AttentionMetadata stand-in: the gate reads only
    # max_query_len off it.
    return SimpleNamespace(max_query_len=max_query_len)


def _ctx(max_query_len: int) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={"layer0": _meta(max_query_len)},
        slot_mapping={},
    )


@pytest.mark.parametrize(
    ("max_query_len", "expected"),
    [
        (None, True),  # no context at all -> stock threshold-only semantics
        (1, False),  # plain decode
        (6, False),  # speculative decode verify (k=5)
        (_DECODE_MAX_QUERY_LEN, False),  # boundary: still decode
        (_DECODE_MAX_QUERY_LEN + 1, True),  # just past the boundary
        (16, True),  # smallest fusable prefill batch
        (2048, True),  # chunked prefill
    ],
)
def test_forward_allows_append_truth_table(max_query_len, expected) -> None:
    if max_query_len is None:
        assert _forward_allows_fused_shared_append() is True
        return
    with override_forward_context(_ctx(max_query_len)):
        assert _forward_allows_fused_shared_append() is expected


def test_forward_allows_append_tolerates_unusual_metadata() -> None:
    # Empty dict, DBO list, and non-int max_query_len all fall back to
    # "allow" (threshold-only semantics) instead of guessing.
    with override_forward_context(
        ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    ):
        assert _forward_allows_fused_shared_append() is True
    with override_forward_context(
        ForwardContext(
            no_compile_layers={},
            attn_metadata=[{"layer0": _meta(_DECODE_CTX)}, {}],
            slot_mapping={},
        )
    ):  # DBO: the first microbatch's metadata decides
        assert _forward_allows_fused_shared_append() is False
    with override_forward_context(
        ForwardContext(
            no_compile_layers={},
            attn_metadata={"layer0": _meta(None)},
            slot_mapping={},
        )
    ):
        assert _forward_allows_fused_shared_append() is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the softplus-sqrt top-k kernel only runs on GPU",
)
@pytest.mark.parametrize("aiter_fast", [True, False])
def test_prefill_only_suppresses_append_during_decode(aiter_fast: bool) -> None:
    """M=64 decode forward: width-6 routes, bit-exact with a plain router."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_experts, topk, n, threshold, m = 384, 6, 1, 16, 64
    gating = torch.randn(m, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)
    hidden = torch.randn(m, 64, device=device, dtype=torch.bfloat16)

    def make_router(prefill_only: bool) -> FusedTopKBiasRouter:
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
        router.shared_expert_append_min_tokens = threshold
        router.shared_expert_append_prefill_only = prefill_only
        return router

    plain = FusedTopKBiasRouter(
        top_k=topk,
        global_num_experts=num_experts,
        e_score_correction_bias=bias.clone(),
        renormalize=True,
        routed_scaling_factor=2.5,
        scoring_func="sqrtsoftplus",
    )

    with (
        patch.object(
            rocm_aiter_ops, "is_fused_moe_enabled", lambda fast=aiter_fast: fast
        ),
        patch.object(envs, "VLLM_MOE_SKIP_PADDING", False),
    ):
        # Decode forward: no append even though M=64 >= threshold.
        with override_forward_context(_ctx(_DECODE_CTX)):
            w, ids = make_router(True)._select_experts(hidden, gating)
        pw, pids = plain._select_experts(hidden, gating)
        assert ids.shape == (m, topk)
        torch.testing.assert_close(w, pw)
        torch.testing.assert_close(ids, pids)

        # Prefill forward (same M): append fires -> fused width.
        with override_forward_context(_ctx(_PREFILL_CTX)):
            w, ids = make_router(True)._select_experts(hidden, gating)
        assert ids.shape == (m, topk + n)

        # No forward context: threshold-only semantics preserved for
        # context-less callers.
        w, ids = make_router(True)._select_experts(hidden, gating)
        assert ids.shape == (m, topk + n)

        # prefill_only=False (stock callers, e.g. the requant fusion):
        # decode forward with M >= threshold still appends (unchanged).
        with override_forward_context(_ctx(_DECODE_CTX)):
            w, ids = make_router(False)._select_experts(hidden, gating)
        assert ids.shape == (m, topk + n)


def test_forward_modular_redirects_width7_decode_routes(monkeypatch) -> None:
    """Belt-and-suspenders: width-(K+1) routes during a decode forward are
    sliced back to routed width and sent to the stock overlapped layout."""
    from vllm.models.deepseek_v4.amd.model import (
        DeepseekV4HeterogeneousSharedRoutedExperts,
    )

    experts = DeepseekV4HeterogeneousSharedRoutedExperts.__new__(
        DeepseekV4HeterogeneousSharedRoutedExperts
    )
    experts.moe_config = SimpleNamespace(experts_per_token=6)
    # The fused path touches these before raising "not prepared"; leaving
    # them unset would turn that proof into an AttributeError.
    experts.shared_w1 = experts.shared_w2 = None
    experts.shared_w1_scale = experts.shared_w2_scale = None

    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_overlapped(self, x, topk_weights, topk_ids):
        calls.append((topk_weights, topk_ids))
        return torch.zeros_like(x)

    monkeypatch.setattr(
        DeepseekV4HeterogeneousSharedRoutedExperts,
        "_forward_overlapped_separate",
        fake_overlapped,
    )

    m, k = 32, 6
    x = torch.zeros(m, 8)
    w7 = torch.zeros(m, k + 1)
    ids7 = torch.zeros(m, k + 1, dtype=torch.int64)

    with override_forward_context(_ctx(_DECODE_CTX)):
        out = experts.forward_modular(x, w7.clone(), ids7.clone())
    assert out is not None
    assert len(calls) == 1, "decode forward must take the redirect"
    rw, rids = calls[0]
    assert rw.shape == (m, k) and rids.shape == (m, k)
    torch.testing.assert_close(rw, w7[:, :-1])
    torch.testing.assert_close(rids, ids7[:, :-1].to(torch.int64))
    assert rids.is_contiguous() and rw.is_contiguous()

    # Prefill forward with the same width-7 routes must NOT redirect: it
    # proceeds toward the fused path (the shared weights are unprepared in
    # this bare instance, which is the observable proof of the dispatch).
    calls.clear()
    with (
        override_forward_context(_ctx(_PREFILL_CTX)),
        pytest.raises(
            RuntimeError, match="Heterogeneous shared weights were not prepared"
        ),
    ):
        experts.forward_modular(x, w7.clone(), ids7.clone())
    assert calls == []

    # Width-6 routes keep the stock decode dispatch in any mode.
    calls.clear()
    w6 = torch.zeros(m, k)
    ids6 = torch.zeros(m, k, dtype=torch.int64)
    with override_forward_context(_ctx(_DECODE_CTX)):
        experts.forward_modular(x, w6.clone(), ids6.clone())
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], w6)
