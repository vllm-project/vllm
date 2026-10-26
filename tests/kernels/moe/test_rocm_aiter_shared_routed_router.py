# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AiterSharedRoutedFusedMoERouter with gate-less fused shared
experts (e.g. DeepSeek-V2 style models), where every routed expert stays
selectable and the shared experts get a constant weight."""

import importlib.util

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

if importlib.util.find_spec("aiter") is None:
    pytest.skip("These tests require AITER to run.", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.model_executor.layers.fused_moe.router.aiter_shared_routed_fused_moe_router import (  # noqa: E501
    AiterSharedRoutedFusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)

NUM_EXPERTS = 128
TOP_K = 4


@pytest.mark.parametrize("has_shared_expert_gate", [True, False])
def test_factory_honors_shared_expert_gate(monkeypatch, has_shared_expert_gate):
    monkeypatch.setattr(
        rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: True
    )
    router = create_fused_moe_router(
        top_k=TOP_K,
        global_num_experts=NUM_EXPERTS,
        num_fused_shared_experts=1,
        has_shared_expert_gate=has_shared_expert_gate,
    )
    assert isinstance(router, AiterSharedRoutedFusedMoERouter)
    assert router.has_shared_expert_gate == has_shared_expert_gate


@pytest.mark.parametrize("num_tokens", [1, 33, 512])
@pytest.mark.parametrize("num_shared", [1, 2])
@pytest.mark.parametrize("shared_expert_weight", [1.0, 0.5])
def test_gateless_routing_matches_softmax_topk(
    monkeypatch, num_tokens, num_shared, shared_expert_weight
):
    monkeypatch.setattr(rocm_aiter_moe, "aiter_topK_meta_data", None)
    rocm_aiter_moe.init_aiter_topK_meta_data.__wrapped__(
        n_routed_experts=NUM_EXPERTS,
        n_shared_experts=num_shared,
        top_k=TOP_K,
        tp_rank=0,
        tp_size=1,
        max_num_tokens=1024,
    )
    meta = rocm_aiter_moe.aiter_topK_meta_data
    assert meta is not None
    router = AiterSharedRoutedFusedMoERouter(
        top_k=TOP_K,
        global_num_experts=NUM_EXPERTS,
        num_fused_shared_experts=num_shared,
        has_shared_expert_gate=False,
        shared_expert_weight=shared_expert_weight,
    )

    torch.manual_seed(0)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 3
    # Make the last routed expert the clear winner for the first token.
    logits[0, -1] = 100.0
    hidden_states = torch.empty(num_tokens, 16, device="cuda")

    weights, ids = router._compute_routing(hidden_states, logits, None)

    assert weights.shape == (num_tokens, TOP_K + num_shared)
    assert ids.shape == (num_tokens, TOP_K + num_shared)
    assert weights.data_ptr() == meta[0].data_ptr()
    assert ids.data_ptr() == meta[1].data_ptr()

    ref_weights, ref_ids = torch.softmax(logits, dim=-1).topk(TOP_K, dim=-1)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    got_ids, order = ids[:, :TOP_K].long().sort(dim=-1)
    exp_ids, exp_order = ref_ids.sort(dim=-1)
    torch.testing.assert_close(got_ids, exp_ids)
    torch.testing.assert_close(
        weights[:, :TOP_K].gather(1, order), ref_weights.gather(1, exp_order)
    )
    assert (ids[0, :TOP_K] == NUM_EXPERTS - 1).any()

    shared_ids = torch.arange(
        NUM_EXPERTS, NUM_EXPERTS + num_shared, dtype=torch.int32, device="cuda"
    )
    assert (ids[:, TOP_K:] == shared_ids).all()
    assert (weights[:, TOP_K:] == shared_expert_weight).all()
