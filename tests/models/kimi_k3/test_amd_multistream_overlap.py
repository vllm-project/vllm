# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity and HIP-graph coverage for AMD Kimi-K3 scheduling overlaps."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.kimi_k3.amd.linear as amd_linear
import vllm.models.kimi_k3.amd.mla as amd_mla
from vllm.models.kimi_k3.amd.linear import KimiMoE
from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper
from vllm.platforms import current_platform


class _TupleScale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x * self.scale, None


_TOP_K = 2


class _TopKRouter:
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states, input_ids
        topk_weights, topk_ids = torch.topk(router_logits.float(), _TOP_K, dim=-1)
        topk_ids = topk_ids.to(topk_indices_dtype or torch.int32)
        return topk_weights.softmax(dim=-1), topk_ids


class _QuantMethod:
    topk_indices_dtype = None


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.routed: torch.Tensor | None = None
        self.shared: torch.Tensor | None = None
        self.router_logits: torch.Tensor | None = None
        self.router = amd_linear._PackedRoutingRouter(_TopKRouter())
        self._quant_method = _QuantMethod()

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        self.routed = hidden_states
        self.shared = shared_experts_input
        self.router_logits = router_logits
        return hidden_states


def _expected_routing(hidden_states: torch.Tensor):
    return _TopKRouter().select_experts(hidden_states, hidden_states * 3.0)


def _parallel(fn0, fn1, event0, event1, stream):
    del event0, event1
    assert stream is not None
    return fn0(), fn1()


def _make_moe(stream=None, device="cpu") -> KimiMoE:
    moe = object.__new__(KimiMoE)
    nn.Module.__init__(moe)
    moe.use_latent_moe = True
    moe.gate = _TupleScale(3.0)
    moe.routed_expert_down_proj = _TupleScale(2.0)
    moe.experts = _Experts()
    moe._direct_grouped_topk = False
    moe._router_down_proj_stream = stream
    moe._router_down_proj_events = (
        (torch.cuda.Event(), torch.cuda.Event())
        if device == "cuda"
        else (object(), object())
    )
    return moe


class _FakeMLA(KimiK3MultiHeadLatentAttentionWrapper):
    def __init__(self, stream=None, device="cpu") -> None:
        nn.Module.__init__(self)
        self.g_proj = _TupleScale(3.0)
        self.o_proj = _TupleScale(5.0)
        self._gate_stream = stream
        self._gate_stream_token_threshold = 128
        self._gate_events = (
            (torch.cuda.Event(), torch.cuda.Event())
            if device == "cuda"
            else (object(), object())
        )

    def _forward_attention_frontend(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions, llama_4_scaling
        return hidden_states * 2.0


def test_router_down_projection_overlap_preserves_inputs(monkeypatch) -> None:
    monkeypatch.setattr(amd_linear, "maybe_execute_in_parallel", _parallel)
    stream = object()
    moe = _make_moe(stream)
    hidden_states = torch.randn(8, 16)

    output = moe(hidden_states)

    experts = moe.experts
    assert isinstance(experts, _Experts)
    torch.testing.assert_close(output, hidden_states * 2.0)
    torch.testing.assert_close(experts.routed, hidden_states * 2.0)
    assert experts.shared is not None
    assert experts.shared.data_ptr() == hidden_states.data_ptr()

    packed = experts.router_logits
    assert packed is not None and packed.shape == (2, 8, _TOP_K)
    topk_weights, topk_ids = experts.router.select_experts(experts.routed, packed)
    expected_weights, expected_ids = _expected_routing(hidden_states)
    torch.testing.assert_close(topk_weights, expected_weights)
    assert torch.equal(topk_ids, expected_ids)


def test_router_down_projection_overlap_is_decode_gated(monkeypatch) -> None:
    def fail_parallel(*args, **kwargs):
        raise AssertionError("large batches must not use the auxiliary stream")

    monkeypatch.setattr(amd_linear, "maybe_execute_in_parallel", fail_parallel)
    moe = _make_moe(object())
    hidden_states = torch.randn(129, 16)

    torch.testing.assert_close(moe(hidden_states), hidden_states * 2.0)
    experts = moe.experts
    assert isinstance(experts, _Experts)
    torch.testing.assert_close(experts.router_logits, hidden_states * 3.0)


@pytest.mark.parametrize("indices_dtype", [None, torch.int32, torch.int64])
def test_packed_routing_round_trip(indices_dtype) -> None:
    router = amd_linear._PackedRoutingRouter(_TopKRouter())
    logits = torch.randn(5, 16)
    topk_weights, topk_ids = _TopKRouter().select_experts(logits, logits)

    packed = amd_linear._PackedRoutingRouter.pack(topk_weights, topk_ids)
    got_weights, got_ids = router.select_experts(logits, packed, indices_dtype)

    torch.testing.assert_close(got_weights, topk_weights)
    assert torch.equal(got_ids.long(), topk_ids.long())
    assert got_ids.dtype == (indices_dtype or torch.int32)
    assert got_weights.is_contiguous() and got_ids.is_contiguous()

    raw_weights, raw_ids = router.select_experts(logits, logits, indices_dtype)
    torch.testing.assert_close(raw_weights, topk_weights)
    assert torch.equal(raw_ids.long(), topk_ids.long())


def test_mla_gate_overlap_matches_sequential(monkeypatch) -> None:
    monkeypatch.setattr(amd_mla, "maybe_execute_in_parallel", _parallel)
    hidden_states = torch.randn(8, 16)
    positions = torch.arange(8)
    parallel = _FakeMLA(object())
    sequential = _FakeMLA(None)

    torch.testing.assert_close(
        parallel(positions, hidden_states),
        sequential(positions, hidden_states),
    )


def test_mla_gate_overlap_is_decode_gated(monkeypatch) -> None:
    def fail_parallel(*args, **kwargs):
        raise AssertionError("large batches must not use the auxiliary stream")

    monkeypatch.setattr(amd_mla, "maybe_execute_in_parallel", fail_parallel)
    mla = _FakeMLA(object())
    hidden_states = torch.randn(129, 16)

    output = mla(torch.arange(129), hidden_states)
    expected = hidden_states * 2.0 * (hidden_states * 3.0).sigmoid() * 5.0
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="AITER grouped top-k requires a ROCm GPU",
)
@pytest.mark.parametrize("num_tokens", [1, 8, 16, 24])
def test_aiter_grouped_topk_packed_matches_select_experts(num_tokens) -> None:
    pytest.importorskip("aiter")
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        rocm_aiter_grouped_topk,
    )

    num_experts, top_k = 256, 8
    router = SimpleNamespace(
        top_k=top_k,
        e_score_correction_bias=torch.randn(num_experts, device="cuda"),
        num_expert_group=8,
        topk_group=4,
        renormalize=True,
        routed_scaling_factor=2.5,
        skip_padding=False,
    )
    logits = torch.randn(num_tokens, num_experts, device="cuda")

    packed = amd_linear._aiter_grouped_topk_packed(router, logits)
    topk_weights, topk_ids = amd_linear._PackedRoutingRouter(
        _TopKRouter()
    ).select_experts(logits, packed)
    ref_weights, ref_ids = rocm_aiter_grouped_topk(
        logits,
        logits,
        top_k,
        router.renormalize,
        router.num_expert_group,
        router.topk_group,
        "sigmoid",
        router.routed_scaling_factor,
        router.e_score_correction_bias,
    )

    torch.testing.assert_close(topk_weights, ref_weights)
    assert torch.equal(topk_ids, ref_ids)


@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="HIP graph capture requires a ROCm GPU",
)
@pytest.mark.parametrize("path", ["moe", "mla"])
def test_amd_k3_overlap_hip_graph_observes_input_mutation(path: str) -> None:
    stream = torch.cuda.Stream()
    x = torch.randn(8, 16, device="cuda")
    positions = torch.arange(8, device="cuda")
    if path == "moe":
        module = _make_moe(stream, device="cuda")
        fn = lambda: module._maybe_overlap_router_and_down_proj(x)
    else:
        module = _FakeMLA(stream, device="cuda")
        fn = lambda: module(positions, x)

    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    torch.accelerator.synchronize()

    first = torch.randn_like(x)
    x.copy_(first)
    graph.replay()
    torch.accelerator.synchronize()
    first_output = captured[0].clone() if path == "moe" else captured.clone()
    first_routing = captured[1].clone() if path == "moe" else None

    second = torch.randn_like(x)
    x.copy_(second)
    graph.replay()
    torch.accelerator.synchronize()
    second_output = captured[0].clone() if path == "moe" else captured.clone()
    second_routing = captured[1].clone() if path == "moe" else None

    if path == "moe":
        torch.testing.assert_close(first_output, first * 2.0)
        torch.testing.assert_close(second_output, second * 2.0)
        router = module.experts.router
        for routing, src in ((first_routing, first), (second_routing, second)):
            topk_weights, topk_ids = router.select_experts(src, routing)
            expected_weights, expected_ids = _expected_routing(src)
            torch.testing.assert_close(topk_weights, expected_weights)
            assert torch.equal(topk_ids.cpu(), expected_ids.cpu())
    else:
        torch.testing.assert_close(
            first_output, first * 2.0 * (first * 3.0).sigmoid() * 5.0
        )
        torch.testing.assert_close(
            second_output, second * 2.0 * (second * 3.0).sigmoid() * 5.0
        )
    assert not torch.equal(first_output, second_output)
