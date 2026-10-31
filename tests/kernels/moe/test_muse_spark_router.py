# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Numeric unit tests for MuseSpark1xRouter and its fused Triton kernel."""

import pytest
import torch

from vllm.models.muse_spark.common.ops.fused_routing import fused_musespark1x_routing
from vllm.models.muse_spark.nvidia.muse_moe_runner import MuseSpark1xRouter

LOGITS = torch.tensor(
    [
        [1.0, 2.0, 0.5, -1.0],
        [0.25, -0.5, 1.5, 0.75],
    ],
    dtype=torch.float32,
)
HIDDEN = torch.zeros(2, 4, dtype=torch.float32)


def _expected(
    logits,
    bias,
    top_k,
    route_scale,
    multiplier=None,
    normalization="sigmoid",
    division_eps=None,
):
    g = logits if multiplier is None else logits * multiplier
    g = torch.sigmoid(g) if normalization == "sigmoid" else torch.softmax(g, dim=-1)
    ids = torch.topk(g + bias, top_k, dim=-1).indices
    w = torch.gather(g, 1, ids)
    denom = w.sum(dim=-1, keepdim=True)
    if division_eps is not None:
        denom = denom + division_eps
    return w / denom * route_scale, ids


def _route(
    top_k=2,
    bias=None,
    route_scale=1.0,
    multiplier=None,
    normalization="sigmoid",
    division_eps=None,
    indices_type=None,
):
    router = MuseSpark1xRouter(
        top_k=top_k,
        num_experts=LOGITS.shape[-1],
        expert_bias=(bias if bias is not None else torch.zeros(LOGITS.shape[-1])),
        route_scale=route_scale,
        router_output_multiplier=multiplier,
        router_score_normalization=normalization,
        router_score_division_eps=division_eps,
    )
    return router._compute_routing(HIDDEN, LOGITS, indices_type)


def test_bias_affects_selection_but_not_gathered_weights():
    bias = torch.tensor([0.0, 0.0, 5.0, 0.0])
    weights, ids = _route(bias=bias, division_eps=1e-5, route_scale=2.0)
    # Unbiased sigmoid top-2 for row 0 would be experts (1, 0); the +5 bias
    # pulls expert 2 into the selection instead.
    assert ids[0].tolist() == [2, 1]
    exp_w, exp_ids = _expected(
        LOGITS, bias, top_k=2, route_scale=2.0, division_eps=1e-5
    )
    torch.testing.assert_close(weights, exp_w)
    assert (ids == exp_ids).all()
    # Had the bias leaked into the gathered scores, expert 2's weight would be
    # ~5.6 / denom, not ~0.62 / denom.
    sig = torch.sigmoid(LOGITS[0, 2])
    denom = sig + torch.sigmoid(LOGITS[0, 1]) + 1e-5
    assert abs(weights[0, 0].item() - (sig / denom * 2.0).item()) < 1e-6


def test_softmax_branch_without_division_eps():
    bias = torch.tensor([0.1, -0.2, 0.0, 0.3])
    weights, ids = _route(bias=bias, normalization="softmax", route_scale=1.5)
    exp_w, exp_ids = _expected(
        LOGITS, bias, top_k=2, route_scale=1.5, normalization="softmax"
    )
    torch.testing.assert_close(weights, exp_w)
    assert (ids == exp_ids).all()


def test_output_multiplier_applied_before_sigmoid():
    weights, ids = _route(multiplier=0.5, division_eps=0.0)
    exp_w, exp_ids = _expected(
        LOGITS,
        torch.zeros(4),
        top_k=2,
        route_scale=1.0,
        multiplier=0.5,
        division_eps=0.0,
    )
    torch.testing.assert_close(weights, exp_w)
    assert (ids == exp_ids).all()
    # Applied AFTER sigmoid instead, the weights would differ (sigmoid is
    # nonlinear): sigmoid(0.5 * x) != 0.5 * sigmoid(x) in general.
    post = torch.sigmoid(LOGITS) * 0.5
    post_w = torch.gather(post, 1, ids)
    post_w = post_w / post_w.sum(dim=-1, keepdim=True)
    assert not torch.allclose(weights, post_w)


def test_output_dtypes():
    weights, ids = _route()
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    _, ids64 = _route(indices_type=torch.int64)
    assert ids64.dtype == torch.int64


def test_invalid_normalization_rejected():
    with pytest.raises(ValueError, match="router_score_normalization"):
        MuseSpark1xRouter(
            top_k=2,
            num_experts=4,
            expert_bias=torch.zeros(4),
            route_scale=1.0,
            router_score_normalization="tanh",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "normalization,multiplier,division_eps",
    [("sigmoid", 2.5, 1e-9), ("softmax", None, None)],
)
@pytest.mark.parametrize(
    "num_tokens,num_experts,top_k", [(1, 32, 8), (777, 384, 6), (2049, 256, 8)]
)
def test_fused_routing_matches_reference(
    normalization, num_tokens, num_experts, top_k, multiplier, division_eps
):
    """Kernel and ``_compute_routing`` pick identical experts; weights to fp32 ulp."""
    torch.manual_seed(0)
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    bias = torch.randn(num_experts, device="cuda")
    router = MuseSpark1xRouter(
        top_k=top_k,
        num_experts=num_experts,
        expert_bias=bias,
        route_scale=2.0,
        router_output_multiplier=multiplier,
        router_score_normalization=normalization,
        router_score_division_eps=division_eps,
    )

    weights, ids = fused_musespark1x_routing(
        logits,
        bias,
        top_k,
        2.0,
        router_output_multiplier=multiplier,
        router_score_normalization=normalization,
        router_score_division_eps=division_eps,
    )
    exp_w, exp_ids = _expected(
        logits, bias, top_k, 2.0, multiplier, normalization, division_eps
    )

    assert torch.equal(ids, exp_ids.to(torch.int32))
    torch.testing.assert_close(weights, exp_w, rtol=1e-5, atol=1e-6)
    routed_w, routed_ids = router._compute_routing(None, logits, None)
    torch.testing.assert_close(routed_w, weights)
    assert torch.equal(routed_ids, ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_routing_writes_into_preallocated_buffers():
    """CUDA-graph capture needs the call to allocate nothing."""
    torch.manual_seed(0)
    num_tokens, num_experts, top_k = 16, 32, 8
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    bias = torch.randn(num_experts, device="cuda")
    # Buffers sized for a larger batch: only the leading rows may be touched.
    weights = torch.full((64, top_k), -7.0, device="cuda")
    ids = torch.full((64, top_k), -7, device="cuda", dtype=torch.int32)

    w_view, id_view = fused_musespark1x_routing(
        logits, bias, top_k, 1.0, topk_weights=weights, topk_ids=ids
    )
    assert w_view.data_ptr() == weights.data_ptr()
    assert id_view.data_ptr() == ids.data_ptr()
    assert (weights[num_tokens:] == -7.0).all()
    assert (ids[num_tokens:] == -7).all()


def _triton_routing_reference(logits, bias, multiplier, scale, eps, padding):
    from vllm.models.muse_spark.common.ops.fused_routing import _routing_kernel
    from vllm.platforms import current_platform

    tokens = logits.shape[0]
    weights = torch.empty((tokens, 8), device=logits.device)
    ids = torch.empty((tokens, 8), device=logits.device, dtype=torch.int32)
    if tokens:
        pdl = current_platform.is_arch_support_pdl()
        _routing_kernel[(tokens,)](
            logits,
            bias,
            weights,
            ids,
            padding,
            0 if padding is None else padding.stride(0),
            tokens,
            256,
            *logits.stride(),
            *weights.stride(),
            *ids.stride(),
            multiplier,
            scale,
            eps,
            TOPK=8,
            TOPK_PADDED=8,
            NUM_EXPERTS_PADDED=256,
            NORMALIZATION_MODE=0,
            HAS_MULTIPLIER=True,
            HAS_EPS=True,
            USE_PDL=pdl,
            num_warps=1,
            **({"launch_pdl": True} if pdl else {}),
        )
    return weights, ids


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires the CUDA routing kernel",
)
@pytest.mark.parametrize("tokens", [0, 1, 12, 16, 48, 128, 2048, 2049, 16384])
@pytest.mark.parametrize("case", ["random", "ties", "near_ties", "negative", "strided"])
def test_cuda_muse_routing_matches_triton(tokens, case):
    """Preserve expert choices, FP32 rounding, padding, and caller-owned buffers."""
    from vllm import _custom_ops as ops

    torch.manual_seed(17)
    width = 260 if case == "strided" else 256
    logits = torch.randn(tokens, width, device="cuda")[:, :256]
    bias = torch.randn(256, device="cuda") * 0.2
    if case == "ties":
        logits.zero_()
        bias.zero_()
    elif case == "near_ties":
        logits.mul_(1e-6)
        bias.zero_()
    elif case == "negative":
        logits.sub_(80)
    padding = (torch.arange(tokens * 2, device="cuda") % 3 == 0)[::2]
    weights = torch.full((tokens + 3, 12), -7.0, device="cuda")
    ids = torch.full((tokens + 3, 12), -7, device="cuda", dtype=torch.int32)
    w, i = weights[:tokens, :8], ids[:tokens, :8]
    ref_w, ref_i = _triton_routing_reference(logits, bias, 0.5, 2.5, 1e-15, padding)
    ops.muse_spark_routing(w, i, logits, bias, 0.5, 2.5, 1e-15, padding)
    assert torch.equal(i, ref_i)
    assert torch.equal(w, ref_w)
    assert (i[padding] == -1).all()
    assert (w[padding] == 0).all()
    out_w, out_i = fused_musespark1x_routing(
        logits,
        bias,
        8,
        2.5,
        router_output_multiplier=0.5,
        router_score_division_eps=1e-15,
        topk_weights=weights,
        topk_ids=ids,
        is_padding=padding,
    )
    assert out_w.untyped_storage().data_ptr() == weights.untyped_storage().data_ptr()
    assert out_i.untyped_storage().data_ptr() == ids.untyped_storage().data_ptr()
    assert torch.equal(out_w, ref_w) and torch.equal(out_i, ref_i)
    assert (weights[tokens:] == -7).all() and (weights[:, 8:] == -7).all()
    assert (ids[tokens:] == -7).all() and (ids[:, 8:] == -7).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires the CUDA routing kernel",
)
def test_cuda_muse_routing_graph_replay():
    """Replays must consume updated logits, bias, and arbitrary padding masks."""
    torch.manual_seed(19)
    logits = torch.randn(48, 256, device="cuda")
    bias = torch.randn(256, device="cuda")
    padding = torch.zeros(48, device="cuda", dtype=torch.bool)
    weights = torch.empty(48, 8, device="cuda")
    ids = torch.empty(48, 8, device="cuda", dtype=torch.int32)

    def route():
        return fused_musespark1x_routing(
            logits,
            bias,
            8,
            1.0,
            router_score_division_eps=1e-15,
            topk_weights=weights,
            topk_ids=ids,
            is_padding=padding,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            route()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        route()
    for offset in range(3):
        logits.normal_()
        bias.normal_()
        padding.copy_(torch.arange(48, device="cuda") % 3 == offset)
        graph.replay()
        ref_w, ref_i = _triton_routing_reference(logits, bias, 1.0, 1.0, 1e-15, padding)
        assert torch.equal(ids, ref_i)
        assert torch.equal(weights, ref_w)
