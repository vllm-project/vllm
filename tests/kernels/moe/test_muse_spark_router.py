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
