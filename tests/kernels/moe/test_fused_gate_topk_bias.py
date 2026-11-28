# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE gate GEMM + sigmoid top-k with a score-correction bias."""

import pytest
import torch

from tests.single_rank_group import SingleRankGroup, override_parallel_groups
from vllm.model_executor.layers.fused_moe.router.fused_gate_topk_bias import (
    fused_gate_sigmoid_topk_bias,
    fused_gate_sigmoid_topk_bias_ref,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(num_tokens, hidden, num_experts, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, hidden, device="cuda", generator=g).to(torch.bfloat16)
    w = torch.randn(num_experts, hidden, device="cuda", generator=g) * hidden**-0.5
    bias = torch.randn(num_experts, device="cuda", generator=g) * 0.1
    return x, w, bias


def _check(got, ref, scores_gap):
    got_w, got_ids = got
    ref_w, ref_ids = ref
    assert got_w.dtype == torch.float32 and got_ids.dtype == torch.int32
    # Only the GEMM's summation order differs, so an id may differ only where
    # the k-th and (k+1)-th biased scores (nearly) tie.
    same = (got_ids.sort(-1).values == ref_ids.sort(-1).values).all(-1)
    assert (same | (scores_gap < 1e-5)).all()
    rows = same.nonzero().squeeze(1)
    order = got_ids[rows].argsort(-1)
    ref_order = ref_ids[rows].argsort(-1)
    torch.testing.assert_close(
        got_w[rows].gather(1, order),
        ref_w[rows].gather(1, ref_order),
        rtol=1e-5,
        atol=1e-6,
    )


def _kth_gap(x, w, bias, topk):
    choice = torch.nn.functional.linear(x.float(), w).sigmoid() + bias
    top = choice.topk(topk + 1, dim=-1).values
    return top[:, topk - 1] - top[:, topk]


@pytest.mark.parametrize("num_tokens", [1, 16, 128, 333, 1024])
@pytest.mark.parametrize("hidden,num_experts,topk", [(6144, 128, 4), (2048, 64, 8)])
@pytest.mark.parametrize("renormalize,scale", [(True, 1.0), (False, 2.5)])
def test_matches_unfused(num_tokens, hidden, num_experts, topk, renormalize, scale):
    x, w, bias = _inputs(num_tokens, hidden, num_experts, seed=num_tokens)
    args = (x, w, bias, topk, renormalize, scale)
    _check(
        fused_gate_sigmoid_topk_bias(*args),
        fused_gate_sigmoid_topk_bias_ref(*args),
        _kth_gap(x, w, bias, topk),
    )


def test_ties_go_to_the_lower_expert():
    x, w, bias = _inputs(4, 256, 16)
    w[8:] = w[:8]  # experts 8..15 duplicate 0..7
    bias[8:] = bias[:8]
    # Top-3 of tied pairs (e, e + 8): the first pair whole, lower id first,
    # then only the lower expert of the second pair.
    _, ids = fused_gate_sigmoid_topk_bias(x, w, bias, 3, True)
    assert (ids[:, 0] < 8).all() and (ids[:, 1] == ids[:, 0] + 8).all()
    assert (ids[:, 2] < 8).all()


def test_padding_rows_are_skipped():
    x, w, bias = _inputs(64, 512, 32)
    is_padding = torch.zeros(80, dtype=torch.bool, device="cuda")
    is_padding[5::7] = True  # a mask longer than the batch is cut to it
    weights, ids = fused_gate_sigmoid_topk_bias(x, w, bias, 4, True, 1.0, is_padding)
    pad = is_padding[:64]
    assert (ids[pad] == -1).all() and (weights[pad] == 0).all()
    assert (ids[~pad] >= 0).all()


def test_empty_batch():
    x, w, bias = _inputs(0, 512, 32)
    weights, ids = fused_gate_sigmoid_topk_bias(x, w, bias, 4, True)
    assert weights.shape == (0, 4) and ids.shape == (0, 4)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="VLLM_ROCM_FUSED_MOE_GATE_TOPK is ROCm-only"
)
@pytest.mark.parametrize("num_tokens", [8, 96])
@override_parallel_groups(SingleRankGroup())
def test_runner_fused_gate_matches_gate_and_router(
    num_tokens, monkeypatch, workspace_init
):
    """A runner holding an fp32 gate gives the same output with the fused gate
    + top-k as with its gate and router."""
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    monkeypatch.setenv("VLLM_ROCM_FUSED_MOE_GATE_TOPK", "1")
    calls = []

    def spy(*args, **kwargs):
        calls.append(fused_gate_sigmoid_topk_bias(*args, **kwargs))
        return calls[-1]

    monkeypatch.setattr(moe_runner, "fused_gate_sigmoid_topk_bias", spy)
    set_random_seed(0)
    torch.set_default_device("cuda")
    hidden, num_experts = 256, 16
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    with set_current_vllm_config(vllm_config):
        gate = GateLinear(
            hidden,
            num_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix="moe.gate",
        )
        bias = torch.nn.Parameter(torch.randn(num_experts) * 0.1)
        moe = FusedMoEFactory(
            gate=gate,
            num_experts=num_experts,
            top_k=4,
            hidden_size=hidden,
            intermediate_size=128,
            scoring_func="sigmoid",
            e_score_correction_bias=bias,
            renormalize=True,
            routed_scaling_factor=2.0,
            params_dtype=torch.bfloat16,
            tp_size=1,
            dp_size=1,
            pcp_size=1,
            prefix="moe",
        )
        with torch.no_grad():
            gate.weight.normal_().mul_(hidden**-0.5)
            moe.routed_experts.w13_weight.normal_().div_(10)
            moe.routed_experts.w2_weight.normal_().div_(10)
        moe.routed_experts.quant_method.process_weights_after_loading(
            moe.routed_experts
        )
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
        with set_forward_context(None, vllm_config, num_tokens=num_tokens):
            moe._fused_gate_topk = False  # the gate and the router
            expected = moe(x, x)
            ref_weights, ref_ids = moe.router.select_experts(x, gate(x)[0], torch.int32)
            moe._fused_gate_topk = None  # decided again: fused
            actual = moe(x, x)
    assert len(calls) == 1
    weights, ids = calls[0]
    # Same experts, in the same order, with the same weights as the router.
    assert torch.equal(ids, ref_ids.to(ids.dtype))
    torch.testing.assert_close(weights, ref_weights, rtol=1e-5, atol=1e-6)
    # AITER's fused MoE accumulates atomically, so its output varies by a BF16
    # ulp from run to run.
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
