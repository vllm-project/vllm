# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's MoE kernels against PyTorch, on weights prepared like
the TRTLLM-Gen FP8 MoE path prepares them."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

EXPERTS, HIDDEN, INTER, TOP_K, LIMIT, SCALE = 288, 4096, 512, 8, 10.0, 2.5
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)


def _quant(x, reciprocal=False, ue8m0=False):
    """FP8 per 128 along the last dim with scale amax / 448; ``reciprocal``
    multiplies by 1 / scale and ``ue8m0`` rounds the scale up to a power of
    two, like per_token_group_quant_fp8."""
    blocks = x.float().unflatten(-1, (-1, 128))
    amax = blocks.abs().amax(-1).clamp_min(1e-10)
    # IEEE division: dividing by a Python scalar multiplies by its reciprocal.
    scale = amax / torch.full_like(amax, 448.0)
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale.clamp_min(1e-10))))
    scaled = (
        blocks * (1 / scale[..., None]) if reciprocal else blocks / scale[..., None]
    )
    return scaled.to(FP8).flatten(-2), scale


def _dequant(q, scale):
    return (q.float().unflatten(-1, (-1, 128)) * scale[..., None]).flatten(-2)


def _weight(w, s):
    """[N, K] FP8 with [N / 128, K / 128] block scales."""
    return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)


def _route(logits, bias):
    scores = logits.float().sigmoid()
    ids = (scores + bias).topk(TOP_K, dim=-1).indices
    weights = scores.gather(1, ids)
    return ids, weights / weights.sum(-1, keepdim=True) * SCALE


def _weights(g):
    """Routed experts ([up; gate] w31 and w2 with FP32 block scales), their
    TRTLLM-Gen copies, and the shared expert in both layouts."""
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_deepseek_fp8_moe_weights,
        swap_w13_to_w31,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.grouped_moe import (
        shared_weights,
    )

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    def scales(*shape):
        return torch.rand(*shape, generator=g, device="cuda") * 0.02

    w13 = (randn(EXPERTS, 2 * INTER, HIDDEN) * 0.5).to(FP8)  # [gate; up]
    w13_s = scales(EXPERTS, 2 * INTER // 128, HIDDEN // 128)
    w2 = (randn(EXPERTS, HIDDEN, INTER) * 0.5).to(FP8)
    w2_s = scales(EXPERTS, HIDDEN // 128, INTER // 128)
    se13 = (randn(2 * INTER, HIDDEN) * 0.5).to(FP8)
    se13_s = scales(2 * INTER // 128, HIDDEN // 128)
    se2 = (randn(HIDDEN, INTER) * 0.5).to(FP8)
    se2_s = scales(HIDDEN // 128, INTER // 128)
    w31, w31_s = swap_w13_to_w31(w13), swap_w13_to_w31(w13_s)
    w31_trtllm, w2_trtllm = _shuffle_deepseek_fp8_moe_weights(w31, w2)
    return {
        "routed": (w31, w31_s, w2, w2_s),
        "trtllm": (w31_trtllm, w31_s, w2_trtllm, w2_s),
        "shared": (swap_w13_to_w31(se13), swap_w13_to_w31(se13_s), se2, se2_s),
        "shared_trtllm": shared_weights(se13, se13_s, se2, se2_s),
    }


@pytest.mark.parametrize(
    ("grouped", "tokens"),
    [(False, 1), (False, 4), (True, 8), (True, 64)],
)
@pytest.mark.parametrize("ue8m0", [False, True])
def test_mega_moe_matches_reference(grouped, tokens, ue8m0):
    """Routing, activation quant and both expert GEMMs against PyTorch."""
    from vllm.models.glm5next.nvidia.specialized_decode.grouped_moe import (
        GroupedExperts,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.mega_moe import MegaExperts

    g = torch.Generator(device="cuda").manual_seed(tokens)
    w = _weights(g)
    args = (*w["trtllm"], TOP_K, LIMIT, SCALE, ue8m0)
    if grouped:
        # The shared expert runs as one more expert of every token.
        mega = GroupedExperts(*args, shared=w["shared_trtllm"])
    else:
        mega = MegaExperts(*args)

    x = torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16()
    logits = torch.randn(tokens, EXPERTS, generator=g, device="cuda")
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
    # The grouped kernel's workspace counters alternate between two sets.
    for _ in range(3 if grouped else 1):
        out, ids, weights, *shared_out = mega(x, logits, bias)
    if grouped:
        # In-kernel finalize: FP32 sum of the weighted routes, plus the shared
        # expert, rounded once.
        final = mega(x, logits, bias, finalize=True)
        rows = out.float().view(tokens, TOP_K, HIDDEN) * weights[..., None]
        expected = rows.sum(1) + shared_out[0].float()
        torch.testing.assert_close(final.float(), expected, atol=1e-6, rtol=8e-3)

    ref_ids, ref_weights = _route(logits, bias)
    order, ref_order = ids.long().sort(-1), ref_ids.sort(-1)
    assert torch.equal(order.values, ref_order.values)
    # The kernel's sigmoid is a few ulp off PyTorch's.
    torch.testing.assert_close(
        weights.gather(1, order.indices),
        ref_weights.gather(1, ref_order.indices),
        atol=0,
        rtol=1e-5,
    )

    x = _dequant(*_quant(x, reciprocal=True, ue8m0=ue8m0))

    def expert(w31, w31_s, w2, w2_s, token: int):
        up, gate = (_weight(w31, w31_s) @ x[token]).split(INTER)
        act = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
        return _weight(w2, w2_s) @ _dequant(*_quant(act))

    w31, w31_s, w2, w2_s = w["routed"]
    ref = torch.stack(
        [
            expert(w31[e], w31_s[e], w2[e], w2_s[e], t)
            for t in range(tokens)
            for e in ids[t].tolist()
        ]
    )
    # BF16 outputs; rare FP8 rounding flips of the hidden move a few elements.
    torch.testing.assert_close(out.float(), ref, atol=5e-3 * ref.abs().max(), rtol=0)
    if grouped:
        se31, se31_s, se2, se2_s = w["shared"]
        ref = torch.stack([expert(se31, se31_s, se2, se2_s, t) for t in range(tokens)])
        torch.testing.assert_close(
            shared_out[0].float(), ref, atol=5e-3 * ref.abs().max(), rtol=0
        )


def _errors(value, ref):
    """Median, p99 and max absolute error."""
    err = (value.double() - ref).abs().flatten()
    return [err.median().item(), err.quantile(0.99).item(), err.max().item()]


def _mlp(w31, w31_s, w2, w2_s, x, quant_hidden):
    """FP64 expert MLP of FP8 (UE8M0) tokens; ``quant_hidden`` rounds the hidden
    to FP8 per 128 like TRTLLM-Gen does."""
    x = _dequant(*_quant(x, ue8m0=True)).double()
    up, gate = (x @ _weight(w31, w31_s).double().t()).split(INTER, dim=-1)
    act = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
    if quant_hidden:
        act = _dequant(*_quant(act)).double()
    return act @ _weight(w2, w2_s).double().t()


@pytest.mark.parametrize("tokens", [8, 64])
def test_grouped_moe_error_within_stock(tokens):
    """The grouped kernel's error against FP64 is no larger (median, p99, max)
    than the stock path's. Routed experts: TRTLLM-Gen, both finalized in FP32
    like the fused boundary, against FP64 with the FP8 tokens and hidden they
    both round to. Shared expert: vLLM's FP8 GEMMs (BF16 gate/up, BF16
    activation, UE8M0 hidden), against FP64 with FP8 tokens."""
    from flashinfer.fused_moe import (
        Fp8QuantizationType,
        WeightLayout,
        trtllm_fp8_block_scale_moe,
    )
    from flashinfer.fused_moe.core import RoutingMethodType

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.models.glm5next.nvidia.specialized_decode.grouped_moe import (
        GroupedExperts,
    )

    g = torch.Generator(device="cuda").manual_seed(100 + tokens)
    w = _weights(g)
    x = torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16()
    logits = torch.randn(tokens, EXPERTS, generator=g, device="cuda")
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1

    grouped = GroupedExperts(
        *w["trtllm"], TOP_K, LIMIT, SCALE, True, shared=w["shared_trtllm"]
    )
    rows, _, weights, shared = grouped(x, logits, bias)
    ours = (rows.float().view(tokens, TOP_K, HIDDEN) * weights[..., None]).sum(1)

    q, q_scale = per_token_group_quant_fp8(
        x, 128, column_major_scales=False, use_ue8m0=True
    )
    w31, w31_s, w2, w2_s = w["trtllm"]
    stock_rows, stock_weights, permute = trtllm_fp8_block_scale_moe(
        routing_logits=logits, routing_bias=bias, hidden_states=q,
        hidden_states_scale=q_scale.t().contiguous(), gemm1_weights=w31,
        gemm1_weights_scale=w31_s, gemm2_weights=w2, gemm2_weights_scale=w2_s,
        num_experts=EXPERTS, top_k=TOP_K, n_group=1, topk_group=1,
        intermediate_size=INTER, local_expert_offset=0, local_num_experts=EXPERTS,
        routed_scaling_factor=SCALE,
        routing_method_type=RoutingMethodType.DeepSeekV3, use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        gemm1_clamp_limit=torch.full((EXPERTS,), LIMIT, device="cuda"),
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8, do_finalize=False,
    )  # fmt: skip
    rows = stock_rows.float()[permute.long().view(tokens, TOP_K)]
    stock = (rows * stock_weights.float().view(tokens, TOP_K, 1)).sum(1)

    ids, ref_weights = _route(logits, bias)
    ref = torch.zeros(tokens, HIDDEN, dtype=torch.float64, device="cuda")
    w31, w31_s, w2, w2_s = w["routed"]
    for t in range(tokens):
        for k, e in enumerate(ids[t].tolist()):
            row = _mlp(w31[e], w31_s[e], w2[e], w2_s[e], x[t : t + 1], True)[0]
            ref[t] += ref_weights[t, k].double() * row
    ours_error, stock_error = _errors(ours, ref), _errors(stock, ref)
    assert all(o <= s for o, s in zip(ours_error, stock_error)), (
        ours_error,
        stock_error,
    )

    w31, w31_s, w2, w2_s = w["shared"]
    ref = _mlp(w31, w31_s, w2, w2_s, x, False)

    def gemm(value, weight, scale):
        value = _dequant(*_quant(value, ue8m0=True))
        return (value @ _weight(weight, scale).t()).bfloat16()

    up, gate = gemm(x, w31, w31_s).float().split(INTER, dim=-1)
    act = (F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)).bfloat16()
    stock_shared = gemm(act, w2, w2_s)
    ours_error, stock_error = _errors(shared, ref), _errors(stock_shared, ref)
    assert all(o <= s for o, s in zip(ours_error, stock_error)), (
        ours_error,
        stock_error,
    )
