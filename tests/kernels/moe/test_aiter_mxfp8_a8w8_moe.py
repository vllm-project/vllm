# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the gfx942 AITER MXFP8 a8w8 MoE backend.

Compares `AiterMxfp8A8W8ExpertsMonolithic` (weights converted by
`convert_to_fp8_moe_kernel_format`) with a BF16 reference built from the same
MXFP8 weights, and checks the e4m3fn -> e4m3fnuz weight conversion bit-exactly.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only.", allow_module_level=True)

from vllm.platforms.rocm import on_gfx942  # noqa: E402

if not on_gfx942():
    pytest.skip("AITER MXFP8 a8w8 MoE is gfx942 only.", allow_module_level=True)

from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_a8w8_moe import (  # noqa: E402
    AiterMxfp8A8W8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (  # noqa: E402
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # noqa: E402
    grouped_topk,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
    mxfp8_e4m3fn_to_e4m3fnuz,
)
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

DEVICE = "cuda"
NUM_EXPERTS = 16
TOPK = 4
HIDDEN = 512
INTER = 384  # w13 is [E, 768, 512], w2 is [E, 512, 384]: non-square on purpose
SCALE = 2.5  # routed_scaling_factor for the sigmoid methods


def _quantize_weight(e: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.randn(e, n, k, device=DEVICE, dtype=torch.bfloat16) / k**0.5
    w_q, w_s = mxfp8_e4m3_quantize(w.view(-1, k))
    return w_q.view(e, n, k), w_s.view(torch.uint8).view(e, n, k // 32)


def _routing_kwargs(routing_method, bias):
    if routing_method == RoutingMethodType.DeepSeekV3:
        return dict(
            num_expert_group=4,
            topk_group=2,
            e_score_correction_bias=bias,
            routed_scaling_factor=SCALE,
        )
    if routing_method == RoutingMethodType.MiniMax2:
        return dict(e_score_correction_bias=bias, routed_scaling_factor=SCALE)
    return {}


def _ref_route(logits, routing_method, bias):
    if routing_method == RoutingMethodType.DeepSeekV3:
        topk_w, topk_ids = grouped_topk(
            logits,
            logits.float(),
            TOPK,
            True,
            num_expert_group=4,
            topk_group=2,
            scoring_func="sigmoid",
            routed_scaling_factor=SCALE,
            e_score_correction_bias=bias,
        )
        return topk_w.float(), topk_ids.long()
    if routing_method == RoutingMethodType.MiniMax2:
        # Select with the bias, weight with the unbiased sigmoid scores.
        s = logits.float().sigmoid()
        _, topk_ids = (s + bias).topk(TOPK, dim=-1)
        topk_w = s.gather(-1, topk_ids)
        return topk_w / topk_w.sum(-1, keepdim=True) * SCALE, topk_ids
    topk_w, topk_ids = logits.float().softmax(-1).topk(TOPK, dim=-1)
    return topk_w / topk_w.sum(-1, keepdim=True), topk_ids


def _qdq(v: torch.Tensor) -> torch.Tensor:
    """MXFP8 round trip, as the kernel does to both GEMM inputs."""
    v_q, v_s = mxfp8_e4m3_quantize(v.to(torch.bfloat16))
    return dequant_mxfp8_to_bf16(v_q, v_s.view(torch.uint8)).float()


def _ref_moe(x, w13, w2, topk_w, topk_ids, on_input, act_fn):
    out = torch.zeros(x.shape[0], HIDDEN, device=DEVICE, dtype=torch.float32)
    for j in range(TOPK):
        e, g = topk_ids[:, j], topk_w[:, j : j + 1]
        # The kernel quantizes x, then applies g after the (linear) first GEMM.
        xi = _qdq(x.float()) * (g if on_input else 1.0)
        gate_up = torch.bmm(w13[e].float(), xi.unsqueeze(-1)).squeeze(-1)
        h = _qdq(act_fn(gate_up))
        y = torch.bmm(w2[e].float(), h.unsqueeze(-1)).squeeze(-1)
        out += y if on_input else g * y
    return out


@pytest.mark.parametrize(
    "routing_method,activation,apply_on_input",
    [
        (RoutingMethodType.Renormalize, MoEActivation.SILU, False),
        (RoutingMethodType.Renormalize, MoEActivation.SILU, True),
        (RoutingMethodType.RenormalizeNaive, MoEActivation.SILU, False),
        (RoutingMethodType.RenormalizeNaive, MoEActivation.SILU, True),
        (RoutingMethodType.MiniMax2, MoEActivation.SILU, False),
        (RoutingMethodType.DeepSeekV3, MoEActivation.SILU, False),
        (RoutingMethodType.Renormalize, MoEActivation.SWIGLUOAI, False),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 37, 300])
def test_aiter_mxfp8_a8w8_moe_matches_reference(
    routing_method, activation, apply_on_input, num_tokens
):
    set_random_seed(0)
    w13_q, w13_s = _quantize_weight(NUM_EXPERTS, 2 * INTER, HIDDEN)
    w2_q, w2_s = _quantize_weight(NUM_EXPERTS, HIDDEN, INTER)
    # The kernel conversion mutates weights in place; keep a reference copy.
    w13_ref = dequant_mxfp8_to_bf16(w13_q.clone(), w13_s)
    w2_ref = dequant_mxfp8_to_bf16(w2_q.clone(), w2_s)
    w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
        Fp8MoeBackend.AITER_MXFP8_A8W8,
        torch.nn.Module(),
        w13_q,
        w2_q,
        w13_s,
        w2_s,
        None,
        None,
    )

    moe_config = FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        intermediate_size=INTER,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        activation=activation,
        device=DEVICE,
        routing_method=routing_method,
        max_num_tokens=512,
    )
    quant_config = FusedMoEQuantConfig.make(
        "mxfp8",
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[1, 32],
        is_scale_swizzled=False,
    )
    experts = AiterMxfp8A8W8ExpertsMonolithic(moe_config, quant_config)

    x = torch.randn(num_tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(NUM_EXPERTS, device=DEVICE) * 0.1
    replay = torch.full((512, TOPK), -1, device=DEVICE, dtype=torch.int16)

    out = experts.apply(
        hidden_states=x,
        w1=w13,
        w2=w2,
        router_logits=logits,
        activation=activation,
        global_num_experts=NUM_EXPERTS,
        expert_map=None,
        a1q_scale=None,
        apply_router_weight_on_input=apply_on_input,
        routing_replay_out=replay,
        **_routing_kwargs(routing_method, bias),
    )

    def act_fn(gate_up):
        h = gate_up.new_empty(gate_up.shape[0], gate_up.shape[1] // 2)
        apply_moe_activation(
            activation, h, gate_up, activation_config=experts.activation_config
        )
        return h

    topk_w, topk_ids = _ref_route(logits, routing_method, bias)
    ref = _ref_moe(x, w13_ref, w2_ref, topk_w, topk_ids, apply_on_input, act_fn)
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 0.03, f"relative error {rel:.4f}"
    # Routed-experts capture reports the experts that were used.
    torch.testing.assert_close(
        replay[:num_tokens].long().sort(-1).values, topk_ids.sort(-1).values
    )


def test_e4m3fn_to_e4m3fnuz_keeps_values():
    set_random_seed(0)
    bits = torch.randint(-128, 128, (4, 64, 64), dtype=torch.int8, device=DEVICE)
    bits[(bits & 0x7F) == 0x7F] = 0  # drop e4m3fn NaN encodings
    bits[0, 0, :8] = -128  # 0x80 is -0 in fn, NaN in fnuz
    w = bits.view(torch.float8_e4m3fn)
    scale = torch.randint(100, 150, (4, 64, 2), dtype=torch.uint8, device=DEVICE)
    scale[0, 0, 0], scale[0, 0, 1], scale[1, 0, 0] = 253, 254, 255

    ref = dequant_mxfp8_to_bf16(w.clone(), scale).float()
    w_fnuz, s_fnuz = mxfp8_e4m3fn_to_e4m3fnuz(w.clone(), scale)
    assert w_fnuz.dtype == torch.float8_e4m3fnuz
    assert not (w_fnuz.view(torch.int8) == -128).any()

    got = (
        w_fnuz.float().view(4, 64, 2, 32)
        * torch.exp2(s_fnuz.float() - 127).unsqueeze(-1)
    ).view(4, 64, 64)
    blk = (s_fnuz != 255).repeat_interleave(32, dim=-1)
    torch.testing.assert_close(got[blk], ref[blk], atol=0, rtol=0)
    # 253 -> 254 stays finite; 254 and 255 cannot be bumped and become NaN.
    assert s_fnuz[0, 0, 0] == 254
    assert s_fnuz[0, 0, 1] == 255 and s_fnuz[1, 0, 0] == 255
