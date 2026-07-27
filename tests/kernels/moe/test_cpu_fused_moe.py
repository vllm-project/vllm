# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm._custom_ops import (
    cpu_fused_moe,
    cpu_fused_moe_int8,
    cpu_prepack_moe_weight,
    cpu_prepack_moe_weight_int8,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
    _CPU_MOE_ACT_FN,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

EXPERT_NUM = [
    8,
]
HIDDEN_DIM = [128, 2880]
INTERMEDIATE_DIM = [128, 2880]
BATCH_SIZE = [1, 64, 256]
ACT = [
    MoEActivation.SILU,
    MoEActivation.SWIGLUOAI,
    MoEActivation.GELU,
    MoEActivation.GELU_TANH,
]
USE_BIAS = [False, True]
ISA = ["vec"]
if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
    ISA.append("neon")
if torch.cpu._is_amx_tile_supported():
    ISA.append("amx")

DTYPE = [torch.bfloat16]


def ref_fused_moe(
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
) -> torch.Tensor:
    len_experts = w13.size(0)

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()

    sorted_tokens = input[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0

    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx].float()
        curr_w13 = w13[i].float()
        curr_w2 = w2[i].float()

        curr_w13_bias = None
        if w13_bias is not None:
            curr_w13_bias = w13_bias[i].float()

        curr_w2_bias = None
        if w2_bias is not None:
            curr_w2_bias = w2_bias[i].float()

        gate_up = torch.nn.functional.linear(
            tokens_for_this_expert, curr_w13, curr_w13_bias
        )
        # Note: to simulate the kernel implementation
        gate_up = _CPU_MOE_ACT_FN[activation](gate_up).to(dtype=input.dtype).float()
        expert_out = torch.nn.functional.linear(gate_up, curr_w2, curr_w2_bias)

        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)

    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .mul_(topk_weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(input.dtype)
    )
    return final_out


def quantize_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize each weight output channel."""
    weight_fp32 = weight.float()
    scale = weight_fp32.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    quantized = (
        (weight_fp32 / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    )
    return quantized, scale


def quantize_per_token(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize each input token."""
    input_fp32 = input.float()
    scale = input_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-7) / 127.0
    quantized = (input_fp32 / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def ref_fused_moe_int8(
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
) -> torch.Tensor:
    """Reference the two dynamically quantized INT8 GEMMs."""
    input_int8, input_scale = quantize_per_token(input)
    expert_num = w13.size(0)

    counts = topk_ids.new_zeros((topk_ids.size(0), expert_num))
    counts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = counts.sum(dim=0).tolist()
    sorted_route_ids = topk_ids.view(-1).argsort()
    sorted_token_ids = sorted_route_ids // topk_ids.size(1)

    outputs = []
    start_idx = 0
    for expert_idx, token_count in enumerate(tokens_per_expert):
        end_idx = start_idx + token_count
        if token_count == 0:
            continue

        token_ids = sorted_token_ids[start_idx:end_idx]
        gate_up = torch.matmul(
            input_int8[token_ids].float(),
            w13[expert_idx].float().T,
        )
        gate_up *= input_scale[token_ids] * w13_scale[expert_idx]
        if w13_bias is not None:
            gate_up += w13_bias[expert_idx].float()

        intermediate = _CPU_MOE_ACT_FN[activation](gate_up).to(input.dtype)
        intermediate_int8, intermediate_scale = quantize_per_token(intermediate)
        output = torch.matmul(
            intermediate_int8.float(),
            w2[expert_idx].float().T,
        )
        output *= intermediate_scale * w2_scale[expert_idx]
        if w2_bias is not None:
            output += w2_bias[expert_idx].float()

        outputs.append(output)
        start_idx = end_idx

    sorted_output = torch.cat(outputs, dim=0)
    routed_output = torch.empty_like(sorted_output)
    routed_output[sorted_route_ids] = sorted_output
    return (
        routed_output.view(*topk_ids.shape, -1)
        .mul_(topk_weights.unsqueeze(-1))
        .sum(dim=1)
        .to(input.dtype)
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("expert_num", EXPERT_NUM)
@pytest.mark.parametrize("hidden_size", HIDDEN_DIM)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_DIM)
@pytest.mark.parametrize("use_bias", USE_BIAS)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("act", ACT)
@pytest.mark.parametrize("isa", ISA)
def test_cpu_fused_moe(
    default_vllm_config,
    batch_size: int,
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    use_bias: bool,
    dtype: torch.dtype,
    act: MoEActivation,
    isa: str,
):
    set_random_seed(0)

    topk_num = max(expert_num // 2, 1)
    up_dim = 2 * intermediate_size

    input = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w13 = torch.randn((expert_num, up_dim, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w2 = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype) / (
        0.5 * intermediate_size**0.5
    )
    router_logits = torch.randn((batch_size, expert_num), dtype=dtype)
    w13_bias = None
    w2_bias = None
    if use_bias:
        w13_bias = torch.randn((expert_num, up_dim), dtype=dtype) / (0.5 * up_dim**0.5)
        w2_bias = torch.randn((expert_num, hidden_size), dtype=dtype) / (
            0.5 * hidden_size**0.5
        )
    score = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk_num)
    topk_ids = topk_ids.to(torch.int32)

    ref_output = ref_fused_moe(
        input,
        w13,
        w2,
        w13_bias,
        w2_bias,
        topk_weight,
        topk_ids,
        act,
    )

    packed_w13 = cpu_prepack_moe_weight(w13, isa)
    packed_w2 = cpu_prepack_moe_weight(w2, isa)
    output = cpu_fused_moe(
        input,
        packed_w13,
        packed_w2,
        w13_bias,
        w2_bias,
        topk_weight,
        topk_ids,
        act.value,
        isa,
    )

    atol, rtol = get_default_atol(output), get_default_rtol(output)
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.ARM,
    reason="Requires Arm CPU",
)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("expert_num", EXPERT_NUM)
@pytest.mark.parametrize("hidden_size", HIDDEN_DIM)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_DIM)
@pytest.mark.parametrize("use_bias", USE_BIAS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("act", ACT)
@pytest.mark.parametrize("isa", ["neon"])
def test_cpu_fused_moe_int8(
    batch_size: int,
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    use_bias: bool,
    dtype: torch.dtype,
    act: MoEActivation,
    isa: str,
):
    set_random_seed(0)
    topk_num = max(expert_num // 2, 1)
    up_dim = 2 * intermediate_size

    input = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w13_fp = torch.randn((expert_num, up_dim, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w2_fp = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype) / (
        0.5 * intermediate_size**0.5
    )
    w13, w13_scale = quantize_per_channel(w13_fp)
    w2, w2_scale = quantize_per_channel(w2_fp)

    w13_bias = None
    w2_bias = None
    if use_bias:
        w13_bias = torch.randn((expert_num, up_dim), dtype=dtype) / (0.5 * up_dim**0.5)
        w2_bias = torch.randn((expert_num, hidden_size), dtype=dtype) / (
            0.5 * hidden_size**0.5
        )

    router_logits = torch.randn((batch_size, expert_num), dtype=dtype)
    score = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score, topk_num)
    topk_ids = topk_ids.to(torch.int32)

    ref_output = ref_fused_moe_int8(
        input,
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        act,
    )
    packed_w13 = cpu_prepack_moe_weight_int8(w13, isa)
    packed_w2 = cpu_prepack_moe_weight_int8(w2, isa)
    output = cpu_fused_moe_int8(
        input,
        packed_w13,
        packed_w2,
        w13_scale,
        w2_scale,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        act.value,
        isa,
    )

    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=2e-2)
