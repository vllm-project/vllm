# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm._custom_ops import cpu_fused_moe, cpu_prepack_moe_weight
from vllm.model_executor.layers.activation import SiluAndMul, SwigluOAIAndMul
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

EXPERT_NUM = [
    8,
]
HIDDEN_DIM = [128, 2880]
INTERMEDIATE_DIM = [128, 2880]
BATCH_SIZE = [1, 64, 256]
ACT = ["silu", "swigluoai"]
USE_BIAS = [True, False]
ISA = ["amx", "vec"] if torch._C._cpu._is_amx_tile_supported() else ["vec"]
DTYPE = [torch.bfloat16]

_CPU_MOE_ACT = {
    "silu": SiluAndMul(),
    "swigluoai": SwigluOAIAndMul(),
}


def ref_fused_moe(
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
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
        gate_up = (
            _CPU_MOE_ACT[activation]
            .forward_native(gate_up)
            .to(dtype=input.dtype)
            .float()
        )
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


@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("expert_num", EXPERT_NUM)
@pytest.mark.parametrize("hidden_size", HIDDEN_DIM)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_DIM)
@pytest.mark.parametrize("use_bias", USE_BIAS)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("act", ACT)
@pytest.mark.parametrize("isa", ISA)
def test_cpu_fused_moe(
    batch_size: int,
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    use_bias: bool,
    dtype: torch.dtype,
    act: str,
    isa: str,
):
    current_platform.seed_everything(0)

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
        act,
        isa,
    )

    atol, rtol = get_default_atol(output), get_default_rtol(output)
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )
