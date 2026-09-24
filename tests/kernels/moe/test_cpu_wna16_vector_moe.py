# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CPU vector GPTQ INT4 MoE fallback."""

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
    CPUExpertsInt4Vec,
    prepare_int4_moe_layer_for_cpu_vec,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    make_wna16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.scalar_type import scalar_types

if not current_platform.is_cpu():
    pytest.skip("CPU-only MoE test", allow_module_level=True)


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("group_size", [64, 128])
def test_gptq_int4_vector_moe_matches_torch(with_bias: bool, group_size: int):
    """The AVX2 fallback computes routed experts with and without bias."""
    if current_platform.get_cpu_architecture() != CpuArchEnum.X86:
        pytest.skip("CPU vector INT4 MoE supports x86 only")
    assert hasattr(torch.ops._C, "cpu_gemm_wna16")

    torch.manual_seed(0)
    m, n, k, e, topk = 5, 128, 256, 4, 2
    hidden = torch.randn(m, k, dtype=torch.bfloat16) / k**0.5
    w1_values = torch.randint(0, 16, (e, k, 2 * n), dtype=torch.int32)
    w2_values = torch.randint(0, 16, (e, n, k), dtype=torch.int32)
    w1_packed = torch.stack(
        [
            pack_quantized_values_into_int32(w, scalar_types.uint4b8, 0)
            for w in w1_values
        ]
    )
    w2_packed = torch.stack(
        [
            pack_quantized_values_into_int32(w, scalar_types.uint4b8, 0)
            for w in w2_values
        ]
    )
    w1_scale = (torch.rand(e, k // group_size, 2 * n) * 0.01 + 0.001).bfloat16()
    w2_scale = (torch.rand(e, n // group_size, k) * 0.01 + 0.001).bfloat16()
    w1_bias = (torch.randn(e, 2 * n) * 0.001).bfloat16() if with_bias else None
    w2_bias = (torch.randn(e, k) * 0.001).bfloat16() if with_bias else None
    topk_weights, topk_ids = torch.topk(torch.softmax(torch.randn(m, e), -1), topk)
    topk_ids = topk_ids.int()

    expected = torch.zeros(m, k)
    for row in range(m):
        for slot in range(topk):
            expert = int(topk_ids[row, slot])
            gate_up_weight = (w1_values[expert].float() - 8) * w1_scale[
                expert
            ].float().repeat_interleave(group_size, 0)
            gate_up = hidden[row : row + 1].float() @ gate_up_weight
            if w1_bias is not None:
                gate_up += w1_bias[expert].float()
            gate, up = gate_up.bfloat16().float().chunk(2, dim=-1)
            activated = (F.silu(gate) * up).bfloat16().float()
            down_weight = (w2_values[expert].float() - 8) * w2_scale[
                expert
            ].float().repeat_interleave(group_size, 0)
            down = activated @ down_weight
            if w2_bias is not None:
                down += w2_bias[expert].float()
            expected[row] += (
                down.bfloat16().float().squeeze(0) * topk_weights[row, slot]
            )

    packed_w1, packed_w2 = prepare_int4_moe_layer_for_cpu_vec(w1_packed, w2_packed)
    quant_config = make_wna16_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        group_size=group_size,
        num_bits=4,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
    experts = CPUExpertsInt4Vec(
        make_dummy_moe_config(
            e, experts_per_token=topk, hidden_dim=k, intermediate_size=n
        ),
        quant_config,
    )
    output = torch.empty_like(hidden)
    experts.apply(
        output,
        hidden,
        packed_w1,
        packed_w2,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        e,
        None,
        None,
        None,
        torch.empty(0),
        torch.empty(0),
        None,
        False,
    )
    torch.testing.assert_close(expected.bfloat16(), output, atol=1e-4, rtol=1e-2)
