# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and inference for quantized HF models supported
on the AutoRound.

Validating the configuration and printing results for manual checking.

Run `pytest tests/quantization/test_auto_round.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.inc import (
    INCGPTQRowParallelTailLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.models.gemma4 import (
    _dequantize_autoround_gptq_router_weight,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

MODELS = [
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",  ##auto_round:auto_gptq
    "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",  ##auto_round:auto_awq
]


@pytest.mark.skipif(
    not current_platform.is_cpu()
    and not current_platform.is_xpu()
    and not current_platform.is_cuda(),
    reason="only supports CPU/XPU/CUDA backend.",
)
@pytest.mark.parametrize("model", MODELS)
def test_auto_round(vllm_runner, model):
    with vllm_runner(model, enforce_eager=True) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=8)
    assert output
    print(f"{output[0][1]}")


def test_autoround_gptq_router_weight_dequantizes_symmetric_zero_point():
    qweight_unpacked = (torch.arange(64, dtype=torch.int32).reshape(8, 8) % 8) + 8
    qzeros_unpacked = torch.full((2, 8), 7, dtype=torch.int32)
    scales = torch.stack(
        (
            torch.linspace(0.5, 1.2, 8),
            torch.linspace(1.5, 2.2, 8),
        )
    )

    qweight = pack_quantized_values_into_int32(
        qweight_unpacked, scalar_types.uint4b8, packed_dim=0
    )
    qzeros = pack_quantized_values_into_int32(
        qzeros_unpacked, scalar_types.uint4b8, packed_dim=1
    )

    weight = _dequantize_autoround_gptq_router_weight(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        num_bits=4,
        group_size=4,
        sym=True,
        params_dtype=torch.float16,
    )

    expected_qzeros = qzeros_unpacked + 1
    row_groups = torch.arange(qweight_unpacked.shape[0]) // 4
    expected = (
        (qweight_unpacked - expected_qzeros[row_groups]) * scales[row_groups]
    ).t()
    torch.testing.assert_close(weight, expected.to(torch.float16))


def test_inc_gptq_row_parallel_tail_fallback_uses_global_group_indices(monkeypatch):
    import vllm.model_executor.layers.quantization.inc as inc
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(inc, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 2)

    method = INCGPTQRowParallelTailLinearMethod(
        weight_bits=4,
        group_size=16,
        sym=True,
    )
    layer = torch.nn.Module()
    layer.input_size_per_partition = 24
    method.create_weights(
        layer,
        input_size_per_partition=24,
        output_partition_sizes=[8],
        input_size=48,
        output_size=8,
        params_dtype=torch.float32,
    )

    assert layer.g_idx.tolist() == [1] * 8 + [2] * 16

    qweight_unpacked = torch.full((24, 8), 9, dtype=torch.int32)
    layer.qweight.data.copy_(
        pack_quantized_values_into_int32(
            qweight_unpacked, scalar_types.uint4b8, packed_dim=0
        )
    )
    layer.scales.data.copy_(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
    )
    method.process_weights_after_loading(layer)

    x = torch.ones(1, 24, dtype=torch.float16)
    output = method.apply(layer, x)

    # qweight 9 minus uint4 symmetric bias 8 gives dequant value 1.
    expected = 8 * layer.scales.data[1] + 16 * layer.scales.data[2]
    expected = expected.unsqueeze(0)
    torch.testing.assert_close(output, expected.to(torch.float16))
