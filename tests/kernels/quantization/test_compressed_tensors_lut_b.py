# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for compressed-tensors LUT-B linear loading and emulation."""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_lut_b import (  # noqa: E501
    CompressedTensorsLutB,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    dequantize_lut_b,
    quantize_lut_b,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)


def test_compressed_tensors_lut_b_qkv_loads_and_emulates_serialized_weights(
    dist_init,
):
    """A fused QKV layer loads CT tiles from its three checkpoint projections."""
    torch.manual_seed(0)
    quant_config = CompressedTensorsConfig.from_config(
        {
            "format": "lut-b",
            "config_groups": {
                "group_0": {
                    "format": "lut-b",
                    "targets": [r"re:.*(q_proj|k_proj|v_proj)$"],
                    "weights": {
                        "type": "codebook",
                        "num_bits": 3,
                        "strategy": "block",
                        "block_structure": [8, 64],
                        "symmetric": True,
                        "dynamic": False,
                        "scale_dtype": "torch.uint8",
                    },
                    "input_activations": None,
                }
            },
        }
    )

    input_size = head_size = 128
    num_heads = num_kv_heads = 2
    with torch.device("cpu"):
        layer = QKVParallelLinear(
            hidden_size=input_size,
            head_size=head_size // num_heads,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=True,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="model.layers.0.self_attn.qkv_proj",
            disable_tp=True,
        )
    assert isinstance(layer.scheme, CompressedTensorsLutB)

    source_weights = {
        shard_id: torch.randn(
            head_size, input_size, dtype=torch.bfloat16, device=layer.bias.device
        )
        for shard_id in ("q", "k", "v")
    }
    quantized = {
        shard_id: quantize_lut_b(weight) for shard_id, weight in source_weights.items()
    }
    for name, tensor_index in (
        ("weight_packed", 0),
        ("weight_codebook", 1),
        ("weight_scale", 2),
    ):
        param = getattr(layer, name)
        for shard_id, tensors in quantized.items():
            param.weight_loader(param, tensors[tensor_index], shard_id)
    for shard_id in ("q", "k", "v"):
        bias = torch.randn(head_size, dtype=torch.bfloat16, device=layer.bias.device)
        layer.weight_loader(layer.bias, bias, shard_id)

    expected_weight = torch.cat(
        [
            dequantize_lut_b(
                quantized[key][0],
                quantized[key][1],
                quantized[key][2],
                out_dtype=torch.bfloat16,
            )
            for key in ("q", "k", "v")
        ]
    )

    layer.scheme.process_weights_after_loading(layer)
    torch.testing.assert_close(
        layer.weight,
        expected_weight,
        rtol=0,
        atol=0,
    )

    x = torch.randn(3, input_size, dtype=torch.bfloat16, device=layer.weight.device)
    x_q, x_scales = mxfp8_e4m3_quantize(x)
    x_qdq = dequant_mxfp8_to_bf16(x_q, x_scales).to(x.dtype)
    expected = F.linear(x_qdq, layer.weight, layer.bias)
    actual, _ = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
