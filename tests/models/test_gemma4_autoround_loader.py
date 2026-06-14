# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Any

import pytest
import torch

try:
    importlib.import_module("vllm.vllm_flash_attn")
except ImportError:
    import sys
    import types

    fake_flash_attn: Any = types.ModuleType("vllm.vllm_flash_attn")
    fake_flash_attn.flash_attn_varlen_func = None
    fake_flash_attn.get_scheduler_metadata = lambda *args, **kwargs: None
    sys.modules["vllm.vllm_flash_attn"] = fake_flash_attn

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.models.gemma4 import (
    _dequantize_autoround_gptq_router_weight,
)
from vllm.scalar_type import scalar_types


def _pack_uint4_rows(values: torch.Tensor) -> torch.Tensor:
    return pack_quantized_values_into_int32(
        values.to(torch.int32),
        scalar_types.uint4b8,
        packed_dim=0,
    )


def _pack_uint4_cols(values: torch.Tensor) -> torch.Tensor:
    return pack_quantized_values_into_int32(
        values.to(torch.int32),
        scalar_types.uint4b8,
        packed_dim=1,
    )


def test_autoround_router_dequant_uses_whole_matrix_for_negative_group_size():
    qweight_unpacked = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 9, 10, 11],
            [5, 6, 7, 8, 9, 10, 11, 12],
            [6, 7, 8, 9, 10, 11, 12, 13],
            [7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [9, 10, 11, 12, 13, 14, 15, 2],
        ],
        dtype=torch.int32,
    )
    qzeros_unpacked = torch.zeros((1, 8), dtype=torch.int32)
    scales = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8) / 10

    result = _dequantize_autoround_gptq_router_weight(
        qweight=_pack_uint4_rows(qweight_unpacked),
        qzeros=_pack_uint4_cols(qzeros_unpacked),
        scales=scales,
        num_bits=4,
        group_size=-1,
        sym=True,
        params_dtype=torch.float16,
    )

    expected = ((qweight_unpacked.to(torch.float32) - 1) * scales[0]).t()
    torch.testing.assert_close(result, expected.to(torch.float16))


def test_autoround_router_dequant_uses_group_specific_scales_and_zeros():
    qweight_unpacked = torch.arange(64, dtype=torch.int32).reshape(8, 8) % 16
    qzeros_unpacked = torch.stack(
        [
            torch.zeros(8, dtype=torch.int32),
            torch.ones(8, dtype=torch.int32),
        ]
    )
    scales = torch.stack(
        [
            torch.full((8,), 0.25, dtype=torch.float32),
            torch.full((8,), 0.5, dtype=torch.float32),
        ]
    )

    result = _dequantize_autoround_gptq_router_weight(
        qweight=_pack_uint4_rows(qweight_unpacked),
        qzeros=_pack_uint4_cols(qzeros_unpacked),
        scales=scales,
        num_bits=4,
        group_size=4,
        sym=True,
        params_dtype=torch.float32,
    )

    expected = torch.empty_like(qweight_unpacked, dtype=torch.float32)
    expected[:4] = (qweight_unpacked[:4].to(torch.float32) - 1) * scales[0]
    expected[4:] = (qweight_unpacked[4:].to(torch.float32) - 2) * scales[1]
    torch.testing.assert_close(result, expected.t())


def test_autoround_router_dequant_rejects_unsupported_bits():
    with pytest.raises(ValueError, match="unsupported num_bits=3"):
        _dequantize_autoround_gptq_router_weight(
            qweight=torch.empty(1, 8, dtype=torch.int32),
            qzeros=torch.empty(1, 1, dtype=torch.int32),
            scales=torch.empty(1, 8),
            num_bits=3,
            group_size=-1,
            sym=True,
            params_dtype=torch.float16,
        )
