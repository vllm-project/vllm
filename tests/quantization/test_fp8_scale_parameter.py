# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.parameter as parameter
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="torch does not expose float8_e8m0fnu",
)
def test_create_fp8_scale_parameter_initializes_e8m0(monkeypatch):
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)

    scale = create_fp8_scale_parameter(
        BlockQuantScaleParameter,
        output_partition_sizes=[128],
        input_size_per_partition=128,
        block_size=[128, 128],
        weight_loader=None,
        scale_dtype=torch.float8_e8m0fnu,
    )

    assert scale.dtype == torch.float8_e8m0fnu
    raw_scale = scale.data.view(torch.uint8)
    assert torch.equal(raw_scale, torch.zeros_like(raw_scale))
