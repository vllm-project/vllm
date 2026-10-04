# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the unquantized-expert-weight diagnostic.

A checkpoint may store a MoE submodule unquantized while omitting it from the
quantization exclude list. The layer is then built for packed low-precision
weights and the loader is handed a floating point tensor of the logical width.
The copy fails on a shape mismatch that names neither the layer nor the cause,
so the loader raises a targeted error first. It must stay silent for every
combination that loads successfully today.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

LAYER = "model.layers.45.mlp.experts"


def check(expert_data: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    RoutedExperts._check_unquantized_expert_weight(
        SimpleNamespace(layer_name=LAYER), expert_data, loaded_weight
    )


def test_packed_destination_with_bf16_checkpoint_raises():
    # NVFP4 packs two values per byte, so the logical width halves.
    expert_data = torch.empty((4096, 1024), dtype=torch.uint8)
    loaded_weight = torch.empty((4096, 2048), dtype=torch.bfloat16)
    with pytest.raises(ValueError) as exc:
        check(expert_data, loaded_weight)
    message = str(exc.value)
    assert LAYER in message
    assert "unquantized" in message
    assert "exclude list" in message


def test_unquantized_load_does_not_raise():
    expert_data = torch.empty((4096, 2048), dtype=torch.bfloat16)
    loaded_weight = torch.empty((4096, 2048), dtype=torch.bfloat16)
    check(expert_data, loaded_weight)


def test_matching_shapes_do_not_raise():
    expert_data = torch.empty((4096, 2048), dtype=torch.uint8)
    loaded_weight = torch.empty((4096, 2048), dtype=torch.bfloat16)
    check(expert_data, loaded_weight)


def test_packed_checkpoint_into_packed_destination_does_not_raise():
    expert_data = torch.empty((4096, 1024), dtype=torch.uint8)
    loaded_weight = torch.empty((4096, 2048), dtype=torch.uint8)
    check(expert_data, loaded_weight)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float16, torch.float32])
def test_float_destination_never_raises(dtype: torch.dtype):
    expert_data = torch.empty((4096, 1024), dtype=dtype)
    loaded_weight = torch.empty((4096, 2048), dtype=torch.bfloat16)
    check(expert_data, loaded_weight)
