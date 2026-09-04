# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arena coverage for Marlin workspaces outside the WNA16 kernel registry."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_mxfp8_layer_for_marlin,
)
from vllm.platforms import current_platform


pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Marlin repack needs an accelerator"
)

N = 256
K = 256


def _base_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.output_size_per_partition = N
    layer.input_size_per_partition = K
    layer.params_dtype = torch.float16
    layer.bias = None
    return layer


def _load_fp4_checkpoint(layer: torch.nn.Module, *, nvfp4: bool) -> None:
    layer.weight = torch.nn.Parameter(
        torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda"),
        requires_grad=False,
    )
    group_size = 16 if nvfp4 else 32
    scale_dtype = torch.float8_e4m3fn if nvfp4 else torch.uint8
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((N, K // group_size), dtype=scale_dtype, device="cuda"),
        requires_grad=False,
    )
    if nvfp4:
        layer.weight_global_scale = torch.nn.Parameter(
            torch.ones((), dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )
    elif hasattr(layer, "weight_global_scale"):
        delattr(layer, "weight_global_scale")


@pytest.mark.parametrize("nvfp4", [False, True], ids=["mxfp4", "nvfp4"])
def test_fp4_marlin_workspace_is_stable_across_pwal(nvfp4):
    layer = _base_layer()
    _load_fp4_checkpoint(layer, nvfp4=nvfp4)
    prepare_fp4_layer_for_marlin(layer)
    before = layer.workspace.data_ptr()
    layer.workspace.fill_(7)

    _load_fp4_checkpoint(layer, nvfp4=nvfp4)
    prepare_fp4_layer_for_marlin(layer)

    assert layer.workspace.data_ptr() == before
    assert torch.count_nonzero(layer.workspace) == 0


def _load_mxfp8_checkpoint(layer: torch.nn.Module) -> None:
    layer.weight = torch.nn.Parameter(
        torch.ones((N, K), dtype=torch.float8_e4m3fn, device="cuda"),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((N, K // 32), dtype=torch.uint8, device="cuda"),
        requires_grad=False,
    )


def test_mxfp8_marlin_workspace_is_stable_across_pwal():
    layer = _base_layer()
    _load_mxfp8_checkpoint(layer)
    prepare_mxfp8_layer_for_marlin(layer)
    before = layer.workspace.data_ptr()
    layer.workspace.fill_(7)

    _load_mxfp8_checkpoint(layer)
    prepare_mxfp8_layer_for_marlin(layer)

    assert layer.workspace.data_ptr() == before
    assert torch.count_nonzero(layer.workspace) == 0
