# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    FlashInferCutlassMxfp8LinearKernel,
    Mxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

if not (
    current_platform.is_cuda()
    and current_platform.is_device_capability_family(100)
    and has_flashinfer()
):
    pytest.skip(
        reason="FlashInfer CUTLASS MXFP8 requires an SM100-family GPU",
        allow_module_level=True,
    )


def _make_layer(weight: torch.Tensor) -> tuple[torch.nn.Module, torch.Tensor]:
    n, k = weight.shape
    weight_mxfp8, weight_scale = mxfp8_e4m3_quantize(
        weight, is_sf_swizzled_layout=False
    )
    weight_scale = weight_scale.view(n, k // MXFP8_BLOCK_SIZE)
    layer = torch.nn.Module()
    # The kernels take the output dtype from the layer, as for a real linear layer.
    layer.params_dtype = weight.dtype
    layer.weight = torch.nn.Parameter(weight_mxfp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    return layer, dequant_mxfp8_to_bf16(weight_mxfp8, weight_scale)


def _check_linear_numerics(kernel: Mxfp8LinearKernel, m: int) -> None:
    torch.manual_seed(0)
    n, k = kernel.config.weight_shape
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.05
    layer, weight_dequant = _make_layer(weight)
    kernel.process_weights_after_loading(layer)

    output = kernel.apply_weights(layer, x)
    assert output.shape == (m, n)

    x_mxfp8, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    x_dequant = dequant_mxfp8_to_bf16(x_mxfp8, x_scale.view(m, k // MXFP8_BLOCK_SIZE))
    reference = x_dequant.float() @ weight_dequant.float().t()
    error = (output.float() - reference).pow(2).mean().sqrt()
    assert error / reference.pow(2).mean().sqrt() < 1e-2


@pytest.mark.parametrize("m", [1, 64])
@pytest.mark.parametrize("n", [130, 1048, 2064, 4608])
@torch.inference_mode()
def test_flashinfer_cutlass_mxfp8_linear_numerics(m: int, n: int) -> None:
    config = Mxfp8LinearLayerConfig(weight_shape=(n, 256))
    _check_linear_numerics(FlashInferCutlassMxfp8LinearKernel(config), m)


@pytest.mark.parametrize("m", [1, 64])
@pytest.mark.parametrize("n", [1036, 2072])
@torch.inference_mode()
def test_selected_mxfp8_linear_numerics(m: int, n: int) -> None:
    """Whichever kernel vLLM selects for a layer computes it, with N not a
    multiple of 8 (1036) or of 32 (2072)."""
    _check_linear_numerics(init_mxfp8_linear_kernel(weight_shape=(n, 256)), m)
