# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer MXFP8 linears with output sizes that tensor-parallel shards leave
unaligned, such as Nemotron-H-4B's Mamba in_proj at TP4 (4124).

mm_mxfp8 cute-dsl needs N % 8 == 0 and returns garbage otherwise; cutlass needs
N % 32 == 0 and raises otherwise.
"""

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
        reason="FlashInfer MXFP8 linears require an SM100-family GPU",
        allow_module_level=True,
    )

_K = 256
# 8 | 4632, but 32 does not.
_UNALIGNED_N = [130, 2188, 4124, 4632]


def _check_numerics(kernel: Mxfp8LinearKernel, m: int) -> None:
    torch.manual_seed(0)
    n, k = kernel.config.weight_shape
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.05
    weight_mxfp8, weight_scale = mxfp8_e4m3_quantize(
        weight, is_sf_swizzled_layout=False
    )
    weight_scale = weight_scale.view(n, k // MXFP8_BLOCK_SIZE)
    weight_dequant = dequant_mxfp8_to_bf16(weight_mxfp8, weight_scale)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight_mxfp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.params_dtype = torch.bfloat16
    kernel.process_weights_after_loading(layer)

    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    output = kernel.apply_weights(layer, x)

    x_mxfp8, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    x_dequant = dequant_mxfp8_to_bf16(x_mxfp8, x_scale.view(m, k // MXFP8_BLOCK_SIZE))
    reference = x_dequant.float() @ weight_dequant.float().t()
    assert output.shape == (m, n)
    error = (output.float() - reference).pow(2).mean().sqrt()
    assert error / reference.pow(2).mean().sqrt() < 1e-2


@pytest.mark.parametrize("m", [1, 64])
@pytest.mark.parametrize("n", _UNALIGNED_N)
@torch.inference_mode()
def test_selected_mxfp8_linear_unaligned_n(m: int, n: int) -> None:
    _check_numerics(init_mxfp8_linear_kernel(weight_shape=(n, _K)), m)


@pytest.mark.parametrize("m", [1, 64])
@pytest.mark.parametrize("n", _UNALIGNED_N)
@torch.inference_mode()
def test_flashinfer_cutlass_mxfp8_linear_unaligned_n(m: int, n: int) -> None:
    config = Mxfp8LinearLayerConfig(weight_shape=(n, _K))
    _check_numerics(FlashInferCutlassMxfp8LinearKernel(config), m)
