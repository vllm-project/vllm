# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FlashInfer MXFP8 linear kernels call mm_mxfp8 with skip_check=True;
the result must match a fully validated mm_mxfp8 call bitwise."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip("FlashInfer MXFP8 linear kernels need SM100", allow_module_level=True)

from vllm.model_executor.kernels.linear import (  # noqa: E402
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    mxfp8_e4m3_quantize,
)
from vllm.utils.flashinfer import has_flashinfer_cutedsl  # noqa: E402


@pytest.mark.parametrize(
    "kernel_cls, backend",
    [
        (FlashInferCutlassMxfp8LinearKernel, "cutlass"),
        (FlashInferCutedslMxfp8LinearKernel, "cute-dsl"),
    ],
)
@pytest.mark.parametrize("m", [1, 300, 2048])
def test_mxfp8_linear_matches_checked_mm_mxfp8(kernel_cls, backend, m):
    import flashinfer

    if backend == "cute-dsl" and not has_flashinfer_cutedsl():
        pytest.skip("FlashInfer CuTe-DSL unavailable")
    torch.manual_seed(0)
    n, k = 384, 1024
    weight_bf16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    weight, weight_scale = mxfp8_e4m3_quantize(weight_bf16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.params_dtype = torch.bfloat16
    kernel = kernel_cls(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x)

    xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    w = layer.weight if layer.weight.shape[0] == k else layer.weight.t()
    ref = flashinfer.mm_mxfp8(
        xq, w, xs, layer.weight_scale, out_dtype=torch.bfloat16, backend=backend
    )
    assert out.shape == (m, n)
    assert torch.equal(out, ref)
