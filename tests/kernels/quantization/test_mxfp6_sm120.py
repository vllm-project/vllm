# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional SM120 integration tests for the native MXFP6 W6A8 backend."""

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MxFp6LinearLayerConfig,
    Mxfp6Sm120LinearKernel,
    init_mxfp6_linear_kernel,
)
from vllm.model_executor.kernels.linear.mxfp6.sm120 import (
    is_mxfp6_sm120_available,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)

pytestmark = pytest.mark.optional

if not is_mxfp6_sm120_available():
    pytest.skip(
        "requires an SM120 GPU and the mxfp6-sm120 extension",
        allow_module_level=True,
    )


@torch.inference_mode()
def test_sm120_w6a8_matches_package_and_replays_in_cuda_graph():
    """Verify vLLM packing, activation quantization, GEMM, and graph replay."""
    import mxfp6

    selected = init_mxfp6_linear_kernel(kMxfp6E3M2Static, kMxfp8Dynamic)
    assert isinstance(selected, Mxfp6Sm120LinearKernel)

    torch.manual_seed(0)
    device = torch.device("cuda")
    rows, output_features, input_features = 4, 256, 512
    x = torch.randn(
        (rows, input_features), device=device, dtype=torch.bfloat16
    ).contiguous()
    weight_source = torch.randn(
        (output_features, input_features), device=device, dtype=torch.bfloat16
    ).contiguous()
    packed_weight = mxfp6.quantize_mxfp6(weight_source)
    logical_scale = mxfp6.unpack_scales(
        packed_weight.scales, output_features, input_features
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        packed_weight.values.reshape(output_features, -1), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(logical_scale, requires_grad=False)
    kernel = Mxfp6Sm120LinearKernel(
        MxFp6LinearLayerConfig(kMxfp6E3M2Static, kMxfp8Dynamic)
    )
    kernel.process_weights_after_loading(layer)

    expected = mxfp6.gemm_w6a8(
        mxfp6.quantize_activation(x), packed_weight, out_dtype=x.dtype
    )
    actual = kernel.apply_weights(layer, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    static_x = x.clone()
    kernel.apply_weights(layer, static_x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = kernel.apply_weights(layer, static_x)
    graph.replay()
    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)

    updated_x = torch.randn_like(static_x)
    updated_expected = mxfp6.gemm_w6a8(
        mxfp6.quantize_activation(updated_x), packed_weight, out_dtype=x.dtype
    )
    static_x.copy_(updated_x)
    graph.replay()
    torch.testing.assert_close(graph_output, updated_expected, rtol=0, atol=0)
