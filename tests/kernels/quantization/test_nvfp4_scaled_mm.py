# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import pytest
import torch
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.nvfp4 import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.nvfp4.cutlass import CutlassNvFp4LinearKernel
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCudnnNvFp4LinearKernel,
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.pytorch import TorchNvFp4LinearKernel
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)
# M > 256 takes the large-M GEMM config.
SHAPES.append((512, 128, 64))

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("token_shape", [(17,), (1, 17), (2, 17)])
@pytest.mark.parametrize(
    "kernel_cls,hidden_size,output_size",
    [
        (cls, 128, 128)
        for cls in (
            FlashInferCutlassNvFp4LinearKernel,
            FlashInferCuteDslNvFp4LinearKernel,
            CutlassNvFp4LinearKernel,
            TorchNvFp4LinearKernel,
            FlashInferCudnnNvFp4LinearKernel,
            FlashInferB12xNvFp4LinearKernel,
        )
    ]
    + [
        (cls, 144, 150)
        for cls in (
            CutlassNvFp4LinearKernel,
            TorchNvFp4LinearKernel,
            FlashInferCutlassNvFp4LinearKernel,
            FlashInferCudnnNvFp4LinearKernel,
            FlashInferCuteDslNvFp4LinearKernel,
            FlashInferB12xNvFp4LinearKernel,
        )
    ],
)
@torch.inference_mode()
def test_nvfp4_preserves_quantized_input_shape(
    token_shape: tuple[int, ...],
    kernel_cls: type[NvFp4LinearKernel],
    hidden_size: int,
    output_size: int,
) -> None:
    """Pre-quantized inputs must match the tensor path and retain token dimensions."""
    supported, reason = kernel_cls.is_supported()
    if not supported:
        pytest.skip(reason)

    set_random_seed(42)
    x = torch.randn(*token_shape, hidden_size, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(output_size, hidden_size, dtype=x.dtype, device=x.device)
    input_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x.abs().max()).float()
    weight_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / weight.abs().max()).float()
    x_fp4, x_blockscale = ops.scaled_fp4_quant(x.reshape(-1, hidden_size), input_scale)
    weight_fp4, weight_blockscale = ops.scaled_fp4_quant(
        weight, weight_scale, is_sf_swizzled_layout=False
    )
    quantized_input = QuantizedActivation(
        data=x_fp4.reshape(*token_shape, hidden_size // 2),
        scale=x_blockscale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kNvfp4Dynamic,
    )
    layer = torch.nn.Module()
    layer.input_size_per_partition = hidden_size
    layer.output_size_per_partition = output_size
    layer.weight = weight_fp4
    layer.weight_scale = weight_blockscale
    layer.input_global_scale_inv = input_scale
    layer.alpha = (input_scale * weight_scale).reciprocal()
    kernel = kernel_cls(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    bias = torch.randn(output_size, dtype=x.dtype, device=x.device)

    expected = kernel.apply_weights(layer, x.reshape(-1, hidden_size), bias)
    actual = kernel.apply_weights(layer, quantized_input, bias)

    assert actual.shape == (*token_shape, output_size)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=0, atol=0)
    with pytest.raises(AssertionError, match="input_quant_key"):
        kernel.apply_weights(
            layer, replace(quantized_input, quant_key=kFp8StaticTensorSym), bias
        )


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("has_bias", [False, True])
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
    has_bias: bool,
) -> None:
    if has_bias and not current_platform.is_device_capability_family(120):
        pytest.skip("The bias epilogue is implemented for SM12x only.")
    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    # ops.scaled_fp4_quant returns swizzled scales, while weights
    # from checkpoints are in linear scales.
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

    # get_ref_results unswizzles the scales internally.
    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        device,
    )
    bias = torch.randn(n, dtype=dtype, device=device) if has_bias else None
    if bias is not None:
        expected_out = expected_out + bias
    out = ops.cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype, bias=bias
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)
