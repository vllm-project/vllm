# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.base import store_partial_sum_prescaled
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.nvfp4.cutlass import CutlassNvFp4LinearKernel
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCutlassNvFp4LinearKernel,
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


@pytest.mark.parametrize(
    "kernel_cls", [CutlassNvFp4LinearKernel, FlashInferCutlassNvFp4LinearKernel]
)
@torch.inference_mode()
def test_nvfp4_linear_fp16_partial_sums(kernel_cls: type[NvFp4LinearKernel]) -> None:
    """FP16 partial sums (VLLM_TP_FP16_PARTIAL_SUMS): alpha and the bias carry
    the 2^-k prescale, with the CUTLASS bias in the GEMM epilogue (SM120) or
    added after it."""
    if not kernel_cls.is_supported()[0]:
        pytest.skip(f"{kernel_cls.__name__} is not supported")
    set_random_seed(0)
    m, k, n = 64, 2048, 1024
    x = torch.randn(m, k, device="cuda").bfloat16()
    w = (torch.randn(n, k, device="cuda") / k**0.5).bfloat16()
    b = torch.randn(n, device="cuda").bfloat16()
    x_global = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x.float().abs().max()).reshape(1)
    w_global = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w.float().abs().max()).reshape(1)
    w_fp4, w_sf = ops.scaled_fp4_quant(w, w_global)  # swizzled, as after loading
    layer = torch.nn.Module()
    layer.weight, layer.weight_scale = w_fp4, w_sf
    layer.alpha = 1.0 / (x_global * w_global)
    layer.input_global_scale_inv = x_global
    layer.input_size_per_partition, layer.output_size_per_partition = k, n

    kernel = kernel_cls(NvFp4LinearLayerConfig())
    epilogues = [False, True] if getattr(kernel, "bias_epilogue", False) else [False]
    for bias_epilogue in epilogues:
        if isinstance(kernel, CutlassNvFp4LinearKernel):
            kernel.bias_epilogue = bias_epilogue
        layer.partial_sum_dtype, layer.partial_sum_scale = None, 1.0
        ref = kernel.apply_weights(layer, x, b)
        layer.partial_sum_dtype, layer.partial_sum_scale = torch.float16, 0.25
        part = kernel.apply_weights(layer, x, b)
        assert ref.dtype == torch.bfloat16 and part.dtype == torch.float16
        rms = float(ref.float().pow(2).mean().sqrt())
        diff = float((part.float() * 4 - ref.float()).abs().mean()) / rms
        assert diff < 1e-2, (bias_epilogue, diff)
        # alpha prescaled after loading: same output as per call; a weight
        # reload refills it in place.
        store_partial_sum_prescaled(layer, "partial_sum_alpha", layer.alpha)
        ptr = layer.partial_sum_alpha.data_ptr()
        torch.testing.assert_close(
            kernel.apply_weights(layer, x, b), part, rtol=0, atol=0
        )
        store_partial_sum_prescaled(layer, "partial_sum_alpha", layer.alpha)
        assert layer.partial_sum_alpha.data_ptr() == ptr
        del layer.partial_sum_alpha
