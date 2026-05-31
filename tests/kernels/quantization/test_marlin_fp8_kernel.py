# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MarlinFP8ScaledMMLinearKernel.

Exercises process_weights_after_loading + apply_weights for square weight
matrices (N == K), which previously suffered silent data corruption due to
a shape-based transpose heuristic that was a no-op when the weight shape
tuple (N, K) == (K, N).

The fix uses contiguity to distinguish the two input layouts:
  - (N, K) contiguous  -- callers that pass checkpoint layout directly
                          (CompressedTensorsW8A16Fp8, Fp8LinearMethod)
                          -> Marlin transposes to (K, N)
  - (K, N) non-contiguous -- callers that pre-transpose with .t()
                             (ModelOptFp8LinearMethod)
                             -> Marlin leaves weight as-is

Run with:
    VLLM_TEST_FORCE_FP8_MARLIN=1 pytest \
        tests/kernels/quantization/test_marlin_fp8_kernel.py -v
"""

import os

import pytest
import torch
from torch.nn import Parameter

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import init_wfp8_a16_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.platforms import current_platform

vllm_config = VllmConfig()

if current_platform.is_rocm():
    pytest.skip(
        "Marlin FP8 kernel is not supported on ROCm.",
        allow_module_level=True,
    )

if not os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN"):
    pytest.skip(
        "Set VLLM_TEST_FORCE_FP8_MARLIN=1 to run Marlin FP8 kernel tests.",
        allow_module_level=True,
    )


def _make_fp8_layer(
    size_n: int,
    size_k: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Build a minimal fake layer with FP8 weight in (N, K) layout and
    per-channel BF16 scales, matching what create_fp8_weight_parameter
    produces at model load time.

    Returns (layer, w_ref) where w_ref is the dequantized BF16 reference
    weight in (N, K) layout for computing the expected output.
    """
    torch.manual_seed(42)

    # Random BF16 weight in (N, K), scale per output channel
    w_bf16 = torch.randn(size_n, size_k, dtype=dtype, device=device)
    scale_per_ch = w_bf16.abs().amax(dim=1, keepdim=True) / 448.0  # (N, 1)

    # Quantize to FP8
    w_fp8 = (w_bf16 / scale_per_ch).to(torch.float8_e4m3fn)
    w_ref = w_fp8.to(dtype) * scale_per_ch  # dequantized reference (N, K)

    layer = torch.nn.Module()
    layer.weight = Parameter(w_fp8, requires_grad=False)
    layer.weight_scale = Parameter(scale_per_ch.to(dtype), requires_grad=False)
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.orig_dtype = dtype

    return layer, w_ref


# (size_m, size_n, size_k) -- square cases (N==K) are the regression targets
@pytest.mark.parametrize(
    "size_m, size_n, size_k",
    [
        # Square: N == K -- the regression case
        (1, 4096, 4096),
        (7, 4096, 4096),
        (16, 4096, 4096),
        # Non-square: N != K -- should still work correctly
        (7, 1024, 4096),
        (7, 12800, 4096),
        (7, 4096, 12800),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_marlin_fp8_square_and_nonsquare(
    size_m: int, size_n: int, size_k: int, dtype: torch.dtype
):
    """Verify that MarlinFP8ScaledMMLinearKernel produces correct outputs for
    both square (N==K) and non-square weight matrices.

    Before the fix, square matrices silently produced correlated-near-zero
    outputs (correlation with reference ~= 0) instead of correct results.
    """
    device = "cuda"

    layer, w_ref = _make_fp8_layer(size_n, size_k, dtype=dtype, device=device)

    with set_current_vllm_config(vllm_config):
        kernel = init_wfp8_a16_linear_kernel(
            weight_quant_key=kFp8StaticChannelSym,
            activation_quant_key=kFp8DynamicTokenSym,
            weight_shape=(size_n, size_k),
            input_dtype=dtype,
            out_dtype=dtype,
            force_kernel=MarlinFP8ScaledMMLinearKernel,
        )
        kernel.process_weights_after_loading(layer)

        x = torch.randn(size_m, size_k, dtype=dtype, device=device)
        y_marlin = kernel.apply_weights(layer, x)

    # Reference: x @ w_ref.T using full BF16 precision
    y_ref = x @ w_ref.T

    # Allow for FP8 quantization noise (~1-2% relative error)
    max_diff = (y_marlin - y_ref).abs().max().item()
    ref_scale = y_ref.abs().max().item()
    rel_err = max_diff / (ref_scale + 1e-6)

    assert rel_err < 0.05, (
        f"N={size_n}, K={size_k}: relative error {rel_err:.4f} too large "
        f"(max_diff={max_diff:.4f}, ref_scale={ref_scale:.4f}). "
        "This likely indicates the weight was not transposed correctly."
    )


@pytest.mark.parametrize(
    "size_m, size_n, size_k",
    [
        # Square -- the regression case
        (1, 4096, 4096),
        (7, 4096, 4096),
        # Non-square
        (7, 1024, 4096),
        (7, 4096, 12800),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_marlin_fp8_pretransposed_input(
    size_m: int, size_n: int, size_k: int, dtype: torch.dtype
):
    """Verify that Marlin correctly handles (K, N) non-contiguous input.

    ModelOptFp8LinearMethod pre-transposes the weight with .t() before
    calling the kernel, producing a (K, N) non-contiguous tensor.  The
    contiguity-based check in process_weights_after_loading must NOT
    transpose again -- a double-transpose would silently corrupt outputs.
    """
    device = "cuda"

    torch.manual_seed(42)

    # Build reference weight in (N, K) layout
    w_bf16 = torch.randn(size_n, size_k, dtype=dtype, device=device)
    scale_per_ch = w_bf16.abs().amax(dim=1, keepdim=True) / 448.0  # (N, 1)
    w_fp8 = (w_bf16 / scale_per_ch).to(torch.float8_e4m3fn)
    w_ref = w_fp8.to(dtype) * scale_per_ch  # (N, K) reference

    # Simulate modelopt pre-transpose: (N, K) -> (K, N) non-contiguous
    w_fp8_kt = w_fp8.t()  # (K, N), non-contiguous
    assert not w_fp8_kt.is_contiguous(), "pre-condition: .t() must be non-contiguous"

    layer = torch.nn.Module()
    layer.weight = Parameter(w_fp8_kt, requires_grad=False)
    layer.weight_scale = Parameter(scale_per_ch.to(dtype), requires_grad=False)
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.orig_dtype = dtype

    with set_current_vllm_config(vllm_config):
        kernel = init_wfp8_a16_linear_kernel(
            weight_quant_key=kFp8StaticChannelSym,
            activation_quant_key=kFp8DynamicTokenSym,
            weight_shape=(size_n, size_k),
            input_dtype=dtype,
            out_dtype=dtype,
            force_kernel=MarlinFP8ScaledMMLinearKernel,
        )
        kernel.process_weights_after_loading(layer)

        x = torch.randn(size_m, size_k, dtype=dtype, device=device)
        y_marlin = kernel.apply_weights(layer, x)

    y_ref = x @ w_ref.T

    max_diff = (y_marlin - y_ref).abs().max().item()
    ref_scale = y_ref.abs().max().item()
    rel_err = max_diff / (ref_scale + 1e-6)

    assert rel_err < 0.05, (
        f"N={size_n}, K={size_k}: relative error {rel_err:.4f} too large "
        f"(max_diff={max_diff:.4f}, ref_scale={ref_scale:.4f}). "
        "This likely indicates a double-transpose regression on pre-transposed input."
    )
