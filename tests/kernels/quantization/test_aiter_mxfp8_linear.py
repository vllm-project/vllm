# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER Triton MXFP8 linear (``AiterMxfp8LinearKernel``) on CDNA4."""

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm()
    or not current_platform.supports_mx()
    or not is_aiter_found_and_supported(),
    reason="AITER MXFP8 linear requires AITER on CDNA4 (gfx95x)",
)

# (name, N, K) for the MiniMax-M3 MXFP8 linears at tensor_parallel=8.
M3_TP8_SHAPES = [
    ("qkv", 1536, 6144),
    ("o_proj", 6144, 1024),
    ("shared_gate_up", 768, 6144),
    ("shared_down", 6144, 384),
]


def _make_layer(n: int, k: int, device: str) -> torch.nn.Module:
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
    weight, weight_scale = _mxfp8_e4m3_quantize_torch(weight_bf16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    return layer


@pytest.mark.parametrize("name,n,k", M3_TP8_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("num_tokens", [1, 2, 7, 16, 31, 64, 128, 1000, 8192])
@torch.inference_mode()
def test_aiter_mxfp8_linear_matches_dequant_reference(name, n, k, num_tokens):
    """Output matches an FP32 GEMM of the same MXFP8-quantized inputs.

    The token counts cover the split-K decode buckets, ragged M and prefill.
    """
    from vllm.model_executor.kernels.linear.mxfp8.aiter import (
        AiterMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )

    if not rocm_aiter_ops.is_triton_gemm_afp8wfp8_tuned(n, k):
        pytest.skip(f"AITER has no tuned MXFP8 config for N={n}, K={k}")

    torch.manual_seed(num_tokens)
    device = "cuda"
    layer = _make_layer(n, k, device)
    kernel = AiterMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, k, device=device, dtype=torch.bfloat16) * 0.5
    out = kernel.apply_weights(layer, x)

    assert out.shape == (num_tokens, n)
    assert torch.isfinite(out).all(), f"{name}: non-finite output"

    x_q, x_scale = mxfp8_e4m3_quantize(x)
    expected = (
        dequant_mxfp8_to_bf16(x_q, x_scale).float()
        @ dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale).float().T
    )
    rel = (out.float() - expected).norm() / expected.norm()
    assert rel < 5e-3, f"{name}: relative error {rel:.4f}"


@torch.inference_mode()
def test_aiter_mxfp8_linear_3d_input_and_bias():
    """A [batch, seq, K] input keeps its leading dims, and the bias is added."""
    from vllm.model_executor.kernels.linear.mxfp8.aiter import (
        AiterMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )

    n, k = 1536, 6144  # qkv at TP=8
    if not rocm_aiter_ops.is_triton_gemm_afp8wfp8_tuned(n, k):
        pytest.skip(f"AITER has no tuned MXFP8 config for N={n}, K={k}")

    torch.manual_seed(0)
    layer = _make_layer(n, k, "cuda")
    kernel = AiterMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)

    x = torch.randn(2, 5, k, device="cuda", dtype=torch.bfloat16) * 0.5
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x, bias)
    flat = kernel.apply_weights(layer, x.reshape(-1, k))

    assert out.shape == (2, 5, n)
    torch.testing.assert_close(out.reshape(-1, n), flat + bias)
