# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CDNA4 MXFP8 linear (``RocmDotScaledMxfp8LinearKernel``).

DeepSeek V4.1 routes every attention linear through this kernel. It used to
carry two BF16-dequantize workarounds on ROCm (one for all ``.attn.`` linears,
one for the row-parallel ``wo_b``, whose comment blamed NaNs at small token
counts). These tests pin the two properties those workarounds stood in for --
finite output and MXFP8-level accuracy -- across the V4.1 attention shapes at
TP=4 and the small token counts decode actually runs.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    _mxfp8_e4m3_quantize_triton,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not current_platform.supports_mx(),
    reason="native MX linear requires CDNA4 (gfx95x)",
)

# (name, N, K) for the DeepSeek V4.1 attention linears at tensor_parallel=4:
# fused wq_a|wkv is replicated, wq_b/wo_a are column-parallel, wo_b is
# row-parallel (so K, not N, is sharded).
V41_ATTN_SHAPES = [
    ("fused_wqa_wkv", 1280 + 512, 5120),
    ("wq_b", 64 * 512 // 4, 1280),
    ("wo_a", 8 * 1024 // 4, 4096),
    ("wo_b", 5120, 8192 // 4),
]


def _make_layer(n: int, k: int, device: str) -> torch.nn.Module:
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
    weight, weight_scale = _mxfp8_e4m3_quantize_torch(weight_bf16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    return layer


@pytest.mark.parametrize("name,n,k", V41_ATTN_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("num_tokens", [1, 2, 7, 16, 31, 32, 128, 1024])
@torch.inference_mode()
def test_rocm_mxfp8_linear_matches_dequant_reference(name, n, k, num_tokens):
    """Output is finite and matches an exact MXFP8 GEMM of the same inputs.

    The reference quantizes the activation with the same quantizer and then
    contracts in FP32, so the tolerance covers only kernel/accumulation error,
    not the MXFP8 quantization error both sides share.
    """
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )

    torch.manual_seed(num_tokens)
    device = "cuda"
    layer = _make_layer(n, k, device)
    kernel = RocmDotScaledMxfp8LinearKernel(Mxfp8LinearLayerConfig(bmm_batch_size=None))
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
def test_rocm_mxfp8_linear_handles_extreme_magnitudes():
    """Rows that are all-zero or near the E4M3 range must not produce NaNs.

    A degenerate block makes the dynamic activation quantizer pick an extreme
    E8M0 scale; a zero row makes it pick the tiny-clamped one. Both feed
    ``tl.dot_scaled`` directly.
    """
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )

    torch.manual_seed(0)
    device = "cuda"
    n, k = 5120, 2048  # wo_b at TP=4
    layer = _make_layer(n, k, device)
    kernel = RocmDotScaledMxfp8LinearKernel(Mxfp8LinearLayerConfig(bmm_batch_size=None))
    kernel.process_weights_after_loading(layer)

    x = torch.randn(8, k, device=device, dtype=torch.bfloat16)
    x[0] = 0.0
    x[1] = 1e-30
    x[2] = 400.0
    x[3, ::2] = 0.0

    out = kernel.apply_weights(layer, x)
    assert torch.isfinite(out).all()
    assert (out[0] == 0).all(), "an all-zero activation row must give a zero row"


@pytest.mark.parametrize("num_rows", [1, 64, 65])
@torch.inference_mode()
def test_rocm_mxfp8_quantizer_matches_torch_on_degenerate_blocks(num_rows):
    """The fused ROCm quantizer must agree with the torch one, zeros included.

    An all-zero 32-element block clamps the E8M0 scale to 0, i.e. 2**-127 --
    subnormal in fp32 and flushed to zero on CDNA. Dividing by it turned the
    block into 0/0 == NaN, which then poisoned every output column of the GEMM
    that consumed it.
    """
    torch.manual_seed(num_rows)
    device = "cuda"
    x = torch.randn(num_rows, 128, device=device, dtype=torch.bfloat16)
    x[0] = 0.0  # all-zero row: every block clamps to scale byte 0
    x[-1, :32] = 0.0  # single zero block in an otherwise normal row
    x[-1, 32:64] = 1e-38  # denormal-ish block

    q_triton, s_triton = _mxfp8_e4m3_quantize_triton(x)
    q_torch, s_torch = _mxfp8_e4m3_quantize_torch(x)

    assert not torch.isnan(q_triton.float()).any()
    torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)
    torch.testing.assert_close(
        q_triton.view(torch.uint8), q_torch.view(torch.uint8), rtol=0, atol=0
    )
