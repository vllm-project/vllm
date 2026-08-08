# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``XPUFp8BlockScaledMMKernel`` weight processing + block GEMM.

Covers the N-padding contract: oneDNN indexes the weight in ``block_n``-sized
tiles, one per scale block along N, so a weight whose N is not a multiple of
``block_n`` must be padded out to the scale's block grid at load time and the
extra columns sliced off the output.

DeepSeek-V3 hits this with ``kv_a_proj_with_mqa`` (N=576 = kv_lora_rank 512 +
qk_rope_head_dim 64) and the fused ``fused_qkv_a_proj`` (N=2112 = q_lora_rank
1536 + 576).

Run ``pytest tests/kernels/quantization/test_xpu_fp8_block_scaled_mm.py -v``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

if not current_platform.is_xpu():
    pytest.skip("skipping XPU-only tests", allow_module_level=True)

# Importing the kernels package registers the ``_xpu_C`` custom ops.
try:  # noqa: SIM105
    import vllm_xpu_kernels._C  # noqa: F401
except Exception:
    pass

if not hasattr(torch.ops, "_xpu_C") or not hasattr(torch.ops._xpu_C, "fp8_gemm"):
    pytest.skip("fp8_gemm op not available", allow_module_level=True)

from vllm.model_executor.kernels.linear.scaled_mm.xpu import (  # noqa: E402
    XPUFp8BlockScaledMMKernel,
)

BLOCK_N = 128
BLOCK_K = 128
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def quant_weight_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize weight [N, K] -> fp8 [N, K] + scales [n_blocks, k_blocks].

    Returns the scale in *checkpoint* layout ([n_blocks, k_blocks]), which is
    what process_weights_after_loading expects to receive.
    """
    N, K = w.shape
    n_t, k_t = cdiv(N, BLOCK_N), cdiv(K, BLOCK_K)
    wp = torch.nn.functional.pad(
        w.float(), (0, k_t * BLOCK_K - K, 0, n_t * BLOCK_N - N)
    )
    wb = wp.view(n_t, BLOCK_N, k_t, BLOCK_K)
    scale = (wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp(min=1e-12)
    q = (wb / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    q = q.view(n_t * BLOCK_N, k_t * BLOCK_K)[:N, :K].contiguous()
    return q, scale.view(n_t, k_t).contiguous()


def dequant_weight_block(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    N, K = q.shape
    n_t, k_t = scale.shape
    qp = torch.nn.functional.pad(
        q.float(), (0, k_t * BLOCK_K - K, 0, n_t * BLOCK_N - N)
    )
    wb = qp.view(n_t, BLOCK_N, k_t, BLOCK_K) * scale.view(n_t, 1, k_t, 1)
    return wb.view(n_t * BLOCK_N, k_t * BLOCK_K)[:N, :K]


def quant_act_per_token_group(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group quantize activation [M, K] -> fp8 + scales [M, k_blocks]."""
    M, K = x.shape
    assert K % BLOCK_K == 0
    xg = x.float().view(M, K // BLOCK_K, BLOCK_K)
    scale = (xg.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    q = (xg / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).view(M, K)
    return q, scale.squeeze(-1).contiguous()


def dequant_act_per_token_group(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = q.shape
    return (q.float().view(M, K // BLOCK_K, BLOCK_K) * scale.unsqueeze(-1)).view(M, K)


def _make_kernel(out_dtype: torch.dtype) -> XPUFp8BlockScaledMMKernel:
    """Build a kernel without the heavy config/QuantFP8 setup.

    process_weights_after_loading and apply_block_scaled_mm only read
    ``weight_group_shape`` and ``config.out_dtype``.
    """
    kernel = XPUFp8BlockScaledMMKernel.__new__(XPUFp8BlockScaledMMKernel)
    kernel.weight_group_shape = GroupShape(BLOCK_N, BLOCK_K)
    kernel.config = SimpleNamespace(out_dtype=out_dtype)
    return kernel


def _make_layer(weight: torch.Tensor, scale: torch.Tensor) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
    return layer


# (N, K). N=576 and N=2112 are not multiples of BLOCK_N and exercise padding.
NK_SIZES = [
    (256, 512),
    (512, 1024),
    (576, 512),
    (2112, 512),
]
M_SIZES = [1, 16]


@pytest.mark.parametrize("M", M_SIZES)
@pytest.mark.parametrize("N,K", NK_SIZES)
def test_xpu_fp8_block_scaled_mm(M: int, N: int, K: int):
    """Block-scaled fp8 GEMM matches a dequantized reference, padded or not."""
    torch.manual_seed(42)
    out_dtype = torch.bfloat16
    device = torch.device("xpu")

    x = torch.randn(M, K, dtype=torch.float32) / (K**0.5)
    w = torch.randn(N, K, dtype=torch.float32) / (K**0.5)

    q_x, s_x = quant_act_per_token_group(x)
    q_w, s_w = quant_weight_block(w)

    ref = dequant_act_per_token_group(q_x, s_x) @ dequant_weight_block(q_w, s_w).t()

    layer = _make_layer(q_w.to(device), s_w.to(device))
    kernel = _make_kernel(out_dtype)
    kernel.process_weights_after_loading(layer)

    # The weight is padded out to the scale's block grid; the scale is not,
    # because ceil(N/block_n) is unchanged by padding N up to a block boundary.
    n_blocks = cdiv(N, BLOCK_N)
    assert layer.weight.shape == (n_blocks * BLOCK_N, K)
    assert layer.weight_scale_inv.shape == (n_blocks, cdiv(K, BLOCK_K))
    assert kernel._output_size == N

    out = kernel.apply_block_scaled_mm(
        A=q_x.to(device),
        B=layer.weight,
        As=s_x.to(device),
        Bs=layer.weight_scale_inv,
    )

    # Padding must be sliced back off, so the caller never sees it.
    assert out.shape == (M, N)
    torch.testing.assert_close(out.float().cpu(), ref.float(), rtol=0.05, atol=0.05)


@pytest.mark.parametrize("N,K", [(576, 512), (2112, 512)])
def test_xpu_fp8_block_scaled_mm_pads_unaligned_n(N: int, K: int):
    """Padding is applied exactly when N is not on a block boundary."""
    torch.manual_seed(0)
    device = torch.device("xpu")

    assert N % BLOCK_N != 0, "shape must be unaligned to be meaningful"

    q_w, s_w = quant_weight_block(torch.randn(N, K, dtype=torch.float32))
    layer = _make_layer(q_w.to(device), s_w.to(device))
    _make_kernel(torch.bfloat16).process_weights_after_loading(layer)

    assert layer.weight.shape[0] == cdiv(N, BLOCK_N) * BLOCK_N > N
    # The padded rows must be zero so they cannot perturb the live output.
    assert layer.weight.data[N:].float().abs().sum() == 0
