# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness and gating tests for the RDNA4 W8A8 block-FP8 HIP GEMM.

Covers ``torch.ops._rocm_C.w8a8_block_fp8_gemm_rdna4`` against the native
torch reference, plus the two gates the kernel is required to enforce: it must
only run on gfx1201, and only for a 128x128 quantisation block.

Run `pytest tests/kernels/quantization/test_rdna4_w8a8_fp8.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA4 W8A8 kernel is ROCm-only", allow_module_level=True)

from tests.kernels.quant_utils import native_w8a8_block_matmul  # noqa: E402
from vllm.model_executor.kernels.linear.scaled_mm.rdna4_w8a8_fp8 import (  # noqa: E402
    BLOCK_K,
    BLOCK_N,
    RDNA4W8A8Fp8BlockScaledMMKernel,
    _widen_padded_weight,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (  # noqa: E402
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
)
from vllm.platforms.rocm import on_gfx1201  # noqa: E402

DEVICE = "cuda"
BLOCK_SIZE = [BLOCK_N, BLOCK_K]

gfx1201_only = pytest.mark.skipif(
    not (
        on_gfx1201()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "w8a8_block_fp8_gemm_rdna4")
    ),
    reason="Requires gfx1201 with the w8a8_block_fp8_gemm_rdna4 op",
)

# M values chosen to land in every branch of the kernel's routing, and on both
# sides of each boundary:
#   <=16 / <=32 / <=64        the three small-M tiles
#   65..383 / 384..767 / 768+ the three large-M tiles
# 129 and 192 additionally pin the large-M grid size, which is where an
# M-rounded grid and an M-exact one disagree.
M_VALUES = [1, 16, 17, 32, 33, 64, 65, 129, 192, 384, 768]

# (N, K). 512x512 gives a SQUARE 4x4 scale grid, which is the one case where a
# transposed weight scale is shape-indistinguishable from a correct one.
# 1024 exercises the N%256 tile, 384 forces the fallback off it.
SHAPES = [(512, 512), (1024, 256), (384, 512)]


def _make_operands(M, N, K, out_dtype, seed=0, k_stride=None):
    """Build A/B/As/Bs and the reference result.

    ``Bs`` is returned in BOTH layouts: vLLM's ``[N/128, K/128]`` for the
    reference, and the kernel's ``[K/128, N/128]``.
    """
    torch.manual_seed(seed)
    fp8_dtype = current_platform.fp8_dtype()
    finfo = torch.finfo(fp8_dtype)
    fp8_max, fp8_min = finfo.max, finfo.min
    factor = 1e-2

    a_f32 = (torch.rand(M, K, dtype=torch.float32, device=DEVICE) - 0.5) * 2 * fp8_max
    A = a_f32.clamp(fp8_min, fp8_max).to(fp8_dtype)

    b_f32 = (torch.rand(N, K, dtype=torch.float32, device=DEVICE) - 0.5) * 2 * fp8_max
    B = b_f32.clamp(fp8_min, fp8_max).to(fp8_dtype)

    n_tiles = N // BLOCK_N
    k_tiles = K // BLOCK_K
    As = torch.rand(M, k_tiles, dtype=torch.float32, device=DEVICE) * factor
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=DEVICE) * factor

    ref = native_w8a8_block_matmul(A, B, As, Bs, BLOCK_SIZE, out_dtype)

    B_kernel = B
    if k_stride is not None:
        # Reproduce the VLLM_ROCM_FP8_PADDING layout: an [N, K] view of a
        # wider buffer. The op takes the full-width tensor.
        assert k_stride > K
        padded = torch.zeros(N, k_stride, dtype=fp8_dtype, device=DEVICE)
        padded[:, :K] = B
        B_kernel = padded

    return A, B_kernel, As, Bs.t().contiguous(), ref


def _run_op(A, B, As, Bs_k_major, out_dtype):
    C = torch.empty((A.shape[0], B.shape[0]), dtype=out_dtype, device=DEVICE)
    torch.ops._rocm_C.w8a8_block_fp8_gemm_rdna4(
        A, B, As, Bs_k_major, C, BLOCK_N, BLOCK_K
    )
    return C


def _rel_diff(out, ref):
    out = out.to(torch.float32)
    ref = ref.to(torch.float32)
    return (torch.mean(torch.abs(out - ref)) / torch.mean(torch.abs(ref))).item()


@gfx1201_only
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("N,K", SHAPES)
@torch.inference_mode()
def test_w8a8_block_fp8_rdna4_matmul(M, N, K):
    out_dtype = torch.bfloat16
    A, B, As, Bs, ref = _make_operands(M, N, K, out_dtype)
    out = _run_op(A, B, As, Bs, out_dtype)

    assert out.shape == (M, N)
    assert torch.isfinite(out.to(torch.float32)).all()
    assert _rel_diff(out, ref) < 1e-3


@gfx1201_only
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("M", [1, 64, 256])
@torch.inference_mode()
def test_w8a8_block_fp8_rdna4_out_dtypes(M, out_dtype):
    N, K = 512, 256
    A, B, As, Bs, ref = _make_operands(M, N, K, out_dtype)
    out = _run_op(A, B, As, Bs, out_dtype)

    assert out.dtype == out_dtype
    assert _rel_diff(out, ref) < 1e-3


@gfx1201_only
@pytest.mark.parametrize("M", [1, 64, 256])
@torch.inference_mode()
def test_w8a8_block_fp8_rdna4_padded_weight_stride(M):
    """A weight row stride wider than K must give the same result as K.

    This is the VLLM_ROCM_FP8_PADDING layout; reading it with the wrong stride
    walks into the padding and produces uncorrelated values rather than an
    error, so the check has to be numeric.
    """
    N, K = 512, 256
    out_dtype = torch.bfloat16

    A, B_wide, As, Bs, ref = _make_operands(M, N, K, out_dtype, k_stride=K + 256)
    assert B_wide.shape == (N, K + 256)
    out = _run_op(A, B_wide, As, Bs, out_dtype)

    assert _rel_diff(out, ref) < 1e-3


@gfx1201_only
@torch.inference_mode()
def test_w8a8_block_fp8_rdna4_square_scale_grid_is_not_transposed():
    """N/128 == K/128 is the case a shape-based orientation guess gets wrong.

    Feeding the op a transposed Bs must change the result; if it does not, the
    kernel is ignoring the layout it was given.
    """
    M, N, K = 64, 512, 512
    out_dtype = torch.bfloat16
    A, B, As, Bs, ref = _make_operands(M, N, K, out_dtype)
    assert Bs.shape[0] == Bs.shape[1]

    good = _run_op(A, B, As, Bs, out_dtype)
    assert _rel_diff(good, ref) < 1e-3

    bad = _run_op(A, B, As, Bs.t().contiguous(), out_dtype)
    assert _rel_diff(bad, ref) > 1e-2


@gfx1201_only
@pytest.mark.parametrize("group_n,group_k", [(64, 128), (128, 64), (64, 64)])
@torch.inference_mode()
def test_w8a8_block_fp8_rdna4_rejects_other_block_sizes(group_n, group_k):
    M, N, K = 64, 512, 256
    A, B, As, Bs, _ = _make_operands(M, N, K, torch.bfloat16)
    C = torch.empty((M, N), dtype=torch.bfloat16, device=DEVICE)

    with pytest.raises(RuntimeError, match="quantisation block"):
        torch.ops._rocm_C.w8a8_block_fp8_gemm_rdna4(A, B, As, Bs, C, group_n, group_k)


@gfx1201_only
@torch.inference_mode()
def test_w8a8_block_fp8_rdna4_rejects_wrong_scale_shape():
    M, N, K = 64, 512, 256
    A, B, As, Bs, _ = _make_operands(M, N, K, torch.bfloat16)
    C = torch.empty((M, N), dtype=torch.bfloat16, device=DEVICE)

    # vLLM's [N/128, K/128] orientation is not what the kernel indexes, and
    # here it is not square either, so it must be rejected outright.
    with pytest.raises(RuntimeError, match="k-major"):
        torch.ops._rocm_C.w8a8_block_fp8_gemm_rdna4(
            A, B, As, Bs.t().contiguous(), C, BLOCK_N, BLOCK_K
        )


def test_widen_padded_weight_recovers_full_stride():
    """Runs anywhere: pure tensor-layout logic, no kernel involved."""
    N, K, pad = 8, 128, 256
    buf = torch.arange(N * (K + pad), dtype=torch.uint8).reshape(N, K + pad)
    view = buf[:, :K]

    assert not view.is_contiguous()
    widened = _widen_padded_weight(view)

    assert widened.shape == (N, K + pad)
    assert widened.data_ptr() == view.data_ptr()
    assert torch.equal(widened[:, :K], view)


def test_widen_padded_weight_passes_contiguous_through():
    w = torch.zeros(8, 128, dtype=torch.uint8)
    assert _widen_padded_weight(w) is w


def _make_config(
    *,
    weight_quant_key=kFp8Static128BlockSym,
    activation_quant_key=kFp8Dynamic128Sym,
    weight_shape=(512, 256),
    out_dtype=torch.bfloat16,
) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        weight_shape=weight_shape,
        input_dtype=torch.bfloat16,
        out_dtype=out_dtype,
    )


def test_is_supported_requires_env_var(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_HIP_W8A8", "0")
    ok, reason = RDNA4W8A8Fp8BlockScaledMMKernel.is_supported()
    assert not ok
    assert "VLLM_ROCM_USE_HIP_W8A8" in reason


@pytest.mark.skipif(on_gfx1201(), reason="checks the non-gfx1201 rejection path")
def test_is_supported_requires_gfx1201(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_HIP_W8A8", "1")
    ok, reason = RDNA4W8A8Fp8BlockScaledMMKernel.is_supported()
    assert not ok
    assert "gfx1201" in reason


@gfx1201_only
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # Only the 128x128 block is implemented, so neither a 64-wide
        # activation group nor a per-token one is accepted.
        ({"activation_quant_key": kFp8Dynamic64Sym}, "activation group_shape"),
        ({"activation_quant_key": kFp8DynamicTokenSym}, "activation group_shape"),
        ({"weight_quant_key": kFp8StaticChannelSym}, "weight group_shape"),
        # Both kernel families require 128-aligned N and K.
        ({"weight_shape": (500, 256)}, "must be divisible"),
        ({"weight_shape": (512, 250)}, "must be divisible"),
    ],
)
def test_can_implement_rejects_unsupported_layers(monkeypatch, kwargs, expected):
    monkeypatch.setenv("VLLM_ROCM_USE_HIP_W8A8", "1")
    ok, reason = RDNA4W8A8Fp8BlockScaledMMKernel.can_implement(_make_config(**kwargs))
    assert not ok
    assert expected in reason


@gfx1201_only
def test_can_implement_accepts_a_supported_layer(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_HIP_W8A8", "1")
    ok, reason = RDNA4W8A8Fp8BlockScaledMMKernel.can_implement(_make_config())
    assert ok, reason
