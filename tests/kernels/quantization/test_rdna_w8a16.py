# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the RDNA W8A16 per-channel op (HIP skinny + Triton).

Run `pytest tests/kernels/quantization/test_rdna_w8a16.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

from vllm.platforms.rocm import on_gfx1x  # noqa: E402

w8a16 = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.rdna_w8a16"
)
MAX_SKINNY_BATCH_SIZE = w8a16.MAX_SKINNY_BATCH_SIZE
LDS_CAPACITY_ELEMENTS = w8a16.LDS_CAPACITY_ELEMENTS

requires_gfx1x = pytest.mark.skipif(
    not on_gfx1x(), reason="RDNA W8A16 is gfx11/gfx12 only"
)
device = "cuda"


def _quantize(n: int, k: int, dtype: torch.dtype, row_pad: int = 0):
    w = torch.randn(n, k, device=device) * 0.02
    scale = (w.abs().amax(dim=1) / 127).to(dtype)
    q = (w / scale.float()[:, None]).round().clamp(-127, 127).to(torch.int8)
    if row_pad:
        padded = torch.zeros(n, k + row_pad, dtype=torch.int8, device=device)
        padded[:, :k] = q
        q = padded[:, :k]
    return q, scale


def _reference(x, q, scale, bias):
    out = x.float() @ (q.float() * scale.float()[:, None]).t()
    if bias is not None:
        out += bias.float()
    return out


def _assert_close(out, ref, dtype):
    # The only rounding beyond the reference is the output cast.
    rtol = 2e-2 if dtype == torch.bfloat16 else 3e-3
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < rtol, f"relative error {err:.3e} >= {rtol}"


@requires_gfx1x
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M", [1, MAX_SKINNY_BATCH_SIZE, MAX_SKINNY_BATCH_SIZE + 1, 17, 64, 257]
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize(
    "N,K,row_pad",
    [(1000, 2560, 0), (4099, 2000, 0), (8192, 4096, 128)],
    ids=["ragged_n", "ragged_k", "padded_rows"],
)
def test_rdna_w8a16_apply_matches_reference(dtype, M, has_bias, N, K, row_pad):
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)
    q, scale = _quantize(N, K, dtype, row_pad)
    x = torch.randn(M, K, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device) if has_bias else None

    out = torch.ops.vllm.rdna_w8a16_apply(x, q, scale, bias, num_compute_units())

    assert out.shape == (M, N) and out.dtype == dtype
    _assert_close(out, _reference(x, q, scale, bias), dtype)


@requires_gfx1x
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", range(1, MAX_SKINNY_BATCH_SIZE + 1))
@pytest.mark.parametrize("N,K", [(151936, 2560), (4096, 4096)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_wvsplitk_int8_matches_reference(dtype, M, N, K, has_bias):
    """Cover the HIP kernel at every batch size it supports, without dispatch."""
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    if M * K > LDS_CAPACITY_ELEMENTS:
        pytest.skip("activations do not fit in LDS")
    set_random_seed(0)
    q, scale = _quantize(N, K, dtype)
    x = torch.randn(M, K, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device) if has_bias else None

    out = ops.wvSplitK_int8(q, x, scale, num_compute_units(), bias)

    assert out.shape == (M, N) and out.dtype == dtype
    ref = _reference(x, q, scale, bias)
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    assert ((out.float() - ref).norm() / ref.norm()).item() < tol


def _bad_wvsplitk_int8_inputs():
    K = 4096
    q = torch.zeros(256, K, dtype=torch.int8, device=device)
    scale = torch.ones(256, dtype=torch.float16, device=device)

    def x(m, k=K, dtype=torch.float16):
        return torch.randn(m, k, dtype=dtype, device=device)

    return {
        "batch_above_max": (q, x(MAX_SKINNY_BATCH_SIZE + 1), scale),
        "k_mismatch": (q, x(2, K // 2), scale),
        "non_contiguous_x": (q, x(K).t()[:2], scale),
        "scale_dtype": (q, x(2), scale.to(torch.bfloat16)),
        "scale_shape": (q, x(2), scale[:-1]),
        "fp16_weight": (q.half(), x(2), scale),
        "over_lds": (
            torch.zeros(256, 8192, dtype=torch.int8, device=device),
            x(MAX_SKINNY_BATCH_SIZE, 8192),
            scale,
        ),
        "bias_shape": (q, x(2), scale, scale[:-1]),
        "bias_2d": (q, x(2), scale, scale.repeat(2, 1)),
    }


@requires_gfx1x
@pytest.mark.parametrize(
    "case",
    [
        "batch_above_max",
        "k_mismatch",
        "non_contiguous_x",
        "scale_dtype",
        "scale_shape",
        "fp16_weight",
        "over_lds",
        "bias_shape",
        "bias_2d",
    ],
)
def test_wvsplitk_int8_rejects_bad_inputs(case):
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    w, x, scale, *bias = _bad_wvsplitk_int8_inputs()[case]
    with pytest.raises(RuntimeError):
        ops.wvSplitK_int8(w, x, scale, num_compute_units(), *bias)
        torch.accelerator.synchronize()


@requires_gfx1x
@pytest.mark.parametrize("M", [2, 64])
def test_triton_w8a16_matches_reference(M):
    """Cover the Triton kernel at batch sizes the op routes to HIP."""
    set_random_seed(0)
    q, scale = _quantize(2048, 1024, torch.bfloat16)
    x = torch.randn(M, 1024, dtype=torch.bfloat16, device=device)

    out = w8a16.triton_w8a16_per_channel_gemm(x, q, scale)

    _assert_close(out, _reference(x, q, scale, None), torch.bfloat16)


@requires_gfx1x
def test_triton_w8a16_weight_offsets_past_int32():
    """A [vocab, hidden] head can exceed 2^31 bytes; addressing must not wrap."""
    K = 4096
    N = 2**31 // K + 64
    if torch.accelerator.get_memory_info()[0] < 3 * N * K:
        pytest.skip("not enough free device memory")
    set_random_seed(0)
    q = torch.zeros(N, K, dtype=torch.int8, device=device)
    q[-64:] = torch.randint(-127, 128, (64, K), dtype=torch.int8, device=device)
    scale = torch.full((N,), 1e-2, dtype=torch.float16, device=device)
    x = torch.randn(MAX_SKINNY_BATCH_SIZE + 1, K, dtype=torch.float16, device=device)

    out = w8a16.triton_w8a16_per_channel_gemm(x, q, scale)

    _assert_close(
        out[:, -64:], _reference(x, q[-64:], scale[-64:], None), torch.float16
    )
    assert out[:, :-64].abs().max() == 0


@requires_gfx1x
@pytest.mark.parametrize("K", [2560, 4096])
@pytest.mark.parametrize("M", [1, MAX_SKINNY_BATCH_SIZE + 1])
def test_empty_int8_weight_layout(K, M):
    """The load-time layout is read correctly by both paths, padded or not."""
    from vllm.platforms.rocm import on_gfx1151
    from vllm.utils.platform_utils import num_compute_units

    set_random_seed(0)
    q, scale = _quantize(512, K, torch.bfloat16)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    laid_out = w8a16.empty_int8_weight(*q.shape, device=q.device).copy_(q)

    assert laid_out.shape == q.shape and torch.equal(laid_out, q)
    expect_pad = on_gfx1151() and K % 1024 == 0
    assert (laid_out.stride(0) > K) == expect_pad
    out = torch.ops.vllm.rdna_w8a16_apply(x, laid_out, scale, None, num_compute_units())
    _assert_close(out, _reference(x, q, scale, None), torch.bfloat16)


@pytest.mark.parametrize(
    "M,K,expected_path",
    [
        (
            MAX_SKINNY_BATCH_SIZE,
            LDS_CAPACITY_ELEMENTS // MAX_SKINNY_BATCH_SIZE // 16 * 16,
            "hip",
        ),
        (MAX_SKINNY_BATCH_SIZE + 1, 1024, "triton"),
        (4, 8192, "hip"),
        (4, 8192 + 16, "triton"),
    ],
    ids=["max_batch", "batch_too_large", "lds_limit", "above_lds"],
)
def test_rdna_w8a16_dispatch_boundaries(M, K, expected_path, monkeypatch):
    import vllm._custom_ops as ops

    N = 16
    x = torch.empty((M, K), dtype=torch.float16)
    w_q = torch.empty((N, K), dtype=torch.int8)
    scale = torch.empty((N,), dtype=torch.float16)
    called = []

    def fake(path):
        def run(*args, **kwargs):
            called.append(path)
            return torch.empty((M, N), dtype=x.dtype)

        return run

    monkeypatch.setattr(ops, "wvSplitK_int8", fake("hip"), raising=False)
    monkeypatch.setattr(w8a16, "triton_w8a16_per_channel_gemm", fake("triton"))

    out = w8a16._rdna_w8a16_apply_impl(x, w_q, scale, None, 40)

    assert out.shape == (M, N)
    assert called == [expected_path]
