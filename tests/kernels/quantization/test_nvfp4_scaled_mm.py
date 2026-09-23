# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
    quant_nvfp4_tensor,
)

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
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

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]

TRITON_NVFP4_GEMM_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks/kernels/triton_nvfp4_gemm.py"
)
# (m, n, k) with k unpacked, unlike SHAPES above whose third entry is packed.
TRITON_SHAPES = [
    (128, 128, 128),
    (256, 128, 128),
    (128, 256, 256),
    (256, 256, 256),
    (512, 512, 256),
    (128, 128, 256),
]
TRITON_TAIL_M = [1, 3, 17, 150]
TRITON_TAIL_N = [100, 130]
# Multiples of 32 that are not multiples of 128.
TRITON_TAIL_K = [160, 320, 2880]
# FP4 values of poison that with_poisoned_k_tail appends to every row.
POISON_K = 256
# The Triton reference is float64, so unlike a bf16 reference it does not round
# the dequantized operands. 1e-1 matches the other NVFP4 tests: the defects
# guarded below are off by O(sqrt(k)), and the fine-grained error is held to
# CUTLASS's in test_triton_nvfp4_gemm_error_vs_cutlass.
TRITON_ATOL = TRITON_RTOL = 1e-1


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
    is_sf_linear_layout=False,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_sf,
        a_global_scale,
        dtype=dtype,
        device=device,
        block_size=block_size,
        is_sf_linear_layout=is_sf_linear_layout,
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4,
        b_sf,
        b_global_scale,
        dtype=dtype,
        device=device,
        block_size=block_size,
        is_sf_linear_layout=is_sf_linear_layout,
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
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
    out = ops.cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)


class NvFp4Operand(NamedTuple):
    fp4: torch.Tensor
    sf: torch.Tensor
    global_scale: torch.Tensor


@pytest.fixture(scope="module")
def triton_fp4_mm() -> Callable[..., torch.Tensor]:
    """triton_scaled_fp4_mm, loaded by path so no sys.path entry is needed."""
    if not HAS_TRITON:
        pytest.skip("Triton is not available.")
    spec = importlib.util.spec_from_file_location(
        "triton_nvfp4_gemm_under_test", TRITON_NVFP4_GEMM_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.triton_scaled_fp4_mm


def linear_nvfp4(rows: int, k: int, dtype: torch.dtype) -> NvFp4Operand:
    """Quantize a random [rows, k] tensor to NVFP4 with linear block scales."""
    x = torch.randn((rows, k), dtype=dtype, device=CUDA_DEVICES[0])
    return NvFp4Operand(*quant_nvfp4_tensor(x, is_sf_swizzled_layout=False))


def linear_ref(a: NvFp4Operand, b: NvFp4Operand) -> torch.Tensor:
    """Dequantize both linear-scale operands and compute A @ B^T in float64."""
    return get_ref_results(
        a.fp4,
        b.fp4,
        a.sf,
        b.sf,
        a.global_scale,
        b.global_scale,
        a.fp4.shape[0],
        b.fp4.shape[0],
        torch.float64,
        16,
        a.fp4.device,
        is_sf_linear_layout=True,
    )


def with_poisoned_k_tail(x: NvFp4Operand) -> NvFp4Operand:
    """Return views of x whose rows are followed by POISON_K poison values.

    Data padding is 0x77 (two +6.0 E2M1 values per byte) and scale padding is
    0x7F (NaN in float8_e4m3fn). The views keep the logical shapes with a wider
    row stride, so a load past a row's last K element reads poison.
    """
    rows, k_bytes = x.fp4.shape
    num_sf = x.sf.shape[1]
    device = x.fp4.device
    fp4 = torch.full(
        (rows, k_bytes + POISON_K // 2), 0x77, dtype=torch.uint8, device=device
    )
    fp4[:, :k_bytes] = x.fp4
    sf = torch.full(
        (rows, num_sf + POISON_K // 16), 0x7F, dtype=torch.uint8, device=device
    )
    sf[:, :num_sf] = x.sf.view(torch.uint8)
    return NvFp4Operand(
        fp4[:, :k_bytes], sf.view(torch.float8_e4m3fn)[:, :num_sf], x.global_scale
    )


def max_rms_error(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    err = out.double() - ref
    return err.abs().max().item(), err.square().mean().sqrt().item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", TRITON_SHAPES)
@torch.inference_mode()
def test_triton_nvfp4_gemm(
    triton_fp4_mm: Callable[..., torch.Tensor],
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> None:
    """Full tiles and full K steps match the reference in both output dtypes."""
    set_random_seed(SEEDS[0])
    m, n, k = shape
    a, b = linear_nvfp4(m, k, dtype), linear_nvfp4(n, k, dtype)
    alpha = 1.0 / (a.global_scale * b.global_scale)
    out = triton_fp4_mm(a.fp4, b.fp4, a.sf, b.sf, alpha, dtype)
    torch.testing.assert_close(
        out, linear_ref(a, b).to(dtype), atol=TRITON_ATOL, rtol=TRITON_RTOL
    )


@pytest.mark.parametrize("m", TRITON_TAIL_M)
@pytest.mark.parametrize("n", TRITON_TAIL_N)
@torch.inference_mode()
def test_triton_nvfp4_gemm_mn_tails(
    triton_fp4_mm: Callable[..., torch.Tensor], m: int, n: int
) -> None:
    """Partial M and N tiles are computed and stored only where valid.

    Guards the store mask and the program-id to tile mapping on edge tiles.
    Unmasked A/B loads past M or N only feed rows and columns the store drops,
    so they do not show in the values and this test passes on such a kernel.
    compute-sanitizer memcheck reports them only with
    PYTORCH_NO_CUDA_MEMORY_CACHING=1, since the caching allocator otherwise
    places the overread inside a live allocation.
    """
    set_random_seed(SEEDS[0])
    k = 256
    dtype = torch.bfloat16
    a, b = linear_nvfp4(m, k, dtype), linear_nvfp4(n, k, dtype)
    alpha = 1.0 / (a.global_scale * b.global_scale)
    out = triton_fp4_mm(a.fp4, b.fp4, a.sf, b.sf, alpha, dtype)
    torch.testing.assert_close(
        out, linear_ref(a, b).to(dtype), atol=TRITON_ATOL, rtol=TRITON_RTOL
    )


@pytest.mark.parametrize("k", TRITON_TAIL_K)
@torch.inference_mode()
def test_triton_nvfp4_gemm_k_tails(
    triton_fp4_mm: Callable[..., torch.Tensor], k: int
) -> None:
    """A K that is not a multiple of the K tile matches the reference.

    Unmasked loads in the last K step add the start of the next row of A, B
    and their scales (past the tensor for the last row) to every output.
    """
    set_random_seed(SEEDS[0])
    m, n = 128, 128
    dtype = torch.bfloat16
    a, b = linear_nvfp4(m, k, dtype), linear_nvfp4(n, k, dtype)
    alpha = 1.0 / (a.global_scale * b.global_scale)
    out = triton_fp4_mm(a.fp4, b.fp4, a.sf, b.sf, alpha, dtype)
    torch.testing.assert_close(
        out, linear_ref(a, b).to(dtype), atol=TRITON_ATOL, rtol=TRITON_RTOL
    )


@pytest.mark.parametrize("k", TRITON_TAIL_K)
@torch.inference_mode()
def test_triton_nvfp4_gemm_k_tail_ignores_row_padding(
    triton_fp4_mm: Callable[..., torch.Tensor], k: int
) -> None:
    """Poison past each row's last K element never reaches the output.

    The operands are row-strided views whose padding holds NaN scales, so an
    unmasked load in the last K step turns every output into NaN instead of
    depending on what the next row holds.
    """
    set_random_seed(SEEDS[0])
    m, n = 128, 128
    dtype = torch.bfloat16
    a, b = linear_nvfp4(m, k, dtype), linear_nvfp4(n, k, dtype)
    a_view, b_view = with_poisoned_k_tail(a), with_poisoned_k_tail(b)
    alpha = 1.0 / (a.global_scale * b.global_scale)
    out = triton_fp4_mm(a_view.fp4, b_view.fp4, a_view.sf, b_view.sf, alpha, dtype)
    assert torch.isfinite(out).all(), f"K={k}: padding past K reached the output"
    torch.testing.assert_close(
        out, linear_ref(a, b).to(dtype), atol=TRITON_ATOL, rtol=TRITON_RTOL
    )


@pytest.mark.skipif(
    not cutlass_fp4_supported(),
    reason="CUTLASS NVFP4 GEMM is not supported on this device.",
)
@pytest.mark.parametrize(
    "shape", [(128, 128, 128), (128, 128, 256), (256, 256, 256), (512, 512, 256)]
)
@torch.inference_mode()
def test_triton_nvfp4_gemm_error_vs_cutlass(
    triton_fp4_mm: Callable[..., torch.Tensor], shape: tuple[int, int, int]
) -> None:
    """Triton's error against the float64 reference is no larger than CUTLASS's.

    Both kernels round the same fp32 sums to bf16, so their max and RMS errors
    should nearly coincide; a K-loop defect shows up as a Triton error well
    above CUTLASS's.
    """
    set_random_seed(SEEDS[0])
    m, n, k = shape
    dtype = torch.bfloat16
    a, b = linear_nvfp4(m, k, dtype), linear_nvfp4(n, k, dtype)
    alpha = 1.0 / (a.global_scale * b.global_scale)
    ref = linear_ref(a, b)
    triton_out = triton_fp4_mm(a.fp4, b.fp4, a.sf, b.sf, alpha, dtype)
    cutlass_out = ops.cutlass_scaled_fp4_mm(
        a.fp4, b.fp4, swizzle_blockscale(a.sf), swizzle_blockscale(b.sf), alpha, dtype
    )
    t_max, t_rms = max_rms_error(triton_out, ref)
    c_max, c_rms = max_rms_error(cutlass_out, ref)
    report = (
        f"M={m} N={n} K={k} error vs float64 reference (max / rms): "
        f"triton {t_max:.3g} / {t_rms:.3g}, cutlass {c_max:.3g} / {c_rms:.3g}"
    )
    print(report)
    assert t_max <= 2 * c_max, report
    assert t_rms <= 2 * c_rms, report


@pytest.mark.skipif(
    not cutlass_fp4_supported(),
    reason="CUTLASS NVFP4 GEMM is not supported on this device.",
)
@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 256, 256), (512, 256, 128)])
@torch.inference_mode()
def test_triton_nvfp4_gemm_matches_cutlass(
    triton_fp4_mm: Callable[..., torch.Tensor], shape: tuple[int, int, int]
) -> None:
    """Triton agrees with CUTLASS on bit-identical NVFP4 operands.

    CUTLASS gets the same data with the scales swizzled by swizzle_blockscale,
    so any difference comes from the GEMMs.
    """
    set_random_seed(SEEDS[0])
    m, n, k = shape
    dtype = torch.bfloat16
    a, b = linear_nvfp4(m, k, dtype), linear_nvfp4(n, k, dtype)
    alpha = 1.0 / (a.global_scale * b.global_scale)
    cutlass_out = ops.cutlass_scaled_fp4_mm(
        a.fp4, b.fp4, swizzle_blockscale(a.sf), swizzle_blockscale(b.sf), alpha, dtype
    )
    triton_out = triton_fp4_mm(a.fp4, b.fp4, a.sf, b.sf, alpha, dtype)
    torch.testing.assert_close(
        triton_out, cutlass_out, atol=TRITON_ATOL, rtol=TRITON_RTOL
    )


@torch.inference_mode()
def test_triton_nvfp4_gemm_zero_activation(
    triton_fp4_mm: Callable[..., torch.Tensor],
) -> None:
    """An all-zero A gives exact zeros, not NaN.

    The 1e-12 guard keeps A's global scale finite, so A quantizes to zero data
    and zero block scales, which the kernel must turn into zeros.
    """
    set_random_seed(SEEDS[0])
    m, n, k = 128, 128, 256
    dtype = torch.bfloat16
    a = torch.zeros((m, k), dtype=dtype, device=CUDA_DEVICES[0])
    a_gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / (a.abs().amax().float() + 1e-12)
    a_fp4, a_sf = ops.scaled_fp4_quant(a, a_gs, is_sf_swizzled_layout=False)
    b = linear_nvfp4(n, k, dtype)
    alpha = 1.0 / (a_gs * b.global_scale)
    out = triton_fp4_mm(a_fp4, b.fp4, a_sf, b.sf, alpha, dtype)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)


@torch.inference_mode()
def test_triton_nvfp4_gemm_cuda_graph(
    triton_fp4_mm: Callable[..., torch.Tensor],
) -> None:
    """The GEMM captures in a CUDA graph with alpha as a device tensor.

    After alpha is updated in place, a replay must match an eager call. A host
    read of alpha at call time (alpha.item()) fails this test whether capture
    rejects it or bakes the captured value into the graph.
    """
    set_random_seed(SEEDS[0])
    m, n, k = 128, 256, 256
    dtype = torch.bfloat16
    a, b = linear_nvfp4(m, k, dtype), linear_nvfp4(n, k, dtype)
    alpha = 1.0 / (a.global_scale * b.global_scale)

    # The eager call also compiles the kernel before capture.
    eager = triton_fp4_mm(a.fp4, b.fp4, a.sf, b.sf, alpha, dtype)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = triton_fp4_mm(a.fp4, b.fp4, a.sf, b.sf, alpha, dtype)
    graph.replay()
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)

    alpha.mul_(2)
    graph.replay()
    eager = triton_fp4_mm(a.fp4, b.fp4, a.sf, b.sf, alpha, dtype)
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
