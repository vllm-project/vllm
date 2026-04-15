# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype

from vllm import _custom_ops as ops
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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gemm_batch_invariant(
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
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

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
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        alpha,
        dtype,
        batch_invariant=True,
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)


# Shapes that cross the normal M-based dispatch thresholds (M16 / M256 /
# default) to verify the batch-invariant kernel is truly identical regardless
# of batch size.
CONSISTENCY_SHAPES = [
    # (M, N, packed_k) — packed_k * 2 = real K.
    # Large K forces multiple K-tile iterations inside the mainloop, which is
    # where different tile/cluster configs can diverge in accumulation order.
    # M=256 (M256 tier) vs M=1 (M16 tier)
    (256, 128, 4096),
    # M=512 (default tier) vs M=1 (M16 tier), crosses M256→default boundary
    (512, 256, 4096),
    # Large square-ish layer resembling real LLM projections
    (256, 256, 2048),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", CONSISTENCY_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gemm_batch_invariant_consistency(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    """All shared rows between a full-batch and a half-batch run must be bitwise
    identical when batch_invariant=True."""
    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2

    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

    # Full batch (M rows)
    a_fp4_full, a_sf_full = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    out_full = ops.cutlass_scaled_fp4_mm(
        a_fp4_full,
        b_fp4,
        a_sf_full,
        b_scale_interleaved,
        alpha,
        dtype,
        batch_invariant=True,
    )

    # Run each row individually and check it matches the full-batch output.
    # Re-quantize each row with the *same* global scale so the FP4 payload is
    # identical; scale-factor tensor dimensions depend on M (128-row tile
    # padding/swizzling), so we call scaled_fp4_quant per row.
    for i in range(m):
        a_row = a_dtype[i : i + 1]
        a_fp4_row, a_sf_row = ops.scaled_fp4_quant(a_row, a_global_scale)
        out_row = ops.cutlass_scaled_fp4_mm(
            a_fp4_row,
            b_fp4,
            a_sf_row,
            b_scale_interleaved,
            alpha,
            dtype,
            batch_invariant=True,
        )

        assert torch.equal(out_full[i], out_row[0]), (
            f"batch_invariant: row {i} differs between M={m} and M=1: "
            f"max_abs_diff={(out_full[i] - out_row[0]).abs().max().item()}"
        )
