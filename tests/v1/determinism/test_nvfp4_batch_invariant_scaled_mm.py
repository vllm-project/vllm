# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 CUTLASS GEMM tests that require ``VLLM_BATCH_INVARIANT=1``.

Must run in a **fresh** pytest process with the env var set before Python starts::

    VLLM_BATCH_INVARIANT=1 pytest \
        tests/v1/determinism/test_nvfp4_batch_invariant_scaled_mm.py -v

Do not share a session with ``tests/kernels/quantization/test_nvfp4_scaled_mm.py``
(no env var) — the cached flag would reflect the default-off path.

The reference correctness test is included here (not only in the default-path
module) because ``VLLM_BATCH_INVARIANT=1`` potentially activates a different kernel.
The two tests are complementary: the reference check catches absolute correctness
bugs; the batch-invariance check catches schedule-dependent bugs that affect
full-batch and single-row runs equally.
"""

import pytest
import torch

import vllm.envs as envs
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )


if not envs.VLLM_BATCH_INVARIANT:
    pytest.skip(
        reason="Set VLLM_BATCH_INVARIANT=1 before starting pytest for this file.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
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
def test_nvfp4_gemm_correctness_vs_reference_batch_invariant_path(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    """Correctness: ``cutlass_scaled_fp4_mm`` vs dequantize-then-matmul reference.

    Complements the default-path tests by running the same reference comparison
    with ``VLLM_BATCH_INVARIANT=1`` active, ensuring the alternate kernel
    produces correct results. Covers ``(M, N, packed_K)`` shapes including
    non-aligned dims in ``PAD_SHAPES``, both fp16 and bf16 output.
    """
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
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)


CONSISTENCY_SHAPES = [
    (256, 128, 4096),
    (512, 256, 4096),
    (256, 256, 2048),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", CONSISTENCY_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gemm_batch_invariance(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    """Batch invariance: each row of a full-``M`` GEMM matches its ``M=1`` counterpart.

    For row ``i``, compares ``cutlass_scaled_fp4_mm`` run once over all ``M``
    rows against a separate call with ``A`` sliced to ``a_dtype[i : i+1]``.
    Catches kernels whose reduction or scheduling depends on ``M`` or adjacent
    rows. Uses larger ``CONSISTENCY_SHAPES`` than the reference test.
    """
    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2  # real K (FP4 elements)

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

    a_fp4_full, a_sf_full = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    out_full = ops.cutlass_scaled_fp4_mm(
        a_fp4_full,
        b_fp4,
        a_sf_full,
        b_scale_interleaved,
        alpha,
        dtype,
    )

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
        )

        assert torch.equal(out_full[i], out_row[0]), (
            f"VLLM_BATCH_INVARIANT: row {i} differs between M={m} and M=1: "
            f"max_abs_diff={(out_full[i] - out_row[0]).abs().max().item()}"
        )
