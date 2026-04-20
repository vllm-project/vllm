# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 CUTLASS GEMM tests that require ``VLLM_BATCH_INVARIANT=1``.

Must run in a **fresh** pytest process:

    pytest tests/v1/determinism/test_nvfp4_batch_invariant_scaled_mm.py -v

Do not share a session with ``tests/kernels/quantization/test_nvfp4_scaled_mm.py``:
the native code caches whether batch invariance is enabled on the first GEMM, and
if ``VLLM_BATCH_INVARIANT`` was not set at that moment, it stays disabled for the
rest of the process.

The reference correctness test is included here (not only in the default-path
module) because ``VLLM_BATCH_INVARIANT=1`` potentially activates a different kernel.
The two tests are complementary: the reference check catches absolute correctness
bugs; the batch-invariance check catches schedule-dependent bugs that affect
full-batch and single-row runs equally.
"""

import os

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
)
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)


CONSISTENCY_SHAPES = [
    (256, 128, 4096),
    (512, 256, 4096),
    (256, 256, 2048),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", CONSISTENCY_SHAPES)
@torch.inference_mode()
def test_nvfp4_gemm_batch_invariance(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> None:
    """Batch invariance: each row of a full-``M`` GEMM matches its ``M=1`` counterpart.

    For row ``i``, compares ``cutlass_scaled_fp4_mm`` run once over all ``M``
    rows against a separate call with ``A`` sliced to ``a_dtype[i : i+1]``.
    Catches kernels whose reduction or scheduling depends on ``M`` or adjacent
    rows. Uses larger ``CONSISTENCY_SHAPES`` than the reference test.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2  # real K (FP4 elements)

    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")

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
