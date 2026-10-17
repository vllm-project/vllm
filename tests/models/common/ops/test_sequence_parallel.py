# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SP correctness of ``mark_sp_region`` (fused/symmetric-memory and unfused
comm+GEMM kernels) against the real Column/RowParallelLinear forward."""

import pytest

from vllm.platforms import current_platform

from .sequence_parallel_test_utils import run_sp_matrix


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="requires CUDA/ROCm/XPU",
)
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize(
    "row_parallel,fused",
    [
        (False, True),
        (False, False),
        (True, True),
        (True, False),
    ],
    ids=["ag-fused", "ag-unfused", "rs-fused", "rs-unfused"],
)
def test_mark_sp_region(row_parallel, fused, world_size):
    run_sp_matrix(row_parallel, fused, world_size)
