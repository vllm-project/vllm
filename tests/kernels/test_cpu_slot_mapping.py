# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

import vllm._C  # noqa: F401, E402

from vllm.triton_utils.dispatcher import KernelOverride  # noqa: E402
from vllm.utils.cpu_triton_utils import (  # noqa: E402
    _compute_slot_mapping_kernel_impl,
)
from vllm.v1.worker.block_table import ComputeSlotMappingKernel  # noqa: E402


@pytest.mark.parametrize(
    "query_start_loc,positions,blocks,expected",
    [
        ([0, 3], [0, 1, 2], [[5, 9]], [80, 81, 82]),
        ([0, 2, 5], [15, 16, 1, 15, 16], [[5, 9], [3, 7]], [95, 144, 49, 63, 112]),
    ],
    ids=["single_request", "multiple_requests_cross_block_boundary"],
)
def test_cpu_slot_mapping_launch_preserves_request_and_block_boundaries(
    query_start_loc, positions, blocks, expected
):
    """The CPU override must receive the launch inputs in the kernel's contract."""
    owner = ComputeSlotMappingKernel()
    owner.kernel = KernelOverride(owner.kernel, _compute_slot_mapping_kernel_impl)
    starts = torch.tensor(query_start_loc, dtype=torch.int32)
    token_positions = torch.tensor(positions, dtype=torch.int64)
    block_table = torch.tensor(blocks, dtype=torch.int32)
    slots = torch.full((len(positions),), -7, dtype=torch.int64)

    owner(
        len(query_start_loc) - 1,
        len(positions),
        len(positions),
        starts,
        token_positions,
        block_table,
        block_table.stride(0),
        slots,
        16,
        1,
        0,
        1,
    )

    assert slots.tolist() == expected
