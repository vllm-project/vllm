# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU regression test for the kpool tail group's slot mapping.

The kpool tail cache holds one circular block per request, so its block table
row is orders of magnitude shorter than the sequence it serves.
`_compute_slot_mappings_kernel` indexes block tables by
`position // kernel_block_size` with an effectively unmasked load, so
this leads to a read of neighboring requests' rows (and for the later requests, outside
the buffer entirely).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.gpu.block_table import BlockTables

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="requires CUDA or ROCm",
)

KPOOL = 4
MAX_MODEL_LEN = 1 << 20
MLA_BLOCK_SIZE = 640
MAX_NUM_REQS = 4
POISON_BLOCK_ID = 1 << 20


def make_tail_spec():
    return CircularBufferSpec(
        block_size=KPOOL,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
    )


def test_kpool_tail_group_never_position_indexes_its_block_table():
    device = torch.device("cuda")
    spec = make_tail_spec()
    tail_width = get_block_table_width(
        spec.max_num_blocks_per_req(None, MAX_MODEL_LEN),
        spec.block_size,
        token_alignment=spec.block_table_token_alignment,
    )
    mla_width = get_block_table_width(MAX_MODEL_LEN // MLA_BLOCK_SIZE, MLA_BLOCK_SIZE)

    # Positions chosen against the tail row: `in_row` still lands inside
    # request 0's own row, `next_row` lands on request 1's poisoned row, and
    # `past_table` is past the whole [MAX_NUM_REQS, tail_width] allocation.
    in_row = KPOOL - 1
    next_row = tail_width * KPOOL
    past_table = MAX_NUM_REQS * tail_width * KPOOL + KPOOL
    assert past_table < MAX_MODEL_LEN
    positions = [0, in_row, next_row, past_table]

    block_tables = BlockTables(
        block_sizes=[MLA_BLOCK_SIZE, spec.block_size],
        max_num_reqs=MAX_NUM_REQS,
        max_num_batched_tokens=len(positions),
        max_num_blocks_per_group=[mla_width, tail_width],
        device=device,
        kernel_block_sizes=[MLA_BLOCK_SIZE, spec.block_size],
        slot_mapping_enabled=[True, spec.uses_slot_mapping],
    )
    mla_blocks = list(range(1, mla_width + 1))
    block_tables.append_block_ids(0, (mla_blocks, [7]), overwrite=True)
    block_tables.append_block_ids(1, ([2], [POISON_BLOCK_ID]), overwrite=True)
    block_tables.apply_staged_writes()

    idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor(
        [0, len(positions)], dtype=torch.int32, device=device
    )
    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        torch.tensor(positions, dtype=torch.int64, device=device),
        num_tokens_padded=len(positions),
    )
    torch.accelerator.synchronize()

    tail_slots = slot_mappings[1].tolist()
    assert tail_slots == [PAD_SLOT_ID] * len(positions)
    assert POISON_BLOCK_ID * KPOOL not in tail_slots

    mla_slots = slot_mappings[0].tolist()
    assert mla_slots == [
        mla_blocks[p // MLA_BLOCK_SIZE] * MLA_BLOCK_SIZE + p % MLA_BLOCK_SIZE
        for p in positions
    ]
