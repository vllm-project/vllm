# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every rank must store every compressed selector state.

The selector cache is replicated. The main KV cache is sharded across DCP
ranks, so its slot mapping holds PAD for a position the rank does not own.
Gating the selector's write on that mapping made one rank store nothing,
because a compressed state lands where ``(position + 1) % ratio == 0`` and
every such position is odd at ratio 8.

The equivalence test in test_qsa_dcp_equivalence.py cannot see this. It builds
each rank's cache by hand from the ownership rule and never calls the builder.
These tests call the builder, both of them.
"""

import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_cache import (
    _build_qsa_metadata_torch,
    build_qsa_metadata_triton,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the metadata builder needs CUDA"
)

RATIO = 8
STORAGE_BLOCK = 98
# Enough tokens that the compressed positions span several logical blocks, so
# the block-table lookup is exercised rather than always landing in block 0.
NUM_TOKENS = 2048
PAD = -1

BUILDERS = [build_qsa_metadata_triton, _build_qsa_metadata_torch]


def _metadata(main_slots):
    query_start_loc = torch.tensor([0, NUM_TOKENS], dtype=torch.int32, device="cuda")
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([NUM_TOKENS], dtype=torch.int32, device="cuda"),
        num_reqs=1,
        num_actual_tokens=NUM_TOKENS,
        max_query_len=NUM_TOKENS,
        max_seq_len=NUM_TOKENS,
        block_table_tensor=torch.arange(64, dtype=torch.int32, device="cuda").unsqueeze(
            0
        ),
        slot_mapping=main_slots,
    )


def _sharded_main_slots(world, rank, interleave=1):
    """What the DCP slot-mapping kernel writes: PAD where the rank is not owner.

    Transcribed from the `is_local` test in vllm/v1/worker/block_table.py.
    """
    positions = torch.arange(NUM_TOKENS, device="cuda")
    owned = (positions // interleave) % world == rank
    return torch.where(
        owned,
        positions.to(torch.int64),
        torch.full_like(positions, PAD, dtype=torch.int64),
    )


def _slot_mapping(builder, main_slots):
    return builder(
        _metadata(main_slots),
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        storage_block_size=STORAGE_BLOCK,
        compress_ratio=RATIO,
    )[3]


@pytest.mark.parametrize("builder", BUILDERS, ids=["triton", "torch"])
@pytest.mark.parametrize("world,interleave", [(2, 1), (2, 4), (4, 1)])
def test_every_rank_stores_every_compressed_state(builder, world, interleave):
    """The defect: at world 2 and interleave 1, rank 0 stored nothing.

    Parametrized over both builders because both carried the gate, and an
    equivalence check between them would have agreed while both were wrong.
    """
    expected = NUM_TOKENS // RATIO

    for rank in range(world):
        slots = _slot_mapping(builder, _sharded_main_slots(world, rank, interleave))
        stored = int((slots >= 0).sum())
        assert stored == expected, (
            f"rank {rank} of {world} stored {stored} states, expected {expected}. "
            "A replicated cache must be written by every rank."
        )


@pytest.mark.parametrize("builder", BUILDERS, ids=["triton", "torch"])
def test_one_rank_is_unchanged(builder):
    """The fix must be a no-op without DCP, where nothing is PAD anyway."""
    whole = torch.arange(NUM_TOKENS, dtype=torch.int64, device="cuda")
    slots = _slot_mapping(builder, whole)
    assert int((slots >= 0).sum()) == NUM_TOKENS // RATIO


def test_the_states_land_on_distinct_slots():
    """A wrong block index would collide rather than drop, so count the slots."""
    slots = _slot_mapping(build_qsa_metadata_triton, _sharded_main_slots(2, 0))
    stored = slots[slots >= 0]
    assert stored.numel() == NUM_TOKENS // RATIO
    assert torch.unique(stored).numel() == stored.numel(), "two states collided"


def test_the_padding_tail_is_still_excluded():
    """`mapped` is what excludes padding, and the fix must not weaken it.

    A cudagraph-padded batch has more actual tokens than mapped ones. States
    past the mapped bound must not be stored, whatever the main slot mapping
    holds there.
    """
    real = 1000
    query_start_loc = torch.tensor([0, real], dtype=torch.int32, device="cuda")
    metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([real], dtype=torch.int32, device="cuda"),
        num_reqs=1,
        num_actual_tokens=NUM_TOKENS,
        max_query_len=real,
        max_seq_len=real,
        block_table_tensor=torch.arange(64, dtype=torch.int32, device="cuda").unsqueeze(
            0
        ),
        # Every token owned, so only `mapped` can exclude the tail.
        slot_mapping=torch.arange(NUM_TOKENS, dtype=torch.int64, device="cuda"),
    )
    slots = build_qsa_metadata_triton(
        metadata,
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        storage_block_size=STORAGE_BLOCK,
        compress_ratio=RATIO,
    )[3]
    assert int((slots >= 0).sum()) == real // RATIO


@pytest.mark.parametrize("builder", BUILDERS, ids=["triton", "torch"])
def test_a_dummy_batch_stays_inert_under_dcp(builder):
    """The two signals must not be confused, in either direction.

    A dummy batch writes nothing even though every rank owns real positions.
    A real batch writes everything even though the mapping is full of PAD.
    """
    all_pad = torch.full((NUM_TOKENS,), PAD, dtype=torch.int64, device="cuda")
    metadata = _metadata(all_pad)
    object.__setattr__(metadata, "is_dummy_batch", True)
    slots = builder(
        metadata,
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cuda"),
        torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda"),
        storage_block_size=STORAGE_BLOCK,
        compress_ratio=RATIO,
    )[3]
    assert int((slots >= 0).sum()) == 0, "a dummy batch must write nothing"
