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
These tests call the builder.
"""

import pytest
import torch

from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the metadata builder needs CUDA"
)

RATIO = 8
STORAGE_BLOCK = 98
PAD = -1


def _metadata(num_tokens, seq_len, slot_mapping, device="cuda"):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        num_reqs=1,
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        max_seq_len=seq_len,
        block_table_tensor=torch.arange(8, dtype=torch.int32, device=device).unsqueeze(
            0
        ),
        slot_mapping=slot_mapping,
    )


def _sharded_main_slots(num_tokens, world, rank, interleave=1, device="cuda"):
    """What the DCP slot-mapping kernel writes: PAD where the rank is not owner."""
    positions = torch.arange(num_tokens, device=device)
    owned = (positions // interleave) % world == rank
    return torch.where(
        owned,
        positions.to(torch.int64),
        torch.full_like(positions, PAD, dtype=torch.int64),
    )


def _build(meta, num_tokens, device="cuda"):
    from vllm.models.qwen4_exp.common.qsa_cache import build_qsa_metadata_triton

    buffers = [
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.zeros(num_tokens, dtype=torch.int64, device=device),
    ]
    _, _, _, slots = build_qsa_metadata_triton(
        meta, *buffers, storage_block_size=STORAGE_BLOCK, compress_ratio=RATIO
    )
    return slots


@requires_gpu
@pytest.mark.parametrize("world,interleave", [(2, 1), (2, 4), (4, 1)])
def test_every_rank_stores_every_compressed_state(world, interleave):
    """The defect: at world 2 and interleave 1, rank 0 stored nothing."""
    num_tokens, seq_len = 256, 256
    expected = (torch.arange(num_tokens) + 1) % RATIO == 0
    expected_count = int(expected.sum())
    assert expected_count > 0

    for rank in range(world):
        main = _sharded_main_slots(num_tokens, world, rank, interleave)
        slots = _build(_metadata(num_tokens, seq_len, main), num_tokens)
        stored = (slots >= 0).sum().item()
        assert stored == expected_count, (
            f"rank {rank} of {world} stored {stored} states, expected "
            f"{expected_count}. A replicated cache must be written by every rank."
        )


@requires_gpu
def test_one_rank_is_unchanged():
    """The fix must be a no-op without DCP, where nothing is PAD anyway."""
    num_tokens, seq_len = 256, 256
    whole = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    sharded = _sharded_main_slots(num_tokens, 1, 0)
    assert torch.equal(whole, sharded), "world 1 owns everything"

    slots = _build(_metadata(num_tokens, seq_len, whole), num_tokens)
    assert (slots >= 0).sum().item() == num_tokens // RATIO


@requires_gpu
@pytest.mark.parametrize("world", [1, 2])
def test_the_fused_and_torch_paths_agree(world):
    """Both paths carried the gate. Both must drop it, and stay identical."""
    from vllm.models.qwen4_exp.common.qsa_cache import _build_qsa_metadata_torch

    num_tokens, seq_len, device = 256, 256, "cuda"
    main = _sharded_main_slots(num_tokens, world, 0)
    fused = _build(_metadata(num_tokens, seq_len, main), num_tokens)

    buffers = [
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.zeros(num_tokens, dtype=torch.int64, device=device),
    ]
    _, _, _, torch_slots = _build_qsa_metadata_torch(
        _metadata(num_tokens, seq_len, main),
        *buffers,
        storage_block_size=STORAGE_BLOCK,
        compress_ratio=RATIO,
    )
    torch.testing.assert_close(fused, torch_slots)
