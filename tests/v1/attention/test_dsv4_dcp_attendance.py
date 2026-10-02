# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4 DCP decode-metadata ownership tests.

The decode metadata builders hand each DCP rank an owned-only, compacted
view of what it attends: SWA window positions (replicated storage,
partitioned attendance) and compressed-cache candidates (sharded storage,
localized slots). For each kernel two properties hold:

* per-rank output matches a pure-python ownership reference, order included
  (the compaction is deterministic, position-ascending);
* the W ranks' rows stitched back together reproduce the dcp=1 row exactly.

The kernels need a GPU; the DCP support-gate tests run anywhere.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.attention import DeepseekV4Attention

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


# ---------------------------------------------------------------------------
# Support gate
# ---------------------------------------------------------------------------


class _PlatformSupported(DeepseekV4Attention):
    @classmethod
    def _dcp_platform_supported(cls) -> bool:
        return True


def _dcp_config(interleave=1, spec=None, eager=True, pcp=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            cp_kv_cache_interleave_size=interleave,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=2,
        ),
        speculative_config=spec,
        model_config=SimpleNamespace(enforce_eager=eager),
    )


@pytest.mark.parametrize(
    ("cls", "config_kwargs", "message_part"),
    [
        (DeepseekV4Attention, {}, "does not implement"),
        (_PlatformSupported, dict(interleave=4), "cp_kv_cache_interleave_size"),
        (_PlatformSupported, dict(spec=object()), "Speculative decoding"),
        (_PlatformSupported, dict(eager=False), "CUDA graph"),
        (_PlatformSupported, dict(pcp=2), "Prefill Context Parallelism"),
    ],
)
def test_dcp_support_gate_clauses(cls, config_kwargs, message_part):
    with pytest.raises(NotImplementedError, match=message_part):
        cls._check_dcp_support(_dcp_config(**config_kwargs))


def test_dcp_support_gate_passes_when_supported():
    _PlatformSupported._check_dcp_support(_dcp_config())


# ---------------------------------------------------------------------------
# Ownership references (pure python, independent of the kernels)
# ---------------------------------------------------------------------------


def _owned(pos: int, world: int, rank: int, interleave: int) -> bool:
    return (pos // interleave) % world == rank


def _local_slot(
    pos: int, block_table_row: list[int], block_size: int, world: int, interleave: int
) -> int:
    """cp_global_to_local_block in python: logical position -> local slot."""
    virtual_block = block_size * world
    block_idx = pos // virtual_block
    virtual_offset = pos - block_idx * virtual_block
    block_offset = (
        virtual_offset // (world * interleave)
    ) * interleave + virtual_offset % interleave
    return block_table_row[block_idx] * block_size + block_offset


def _global_slot(pos: int, block_table_row: list[int], block_size: int) -> int:
    return block_table_row[pos // block_size] * block_size + pos % block_size


# ---------------------------------------------------------------------------
# SWA attendance (replicated storage: global slots, partitioned attendance)
# ---------------------------------------------------------------------------

_WORLDS = [2, 4]


def _run_swa_kernel(device, world, rank, *, window, block_size, seq_lens, width):
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _COMPUTE_SWA_INDICES_AND_LENS_KERNEL,
    )

    num_tokens = len(seq_lens)
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    block_table = torch.arange(
        100, 100 + num_tokens * max_blocks, dtype=torch.int32, device=device
    ).reshape(num_tokens, max_blocks)
    swa_indices = torch.full((num_tokens, width), -7, dtype=torch.int32, device=device)
    swa_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    # One decode token per request.
    query_start_loc = torch.arange(num_tokens + 1, dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    token_to_req = torch.arange(num_tokens, dtype=torch.int32, device=device)
    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    replay_start = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(
        swa_indices,
        swa_lens,
        window,
        width,
        swa_lens,  # unused (HAS_IMAGE=False)
        swa_lens,  # unused (HAS_IMAGE=False)
        query_start_loc,
        seq_lens_t,
        token_to_req,
        is_valid,
        block_table,
        block_size,
        replay_start,
        num_tokens=num_tokens,
        token_offset=0,
        dcp_world_size=world,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=1,
    )
    return swa_indices.cpu(), swa_lens.cpu(), block_table.cpu()


@GPU
@pytest.mark.parametrize("world", _WORLDS)
def test_swa_attendance_partition(world):
    device = torch.device("cuda")
    window, block_size, width = 8, 4, 8
    seq_lens = [13, 21, 3]  # incl. one shorter than the window

    base_idx, base_lens, block_table = _run_swa_kernel(
        device, 1, 0, window=window, block_size=block_size,
        seq_lens=seq_lens, width=width,
    )
    per_rank = [
        _run_swa_kernel(
            device, world, rank, window=window, block_size=block_size,
            seq_lens=seq_lens, width=width,
        )
        for rank in range(world)
    ]

    for tok, seq_len in enumerate(seq_lens):
        pos = seq_len - 1
        start = max(pos - (window - 1), 0)
        positions = list(range(start, pos + 1))
        table_row = block_table[tok].tolist()
        assert base_lens[tok] == len(positions)

        stitched = []
        for rank, (idx, lens, _) in enumerate(per_rank):
            owned_pos = [p for p in positions if _owned(p, world, rank, 1)]
            expected = [_global_slot(p, table_row, block_size) for p in owned_pos]
            n = int(lens[tok])
            assert n == len(expected), f"tok {tok} rank {rank}"
            # Compacted prefix in position order; -1 tail.
            assert idx[tok, :n].tolist() == expected
            assert (idx[tok, n:] == -1).all()
            stitched += expected
        assert sorted(stitched) == sorted(base_idx[tok, : base_lens[tok]].tolist())


# ---------------------------------------------------------------------------
# C4A candidate localization (sharded storage: owned-only, local slots)
# ---------------------------------------------------------------------------


def _c4a_case(device):
    # Candidate ids in compressed-slot space, -1 padded; one row crams all
    # candidates into rank-0 ownership at world=2/4 (pigeonhole edge).
    topk_indices = torch.tensor(
        [
            [5, 0, 12, 3, -1, -1, -1, -1],
            [8, 16, 0, 4, 12, -1, -1, -1],  # all even: rank0-owned at any world
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    block_size = 4  # compressed slots per block
    num_tokens = topk_indices.shape[0]
    block_table = torch.arange(
        50, 50 + num_tokens * 8, dtype=torch.int32, device=device
    ).reshape(num_tokens, 8)
    token_to_req = torch.arange(num_tokens, dtype=torch.int32, device=device)
    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    return topk_indices, block_table, block_size, token_to_req, is_valid


@GPU
@pytest.mark.parametrize("world", _WORLDS)
def test_c4a_candidate_localization(world):
    from vllm.models.deepseek_v4.amd.rocm import (
        compute_global_topk_ragged_indices_and_indptr,
    )

    device = torch.device("cuda")
    topk_indices, block_table, block_size, token_to_req, is_valid = _c4a_case(device)
    num_tokens = topk_indices.shape[0]
    rows = topk_indices.tolist()

    for rank in range(world):
        ragged, indptr, lens = compute_global_topk_ragged_indices_and_indptr(
            topk_indices,
            token_to_req,
            block_table,
            block_size,
            is_valid,
            dcp_world_size=world,
            dcp_rank=rank,
            cp_kv_cache_interleave_size=1,
        )
        indptr = indptr.cpu()
        ragged = ragged.cpu()
        for tok in range(num_tokens):
            owned_ids = [
                c for c in rows[tok] if c >= 0 and _owned(c, world, rank, 1)
            ]
            expected = [
                _local_slot(c, block_table[tok].tolist(), block_size, world, 1)
                for c in owned_ids
            ]
            assert int(lens[tok]) == len(expected)
            got = ragged[indptr[tok] : indptr[tok + 1]].tolist()
            assert got == expected, f"tok {tok} rank {rank}"

    # Pigeonhole row: every candidate owned by rank 0.
    _, _, lens0 = compute_global_topk_ragged_indices_and_indptr(
        topk_indices, token_to_req, block_table, block_size, is_valid,
        dcp_world_size=world, dcp_rank=0, cp_kv_cache_interleave_size=1,
    )
    assert int(lens0[1]) == 5
    _, _, lens1 = compute_global_topk_ragged_indices_and_indptr(
        topk_indices, token_to_req, block_table, block_size, is_valid,
        dcp_world_size=world, dcp_rank=1, cp_kv_cache_interleave_size=1,
    )
    assert int(lens1[1]) == 0


# ---------------------------------------------------------------------------
# C128A enumeration (sharded storage: owned-only, local slots)
# ---------------------------------------------------------------------------


@GPU
@pytest.mark.parametrize("world", _WORLDS)
def test_c128a_owned_enumeration(world):
    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

    device = torch.device("cuda")
    compress_ratio = 128
    block_size = 2  # compressed slots per block
    # build_c128a_topk_metadata requires _C128A_TOPK_ALIGNMENT (128) multiples.
    max_compressed = 128
    positions = torch.tensor([1023, 2047, 127], dtype=torch.int64, device=device)
    num_tokens = positions.shape[0]
    block_table = torch.arange(
        30, 30 + num_tokens * 8, dtype=torch.int32, device=device
    ).reshape(num_tokens, 8)
    token_to_req = torch.arange(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device=device)

    def run(w, r):
        global_buf = torch.full(
            (num_tokens, max_compressed), -7, dtype=torch.int32, device=device
        )
        lens_buf = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        prefill_buf = torch.empty(
            (1, max_compressed), dtype=torch.int32, device=device
        )
        dense, lens, _ = build_c128a_topk_metadata(
            positions,
            compress_ratio,
            num_tokens,  # all decode
            token_to_req,
            block_table,
            block_size,
            slot_mapping,
            global_buf,
            lens_buf,
            prefill_buf,
            max_compressed_tokens=max_compressed,
            dcp_world_size=w,
            dcp_rank=r,
            cp_kv_cache_interleave_size=1,
        )
        return dense.cpu(), lens.cpu()

    base_dense, base_lens = run(1, 0)

    for tok in range(num_tokens):
        num_compressed = min(
            (int(positions[tok]) + 1) // compress_ratio, max_compressed
        )
        table_row = block_table[tok].tolist()
        assert int(base_lens[tok]) == num_compressed

        stitched = []
        for rank in range(world):
            dense, lens = run(world, rank)
            owned_pos = [
                p for p in range(num_compressed) if _owned(p, world, rank, 1)
            ]
            expected = [
                _local_slot(p, table_row, block_size, world, 1) for p in owned_pos
            ]
            n = int(lens[tok])
            assert n == len(expected), f"tok {tok} rank {rank}"
            assert dense[tok, :n].tolist() == expected
            assert (dense[tok, n:] == -1).all()
            stitched += owned_pos
        assert sorted(stitched) == list(range(num_compressed))
