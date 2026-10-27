# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dummy path's buffer builds stage into the decode prologue batch.

A DP rank with no tokens still builds a dummy batch every step; its three
launches (dummy-buffer init, block-table row zeroing, slot-mapping pad fill)
must produce byte-identical buffers when staged into one prologue launch.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.decode_prologue import DecodePrologueBatch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    _fill_is_padding_kernel,
)


def snapshot_dummy_buffers(
    batch: InputBatch, buffers: InputBuffers, num_reqs: int, num_tokens: int
) -> dict[str, torch.Tensor]:
    max_num_reqs = buffers.seq_lens.shape[0]
    return {
        "idx_mapping": batch.idx_mapping.clone(),
        "expanded_local_pos": batch.expanded_local_pos.clone(),
        "cu_num_logits": batch.cu_num_logits.clone(),
        "logits_indices": batch.logits_indices.clone(),
        "seq_lens": buffers.seq_lens.clone(),
        "query_start_loc": buffers.query_start_loc.clone(),
        "input_ids": buffers.input_ids[:num_tokens].clone(),
        "positions": buffers.positions[:num_tokens].clone(),
        "is_padding": buffers.is_padding[:num_tokens].clone(),
        "seq_lens_tail": buffers.seq_lens[num_reqs:max_num_reqs].clone(),
        "qsl_tail": buffers.query_start_loc[num_reqs + 1 :].clone(),
    }


def poison_dummy_buffers(buffers: InputBuffers, num_tokens: int) -> None:
    buffers.seq_lens.fill_(-3)
    buffers.query_start_loc.fill_(-5)
    buffers.input_ids[:num_tokens].fill_(77)
    buffers.positions[:num_tokens].fill_(9)
    buffers.is_padding[:num_tokens].fill_(True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_staged_dummy_batch_build_matches_the_direct_kernel():
    device = torch.device("cuda")
    num_reqs, num_tokens = 4, 10

    direct_buffers = InputBuffers(8, 16, device)
    direct = InputBatch.make_dummy(num_reqs, num_tokens, direct_buffers)
    expected = snapshot_dummy_buffers(direct, direct_buffers, num_reqs, num_tokens)

    staged_buffers = InputBuffers(8, 16, device)
    prologue = DecodePrologueBatch(device)
    prologue.begin(decode_only=True)
    # A second row forces the batched launch; a lone row would take the
    # direct fallback path.
    scratch = torch.full(6, -11, dtype=torch.int64, device=device)
    prologue.stage_fill_i64(fuse=True, tensor=scratch, value=-1)
    staged = InputBatch.make_dummy(
        num_reqs, num_tokens, staged_buffers, prologue=prologue
    )
    prologue.flush()

    assert not prologue.open
    assert (scratch == -1).all()
    actual = snapshot_dummy_buffers(staged, staged_buffers, num_reqs, num_tokens)
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_staged_dummy_zero_and_slot_fill_match_the_direct_launches():
    device = torch.device("cuda")

    def make_tables() -> BlockTables:
        tables = BlockTables(
            block_sizes=[16, 8],
            max_num_reqs=4,
            max_num_batched_tokens=64,
            max_num_blocks_per_group=[8, 6],
            device=device,
        )
        tables.append_block_ids(
            req_index=0, new_block_ids=([1, 2], [3]), overwrite=True
        )
        tables.append_block_ids(
            req_index=1, new_block_ids=([4], [5, 6]), overwrite=True
        )
        tables.apply_staged_writes()
        idx_mapping = torch.arange(2, dtype=torch.int32, device=device)
        tables.gather_block_tables(idx_mapping, num_reqs_padded=2)
        return tables

    direct = make_tables()
    direct_dummy = direct.get_dummy_block_tables(num_reqs=1)
    direct.get_dummy_slot_mappings(num_tokens=5)
    expected_tables = [t.clone() for t in direct_dummy]
    expected_slots = direct.slot_mappings.clone()
    # A larger earlier batch leaves dirty rows past num_reqs; the zero must
    # clear those too.
    direct._num_dirty_rows = 3
    direct.get_dummy_block_tables(num_reqs=1)
    torch.accelerator.synchronize()
    expected_dirty = [t.clone() for t in direct.input_block_tables]

    staged = make_tables()
    staged._num_dirty_rows = 3
    prologue = DecodePrologueBatch(device)
    prologue.begin(decode_only=True)
    staged_dummy = staged.get_dummy_block_tables(num_reqs=1, prologue=prologue)
    staged_slots = staged.get_dummy_slot_mappings(num_tokens=5, prologue=prologue)
    prologue.flush()
    torch.accelerator.synchronize()

    assert not prologue.open
    for expected, actual in zip(expected_dirty, staged.input_block_tables):
        assert torch.equal(expected, actual)
    for expected, actual in zip(expected_tables, staged_dummy):
        assert torch.equal(expected, actual[: expected.shape[0]])
    assert torch.equal(staged.slot_mappings, expected_slots)
    assert (staged_slots == PAD_SLOT_ID).all()
    # CUDA graph invariant: same persistent tensor, not a fresh allocation.
    assert staged_dummy[0].data_ptr() == staged.input_block_tables[0].data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_fused_is_padding_fill_matches_the_two_range_fills():
    device = torch.device("cuda")
    num_tokens, num_padded = 7, 12

    fused = torch.zeros(num_padded, dtype=torch.bool, device=device)
    _fill_is_padding_kernel[(num_padded,)](
        fused, num_tokens, num_padded, BLOCK_SIZE=1024
    )

    reference = torch.zeros(num_padded, dtype=torch.bool, device=device)
    reference[:num_tokens].fill_(False)
    reference[num_tokens:].fill_(True)

    assert torch.equal(fused, reference)
