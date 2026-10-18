# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _run_prepare(
    *,
    target_positions: list[int],
    block_table_values: list[int],
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_interleave: int = 1,
    draft_dcp_size: int | None = None,
    extra_kv_groups: tuple[tuple[list[int], int], ...] = (),
    draft_group_ids: tuple[int, ...] = (0,),
):
    """Run the prepare of ``propose`` for one request. KV-cache group 0 has
    ``block_table_values`` and block size 4; ``extra_kv_groups`` adds groups of
    (block table, block size), and draft group ``i`` uses KV-cache group
    ``draft_group_ids[i]``."""
    device = torch.device("cuda")
    max_num_reqs = 4
    max_num_tokens = 16
    num_speculative_steps = 3

    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_num_tokens,), -1, dtype=torch.int32, device=device),
        positions=torch.full((max_num_tokens,), -1, dtype=torch.int64, device=device),
        is_padding=torch.zeros(max_num_tokens, dtype=torch.bool, device=device),
        query_start_loc=torch.full(
            (max_num_reqs + 1,), -1, dtype=torch.int32, device=device
        ),
        seq_lens=torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device),
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=4,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([target_positions[-1] + 1]),
        positions=torch.tensor(target_positions, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([2], dtype=torch.int32, device=device),
    )
    kv_groups = [(block_table_values, 4), *extra_kv_groups]
    query_slot_mappings = torch.full(
        (len(kv_groups), max_num_tokens), -2, dtype=torch.int64, device=device
    )
    context_positions = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_slot_mappings = torch.full(
        (len(draft_group_ids), max_num_tokens), -2, dtype=torch.int64, device=device
    )
    sample_indices = torch.full(
        (max_num_reqs * num_speculative_steps,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    sample_pos = torch.full_like(sample_indices, -1)
    sample_idx_mapping = torch.full(
        sample_indices.shape, -1, dtype=torch.int32, device=device
    )
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
    seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
    input_temperature = torch.tensor(
        [0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device
    )
    input_seeds = torch.tensor([0, 0, 17, 0], dtype=torch.int64, device=device)
    last_sampled = torch.tensor([0, 0, 99, 0], dtype=torch.int64, device=device)
    next_prefill_tokens = torch.zeros_like(last_sampled)
    block_tables = [
        torch.tensor([values], dtype=torch.int32, device=device)
        for values, _ in kv_groups
    ]

    class InputsPrepared(Exception):
        pass

    draft = SimpleNamespace(
        input_buffers=input_buffers,
        context_positions=context_positions,
        _context_slot_mappings=context_slot_mappings,
        sample_indices=sample_indices,
        sample_pos=sample_pos,
        sample_idx_mapping=sample_idx_mapping,
        temperature=temperature,
        seeds=seeds,
        hidden_states=torch.zeros(4, 1, device=device),
        prepare_context_anchor=lambda *args: None,
        query_cudagraph_manager=None,
        dp_size=1,
        dp_rank=0,
        pcp_manager=None,
        draft_kv_cache_group_id=draft_group_ids[0],
        draft_kv_cache_group_ids=list(draft_group_ids),
        _draft_group_ids=torch.tensor(
            draft_group_ids, dtype=torch.int32, device=device
        ),
        _layer_group_idx=None,
        block_tables=SimpleNamespace(
            slot_mappings=query_slot_mappings,
            input_block_table_ptrs=torch.tensor(
                [table.data_ptr() for table in block_tables],
                dtype=torch.uint64,
                device=device,
            ),
            block_table_strides=torch.tensor(
                [table.stride(0) for table in block_tables],
                dtype=torch.int64,
                device=device,
            ),
            block_sizes_tensor=torch.tensor(
                [size for _, size in kv_groups], dtype=torch.int32, device=device
            ),
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_interleave=cp_interleave,
        ),
        dcp_size=cp_size if draft_dcp_size is None else draft_dcp_size,
        parallel_drafting_token_id=123,
        num_query_per_req=num_speculative_steps,
        num_speculative_steps=num_speculative_steps,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        max_model_len=128,
        sample_from_anchor=True,
        inputs_prepared=False,
        context_side_stream=None,
    )
    draft.prepare_draft_inputs = MethodType(
        DFlashSpeculator.prepare_draft_inputs, draft
    )
    with (
        patch(
            "vllm.v1.worker.gpu.spec_decode.dflash.speculator.dispatch_cg_and_sync_dp",
            side_effect=InputsPrepared,
        ),
        pytest.raises(InputsPrepared),
    ):
        DFlashSpeculator.propose(
            draft,
            input_batch,
            {},
            {},
            draft.hidden_states,
            None,
            torch.tensor([1], dtype=torch.int32, device=device),
            torch.tensor([2], dtype=torch.int32, device=device),
            last_sampled,
            next_prefill_tokens,
            input_temperature,
            input_seeds,
        )
    torch.accelerator.synchronize()
    return SimpleNamespace(
        input_buffers=input_buffers,
        query_slot_mapping=query_slot_mappings[0].cpu(),
        query_slot_mappings=query_slot_mappings.cpu(),
        context_positions=context_positions.cpu(),
        context_slot_mapping=context_slot_mappings[0].cpu(),
        context_slot_mappings=context_slot_mappings.cpu(),
        sample_indices=sample_indices.cpu(),
        sample_pos=sample_pos.cpu(),
        sample_idx_mapping=sample_idx_mapping.cpu(),
        temperature=temperature.cpu(),
        seeds=seeds.cpu(),
    )


@pytest.mark.parametrize("cp_rank,cp_size", [(0, 1), (0, 2), (1, 2)])
def test_prepare_dflash_inputs_excludes_rejected_context_suffix(cp_rank, cp_size):
    # Positions 10/11 use physical block 7. Rejected positions 12/13 would use
    # block 8, but must be PAD context rather than contaminating draft KV.
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
        cp_rank=cp_rank,
        cp_size=cp_size,
        cp_interleave=2,
        draft_dcp_size=1,
    )

    assert out.context_positions[:4].tolist() == [10, 11, 0, 0]
    assert out.context_slot_mapping[:4].tolist() == [30, 31, PAD_SLOT_ID, PAD_SLOT_ID]

    # The replacement query starts immediately after the two valid rows and
    # advances from the last accepted position (11).
    assert out.input_buffers.input_ids[:3].cpu().tolist() == [99, 123, 123]
    assert out.input_buffers.positions[:3].cpu().tolist() == [12, 13, 14]
    assert out.query_slot_mapping[:3].tolist() == [32, 33, 34]
    assert out.sample_indices[:3].tolist() == [0, 1, 2]
    assert out.sample_pos[:3].tolist() == [13, 14, 15]
    assert out.sample_idx_mapping[:3].tolist() == [2, 2, 2]
    assert out.temperature[2].item() == 1.0
    assert out.seeds[2].item() == 17
    assert not out.input_buffers.is_padding[:3].any()
    assert out.input_buffers.is_padding[3:].all()
    assert (out.input_buffers.input_ids[3:] == 0).all()
    assert (out.input_buffers.positions[3:] == 0).all()
    assert (out.query_slot_mapping[3:] == PAD_SLOT_ID).all()


def test_prepare_dflash_inputs_excludes_rejected_context_suffix_with_dcp():
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 7, 8, 9],
        cp_rank=1,
        cp_size=2,
        cp_interleave=2,
    )

    assert out.context_positions[:4].tolist() == [10, 11, 0, 0]
    assert out.context_slot_mapping[:4].tolist() == [28, 29, PAD_SLOT_ID, PAD_SLOT_ID]
    assert out.query_slot_mapping[:3].tolist() == [PAD_SLOT_ID, PAD_SLOT_ID, 30]


def test_prepare_dflash_inputs_never_writes_the_null_block():
    # The valid context uses logical block 0 and the replacement query uses
    # logical block 1. Both map to the null block and must remain unwritable.
    out = _run_prepare(
        target_positions=[2, 3, 4, 5],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out.context_slot_mapping[:4].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
    assert out.query_slot_mapping[:3].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]


def test_prepare_dflash_inputs_writes_each_draft_group():
    # Three draft groups over KV-cache groups 2, 0 and 1, with their own block
    # tables and block sizes, in one launch: each writes the slot mappings of
    # its own table, and the shared outputs are those of a single group.
    tables = {
        0: ([0, 0, 7, 8, 9, 10, 11, 12], 4),
        1: ([0, 5, 6], 8),
        2: ([3, 4, 1, 2, 13, 14, 15, 16], 2),
    }
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=tables[0][0],
        extra_kv_groups=(tables[1], tables[2]),
        draft_group_ids=(2, 0, 1),
    )
    for draft_group, kv_group in enumerate((2, 0, 1)):
        values, size = tables[kv_group]
        single = _run_prepare(
            target_positions=[10, 11, 12, 13],
            block_table_values=values,
            extra_kv_groups=((values, size),),
            draft_group_ids=(1,),
        )
        assert torch.equal(
            out.query_slot_mappings[kv_group], single.query_slot_mappings[1]
        ), kv_group
        assert torch.equal(
            out.context_slot_mappings[draft_group], single.context_slot_mappings[0]
        ), draft_group
        for name in ("context_positions", "sample_indices", "sample_pos"):
            assert torch.equal(getattr(out, name), getattr(single, name)), name
        for name in ("input_ids", "positions", "is_padding", "query_start_loc"):
            assert torch.equal(
                getattr(out.input_buffers, name), getattr(single.input_buffers, name)
            ), name
    # Block sizes 2, 4 and 8 put query position 12 in different slots.
    assert out.query_slot_mappings[:, 0].tolist() == [32, 44, 30]
