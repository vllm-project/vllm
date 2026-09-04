# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    prepare_dflash_inputs,
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
):
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
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        positions=torch.tensor(target_positions, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([2], dtype=torch.int32, device=device),
    )
    query_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    context_positions = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
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
    block_table = torch.tensor([block_table_values], dtype=torch.int32, device=device)

    prepare_dflash_inputs(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([2], dtype=torch.int32, device=device),
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        4,
        cp_rank,
        cp_size,
        cp_interleave,
        123,
        num_speculative_steps,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        128,
        sample_from_anchor=True,
    )
    torch.accelerator.synchronize()
    return SimpleNamespace(
        input_buffers=input_buffers,
        query_slot_mapping=query_slot_mapping.cpu(),
        context_positions=context_positions.cpu(),
        context_slot_mapping=context_slot_mapping.cpu(),
        sample_indices=sample_indices.cpu(),
        sample_pos=sample_pos.cpu(),
        sample_idx_mapping=sample_idx_mapping.cpu(),
        temperature=temperature.cpu(),
        seeds=seeds.cpu(),
    )


def test_prepare_dflash_inputs_excludes_rejected_context_suffix():
    # Positions 10/11 use physical block 7. Rejected positions 12/13 would use
    # block 8, but must be PAD context rather than contaminating draft KV.
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
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
    assert out.input_buffers.input_ids[3:].cpu().tolist() == [0] * 13
    assert out.input_buffers.positions[3:].cpu().tolist() == [0] * 13


def test_prepare_dflash_inputs_compacts_noncontiguous_request_slots():
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
        num_reqs=2,
        num_scheduled_tokens=np.array([4, 4], dtype=np.int32),
        positions=torch.tensor(
            [10, 11, 12, 13, 20, 21, 22, 23],
            dtype=torch.int64,
            device=device,
        ),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        # Active batch rows are compact, while request state remains in slots 3 and 1.
        idx_mapping=torch.tensor([3, 1], dtype=torch.int32, device=device),
    )
    query_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    context_positions = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
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
        [0.0, 0.5, 0.0, 1.0], dtype=torch.float32, device=device
    )
    input_seeds = torch.tensor([0, 11, 0, 33], dtype=torch.int64, device=device)
    last_sampled = torch.tensor([0, 77, 0, 99], dtype=torch.int64, device=device)
    next_prefill_tokens = torch.zeros_like(last_sampled)
    block_table = torch.tensor(
        [[0, 0, 7, 8, 9, 10, 11, 12], [0, 0, 13, 14, 15, 16, 17, 18]],
        dtype=torch.int32,
        device=device,
    )

    prepare_dflash_inputs(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        torch.tensor([1, 1], dtype=torch.int32, device=device),
        torch.tensor([2, 1], dtype=torch.int32, device=device),
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        4,
        0,
        1,
        1,
        123,
        num_speculative_steps,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        128,
        sample_from_anchor=True,
    )
    torch.accelerator.synchronize()

    # Query rows follow compact batch order, but every persistent state lookup
    # follows idx_mapping instead of accidentally using the compact row index.
    assert input_buffers.input_ids[:6].cpu().tolist() == [99, 123, 123, 77, 123, 123]
    assert input_buffers.positions[:6].cpu().tolist() == [12, 13, 14, 23, 24, 25]
    assert sample_indices[:6].cpu().tolist() == [0, 1, 2, 3, 4, 5]
    assert sample_idx_mapping[:6].cpu().tolist() == [3, 3, 3, 1, 1, 1]
    assert temperature.cpu().tolist() == [0.0, 0.5, 0.0, 1.0]
    assert seeds.cpu().tolist() == [0, 11, 0, 33]
    assert not input_buffers.is_padding[:6].any()
    assert input_buffers.is_padding[6:].all()


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


def test_dflash_forward_context_receives_draft_padding_mask(monkeypatch):
    device = torch.device("cuda")
    input_buffers = SimpleNamespace(
        input_ids=torch.tensor([11, 12, 0, 0], dtype=torch.int32, device=device),
        positions=torch.tensor([7, 8, 0, 0], dtype=torch.int64, device=device),
        is_padding=torch.tensor([False, False, True, True], device=device),
    )
    observed = None

    @contextmanager
    def fake_set_forward_context(*args, **kwargs):
        nonlocal observed
        observed = kwargs["is_padding"].clone()
        yield

    monkeypatch.setattr(
        dflash_speculator, "set_forward_context", fake_set_forward_context
    )
    speculator = SimpleNamespace(
        input_buffers=input_buffers,
        vllm_config=SimpleNamespace(),
        model=lambda **kwargs: kwargs["input_ids"],
    )

    result = DFlashSpeculator._run_model(
        speculator,
        num_tokens=4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
    )

    assert result.tolist() == [11, 12, 0, 0]
    assert observed is not None
    assert observed.tolist() == [False, False, True, True]
