# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    select_context_window_indices,
    _prepare_dflash_inputs_kernel,
)

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="requires a CUDA or XPU device",
)


@pytest.mark.parametrize("lengths", [[16384], [6, 16348, 1, 29], [0, 132, 133, 134]])
@pytest.mark.parametrize("pointer_offset", [0, 1])
def test_context_suffix_keeps_rejected_anchor_window(lengths, pointer_offset):
    """Packing preserves every cache row reachable after any rejection count."""
    window, speculative = 128, 5
    keep = window + speculative
    boundaries = np.concatenate(([0], np.cumsum(lengths)))
    storage = torch.full(
        (len(boundaries) + pointer_offset,),
        -1,
        dtype=torch.int32,
        device=current_platform.device_type,
    )
    query_start_loc = storage[pointer_offset:]
    query_start_loc.copy_(torch.tensor(boundaries))
    indices, valid = select_context_window_indices(query_start_loc, len(lengths), keep)
    expected_indices = torch.zeros_like(indices, device="cpu")
    expected_valid = torch.zeros_like(valid, device="cpu")
    for req, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        rows = torch.arange(max(begin, end - keep), end)
        start = req * keep
        expected_indices[start : start + len(rows)] = rows
        expected_valid[start : start + len(rows)] = True
        selected = set(rows.tolist())
        for rejected in range(min(speculative, end - begin) + 1):
            accepted_end = end - rejected
            needed = set(range(max(begin, accepted_end - window), accepted_end))
            assert needed <= selected
            for graph_prefix in (0, 48):
                eager = {i for i in selected if i >= graph_prefix and i < accepted_end}
                graph = set(range(begin, min(graph_prefix, accepted_end)))
                assert needed <= eager | graph
                assert eager.isdisjoint(graph)
    torch.testing.assert_close(indices.cpu(), expected_indices, atol=0, rtol=0)
    torch.testing.assert_close(valid.cpu(), expected_valid, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires CUDA graph replay"
)
def test_context_suffix_uses_replayed_gpu_boundaries():
    """Adaptive verification can change request boundaries without CPU readback."""
    device = current_platform.device_type
    boundaries = torch.tensor([0, 8, 4096, 4102], dtype=torch.int32, device=device)
    select_context_window_indices(boundaries, 3, 133)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices, valid = select_context_window_indices(boundaries, 3, 133)
    boundaries.copy_(torch.tensor([0, 3, 4091, 4092]))
    graph.replay()
    actual = indices[valid].cpu()
    expected = torch.cat(
        [torch.arange(3), torch.arange(3958, 4091), torch.tensor([4091])]
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _run_prepare(
    *,
    target_positions: list[int],
    block_table_values: list[int],
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_interleave: int = 1,
    draft_dcp_size: int | None = None,
    num_scheduled_tokens: int = 4,
):
    device = torch.device(current_platform.device_type)
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
        num_scheduled_tokens=np.array([num_scheduled_tokens], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([target_positions[-1] + 1]),
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

    class InputsPrepared(Exception):
        pass

    draft = SimpleNamespace(
        input_buffers=input_buffers,
        context_positions=context_positions,
        _context_slot_mappings=context_slot_mapping.unsqueeze(0),
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
        draft_kv_cache_group_id=0,
        draft_kv_cache_group_ids=[0],
        _layer_group_idx=None,
        block_tables=SimpleNamespace(
            slot_mappings=query_slot_mapping.unsqueeze(0),
            input_block_tables=[block_table],
            block_sizes=[4],
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
        query_slot_mapping=query_slot_mapping.cpu(),
        context_positions=context_positions.cpu(),
        context_slot_mapping=context_slot_mapping.cpu(),
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


def test_prepare_dflash_inputs_compiles_once_for_all_query_lengths():
    def compiled_variants() -> set:
        return {
            key
            for caches in _prepare_dflash_inputs_kernel.device_caches.values()
            for key in caches[0]
        }

    kwargs = dict(target_positions=[10, 11, 12, 13], block_table_values=[0, 7, 8])
    _run_prepare(**kwargs)
    warmed = compiled_variants()
    _run_prepare(**kwargs, num_scheduled_tokens=300)
    assert compiled_variants() == warmed
