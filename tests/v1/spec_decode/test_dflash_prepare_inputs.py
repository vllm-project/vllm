# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as speculator_module
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    _prepare_dflash_inputs_kernel,
)

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="requires a CUDA or XPU device",
)


@pytest.fixture(autouse=True)
def _force_fastpaths_supported(monkeypatch):
    # The draft-prepare fastpaths are ROCm-gated in the speculator; these
    # tests pin their contracts on any GPU.
    monkeypatch.setattr(
        speculator_module, "_draft_prepare_fastpaths_supported", lambda: True
    )


def _run_prepare(
    *,
    target_positions: list[int],
    block_table_values: list[int],
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_interleave: int = 1,
    draft_dcp_size: int | None = None,
    num_scheduled_tokens: int = 4,
    draft_kv_cache_group_ids: list[int] | None = None,
):
    # With draft_kv_cache_group_ids the speculator prepares inputs once per
    # draft KV group: the returned all_query_slot_mappings /
    # all_context_slot_mappings carry one row per group (plus any rows for
    # group ids below the first draft group, which nothing may write), and
    # query_slot_mapping / context_slot_mapping are the first group's rows.
    device = torch.device(current_platform.device_type)
    max_num_reqs = 4
    max_num_tokens = 16
    num_speculative_steps = 3
    group_ids = [0] if draft_kv_cache_group_ids is None else draft_kv_cache_group_ids
    n_rows = max(group_ids) + 1

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
        (n_rows, max_num_tokens), -2, dtype=torch.int64, device=device
    )
    context_positions = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_slot_mapping = torch.full(
        (len(group_ids), max_num_tokens), -2, dtype=torch.int64, device=device
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
        _context_slot_mappings=context_slot_mapping,
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
        draft_kv_cache_group_id=group_ids[0],
        draft_kv_cache_group_ids=group_ids,
        _layer_group_idx=None,
        block_tables=SimpleNamespace(
            slot_mappings=query_slot_mapping,
            input_block_tables=[block_table] * n_rows,
            block_sizes=[4] * n_rows,
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
        query_slot_mapping=query_slot_mapping[group_ids[0]].cpu(),
        all_query_slot_mappings=query_slot_mapping.cpu(),
        context_positions=context_positions.cpu(),
        context_slot_mapping=context_slot_mapping[0].cpu(),
        all_context_slot_mappings=context_slot_mapping.cpu(),
        sample_indices=sample_indices.cpu(),
        sample_pos=sample_pos.cpu(),
        sample_idx_mapping=sample_idx_mapping.cpu(),
        temperature=temperature.cpu(),
        seeds=seeds.cpu(),
    )


@pytest.mark.parametrize(
    "pad_fix",
    [pytest.param(True, id="pad-fix-on"), pytest.param(False, id="pad-fix-off")],
)
@pytest.mark.parametrize("cp_rank,cp_size", [(0, 1), (0, 2), (1, 2)])
def test_prepare_dflash_inputs_excludes_rejected_context_suffix(
    cp_rank, cp_size, pad_fix, monkeypatch
):
    # Positions 10/11 use physical block 7. Rejected positions 12/13 would use
    # block 8, but must be PAD context rather than contaminating draft KV.
    if pad_fix:
        monkeypatch.delenv("VLLM_DEBUG_DISABLE_DFLASH_PREPARE_PAD_FIX", raising=False)
    else:
        monkeypatch.setenv("VLLM_DEBUG_DISABLE_DFLASH_PREPARE_PAD_FIX", "1")
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
    # Pad-tail contract: with the fix the tail is one parallel launch
    # bounded by the draft replay read range [q_pad_start, HWM) = [3, 12);
    # the stock in-kernel tail pads the full [3, 16) buffer.
    pad_end = 12 if pad_fix else 16
    assert out.input_buffers.is_padding[3:pad_end].all()
    assert (out.input_buffers.input_ids[3:pad_end] == 0).all()
    assert (out.input_buffers.positions[3:pad_end] == 0).all()
    assert (out.query_slot_mapping[3:pad_end] == PAD_SLOT_ID).all()
    if pad_fix:
        assert not out.input_buffers.is_padding[12:].any()
        assert (out.input_buffers.input_ids[12:] == -1).all()
        assert (out.input_buffers.positions[12:] == -1).all()
        assert (out.query_slot_mapping[12:] == -2).all()


def test_prepare_dflash_inputs_two_draft_groups_dedup_and_pad(monkeypatch):
    # Draft KV groups at non-zero ids with a non-draft row 0 in between: the
    # dedup must keep the shared outputs identical to the per-group
    # recompute, and the parallel pad launch must touch only the draft
    # group rows over [q_pad_start, HWM) -- never row 0.
    outs = {}
    for dedup_on in (True, False):
        if dedup_on:
            monkeypatch.delenv("VLLM_DEBUG_DISABLE_DRAFT_PREPARE_DEDUP", raising=False)
            monkeypatch.delenv(
                "VLLM_DEBUG_DISABLE_DFLASH_PREPARE_PAD_FIX", raising=False
            )
        else:
            monkeypatch.setenv("VLLM_DEBUG_DISABLE_DRAFT_PREPARE_DEDUP", "1")
            monkeypatch.setenv("VLLM_DEBUG_DISABLE_DFLASH_PREPARE_PAD_FIX", "1")
        outs[dedup_on] = _run_prepare(
            target_positions=[10, 11, 12, 13],
            block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
            cp_interleave=2,
            draft_dcp_size=1,
            draft_kv_cache_group_ids=[1, 2],
        )
    on, off = outs[True], outs[False]
    # Shared outputs stay bit-identical to the per-group recompute over the
    # replay read range [0, HWM) (the pad fix intentionally leaves [12, 16)
    # untouched; the single-group test pins that contract).
    for field in ("input_ids", "positions", "is_padding"):
        assert torch.equal(
            getattr(on.input_buffers, field)[:12],
            getattr(off.input_buffers, field)[:12],
        )
    for field in (
        "context_positions",
        "sample_indices",
        "sample_pos",
        "sample_idx_mapping",
    ):
        assert torch.equal(getattr(on, field), getattr(off, field))
    assert torch.equal(
        on.all_query_slot_mappings[:, :12], off.all_query_slot_mappings[:, :12]
    )
    assert torch.equal(on.all_context_slot_mappings, off.all_context_slot_mappings)
    # Same exact contract as the single-group case, on every group row.
    assert on.input_buffers.input_ids[:3].cpu().tolist() == [99, 123, 123]
    assert on.context_slot_mapping[:4].tolist() == [30, 31, PAD_SLOT_ID, PAD_SLOT_ID]
    for row in (1, 2):
        assert on.all_query_slot_mappings[row][:3].tolist() == [32, 33, 34]
        assert (on.all_query_slot_mappings[row][3:12] == PAD_SLOT_ID).all()
        assert (on.all_query_slot_mappings[row][12:] == -2).all()
    # The stock arm recomputes per group and pads the full tail on both rows.
    for row in (1, 2):
        assert off.all_query_slot_mappings[row][:3].tolist() == [32, 33, 34]
        assert (off.all_query_slot_mappings[row][3:16] == PAD_SLOT_ID).all()
    # Row 0 (not a draft group) is never written by either arm.
    assert (on.all_query_slot_mappings[0] == -2).all()
    assert (off.all_query_slot_mappings[0] == -2).all()


def test_draft_prepare_fastpaths_off_platform_keeps_stock_paths(monkeypatch):
    # Non-ROCm platforms keep the stock paths: every draft group call
    # recomputes the shared outputs and pads the full in-kernel tail.
    monkeypatch.setattr(
        speculator_module, "_draft_prepare_fastpaths_supported", lambda: False
    )
    monkeypatch.delenv("VLLM_DEBUG_DISABLE_DRAFT_PREPARE_DEDUP", raising=False)
    monkeypatch.delenv("VLLM_DEBUG_DISABLE_DFLASH_PREPARE_PAD_FIX", raising=False)
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
        cp_interleave=2,
        draft_dcp_size=1,
        draft_kv_cache_group_ids=[1, 2],
    )
    assert out.input_buffers.input_ids[:3].cpu().tolist() == [99, 123, 123]
    assert out.input_buffers.is_padding[3:16].all()
    for row in (1, 2):
        assert out.all_query_slot_mappings[row][:3].tolist() == [32, 33, 34]
        assert (out.all_query_slot_mappings[row][3:16] == PAD_SLOT_ID).all()
    assert (out.all_query_slot_mappings[0] == -2).all()


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
