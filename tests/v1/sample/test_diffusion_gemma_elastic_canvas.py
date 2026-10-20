# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionGemma canvas swap (preempted blocks resume from a host-RAM stash)
and canvas packing (two tenants share one canvas behind span isolation)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaModelState,
    DiffusionGemmaRequestStates,
    DiffusionSampler,
    _parse_pack_spans,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CL = 16
VOCAB = 64
HIDDEN = 8
MAX_REQS = 4
# lenA=3, lenB=2, split=8 -> packed prompts are 5 tokens long.
PACK_CFG = (3, 2, 8)


def _states(cap: int = 8) -> DiffusionGemmaRequestStates:
    states = DiffusionGemmaRequestStates(
        max_num_reqs=MAX_REQS,
        canvas_length=CL,
        vocab_size=VOCAB,
        max_denoising_steps=48,
        device=DEVICE,
        hidden_size=HIDDEN,
        stability_threshold=2,
    )
    states.canvas_swap = True
    states.swap_cap = cap
    return states


def _model_state(
    states: DiffusionGemmaRequestStates,
    pack_cfg: tuple[int, int, int] | None = None,
    selective: bool = False,
) -> DiffusionGemmaModelState:
    # Only the request bookkeeping is under test; skip the model-bound __init__.
    ms = object.__new__(DiffusionGemmaModelState)
    ms.device = DEVICE
    ms.diffusion_states = states
    ms._req_id_to_index = {}
    ms._pack_armed = pack_cfg is not None
    ms._pack_cfg = pack_cfg
    ms._pack_selective = selective
    if pack_cfg is not None:
        ms._span_master = torch.zeros(MAX_REQS, 4, 3, dtype=torch.int32, device=DEVICE)
        ms._span_buf = torch.zeros_like(ms._span_master)
    return ms


def _req(req_id: str, prompt_len: int = 5) -> SimpleNamespace:
    return SimpleNamespace(req_id=req_id, prompt_token_ids=[1] * prompt_len)


def _run_block(states: DiffusionGemmaRequestStates, slot: int, commit_next: bool):
    """Put ``slot`` in the middle of a block: some denoise steps taken, and
    optionally converged and waiting to commit."""
    g = torch.Generator().manual_seed(slot)
    states.canvas[slot] = torch.randint(0, VOCAB, (CL,), generator=g)
    states.argmax_canvas[slot] = torch.randint(0, VOCAB, (CL,), generator=g)
    states.step[slot] = 23
    states.is_encoder_phase[slot] = commit_next
    states.confident[slot] = True
    states.accepted_canvas_history[slot] = torch.randint(
        0, VOCAB, (states.stability_threshold, CL), generator=g
    )
    states.accepted_canvas_history_len[slot] = 2
    states.self_conditioning_embeds[slot] = torch.randn(CL, HIDDEN, generator=g)


def _snapshot(states: DiffusionGemmaRequestStates, slot: int) -> dict:
    return {f: getattr(states, f)[slot].clone() for f in states._SWAP_FIELDS}


def _assert_restored(states: DiffusionGemmaRequestStates, slot: int, want: dict):
    for f in states._SWAP_FIELDS:
        assert torch.equal(getattr(states, f)[slot], want[f]), f


def _preempt(ms: DiffusionGemmaModelState, req_id: str) -> None:
    ms.note_preempted({req_id})
    ms.remove_request(req_id)


def test_stash_restore_round_trip():
    states = _states()
    states.add_request(1)
    _run_block(states, 1, commit_next=False)
    want = _snapshot(states, 1)

    entry = states.stash_slot(1)
    assert all(v.device.type == "cpu" for k, v in entry.items() if k != "logprobs")
    states.remove_request(1)
    states.add_request(1)
    assert not torch.equal(states.canvas[1], want["canvas"])

    states.restore_slot(1, entry)
    _assert_restored(states, 1, want)


def test_preempt_stashes_and_resume_restores():
    states = _states()
    ms = _model_state(states)
    ms.add_request(0, _req("A"))
    _run_block(states, 0, commit_next=False)
    want = _snapshot(states, 0)

    _preempt(ms, "A")
    assert "A" in states.canvas_stash

    ms.add_request(2, _req("A"))
    assert states.pending_restore == {2: "A"}
    sampler = object.__new__(DiffusionSampler)
    sampler.diffusion_states = states
    assert sampler._restore_stashed_canvases(np.array([2])) == []
    _assert_restored(states, 2, want)
    assert not states.canvas_stash and not states.pending_restore


def test_converged_block_resumes_at_commit():
    states = _states()
    ms = _model_state(states)
    ms.add_request(0, _req("A"))
    _run_block(states, 0, commit_next=True)
    _preempt(ms, "A")

    ms.add_request(1, _req("A"))
    sampler = object.__new__(DiffusionSampler)
    sampler.diffusion_states = states
    assert sampler._restore_stashed_canvases(np.array([1])) == [1]


def test_repreempt_during_reprefill_keeps_stash():
    states = _states()
    ms = _model_state(states)
    ms.add_request(0, _req("A"))
    _run_block(states, 0, commit_next=False)
    want_canvas = states.canvas[0].clone()
    _preempt(ms, "A")

    # Resumed, then preempted again before its re-prefill finished: the slot
    # holds a fresh canvas that must not replace the stashed block.
    ms.add_request(1, _req("A"))
    _preempt(ms, "A")
    assert torch.equal(states.canvas_stash["A"]["canvas"].to(DEVICE), want_canvas)
    assert not states.pending_restore


def test_finished_request_drops_stash():
    states = _states()
    ms = _model_state(states)
    ms.add_request(0, _req("A"))
    _run_block(states, 0, commit_next=False)
    _preempt(ms, "A")
    ms.add_request(1, _req("A"))
    ms.remove_request("A")
    assert "A" not in states.canvas_stash
    assert not states.pending_restore


def test_stash_is_fifo_bounded():
    states = _states(cap=2)
    ms = _model_state(states)
    for slot, req_id in enumerate("ABC"):
        ms.add_request(slot, _req(req_id))
        _run_block(states, slot, commit_next=False)
    for req_id in "ABC":
        _preempt(ms, req_id)
    assert list(states.canvas_stash) == ["B", "C"]


def test_pending_logprobs_survive_preemption():
    states = _states()
    ms = _model_state(states)
    ms.add_request(0, _req("A"))
    _run_block(states, 0, commit_next=True)
    sentinel = object()
    states.pending_logprobs[0] = sentinel
    _preempt(ms, "A")

    ms.add_request(3, _req("A"))
    sampler = object.__new__(DiffusionSampler)
    sampler.diffusion_states = states
    sampler._restore_stashed_canvases(np.array([3]))
    assert states.pending_logprobs[3] is sentinel


def _span_rows(ms: DiffusionGemmaModelState, slot: int) -> list[tuple[int, ...]]:
    rows = ms._span_master[slot].tolist()
    return [tuple(r) for r in rows if r[0] < r[1]]


def test_parse_pack_spans():
    assert _parse_pack_spans("", CL) == (False, None, False)
    assert _parse_pack_spans("inert", CL) == (True, None, False)
    assert _parse_pack_spans("3,2,8", CL) == (True, (3, 2, 8), False)
    assert _parse_pack_spans("3,2,8,selective", CL) == (True, (3, 2, 8), True)
    for bad in ("3,2", "3,2,16", "3,2,8,all", "0,2,8"):
        with pytest.raises(ValueError):
            _parse_pack_spans(bad, CL)


def test_packed_prompt_gets_both_tenants():
    ms = _model_state(_states(), pack_cfg=PACK_CFG)
    ms.add_request(0, _req("pair"))
    assert _span_rows(ms, 0) == [(0, 3, 0), (3, 5, 1), (5, 13, 0), (13, 21, 1)]
    ms.remove_request("pair")
    assert _span_rows(ms, 0) == []


def test_unpackable_prompt():
    ms = _model_state(_states(), pack_cfg=PACK_CFG)
    with pytest.raises(ValueError, match="lenA"):
        ms.add_request(0, _req("solo", prompt_len=9))

    ms = _model_state(_states(), pack_cfg=PACK_CFG, selective=True)
    ms.add_request(0, _req("solo", prompt_len=9))
    assert _span_rows(ms, 0) == []


def test_packed_slot_resumes_with_both_tenants():
    states = _states()
    ms = _model_state(states, pack_cfg=PACK_CFG, selective=True)
    ms.add_request(0, _req("pair"))
    ms.add_request(1, _req("solo", prompt_len=9))
    want_rows = _span_rows(ms, 0)
    for slot in (0, 1):
        _run_block(states, slot, commit_next=False)
    want = {slot: _snapshot(states, slot) for slot in (0, 1)}
    _preempt(ms, "pair")
    _preempt(ms, "solo")

    ms.add_request(2, _req("solo", prompt_len=9))
    ms.add_request(3, _req("pair"))
    assert _span_rows(ms, 2) == []
    assert _span_rows(ms, 3) == want_rows
    sampler = object.__new__(DiffusionSampler)
    sampler.diffusion_states = states
    sampler._restore_stashed_canvases(np.array([2, 3]))
    _assert_restored(states, 2, want[1])
    _assert_restored(states, 3, want[0])


def test_packed_positions_follow_each_tenant():
    ms = _model_state(_states(), pack_cfg=PACK_CFG, selective=True)
    ms.add_request(0, _req("pair"))
    ms.add_request(1, _req("solo", prompt_len=9))
    # Row 0: the packed slot's canvas pass. Row 1: an unpacked canvas pass.
    positions = torch.cat([torch.arange(5, 5 + CL), torch.arange(9, 9 + CL)]).to(DEVICE)
    batch = SimpleNamespace(
        positions=positions.clone(),
        query_start_loc=torch.tensor([0, CL, 2 * CL], device=DEVICE),
        num_tokens=2 * CL,
    )
    slots = torch.tensor([0, 1], device=DEVICE)
    ms._span_buf[:2] = ms._span_master[slots]

    ms._remap_packed_positions(batch, actual_num_reqs=2)

    # Tenant A's span continues its 3-token prompt, B's its 2-token prompt.
    want_packed = list(range(3, 11)) + list(range(2, 10))
    assert batch.positions[:CL].tolist() == want_packed
    assert torch.equal(batch.positions[CL:], positions[CL:])


def test_packed_prefill_positions():
    ms = _model_state(_states(), pack_cfg=PACK_CFG)
    ms.add_request(0, _req("pair"))
    batch = SimpleNamespace(
        positions=torch.arange(5, device=DEVICE),
        query_start_loc=torch.tensor([0, 5], device=DEVICE),
        num_tokens=5,
    )
    ms._span_buf[:1] = ms._span_master[:1]
    ms._remap_packed_positions(batch, actual_num_reqs=1)
    assert batch.positions.tolist() == [0, 1, 2, 0, 1]
