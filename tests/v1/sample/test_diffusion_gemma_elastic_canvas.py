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
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CL = 16
VOCAB = 64
HIDDEN = 8
MAX_REQS = 4
# lenA=3, lenB=2, split=8 -> packed prompts are 5 tokens long.
PACK_CFG = (3, 2, 8)


def _model_state(
    cap: int = 8,
    pack_cfg: tuple[int, int, int] | None = None,
    selective: bool = False,
) -> DiffusionGemmaModelState:
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


def _start_block(
    ms: DiffusionGemmaModelState, slot: int, req_id: str, commit_next: bool
) -> dict[str, torch.Tensor]:
    """Add ``req_id`` at ``slot`` mid-block and return its swapped state."""
    ms.add_request(slot, _req(req_id))
    states = ms.diffusion_states
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
    return {f: getattr(states, f)[slot].clone() for f in states._SWAP_FIELDS}


def _preempt(ms: DiffusionGemmaModelState, req_id: str) -> None:
    ms.note_preempted({req_id})
    ms.remove_request(req_id)


def _resume(ms: DiffusionGemmaModelState, slot: int, req_id: str) -> list[int]:
    ms.add_request(slot, _req(req_id))
    sampler = object.__new__(DiffusionSampler)
    sampler.diffusion_states = ms.diffusion_states
    return sampler._restore_stashed_canvases(np.array([slot]))


def test_preempted_block_resumes_in_new_slot():
    ms = _model_state()
    want = _start_block(ms, 0, "A", commit_next=False)
    _preempt(ms, "A")

    assert _resume(ms, 2, "A") == []
    states = ms.diffusion_states
    for f in states._SWAP_FIELDS:
        assert torch.equal(getattr(states, f)[2], want[f]), f
    assert not states.canvas_stash and not states.pending_restore


def test_converged_block_resumes_at_commit_with_logprobs():
    ms = _model_state()
    _start_block(ms, 0, "A", commit_next=True)
    pending = object()
    ms.diffusion_states.pending_logprobs[0] = pending
    _preempt(ms, "A")

    assert _resume(ms, 3, "A") == [3]
    assert ms.diffusion_states.pending_logprobs[3] is pending


def test_repreempt_during_reprefill_keeps_stash():
    ms = _model_state()
    want = _start_block(ms, 0, "A", commit_next=False)
    _preempt(ms, "A")
    # Preempted again before the re-prefill restored the block: the slot's
    # fresh canvas must not replace the stash.
    ms.add_request(1, _req("A"))
    _preempt(ms, "A")

    stash = ms.diffusion_states.canvas_stash["A"]
    assert torch.equal(stash["canvas"].to(DEVICE), want["canvas"])


def test_stash_is_fifo_bounded():
    ms = _model_state(cap=2)
    for slot, req_id in enumerate("ABC"):
        _start_block(ms, slot, req_id, commit_next=False)
    for req_id in "ABC":
        _preempt(ms, req_id)
    assert list(ms.diffusion_states.canvas_stash) == ["B", "C"]


def _span_rows(ms: DiffusionGemmaModelState, slot: int) -> list[tuple[int, ...]]:
    return [tuple(r) for r in ms._span_master[slot].tolist() if r[0] < r[1]]


def test_span_rows_for_packed_and_unpacked_prompts():
    ms = _model_state(pack_cfg=PACK_CFG)
    ms.add_request(0, _req("pair"))
    assert _span_rows(ms, 0) == [(0, 3, 0), (3, 5, 1), (5, 13, 0), (13, 21, 1)]
    with pytest.raises(ValueError, match="lenA"):
        ms.add_request(1, _req("solo", prompt_len=9))

    ms = _model_state(pack_cfg=PACK_CFG, selective=True)
    ms.add_request(1, _req("solo", prompt_len=9))
    assert _span_rows(ms, 1) == []


@pytest.mark.parametrize("phase", ["prefill", "canvas"])
def test_packed_positions_follow_each_tenant(phase):
    ms = _model_state(pack_cfg=PACK_CFG, selective=True)
    ms.add_request(0, _req("pair"))
    ms.add_request(1, _req("solo", prompt_len=9))
    # Row 0 is the packed slot; row 1 is unpacked and must not move.
    if phase == "prefill":
        packed = torch.arange(5)
        want_packed = [0, 1, 2, 0, 1]
    else:
        packed = torch.arange(5, 5 + CL)
        want_packed = list(range(3, 11)) + list(range(2, 10))
    solo = torch.arange(9, 9 + CL)
    positions = torch.cat([packed, solo]).to(DEVICE)
    n = positions.numel()
    batch = SimpleNamespace(
        positions=positions.clone(),
        query_start_loc=torch.tensor([0, packed.numel(), n], device=DEVICE),
        num_tokens=n,
    )
    ms._span_buf[:2] = ms._span_master[:2]

    ms._remap_packed_positions(batch, actual_num_reqs=2)

    assert batch.positions[: packed.numel()].tolist() == want_packed
    assert batch.positions[packed.numel() :].tolist() == solo.tolist()
