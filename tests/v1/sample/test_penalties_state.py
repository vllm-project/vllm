# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alignment tests for the pooled penalties state.

The reference implementation recomputes penalties from the full Python-side
token histories (the semantics of the old per-step rebuild), so any divergence
in the incremental bookkeeping (admission, commit, slot reuse, row moves,
draft prefixes) shows up as a logits mismatch.
"""

import random

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.penalties import PenaltiesState

VOCAB_SIZE = 2048
MAX_NUM_REQS = 64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class MirrorRequest:
    def __init__(self, rng: random.Random, with_penalties: bool):
        self.prompt = [rng.randrange(VOCAB_SIZE) for _ in range(rng.randint(1, 50))]
        self.output = [rng.randrange(VOCAB_SIZE) for _ in range(rng.randint(0, 30))]
        if with_penalties:
            self.params = SamplingParams(
                repetition_penalty=rng.choice([1.0, 1.05, 1.5]),
                frequency_penalty=rng.choice([0.0, 0.2, 1.1]),
                presence_penalty=rng.choice([0.0, 0.5, -0.4]),
            )
            if not self.needs_penalties():
                self.params.presence_penalty = 0.7
        else:
            self.params = SamplingParams()

    def needs_penalties(self) -> bool:
        return (
            self.params.repetition_penalty != 1.0
            or self.params.frequency_penalty != 0.0
            or self.params.presence_penalty != 0.0
        )

    def token_ids(self) -> np.ndarray:
        return np.array(self.prompt + self.output, dtype=np.int64)


def reference_apply(
    logits: torch.Tensor,
    rows: list[MirrorRequest],
    draft_prefixes: list[list[int]] | None = None,
) -> torch.Tensor:
    logits = logits.clone().float()
    for i, req in enumerate(rows):
        if not req.needs_penalties():
            continue
        output = list(req.output)
        if draft_prefixes is not None:
            output += draft_prefixes[i]
        counts = torch.bincount(
            torch.tensor(output, dtype=torch.int64, device="cpu"),
            minlength=VOCAB_SIZE,
        ).to(logits.device)
        output_mask = counts > 0
        prompt_mask = (
            torch.bincount(
                torch.tensor(req.prompt, dtype=torch.int64, device="cpu"),
                minlength=VOCAB_SIZE,
            ).to(logits.device)
            > 0
        )

        row = logits[i]
        rep = req.params.repetition_penalty
        if rep != 1.0:
            penalized = prompt_mask | output_mask
            scale = torch.where(
                penalized,
                torch.tensor(rep, device=logits.device),
                torch.tensor(1.0, device=logits.device),
            )
            row *= torch.where(row > 0, 1.0 / scale, scale)
        row -= req.params.frequency_penalty * counts
        row -= req.params.presence_penalty * output_mask
    return logits


def make_penalty_tensors(rows: list[MirrorRequest], device):
    rep = torch.tensor(
        [r.params.repetition_penalty for r in rows], dtype=torch.float32, device=device
    )
    freq = torch.tensor(
        [r.params.frequency_penalty for r in rows], dtype=torch.float32, device=device
    )
    pres = torch.tensor(
        [r.params.presence_penalty for r in rows], dtype=torch.float32, device=device
    )
    return rep, freq, pres


def assert_apply_matches(
    state: PenaltiesState,
    rows: list[MirrorRequest],
    device,
    rng: random.Random,
):
    num_reqs = len(rows)
    slot_mapping = state.make_slot_mapping(num_reqs)
    logits = torch.randn(num_reqs, VOCAB_SIZE, dtype=torch.float32, device=device)
    expected = reference_apply(logits.cpu(), rows)
    rep, freq, pres = make_penalty_tensors(rows, device)
    state.apply(logits, slot_mapping, rep, freq, pres)
    torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_random_episodes(device):
    """Adds, removals, condense-style moves, swaps, commits across steps."""
    rng = random.Random(1234)
    device = torch.device(device)
    state = PenaltiesState(MAX_NUM_REQS, VOCAB_SIZE, device)
    rows: list[MirrorRequest] = []

    def add_request(req: MirrorRequest):
        rows.append(req)
        state.add_request(len(rows) - 1, req.params, req.token_ids(), len(req.prompt))

    for _ in range(4):
        add_request(MirrorRequest(rng, with_penalties=rng.random() < 0.7))

    for step in range(40):
        # Random batch mutations, mirroring InputBatch behavior.
        if rng.random() < 0.3 and len(rows) < MAX_NUM_REQS - 2:
            add_request(MirrorRequest(rng, with_penalties=rng.random() < 0.7))
        if rng.random() < 0.2 and len(rows) > 2:
            # Remove + condense: last row moves into the gap.
            victim = rng.randrange(len(rows))
            last = len(rows) - 1
            state.remove_row(victim)
            if victim != last:
                state.move_row(last, victim)
                rows[victim] = rows[last]
            rows.pop()
        if rng.random() < 0.2 and len(rows) > 2:
            i1, i2 = rng.sample(range(len(rows)), 2)
            state.swap_rows(i1, i2)
            rows[i1], rows[i2] = rows[i2], rows[i1]

        assert_apply_matches(state, rows, device, rng)

        # Commit this step's sampled tokens (with holes and discards).
        num_reqs = len(rows)
        num_sampled = rng.randint(1, 4)
        sampled = torch.full((num_reqs, num_sampled), -1, dtype=torch.int64)
        discard = torch.zeros(num_reqs, dtype=torch.bool)
        for i, req in enumerate(rows):
            if rng.random() < 0.1:
                discard[i] = True
                sampled[i, 0] = rng.randrange(VOCAB_SIZE)
                continue
            n_accept = rng.randint(0, num_sampled)
            for j in range(n_accept):
                tok = rng.randrange(VOCAB_SIZE)
                sampled[i, j] = tok
                req.output.append(tok)
        slot_mapping = state.make_slot_mapping(num_reqs)
        state.commit(sampled.to(device), slot_mapping, discard.to(device).int())


@pytest.mark.parametrize("device", DEVICES)
def test_spec_decode_draft_and_bonus_rows(device):
    """Expanded draft-verification rows and bonus rows see draft prefixes."""
    rng = random.Random(99)
    device = torch.device(device)
    state = PenaltiesState(MAX_NUM_REQS, VOCAB_SIZE, device)
    rows = [MirrorRequest(rng, with_penalties=i % 3 != 2) for i in range(6)]
    for i, req in enumerate(rows):
        state.add_request(i, req.params, req.token_ids(), len(req.prompt))

    num_drafts = [rng.randint(0, 5) for _ in rows]
    flat_drafts = []
    starts = []
    for k in num_drafts:
        starts.append(len(flat_drafts))
        flat_drafts.extend(rng.randrange(VOCAB_SIZE) for _ in range(k))
    draft_token_ids = torch.tensor(flat_drafts, dtype=torch.int64, device=device)

    slot_mapping = state.make_slot_mapping(len(rows))
    rep, freq, pres = make_penalty_tensors(rows, device)

    # Draft-verification rows: request i contributes num_drafts[i] rows;
    # the row for draft position p sees drafts [0, p).
    row_to_req, prefix_lens, draft_starts = [], [], []
    expanded_rows, expanded_prefixes = [], []
    for i, k in enumerate(num_drafts):
        for p in range(k):
            row_to_req.append(i)
            prefix_lens.append(p)
            draft_starts.append(starts[i])
            expanded_rows.append(rows[i])
            expanded_prefixes.append(flat_drafts[starts[i] : starts[i] + p])
    logits = torch.randn(
        len(row_to_req), VOCAB_SIZE, dtype=torch.float32, device=device
    )
    expected = reference_apply(logits.cpu(), expanded_rows, expanded_prefixes)
    state.apply(
        logits,
        slot_mapping,
        rep,
        freq,
        pres,
        row_to_req=torch.tensor(row_to_req, dtype=torch.int32, device=device),
        prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32, device=device),
        draft_token_ids=draft_token_ids,
        draft_starts=torch.tensor(draft_starts, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-4)

    # Bonus rows: one per request, sees all of its drafts.
    logits = torch.randn(len(rows), VOCAB_SIZE, dtype=torch.float32, device=device)
    expected = reference_apply(
        logits.cpu(),
        rows,
        [flat_drafts[starts[i] : starts[i] + k] for i, k in enumerate(num_drafts)],
    )
    state.apply(
        logits,
        slot_mapping,
        rep,
        freq,
        pres,
        prefix_lens=torch.tensor(num_drafts, dtype=torch.int32, device=device),
        draft_token_ids=draft_token_ids,
        draft_starts=torch.tensor(starts, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(logits.cpu(), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_slot_reuse_no_contamination(device):
    """A recycled slot must not leak the previous request's statistics."""
    rng = random.Random(7)
    device = torch.device(device)
    state = PenaltiesState(MAX_NUM_REQS, VOCAB_SIZE, device)

    heavy = MirrorRequest(rng, with_penalties=True)
    heavy.output = [42] * 100
    state.add_request(0, heavy.params, heavy.token_ids(), len(heavy.prompt))
    state.make_slot_mapping(1)
    state.remove_row(0)

    fresh = MirrorRequest(rng, with_penalties=True)
    fresh.output = []
    state.add_request(0, fresh.params, fresh.token_ids(), len(fresh.prompt))
    assert_apply_matches(state, [fresh], device, rng)


@pytest.mark.parametrize("device", DEVICES)
def test_pool_growth(device):
    """Growing past the initial capacity preserves existing statistics."""
    rng = random.Random(21)
    device = torch.device(device)
    state = PenaltiesState(MAX_NUM_REQS, VOCAB_SIZE, device)
    rows = []
    for i in range(30):  # > _INITIAL_POOL_SLOTS, forces two growths
        req = MirrorRequest(rng, with_penalties=True)
        rows.append(req)
        state.add_request(i, req.params, req.token_ids(), len(req.prompt))
        if i % 10 == 9:
            assert_apply_matches(state, rows, device, rng)
    assert_apply_matches(state, rows, device, rng)
