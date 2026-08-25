# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-side mRoPE streaming helpers on GPUModelRunner.

Exercises `_slice_mrope_positions_for_evicted_ranges` (drop evicted columns)
and `_extend_mrope_positions_for_streaming_chunk` (append the new chunk's
positions from max_cached_position + 1) in isolation — no real model or
scheduler. Synthetic `CachedRequestState`s are driven through both; the
extend tests use a tiny `SupportsMRoPE` stub via a fake runner that exposes
only `get_model()`.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test


def _make_req_state(num_tokens: int) -> CachedRequestState:
    positions = torch.tensor(
        [[i for i in range(num_tokens)] for _ in range(3)],
        dtype=torch.long,
    )
    return CachedRequestState(
        req_id="r-slice",
        prompt_token_ids=list(range(num_tokens)),
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([],),
        num_computed_tokens=num_tokens,
        output_token_ids=[],
        mrope_positions=positions,
        mrope_position_delta=0,
        max_cached_position=num_tokens - 1,
    )


def _slice_positions(state, ranges):
    GPUModelRunner._slice_mrope_positions_for_evicted_ranges(state, ranges)


def _row(state):
    return state.mrope_positions[0].tolist()


def test_single_middle_range():
    state = _make_req_state(16)
    _slice_positions(state, [(4, 8)])
    assert state.mrope_positions.shape == (3, 12)
    assert _row(state) == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]


def test_prefix_range():
    state = _make_req_state(8)
    _slice_positions(state, [(0, 3)])
    assert state.mrope_positions.shape == (3, 5)
    assert _row(state) == [3, 4, 5, 6, 7]


def test_suffix_range():
    state = _make_req_state(8)
    _slice_positions(state, [(5, 8)])
    assert state.mrope_positions.shape == (3, 5)
    assert _row(state) == [0, 1, 2, 3, 4]


def test_sequential_ranges_are_applied_post_shrink():
    """Two evictions in one chunk: first slices [4,8), then the SECOND
    range refers to the post-first-shrink tensor — applied AFTER the
    first range. Validates the sequential-apply contract."""
    state = _make_req_state(16)
    # First: drop [4, 8) → tensor becomes [0,1,2,3, 8,9,...,15] (12 cols)
    # Second: drop [4, 6) of the SHRUNK tensor → drop [8,9) of original
    #          → final tensor: [0,1,2,3, 10,11,12,13,14,15]
    _slice_positions(state, [(4, 8), (4, 6)])
    assert state.mrope_positions.shape == (3, 10)
    assert _row(state) == [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]


def test_range_extending_past_width_is_clipped():
    state = _make_req_state(8)
    _slice_positions(state, [(5, 100)])
    assert state.mrope_positions.shape == (3, 5)
    assert _row(state) == [0, 1, 2, 3, 4]


def test_empty_and_invalid_ranges_are_no_op():
    state = _make_req_state(4)
    _slice_positions(state, [])
    _slice_positions(state, [(2, 2)])
    _slice_positions(state, [(3, 1)])  # invalid order
    assert _row(state) == [0, 1, 2, 3]


def test_no_op_when_mrope_positions_is_none():
    state = _make_req_state(4)
    state.mrope_positions = None
    _slice_positions(state, [(0, 2)])
    assert state.mrope_positions is None


# ---------------------------------------------------------------------------
# GPUModelRunner._extend_mrope_positions_for_streaming_chunk
#
# Appends positions for the new chunk only (tokens past the prior chunk's
# mrope_positions width), starting at max_cached_position + 1. Driven with a
# tiny SupportsMRoPE stub and a fake runner exposing only get_model().
# ---------------------------------------------------------------------------


class _StubMRoPEModel:
    """Minimal SupportsMRoPE model: every token gets linear (p, p, p)."""

    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        n = len(input_tokens)
        positions = (
            torch.tensor([[i, i, i] for i in range(n)], dtype=torch.long)
            .t()
            .contiguous()
        )
        max_val = int(positions.max().item()) if n else -1
        return positions, max_val + 1 - n


class _FakeRunner:
    """Stands in for GPUModelRunner — provides the only hooks _extend touches:
    get_model() (main path) and _init_mrope_positions() (the None fallback,
    delegated to the real method, which itself only needs get_model())."""

    def __init__(self, model):
        self._model = model

    def get_model(self):
        return self._model

    def _init_mrope_positions(self, req_state):
        GPUModelRunner._init_mrope_positions(self, req_state)


def _extend(model, state):
    GPUModelRunner._extend_mrope_positions_for_streaming_chunk(
        _FakeRunner(model), state
    )


def _make_extend_state(prev_width: int, num_prompt_tokens: int) -> CachedRequestState:
    """A session with `prev_width` cached positions and a prompt grown to
    `num_prompt_tokens` (the new chunk is the tail past prev_width)."""
    positions = torch.tensor(
        [list(range(prev_width)) for _ in range(3)], dtype=torch.long
    )
    state = CachedRequestState(
        req_id="r-extend",
        prompt_token_ids=list(range(num_prompt_tokens)),
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([],),
        num_computed_tokens=prev_width,
        output_token_ids=[],
        mrope_positions=positions,
        mrope_position_delta=0,
        max_cached_position=prev_width - 1,
    )
    # Set by the worker (`_update_streaming_request`) before _extend runs.
    state.num_prompt_tokens = num_prompt_tokens
    return state


def test_extend_appends_new_chunk_above_cached_max():
    state = _make_extend_state(prev_width=4, num_prompt_tokens=7)
    _extend(_StubMRoPEModel(), state)
    # 3 new columns appended; they start strictly above the prior max (3).
    assert state.mrope_positions.shape == (3, 7)
    assert _row(state) == [0, 1, 2, 3, 4, 5, 6]
    assert min(_row(state)[4:]) > 3
    assert state.max_cached_position == 6
    # Decode continuation delta = new_max + 1 - num_prompt_tokens.
    assert state.mrope_position_delta == 0


def test_extend_noop_when_no_new_tokens():
    state = _make_extend_state(prev_width=5, num_prompt_tokens=5)
    _extend(_StubMRoPEModel(), state)
    assert state.mrope_positions.shape == (3, 5)
    assert _row(state) == [0, 1, 2, 3, 4]
    assert state.max_cached_position == 4


def test_extend_falls_back_to_full_init_when_positions_missing():
    """No cached positions (e.g. a re-prefill reset null'd them): _extend
    must fall back to a full _init_mrope_positions over the whole prompt."""
    state = _make_extend_state(prev_width=4, num_prompt_tokens=6)
    state.mrope_positions = None
    state.max_cached_position = None
    _extend(_StubMRoPEModel(), state)
    assert state.mrope_positions.shape == (3, 6)
    assert _row(state) == [0, 1, 2, 3, 4, 5]
    assert state.max_cached_position == 5
