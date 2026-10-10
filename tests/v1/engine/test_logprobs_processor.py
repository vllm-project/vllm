# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LogprobsProcessor.

These tests exercise the truncation invariant that the MRV2 sampler relies
on: when the sampler returns a row wider than a request's own
`num_logprobs + 1` (because another request in the batch needed a wider
row), the trailing positions are populated with sentinel values
(`token_id=0`, `logprob=-inf`). LogprobsProcessor must read only the first
`num_logprobs + 1` entries so those sentinels never reach the user.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.logprobs import FlatLogprobs, create_sample_logprobs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists


def _make_processor(
    num_logprobs: int, flat_logprobs: bool = False
) -> LogprobsProcessor:
    return LogprobsProcessor(
        tokenizer=None,
        logprobs=create_sample_logprobs(flat_logprobs=flat_logprobs),
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=num_logprobs,
        num_prompt_logprobs=None,
    )


def test_drops_trailing_sentinel_columns():
    """A request that asked for 3 custom token logprobs but ended up in a
    batch padded to width 5 must not surface the trailing -inf entries."""
    processor = _make_processor(num_logprobs=3)

    sampled = 42
    # Layout: [sampled, custom_1, custom_2, custom_3, SENTINEL, SENTINEL]
    # Use float32-exact values so cumulative_logprob compares cleanly.
    token_ids = np.array([[sampled, 100, 200, 300, 0, 0]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.0, -2.0, -3.0, -np.inf, -np.inf]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(LogprobsLists(token_ids, logprobs, ranks))

    assert len(processor.logprobs) == 1
    pos = processor.logprobs[0]
    # Exactly sampled + 3 requested tokens; trailing sentinels dropped.
    assert set(pos.keys()) == {sampled, 100, 200, 300}
    assert 0 not in pos
    assert all(np.isfinite(lp.logprob) for lp in pos.values())
    # cumulative_logprob comes from the sampled token's logprob only.
    assert processor.cumulative_logprob == -0.5


def test_accepts_exactly_sized_row():
    """When the row is exactly num_logprobs+1, no truncation needed."""
    processor = _make_processor(num_logprobs=2)

    token_ids = np.array([[7, 11, 13]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.5, -2.5]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(LogprobsLists(token_ids, logprobs, ranks))

    pos = processor.logprobs[0]
    assert set(pos.keys()) == {7, 11, 13}


def test_prompt_token_id_logprobs_are_popped_once():
    """DELTA outputs carry fixed-ID prompt scores exactly once."""
    processor = _make_processor(num_logprobs=1)
    assert processor.pop_prompt_token_id_logprobs() is None

    processor.update_from_output(
        EngineCoreOutput(
            request_id="req-0",
            new_token_ids=[],
            prompt_token_id_logprobs=torch.tensor(
                [[-0.5, -1.5], [-2.5, -3.5]], dtype=torch.float32
            ),
        )
    )

    scores = processor.pop_prompt_token_id_logprobs()
    assert scores is not None
    assert scores.tolist() == [[-0.5, -1.5], [-2.5, -3.5]]
    assert processor.pop_prompt_token_id_logprobs() is None


def _engine_steps(seed: int, width: int, num_steps: int) -> list[LogprobsLists]:
    """Engine steps of 1-3 rows, some repeating the sampled id in the top-k."""
    rng = np.random.default_rng(seed)
    steps = []
    for _ in range(num_steps):
        rows = int(rng.integers(1, 4))
        token_ids = rng.integers(0, 1000, (rows, width)).astype(np.int32)
        dup = (rng.random(rows) < 0.3) & (width > 1)
        token_ids[dup, -1] = token_ids[dup, 0]
        logprobs = -rng.random((rows, width)).astype(np.float32) * 10
        ranks = rng.integers(0, 50, rows).astype(np.int64)
        steps.append(LogprobsLists(token_ids, logprobs, ranks))
    return steps


@pytest.mark.parametrize("num_logprobs,width", [(0, 1), (3, 4), (3, 6), (-1, 5)])
def test_flat_rows_match_list_logprobs(num_logprobs, width):
    """FlatLogprobs keeps the engine rows but reads like the list path:
    same positions (first-occurrence keys, last-occurrence values, ranks),
    same cumulative logprob."""
    expected = _make_processor(num_logprobs)
    actual = _make_processor(num_logprobs, flat_logprobs=True)
    for step in _engine_steps(0, width, 40):
        expected._update_sample_logprobs(step)
        actual._update_sample_logprobs(step)

    assert isinstance(actual.logprobs, FlatLogprobs)
    assert list(actual.logprobs) == expected.logprobs
    assert [actual.logprobs[i] for i in range(len(expected.logprobs))] == (
        expected.logprobs
    )
    assert list(actual.logprobs[-3:]) == expected.logprobs[-3:]
    assert actual.cumulative_logprob == expected.cumulative_logprob

    rows = actual.logprobs.rows()
    assert rows is not None
    token_ids, logprobs, ranks = rows
    steps = _engine_steps(0, width, 40)
    slots = width if num_logprobs == -1 else num_logprobs + 1
    assert token_ids.dtype == np.dtype("<i4") and logprobs.dtype == np.dtype("<f4")
    assert token_ids.flags.c_contiguous and logprobs.flags.c_contiguous
    np.testing.assert_array_equal(
        token_ids, np.concatenate([s.logprob_token_ids[:, :slots] for s in steps])
    )
    np.testing.assert_array_equal(
        logprobs, np.concatenate([s.logprobs[:, :slots] for s in steps])
    )
    np.testing.assert_array_equal(
        ranks, np.concatenate([s.sampled_token_ranks for s in steps])
    )


def test_flat_irregular_rows_have_no_engine_rows():
    """A row narrower than the stored ones (a co-batched request replaced the
    batch's logprob tensors) still reads like the list path, but rows() is
    None."""
    expected = _make_processor(3)
    actual = _make_processor(3, flat_logprobs=True)
    steps = _engine_steps(1, 4, 3) + _engine_steps(2, 2, 2) + _engine_steps(3, 4, 2)
    for step in steps:
        expected._update_sample_logprobs(step)
        actual._update_sample_logprobs(step)

    assert isinstance(actual.logprobs, FlatLogprobs)
    assert list(actual.logprobs) == expected.logprobs
    assert actual.logprobs.rows() is None


def test_sample_logprobs_skip_detokenize():
    """_detokenize_logprobs=False keeps no decoded tokens, even with a
    tokenizer (which would fail here if used)."""
    params = SamplingParams(logprobs=2, flat_logprobs=True)
    params._detokenize_logprobs = False
    processor = LogprobsProcessor.from_new_request(
        tokenizer=object(), request=SimpleNamespace(sampling_params=params)
    )
    for step in _engine_steps(4, 3, 3):
        processor._update_sample_logprobs(step)
    assert isinstance(processor.logprobs, FlatLogprobs)
    assert set(processor.logprobs.decoded_tokens) == {None}
