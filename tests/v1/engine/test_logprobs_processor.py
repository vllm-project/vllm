# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LogprobsProcessor."""

import numpy as np
import pytest

from vllm.logprobs import create_prompt_logprobs, create_sample_logprobs
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

pytestmark = pytest.mark.skip_global_cleanup


def _make_processor(num_logprobs: int) -> LogprobsProcessor:
    return LogprobsProcessor(
        tokenizer=None,
        logprobs=create_sample_logprobs(flat_logprobs=False),
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


class TrackingArray(np.ndarray):
    def __new__(cls, values):
        array = np.asarray(values).view(cls)
        array.materialized_shapes = []
        return array

    def __array_finalize__(self, source):
        self.materialized_shapes = getattr(source, "materialized_shapes", [])

    def tolist(self):
        self.materialized_shapes.append(self.shape)
        return super().tolist()


class FakeTokenizer:
    def decode(self, token_ids):
        return f"tok{token_ids[0]}"


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_update_sample_logprobs_materializes_only_requested_columns(flat_logprobs):
    processor = LogprobsProcessor(
        tokenizer=None,
        logprobs=create_sample_logprobs(flat_logprobs=flat_logprobs),
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=2,
        num_prompt_logprobs=None,
    )

    token_ids = TrackingArray([[42, 7, 9, 0, 0], [43, 10, 11, 0, 0]])
    logprobs = TrackingArray(
        [[-0.1, -1.0, -2.0, -np.inf, -np.inf], [-0.3, -1.3, -2.3, -np.inf, -np.inf]]
    )
    ranks = TrackingArray([4, 5])
    processor._update_sample_logprobs(LogprobsLists(token_ids, logprobs, ranks))

    assert token_ids.materialized_shapes == [(2, 3)]
    assert logprobs.materialized_shapes == [(2, 3)]
    assert ranks.materialized_shapes == [(2,)]

    assert processor.cumulative_logprob == pytest.approx(-0.4)
    positions = list(processor.logprobs)
    assert [list(position) for position in positions] == [[42, 7, 9], [43, 10, 11]]
    assert [position[ids[0]].rank for position, ids in zip(positions, token_ids)] == [
        4,
        5,
    ]
    for position, expected in zip(positions, logprobs):
        assert [item.logprob for item in position.values()] == pytest.approx(
            expected[:3]
        )


@pytest.mark.parametrize("tokenizer", [None, FakeTokenizer()])
def test_update_prompt_logprobs_materializes_requested_columns_once(tokenizer):
    processor = LogprobsProcessor(
        tokenizer=tokenizer,
        logprobs=None,
        prompt_logprobs=create_prompt_logprobs(flat_logprobs=False),
        cumulative_logprob=None,
        num_logprobs=None,
        num_prompt_logprobs=2,
    )

    token_ids = TrackingArray([[101, 102, 103, 0, 0], [201, 202, 203, 0, 0]])
    logprobs = TrackingArray(
        [[-0.1, -1.0, -2.0, -np.inf, -np.inf], [-0.2, -1.2, -2.2, -np.inf, -np.inf]]
    )
    ranks = TrackingArray([3, 4])
    processor._update_prompt_logprobs(LogprobsTensors(token_ids, logprobs, ranks))

    assert token_ids.materialized_shapes == [(2, 3)]
    assert logprobs.materialized_shapes == [(2, 3)]
    assert ranks.materialized_shapes == [(2,)]

    positions = processor.prompt_logprobs[1:]
    assert [list(position) for position in positions] == [
        [101, 102, 103],
        [201, 202, 203],
    ]
    assert [position[ids[0]].rank for position, ids in zip(positions, token_ids)] == [
        3,
        4,
    ]
    for position, expected in zip(positions, logprobs):
        assert [item.logprob for item in position.values()] == pytest.approx(
            expected[:3]
        )
        assert [item.decoded_token for item in position.values()] == [
            f"tok{token_id}" if tokenizer is not None else None for token_id in position
        ]
