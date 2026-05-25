# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LogprobsProcessor."""

import numpy as np
import pytest

from vllm.logprobs import Logprob, create_prompt_logprobs, create_sample_logprobs
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists, LogprobsTensors


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
    logprobs = np.array([[-0.5, -1.0, -2.0, -3.0, -np.inf, -np.inf]],
                        dtype=np.float32)
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


class NoToListArray(np.ndarray):
    def tolist(self):
        raise AssertionError("LogprobsProcessor should not materialize rows early")


class NoToListMatrix:
    def __init__(self, values):
        self.values = values

    @property
    def shape(self):
        return (len(self.values), len(self.values[0]))

    def __getitem__(self, index):
        return self.values[index]

    def tolist(self):
        raise AssertionError("LogprobsProcessor should not materialize tensors early")


class NoToListVector:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def tolist(self):
        raise AssertionError("LogprobsProcessor should not materialize tensors early")


def _no_tolist_array(values):
    return np.asarray(values).view(NoToListArray)


class FakeTokenizer:
    def decode(self, token_ids):
        return f"tok{token_ids[0]}"


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_update_sample_logprobs_iterates_numpy_rows_without_tolist(flat_logprobs):
    processor = LogprobsProcessor(
        tokenizer=None,
        logprobs=create_sample_logprobs(flat_logprobs=flat_logprobs),
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=2,
        num_prompt_logprobs=None,
    )

    processor._update_sample_logprobs(
        LogprobsLists(
            logprob_token_ids=_no_tolist_array([[42, 7, 9], [43, 10, 11]]),
            logprobs=_no_tolist_array([[-0.1, -1.0, -2.0], [-0.3, -1.3, -2.3]]),
            sampled_token_ranks=_no_tolist_array([4, 5]),
        ))

    assert processor.cumulative_logprob == pytest.approx(-0.4)
    assert list(processor.logprobs) == [
        {
            42: Logprob(logprob=-0.1, rank=4, decoded_token=None),
            7: Logprob(logprob=-1.0, rank=1, decoded_token=None),
            9: Logprob(logprob=-2.0, rank=2, decoded_token=None),
        },
        {
            43: Logprob(logprob=-0.3, rank=5, decoded_token=None),
            10: Logprob(logprob=-1.3, rank=1, decoded_token=None),
            11: Logprob(logprob=-2.3, rank=2, decoded_token=None),
        },
    ]


@pytest.mark.parametrize("tokenizer", [None, FakeTokenizer()])
def test_update_prompt_logprobs_iterates_tensor_rows_without_tolist(tokenizer):
    processor = LogprobsProcessor(
        tokenizer=tokenizer,
        logprobs=None,
        prompt_logprobs=create_prompt_logprobs(flat_logprobs=False),
        cumulative_logprob=None,
        num_logprobs=None,
        num_prompt_logprobs=2,
    )

    processor._update_prompt_logprobs(
        LogprobsTensors(
            logprob_token_ids=NoToListMatrix([[101, 102, 103], [201, 202, 203]]),
            logprobs=NoToListMatrix([[-0.1, -1.0, -2.0],
                                     [-0.2, -1.2, -2.2]]),
            selected_token_ranks=NoToListVector([3, 4]),
        ))

    expected_decoded_tokens = (
        {token_id: f"tok{token_id}" for token_id in [101, 102, 103, 201, 202, 203]}
        if tokenizer is not None else {})
    assert processor.prompt_logprobs == [
        None,
        {
            101: Logprob(
                logprob=-0.1,
                rank=3,
                decoded_token=expected_decoded_tokens.get(101),
            ),
            102: Logprob(
                logprob=-1.0,
                rank=1,
                decoded_token=expected_decoded_tokens.get(102),
            ),
            103: Logprob(
                logprob=-2.0,
                rank=2,
                decoded_token=expected_decoded_tokens.get(103),
            ),
        },
        {
            201: Logprob(
                logprob=-0.2,
                rank=4,
                decoded_token=expected_decoded_tokens.get(201),
            ),
            202: Logprob(
                logprob=-1.2,
                rank=1,
                decoded_token=expected_decoded_tokens.get(202),
            ),
            203: Logprob(
                logprob=-2.2,
                rank=2,
                decoded_token=expected_decoded_tokens.get(203),
            ),
        },
    ]
