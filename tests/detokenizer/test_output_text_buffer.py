# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The detokenizer buffers output text and joins it lazily. These tests cover
outputs long enough to span several buffered chunks."""

import random

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import (
    _OUTPUT_TEXT_CHUNK_PIECES,
    BaseIncrementalDetokenizer,
)

NUM_TOKENS = 5 * _OUTPUT_TEXT_CHUNK_PIECES + 7


class _PiecesDetokenizer(BaseIncrementalDetokenizer):
    """Decodes token id i to pieces[i]."""

    def __init__(self, request: EngineCoreRequest, pieces: list[str]):
        super().__init__(request)
        self.pieces = pieces

    def decode_next(self, next_token_id: int) -> str:
        return self.pieces[next_token_id]


def _make_request(stop=None, include_stop_str_in_output=False, min_tokens=0):
    params = SamplingParams(
        stop=stop,
        include_stop_str_in_output=include_stop_str_in_output,
        min_tokens=min_tokens,
    )
    return EngineCoreRequest(
        request_id="test",
        prompt_token_ids=[],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _random_pieces(seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = "abc xyz\né中\U0001f600"
    return [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 4)))
        for _ in range(NUM_TOKENS)
    ]


def _steps(num_tokens: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    steps, start = [], 0
    while start < num_tokens:
        size = rng.choice([1, 1, 1, 2, 3, 5])
        steps.append(list(range(start, min(start + size, num_tokens))))
        start += size
    return steps


@pytest.mark.parametrize("seed", range(3))
def test_output_text_matches_decoded_pieces(seed):
    pieces = _random_pieces(seed)
    detok = _PiecesDetokenizer(_make_request(), pieces)
    expected = ""
    for step in _steps(len(pieces), seed):
        assert detok.update(step, False) is None
        expected += "".join(pieces[i] for i in step)
        assert detok.output_text == expected
    assert detok.get_next_output_text(finished=True, delta=False) == expected


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("stop", [None, ["zz\n"]])
def test_streamed_deltas_reassemble_output_text(seed, stop):
    pieces = _random_pieces(seed)
    detok = _PiecesDetokenizer(_make_request(stop=stop), pieces)
    buffer_length = detok.stop_buffer_length
    streamed = ""
    for step in _steps(len(pieces), seed):
        if detok.update(step, False) is not None:
            break
        streamed += detok.get_next_output_text(finished=False, delta=True)
        # Characters that could still start a stop string are held back.
        full = detok.output_text
        assert streamed == full[: max(len(full) - buffer_length, 0)]
    streamed += detok.get_next_output_text(finished=True, delta=True)
    assert streamed == detok.output_text


@pytest.mark.parametrize("include_stop_str_in_output", [False, True])
@pytest.mark.parametrize("step_size", [1, 3])
def test_stop_string_after_several_chunks(include_stop_str_in_output, step_size):
    # The stop string straddles token boundaries well past the first chunk.
    stop_at = 3 * _OUTPUT_TEXT_CHUNK_PIECES + 5
    pieces = ["a"] * NUM_TOKENS
    pieces[stop_at : stop_at + 3] = ["xS", "T", "OPy"]
    detok = _PiecesDetokenizer(
        _make_request(
            stop=["STOP"], include_stop_str_in_output=include_stop_str_in_output
        ),
        pieces,
    )
    result = None
    for start in range(0, NUM_TOKENS, step_size):
        result = detok.update(list(range(start, start + step_size)), False)
        if result is not None:
            break

    assert result == "STOP"
    prefix = "a" * stop_at + "x"
    expected_text = prefix + "STOP" if include_stop_str_in_output else prefix
    assert detok.output_text == expected_text
    # Tokens past the one completing the stop string are dropped.
    assert detok.output_token_ids == list(range(stop_at + 3))
    assert detok.get_next_output_text(finished=True, delta=False) == expected_text
