# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for stop token id handling in IncrementalDetokenizer.

These tests do not load a model or need a GPU. They pin down that
include_stop_str_in_output=False strips the matched stop token from
output text and streamed deltas, including when the caller does not
set stop_terminated (the engine-core flag used for EOS and stop ids).
"""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer

pytestmark = pytest.mark.skip_global_cleanup


class _DummyDetokenizer(BaseIncrementalDetokenizer):
    def decode_next(self, next_token_id: int) -> str:
        return chr(next_token_id)


def _make_request(
    *,
    stop: list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    include_stop_str_in_output: bool = False,
    min_tokens: int = 0,
) -> EngineCoreRequest:
    params = SamplingParams(
        stop=stop,
        stop_token_ids=stop_token_ids,
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


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


@pytest.mark.parametrize("stop_terminated", [True, False])
def test_stop_token_id_excluded_from_output(stop_terminated: bool):
    """Stop token text is omitted when include_stop_str_in_output=False.

    The engine-core skip path sets stop_terminated=True. The detokenizer
    must also honor stop_token_ids when that flag is false, otherwise
    the matched token remains in output_text and streamed deltas.
    """
    stop_id = ord("!")
    req = _make_request(
        stop_token_ids=[stop_id],
        include_stop_str_in_output=False,
    )
    detok = _DummyDetokenizer(req)

    assert detok.update(_ids("hello"), False) is None
    assert detok.update([stop_id], stop_terminated) is None
    assert detok.output_text == "hello"
    assert detok.output_token_ids == _ids("hello!")
    assert detok.get_next_output_text(finished=True, delta=False) == "hello"


@pytest.mark.parametrize("stop_terminated", [True, False])
def test_stop_token_id_included_in_output(stop_terminated: bool):
    stop_id = ord("!")
    req = _make_request(
        stop_token_ids=[stop_id],
        include_stop_str_in_output=True,
    )
    detok = _DummyDetokenizer(req)

    assert detok.update(_ids("hello"), False) is None
    assert detok.update([stop_id], stop_terminated) is None
    assert detok.output_text == "hello!"
    assert detok.output_token_ids == _ids("hello!")
    assert detok.get_next_output_text(finished=True, delta=False) == "hello!"


def test_stop_token_id_not_leaked_in_streamed_deltas():
    stop_id = ord("!")
    req = _make_request(
        stop_token_ids=[stop_id],
        include_stop_str_in_output=False,
    )
    detok = _DummyDetokenizer(req)

    detok.update(_ids("hel"), False)
    assert detok.get_next_output_text(finished=False, delta=True) == "hel"

    detok.update(_ids("lo"), False)
    assert detok.get_next_output_text(finished=False, delta=True) == "lo"

    # stop_terminated=False would previously detokenize the stop token
    # and emit it as a delta before the request was marked finished.
    detok.update([stop_id], stop_terminated=False)
    assert detok.get_next_output_text(finished=True, delta=True) == ""
    assert detok.output_text == "hello"


def test_stop_token_id_in_same_step_as_prior_tokens():
    stop_id = ord("!")
    req = _make_request(
        stop_token_ids=[stop_id],
        include_stop_str_in_output=False,
    )
    detok = _DummyDetokenizer(req)

    detok.update(_ids("hello!more"), stop_terminated=False)
    assert detok.output_text == "hello"
    assert detok.output_token_ids == _ids("hello!more")


def test_include_stop_token_drops_suffix_in_same_step():
    """include=True keeps the stop token text and drops tokens after it."""
    stop_id = ord("!")
    req = _make_request(
        stop_token_ids=[stop_id],
        include_stop_str_in_output=True,
    )
    detok = _DummyDetokenizer(req)

    detok.update(_ids("hello!more"), stop_terminated=False)
    assert detok.output_text == "hello!"
    assert detok.output_token_ids == _ids("hello!more")
    assert detok.get_next_output_text(finished=True, delta=False) == "hello!"


def test_stop_token_id_before_min_tokens_is_ordinary_output():
    """A stop id before min_tokens is decoded as normal text."""
    stop_id = ord("!")
    req = _make_request(
        stop_token_ids=[stop_id],
        include_stop_str_in_output=False,
        min_tokens=5,
    )
    detok = _DummyDetokenizer(req)

    detok.update(_ids("hi!x"), stop_terminated=False)
    assert detok.output_text == "hi!x"
    assert detok.output_token_ids == _ids("hi!x")

    # 4 tokens so far; the next stop id is still below min_tokens=5.
    detok.update(_ids("y"), stop_terminated=False)
    assert detok.output_text == "hi!xy"

    detok.update(_ids("!z"), stop_terminated=False)
    assert detok.output_text == "hi!xy"
    assert detok.output_token_ids == _ids("hi!xy!z")


def test_stop_strings_still_truncate_when_stop_token_ids_set():
    stop_id = ord("Z")
    req = _make_request(
        stop=["cd"],
        stop_token_ids=[stop_id],
        include_stop_str_in_output=False,
    )
    detok = _DummyDetokenizer(req)

    result = detok.update(_ids("abcdeZ"), stop_terminated=True)
    assert result == "cd"
    assert detok.output_text == "ab"
    assert detok.output_token_ids == _ids("abcdeZ")
