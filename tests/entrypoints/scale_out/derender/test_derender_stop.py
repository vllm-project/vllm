# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Batch and streaming derender must hide stop text the coupled server hides."""

import asyncio
from types import SimpleNamespace

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    DerenderStreamState,
    GenerateResponse,
    GenerateResponseChoice,
    GenerateResponseStreamChoice,
    GenerateStreamResponse,
)
from vllm.renderers.online_derenderer import (
    OnlineDerenderer,
    apply_streaming_stop,
    decode_with_stop,
    normalize_stop_strings,
    truncate_at_stop_string,
)

WORDS = {1: "one", 2: "two", 3: "three", 4: "number"}


class _Tokenizer:
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(WORDS[token_id] for token_id in token_ids)


def _derenderer(eos_token_id=None, generation_eos=None):
    model_config = None
    if generation_eos is not None:
        model_config = SimpleNamespace(
            try_get_generation_config=lambda: {"eos_token_id": generation_eos}
        )
    return SimpleNamespace(
        parser=None,
        model_config=model_config,
        renderer=SimpleNamespace(
            get_tokenizer=lambda: _Tokenizer(eos_token_id=eos_token_id)
        ),
    )


def _chat(token_ids, *, eos_token_id=None, generation_eos=None, **kwargs):
    response = GenerateResponse(
        request_id="chatcmpl-stop",
        choices=[
            GenerateResponseChoice(
                index=0,
                token_ids=token_ids,
                finish_reason=kwargs.pop("finish_reason", "stop"),
            )
        ],
    )
    request = ChatCompletionRequest(
        model="tiny",
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )
    choices = OnlineDerenderer._derender_chat(
        _derenderer(eos_token_id=eos_token_id, generation_eos=generation_eos),
        response,
        request,
    )
    return choices[0]


def test_chat_stop_string_is_left_out_of_the_text():
    choice = _chat([1, 2, 3], stop=["three"])
    # Same cut as the engine: the stop word goes, the space before it stays.
    assert choice.message.content == "one two "
    assert choice.stop_reason == "three"


def test_chat_keeps_the_stop_string_when_asked():
    choice = _chat([1, 2, 3], stop=["three"], include_stop_str_in_output=True)
    assert choice.message.content == "one two three"
    assert choice.stop_reason == "three"


def test_chat_keeps_the_last_token_when_it_is_not_a_stop_token():
    # finish_reason stop alone is not enough. Round-trip fixtures encode
    # real text and do not append a stop id.
    choice = _chat([1, 2, 3])
    assert choice.message.content == "one two three"
    assert choice.stop_reason is None


def test_chat_client_stop_token_drops_the_last_id():
    choice = _chat([1, 2, 3], stop_token_ids=[3])
    assert choice.message.content == "one two"
    assert choice.stop_reason == 3


def test_chat_primary_eos_drops_the_last_id_and_leaves_stop_reason_empty():
    choice = _chat([1, 2, 3], eos_token_id=3)
    assert choice.message.content == "one two"
    assert choice.stop_reason is None


def test_chat_generation_config_eos_drops_the_last_id():
    choice = _chat([1, 2, 3], generation_eos=[3, 9])
    assert choice.message.content == "one two"
    assert choice.stop_reason == 3


def test_chat_keeps_the_stop_token_when_asked():
    choice = _chat([1, 2, 3], stop_token_ids=[3], include_stop_str_in_output=True)
    assert choice.message.content == "one two three"
    assert choice.stop_reason == 3


def test_chat_token_stop_runs_before_the_string_scan():
    # The engine drops the stop token, then looks for a stop string in what remains.
    choice = _chat([1, 2, 3], eos_token_id=3, stop=["two"])
    assert choice.message.content == "one "
    assert choice.stop_reason == "two"


def test_length_finish_keeps_the_last_token():
    choice = _chat([1, 2, 3], finish_reason="length")
    assert choice.message.content == "one two three"
    assert choice.stop_reason is None


def test_completion_stop_string_is_left_out_of_the_text():
    response = GenerateResponse(
        request_id="cmpl-stop",
        choices=[
            GenerateResponseChoice(
                index=0,
                token_ids=[1, 4],
                finish_reason="stop",
            )
        ],
    )
    request = CompletionRequest(model="tiny", prompt="hi", stop=["number"])
    choices, _, _ = OnlineDerenderer._derender_completion(
        _derenderer(),
        [response],
        completion_request=request,
    )
    assert choices[0].text == "one "
    assert choices[0].stop_reason == "number"


def test_streaming_stop_strings_are_rejected_by_the_helper():
    assert normalize_stop_strings(["three"]) == ["three"]
    assert normalize_stop_strings([]) == []
    assert normalize_stop_strings(None) == []


def test_stop_helpers_match_the_coupled_rule():
    text, ids, reason = decode_with_stop(
        _Tokenizer(),
        [1, 2, 3],
        skip_special_tokens=True,
        finish_reason="stop",
        stop=None,
        include_stop_str_in_output=False,
        stop_token_ids=[3],
    )
    assert text == "one two"
    assert ids == [1, 2]
    assert reason == 3
    kept, kept_ids, kept_reason = decode_with_stop(
        _Tokenizer(),
        [1, 2, 3],
        skip_special_tokens=True,
        finish_reason="stop",
        stop=["three"],
        include_stop_str_in_output=False,
    )
    assert kept == "one two "
    assert kept_ids == [1, 2, 3]
    assert kept_reason == "three"
    text, reason = truncate_at_stop_string("one two three", ["three"], False)
    assert text == "one two "
    assert reason == "three"


def test_stream_holds_a_stop_prefix_until_the_word_arrives():
    state = DerenderStreamState()
    emit, state, reason = apply_streaming_stop(
        state,
        "one two ",
        stop=["three"],
        include_in_output=False,
        finished=False,
        token_stripped=False,
        token_stop_reason=None,
    )
    assert emit == "one "
    assert state.held_text == "two "
    assert reason is None

    emit, state, reason = apply_streaming_stop(
        state,
        "three",
        stop=["three"],
        include_in_output=False,
        finished=True,
        token_stripped=False,
        token_stop_reason=None,
    )
    assert emit == "two "
    assert state.held_text == ""
    assert state.stop_fired
    assert reason == "three"

    extra, state, extra_reason = apply_streaming_stop(
        state,
        "more",
        stop=["three"],
        include_in_output=False,
        finished=False,
        token_stripped=False,
        token_stop_reason=None,
    )
    assert extra == ""
    assert extra_reason is None


def test_stream_finish_flushes_the_hold_when_the_stop_never_arrives():
    state = DerenderStreamState(held_text="two ")
    emit, state, reason = apply_streaming_stop(
        state,
        "",
        stop=["three"],
        include_in_output=False,
        finished=True,
        token_stripped=False,
        token_stop_reason=None,
    )
    assert emit == "two "
    assert state.held_text == ""
    assert reason is None


def _stream_chunk(token_ids, finish_reason=None):
    return GenerateStreamResponse(
        request_id="cmpl-stream-stop",
        choices=[
            GenerateResponseStreamChoice(
                index=0,
                token_ids=token_ids,
                finish_reason=finish_reason,
            )
        ],
    )


def test_completion_stream_cuts_a_stop_word_split_across_chunks():
    pieces = iter(["one two ", "three"])
    seen: list[list[int]] = []

    async def detok(tokenizer, delta_token_ids, stream_state, skip_special_tokens=True):
        seen.append(list(delta_token_ids))
        return next(pieces), stream_state

    derenderer = SimpleNamespace(
        _detokenize_delta_async=detok,
        renderer=SimpleNamespace(get_tokenizer=lambda: _Tokenizer()),
        model_config=None,
    )
    request = CompletionRequest(model="tiny", prompt="hi", stop=["three"])

    async def run():
        first, stream_state = await OnlineDerenderer.derender_completion_stream(
            derenderer,
            "tiny",
            _stream_chunk([1, 2]),
            completion_request=request,
        )
        second, stream_state = await OnlineDerenderer.derender_completion_stream(
            derenderer,
            "tiny",
            _stream_chunk([3], finish_reason="stop"),
            state=stream_state,
            completion_request=request,
        )
        return first, second, stream_state

    first, second, stream_state = asyncio.run(run())
    assert first.choices[0].text == "one "
    assert first.choices[0].stop_reason is None
    assert second.choices[0].text == "two "
    assert second.choices[0].stop_reason == "three"
    assert stream_state.stop_fired
    assert seen == [[1, 2], [3]]
