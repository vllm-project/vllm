# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Batch derender must hide stop text the coupled server already hides."""

from types import SimpleNamespace

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.renderers.online_derenderer import (
    OnlineDerenderer,
    normalize_stop_strings,
    output_token_ids_for_text,
    truncate_at_stop_string,
)

WORDS = {1: "one", 2: "two", 3: "three", 4: "number"}


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(WORDS[token_id] for token_id in token_ids)


def _derenderer():
    return SimpleNamespace(
        parser=None,
        renderer=SimpleNamespace(get_tokenizer=lambda: _Tokenizer()),
    )


def _chat(token_ids, **kwargs):
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
    choices = OnlineDerenderer._derender_chat(_derenderer(), response, request)
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


def test_chat_stop_token_is_left_out_of_the_text():
    choice = _chat([1, 2, 3], stop_token_ids=[3])
    assert choice.message.content == "one two"
    assert choice.stop_reason == 3


def test_length_finish_keeps_a_matching_stop_token():
    choice = _chat([1, 2, 3], stop_token_ids=[3], finish_reason="length")
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
    ids, stopped = output_token_ids_for_text(
        [1, 2, 3],
        finish_reason="stop",
        stop_token_ids=[3],
        include_stop_str_in_output=False,
    )
    assert ids == [1, 2]
    assert stopped == 3
    kept, kept_reason = output_token_ids_for_text(
        [1, 2, 3],
        finish_reason="stop",
        stop_token_ids=[3],
        include_stop_str_in_output=True,
    )
    assert kept == [1, 2, 3]
    assert kept_reason == 3
    text, reason = truncate_at_stop_string("one two three", ["three"], False)
    assert text == "one two "
    assert reason == "three"
