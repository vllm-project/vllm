# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client stop strings must stay dormant inside the reasoning segment.

With think-in-prompt templates (DeepSeek V4, Qwen3 style) the prompt ends with
the reasoning start token, so generation begins inside ``<think>``. Evaluating
client ``stop`` strings there truncates the chain-of-thought whenever it
restates a stop phrase, the end marker never arrives, and the reasoning parser
yields ``content: None``. See gh-issue: 52393.
"""

import pytest
from transformers import AutoTokenizer

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import IncrementalDetokenizer

TOKENIZER_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _make_request(prompt_token_ids: list[int], stop: list[str]) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="test",
        external_req_id="test-ext",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(stop=stop, min_tokens=0),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _detokenizer(tokenizer, *, prompt_ends_in_think: bool, stop: list[str]):
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    assert think_id is not None and think_id >= 0
    prompt = tokenizer.encode("Question: what is 2+2?")
    if prompt_ends_in_think:
        prompt = prompt + [think_id]
    return IncrementalDetokenizer.from_new_request(tokenizer, _make_request(prompt, stop))


def _feed(detokenizer, token_ids: list[int]):
    """Feed tokens one at a time; return the first stop string hit, else None."""
    for token_id in token_ids:
        stop_string = detokenizer.update([token_id], False)
        if stop_string is not None:
            return stop_string
    return None


def test_stop_dormant_inside_reasoning(tokenizer):
    """A stop phrase restated by the CoT must not fire before ``</think>``,
    and the same phrase must still fire in content after the close."""
    detokenizer = _detokenizer(tokenizer, prompt_ends_in_think=True, stop=["Question:"])
    reasoning = tokenizer.encode(
        "Let me restate the problem. Question: what is 2+2? It is four.",
        add_special_tokens=False,
    )
    assert _feed(detokenizer, reasoning) is None, (
        "stop fired inside the open reasoning segment"
    )
    close = tokenizer.encode("</think>", add_special_tokens=False)
    assert _feed(detokenizer, close) is None
    content = tokenizer.encode(
        "The answer is 4. Question: next?", add_special_tokens=False
    )
    assert _feed(detokenizer, content) == "Question:", (
        "stop must still fire in content after the reasoning segment closes"
    )


def test_stop_after_marker_in_same_chunk(tokenizer):
    """Spec decoding delivers multi-token chunks: the chunk carrying the end
    marker also carries the reasoning tail, and a stop phrase in that tail
    must not fire — only text following the marker is eligible."""
    detokenizer = _detokenizer(tokenizer, prompt_ends_in_think=True, stop=["Question:"])
    one_chunk = tokenizer.encode(
        "Almost done. Question: restated once more.</think>The answer is 4.",
        add_special_tokens=False,
    )
    stop_string = detokenizer.update(one_chunk, False)
    assert stop_string is None, (
        "stop in the pre-marker reasoning tail of the closing chunk fired"
    )
    trailing = tokenizer.encode(" Question: next?", add_special_tokens=False)
    assert _feed(detokenizer, trailing) == "Question:"


def test_non_thinking_request_unaffected(tokenizer):
    """A prompt that does not end inside reasoning keeps stock behavior."""
    detokenizer = _detokenizer(tokenizer, prompt_ends_in_think=False, stop=["Question:"])
    output = tokenizer.encode(
        "Sure. Question: what next?", add_special_tokens=False
    )
    assert _feed(detokenizer, output) == "Question:"


def test_opt_out_env(tokenizer, monkeypatch):
    """``VLLM_SUPPRESS_STOPS_IN_REASONING=0`` restores stock behavior."""
    monkeypatch.setenv("VLLM_SUPPRESS_STOPS_IN_REASONING", "0")
    detokenizer = _detokenizer(tokenizer, prompt_ends_in_think=True, stop=["Question:"])
    reasoning = tokenizer.encode(
        "Restating: Question: what is 2+2?", add_special_tokens=False
    )
    assert _feed(detokenizer, reasoning) == "Question:"
