# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from transformers import AutoTokenizer

from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import (
    BaseIncrementalDetokenizer,
    IncrementalDetokenizer,
    ReasoningEndFinder,
)
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.outputs import LogprobsLists, SamplingMaskLists


@pytest.fixture(params=[True, False])
def include_stop_str_in_output(request):
    return request.param


class _DummyDetokenizer(BaseIncrementalDetokenizer):
    def __init__(
        self,
        request: EngineCoreRequest,
        reasoning_end_finder: ReasoningEndFinder | None = None,
    ):
        super().__init__(request, reasoning_end_finder)

    def decode_next(self, next_token_id: int) -> str:
        # Map token id to single ASCII character for deterministic testing.
        return chr(next_token_id)


class _PipeEndsReasoning:
    """Token-id reasoner whose reasoning ends at "|"."""

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return ord("|") in delta_ids


def _make_request(stop, include_stop_str_in_output: bool, min_tokens: int = 0):
    params = SamplingParams(
        stop=stop,
        include_stop_str_in_output=include_stop_str_in_output,
        min_tokens=min_tokens,
    )
    # Keep other fields minimal for unit test purposes.
    req = EngineCoreRequest(
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
    return req


def test_stop_string_while_stop_token_terminates(include_stop_str_in_output: bool):
    """This test verifies that the detokenizer correctly handles the case where
    the generated token sequence contains both:
    - a stop token
    - an <eos> token

    The detokenizer should respect the stop string and truncate the output
    accordingly.

    Imagine the following sequence:
    - "abcdeZ" is generated, where "Z" is the <eos> token.
    - "cd" is the stop string.

    If include_stop_str_in_output=False, the detokenizer should truncate the
    output to "ab" because the stop string "cd" is excluded.
    If include_stop_str_in_output=True, the detokenizer should include the stop
    string "cd" in the output, resulting in "abcd".


    This verifies the behavioral change introduced in BaseIncrementalDetokenizer
    where stop-string evaluation occurs before the early-return on
    stop_terminated.
    """
    # Generate text "abcdeZ" and tokenize it.
    generated_text = "abcde"
    eos_token = "Z"
    stop_string = "cd"
    generated_text = generated_text + eos_token
    token_ids = [ord(c) for c in generated_text]

    # Create a request with the stop string and initialize the detokenizer.
    req = _make_request(
        stop=[stop_string], include_stop_str_in_output=include_stop_str_in_output
    )
    detok = _DummyDetokenizer(req)

    # Simulate that the last token ('Z') is a stop token (stop_terminated=True).
    result = detok.update(new_token_ids=token_ids, stop_terminated=True)

    # The update should report the matched stop string.
    assert result == stop_string

    # Output text should reflect stop-string handling:
    # - include_stop_str_in_output=False => exclude "cd" => "ab"
    # - include_stop_str_in_output=True  => include "cd" => "abcd"
    expected_text = "abcd" if include_stop_str_in_output else "ab"
    assert detok.output_text == expected_text

    # Tokens after the stop string, including the stop token, are dropped.
    assert detok.output_token_ids == token_ids[:4]

    # get_next_output_text should return the full text when finished=True.
    # (Buffering only applies during streaming when finished=False.)
    assert detok.get_next_output_text(finished=True, delta=False) == expected_text


@pytest.mark.parametrize(
    "min_tokens,num_kept",
    [(0, 2), (1, 2), (2, 4), (3, 4), (4, None)],
)
@pytest.mark.parametrize("num_prev_tokens", [0, 1])
def test_stop_string_min_tokens_within_step(
    include_stop_str_in_output: bool, min_tokens, num_kept, num_prev_tokens
):
    """Stop strings completing within min_tokens are ignored, even when the
    min_tokens boundary falls in the middle of a multi-token step."""
    token_ids = [ord(c) for c in "cdcdx"]
    req = _make_request(
        stop=["cd"],
        include_stop_str_in_output=include_stop_str_in_output,
        min_tokens=min_tokens,
    )
    detok = _DummyDetokenizer(req)

    if num_prev_tokens:
        assert detok.update(token_ids[:num_prev_tokens], False) is None
    result = detok.update(token_ids[num_prev_tokens:], False)

    if num_kept is None:
        assert result is None
        assert detok.output_text == "cdcdx"
        assert detok.output_token_ids == token_ids
        return
    assert result == "cd"
    text_len = num_kept if include_stop_str_in_output else num_kept - 2
    assert detok.output_text == "cdcdx"[:text_len]
    assert detok.output_token_ids == token_ids[:num_kept]


@pytest.mark.parametrize(
    "decoded_tokens,keep",
    [
        (["a", "b", "c", "d", "e", "f"], 4),
        (["ab", "cdx", "ef"], 2),
        (["ab", "", "c", "", "d", "", "ef"], 5),
    ],
)
@pytest.mark.parametrize("stop_terminated", [False, True])
def test_stop_string_trims_speculative_overflow(
    include_stop_str_in_output: bool,
    decoded_tokens,
    keep,
    stop_terminated: bool,
    monkeypatch,
):
    if stop_terminated:
        # The engine also stopped on a stop token at the end of the batch.
        decoded_tokens = decoded_tokens + [""]
    token_ids = list(range(len(decoded_tokens)))
    stop_string = "cd"
    expected_token_ids = token_ids[:keep]

    req = _make_request(
        stop=[stop_string], include_stop_str_in_output=include_stop_str_in_output
    )
    req.external_req_id = req.request_id
    assert req.sampling_params is not None
    req.sampling_params.logprobs = 0
    detok = _DummyDetokenizer(req)
    monkeypatch.setattr(detok, "decode_next", decoded_tokens.__getitem__)
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    processor.add_request(req, prompt=None)
    processor.request_states[req.request_id].detokenizer = detok

    outputs = processor.process_outputs(
        [
            EngineCoreOutput(
                request_id=req.request_id,
                new_token_ids=token_ids,
                finish_reason=FinishReason.STOP if stop_terminated else None,
                new_logprobs=LogprobsLists(
                    np.array(token_ids).reshape(-1, 1),
                    np.zeros((len(token_ids), 1)),
                    np.ones(len(token_ids), dtype=int),
                    None,
                ),
                new_sampling_mask=SamplingMaskLists(
                    np.array(token_ids), np.arange(len(token_ids) + 1)
                ),
                routed_experts=np.array(token_ids).reshape(-1, 1, 1),
            )
        ]
    )
    result = outputs.request_outputs[0].outputs[0]

    assert result.stop_reason == stop_string
    expected_text = "abcd" if include_stop_str_in_output else "ab"
    assert detok.output_text == expected_text
    assert detok.output_token_ids == expected_token_ids
    assert result.token_ids == expected_token_ids
    assert result.logprobs is not None
    assert len(result.logprobs) == len(expected_token_ids)
    assert result.sampling_mask is not None
    assert result.sampling_mask.token_ids == [[token] for token in expected_token_ids]
    assert result.routed_experts is not None
    assert result.routed_experts.ravel().tolist() == expected_token_ids


@pytest.mark.parametrize("step", [1, 3, 64])
def test_stop_string_dormant_until_reasoning_ends(
    include_stop_str_in_output: bool, step: int
):
    """A stop string in the reasoning must not end the request; it still
    applies to the response, however the tokens are batched."""
    token_ids = [ord(c) for c in "a cd b|x cd y"]
    req = _make_request(
        stop=["cd"], include_stop_str_in_output=include_stop_str_in_output
    )
    detok = _DummyDetokenizer(req, ReasoningEndFinder(_PipeEndsReasoning(), []))

    for i in range(0, len(token_ids), step):
        if result := detok.update(token_ids[i : i + step], False):
            break

    assert result == "cd"
    assert detok.output_text == (
        "a cd b|x cd" if include_stop_str_in_output else "a cd b|x "
    )


def test_stop_string_cannot_span_reasoning_end(monkeypatch):
    """When the end marker decodes to no text, a stop string formed across
    the boundary must not match; one wholly in the response does."""
    req = _make_request(stop=["cd"], include_stop_str_in_output=False)
    detok = _DummyDetokenizer(req, ReasoningEndFinder(_PipeEndsReasoning(), []))
    monkeypatch.setattr(detok, "decode_next", lambda t: "" if t == ord("|") else chr(t))

    assert detok.update([ord(c) for c in "abc|d"], False) is None
    assert detok.update([ord(c) for c in "xcd"], False) == "cd"
    assert detok.output_text == "abcdx"


@pytest.mark.parametrize("reasoning_ended", [False, None])
def test_stop_string_after_qwen3_reasoning(reasoning_ended: bool | None):
    """With a registered reasoning parser, stop strings skip the reasoning
    only when the serving layer reports the prompt leaves the model in it."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    req = _make_request(stop=["</block>"], include_stop_str_in_output=False)
    req.prompt_token_ids = tokenizer.encode("<|im_start|>assistant\n<think>\n")
    req.reasoning_ended = reasoning_ended
    detok = IncrementalDetokenizer.from_new_request(
        tokenizer, req, ReasoningParserManager.get_reasoning_parser("qwen3")
    )
    assert isinstance(detok, BaseIncrementalDetokenizer)

    output = "Safe, so <block>no</block>.\n</think>\n\n<block>no</block>"
    assert detok.update(tokenizer.encode(output), False) == "</block>"
    if reasoning_ended is False:
        assert detok.output_text == "Safe, so <block>no</block>.\n</think>\n\n<block>no"
    else:
        assert detok.output_text == "Safe, so <block>no"
