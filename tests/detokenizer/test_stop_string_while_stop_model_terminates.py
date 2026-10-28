# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import (
    BaseIncrementalDetokenizer,
    ContentFilter,
    IncrementalDetokenizer,
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
        content_filter: ContentFilter | None = None,
    ):
        super().__init__(request, content_filter)

    def decode_next(self, next_token_id: int) -> str:
        # Map token id to single ASCII character for deterministic testing.
        return chr(next_token_id)


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

    # get_next_output should return the full text when finished=True.
    # (Buffering only applies during streaming when finished=False.)
    assert detok.get_next_output(finished=True, delta=False) == (expected_text, 0)


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


@pytest.mark.parametrize(
    "decoded_tokens,num_kept",
    [
        (["Say", " <", "tool", "_call", ">", "{}", "</", "tool", "_", ">"], None),
        (["Say", " <", "tool", "_call", ">", "{}", "</", "block>", "x", "y"], 8),
    ],
)
@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.CUMULATIVE]
)
@pytest.mark.parametrize("step", [1, 3])
def test_streamed_tokens_match_streamed_text(
    include_stop_str_in_output: bool,
    decoded_tokens: list[str],
    num_kept: int | None,
    output_kind: RequestOutputKind,
    step: int,
    monkeypatch,
):
    token_ids = list(range(len(decoded_tokens)))
    expected_token_ids = token_ids[:num_kept]
    expected_text = "".join(decoded_tokens[:num_kept])
    if num_kept is not None and not include_stop_str_in_output:
        expected_text = expected_text.removesuffix("</block>")
    req = _make_request(
        stop=["</block>"], include_stop_str_in_output=include_stop_str_in_output
    )
    req.external_req_id = req.request_id
    assert req.sampling_params is not None
    req.sampling_params.logprobs = 0
    req.sampling_params.output_kind = output_kind
    detok = _DummyDetokenizer(req)
    monkeypatch.setattr(detok, "decode_next", decoded_tokens.__getitem__)
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    processor.add_request(req, prompt=None)
    processor.request_states[req.request_id].detokenizer = detok

    sent_text, sent_token_ids = "", []
    for start in range(0, len(token_ids), step):
        new_token_ids = token_ids[start : start + step]
        last = start + step >= len(token_ids)
        outputs = processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id=req.request_id,
                    new_token_ids=new_token_ids,
                    finish_reason=FinishReason.LENGTH if last else None,
                    new_logprobs=LogprobsLists(
                        np.array(new_token_ids).reshape(-1, 1),
                        np.zeros((len(new_token_ids), 1)),
                        np.ones(len(new_token_ids), dtype=int),
                        None,
                    ),
                    new_sampling_mask=SamplingMaskLists(
                        np.array(new_token_ids), np.arange(len(new_token_ids) + 1)
                    ),
                )
            ]
        )
        result = outputs.request_outputs[0].outputs[0]
        finished = result.finish_reason is not None
        if output_kind == RequestOutputKind.DELTA:
            sent_text += result.text
            sent_token_ids += result.token_ids
            assert result.logprobs is not None
            assert [next(iter(lp)) for lp in result.logprobs] == result.token_ids
            if result.sampling_mask is not None:
                assert result.sampling_mask.token_ids == [
                    [token] for token in result.token_ids
                ]
        else:
            sent_text, sent_token_ids = result.text, list(result.token_ids)
            assert result.logprobs is not None
            assert len(result.logprobs) == len(result.token_ids)
        if finished:
            break
        assert "".join(decoded_tokens[t] for t in sent_token_ids) == sent_text
        if not include_stop_str_in_output:
            assert len(sent_text) <= max(len(detok.output_text) - 7, 0)

    assert finished
    assert sent_token_ids == expected_token_ids
    assert sent_text == expected_text


def test_released_tokens_are_not_held_again(monkeypatch):
    decoded_tokens = ["ab", "</bl", "ock>", "xy", "zw"]
    req = _make_request(stop=["</block>"], include_stop_str_in_output=False)
    detok = _DummyDetokenizer(req)
    monkeypatch.setattr(detok, "decode_next", decoded_tokens.__getitem__)

    assert detok.update([0, 1, 2], False) == "</block>"
    assert detok.get_next_output(finished=True, delta=True) == ("ab", 0)

    detok.update([3], False)
    assert detok.get_next_output(finished=False, delta=True) == ("", 1)
    detok.update([4], False)
    assert detok.get_next_output(finished=False, delta=True) == ("", 2)
    assert detok.get_next_output(finished=True, delta=True) == ("xyzw", 0)


def _pipe_content_filter(
    lag: int = 0, strip: str = "", prefix: str = ""
) -> ContentFilter:
    in_content = False
    pending = ""

    def content_of(delta_text: str, delta_token_ids: list[int], finished: bool):
        nonlocal in_content, pending
        for char in delta_text:
            if in_content and char not in strip:
                pending += char
            if not in_content and char == "|":
                in_content = True
                pending = prefix
        released = len(pending) if finished else max(len(pending) - lag, 0)
        content, pending = pending[:released], pending[released:]
        return content

    return content_of


def _update_in_steps(detok: BaseIncrementalDetokenizer, text: str, step: int):
    token_ids = [ord(c) for c in text]
    for i in range(0, len(token_ids), step):
        if stop_string := detok.update(token_ids[i : i + step], False):
            return stop_string
    return None


@pytest.mark.parametrize("step", [1, 3, 64])
@pytest.mark.parametrize("lag", [0, 2])
def test_stop_string_matches_only_content(
    include_stop_str_in_output: bool, step: int, lag: int
):
    req = _make_request(
        stop=["cd"], include_stop_str_in_output=include_stop_str_in_output
    )
    detok = _DummyDetokenizer(req, _pipe_content_filter(lag))

    assert _update_in_steps(detok, "a cd b|x cd y cd", step) == "cd"
    assert detok.output_text == (
        "a cd b|x cd" if include_stop_str_in_output else "a cd b|x"
    )


@pytest.mark.parametrize("step", [1, 64])
@pytest.mark.parametrize(
    "strip, prefix, output, expected_text, expected_text_with_stop",
    [
        ("<>", "", "a cd|x c<>d y", "a cd|x", "a cd|x c<>d"),
        ("", "<p>", "a cd|x cd y", "a cd|x", "a cd|x cd"),
    ],
)
def test_stop_string_in_content_not_verbatim(
    include_stop_str_in_output: bool,
    step: int,
    strip: str,
    prefix: str,
    output: str,
    expected_text: str,
    expected_text_with_stop: str,
):
    req = _make_request(
        stop=["cd"], include_stop_str_in_output=include_stop_str_in_output
    )
    detok = _DummyDetokenizer(req, _pipe_content_filter(strip=strip, prefix=prefix))

    assert _update_in_steps(detok, output, step) == "cd"
    assert detok.output_text == (
        expected_text_with_stop if include_stop_str_in_output else expected_text
    )


def test_stop_string_content_filter_failure_matches_all_text():
    def failing_filter(delta_text, delta_token_ids, finished):
        raise RuntimeError("parser error")

    req = _make_request(stop=["cd"], include_stop_str_in_output=False)
    detok = _DummyDetokenizer(req, failing_filter)

    assert _update_in_steps(detok, "a cd b", 1) == "cd"
    assert detok.output_text == "a "


@pytest.mark.parametrize(
    "reasoning_parser, model, prompt, output, expected_text",
    [
        (
            "qwen3",
            "Qwen/Qwen3-0.6B",
            "<|im_start|>asssitant\n<think>\n",
            "Say <block>no</block>.\n</think>\n\n<block>no</block> Done.",
            "Say <block>no</block>.\n</think>\n\n<block>no",
        ),
        (
            "muse_glimmer",
            "meta-models/Muse-Glimmer-30B",
            "<|start|>user<|message|>hi<|eot|><|start|>assistant",
            " to=self<|message|>Say <block>no</block>.<|eom|><|start|>assistant"
            " to=user<|message|><block>no</block> Done.<|eot|>",
            " to=self<|message|>Say <block>no</block>.<|eom|><|start|>assistant"
            " to=user<|message|><block>no",
        ),
    ],
)
def test_stop_string_skips_reasoning_with_parser_content_filter(
    reasoning_parser: str, model: str, prompt: str, output: str, expected_text: str
):
    tokenizer = AutoTokenizer.from_pretrained(model)
    req = _make_request(stop=["</block>"], include_stop_str_in_output=False)
    req.sampling_params.skip_special_tokens = False
    req.prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    parser_cls = ParserManager.get_parser(reasoning_parser_name=reasoning_parser)
    assert parser_cls is not None
    content_filter = parser_cls(tokenizer).stop_string_content_filter(
        ChatCompletionRequest(model="test", messages=[]), req.prompt_token_ids
    )
    detok = IncrementalDetokenizer.from_new_request(tokenizer, req, content_filter)
    assert isinstance(detok, BaseIncrementalDetokenizer)

    token_ids = tokenizer.encode(output, add_special_tokens=False)
    assert detok.update(token_ids, False) == "</block>"
    assert detok.output_text == expected_text


_TOOL_CALL = (
    '<tool_call>\n{"name": "run", "arguments": {"cmd": "echo </block>"}}\n</tool_call>'
)


@pytest.mark.parametrize(
    "reasoning_parser, output",
    [
        (None, _TOOL_CALL),
        ("qwen3", "Say <block>no</block>.</think>\n\n" + _TOOL_CALL),
    ],
)
def test_stop_string_skips_tool_calls_with_parser_content_filter(
    reasoning_parser: str | None, output: str
):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        tools=[{"type": "function", "function": {"name": "run", "parameters": {}}}],
        tool_choice="auto",
    )
    req = _make_request(stop=["</block>"], include_stop_str_in_output=False)
    req.sampling_params.skip_special_tokens = False
    req.prompt_token_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name="hermes",
        reasoning_parser_name=reasoning_parser,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    content_filter = parser_cls(tokenizer, request.tools).stop_string_content_filter(
        request, req.prompt_token_ids
    )
    detok = IncrementalDetokenizer.from_new_request(tokenizer, req, content_filter)
    assert isinstance(detok, BaseIncrementalDetokenizer)

    for token_id in tokenizer.encode(output, add_special_tokens=False):
        assert detok.update([token_id], False) is None
    assert detok.output_text == output
