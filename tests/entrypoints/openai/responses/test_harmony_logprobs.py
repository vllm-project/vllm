# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output text logprobs for gpt-oss (Harmony) in the Responses API.

Streaming and non-streaming must report the same logprobs, so these tests
feed the same tokens to HarmonyContext in one chunk and one token at a time
and compare the results.
"""

import pytest
from openai.types.responses.response_output_text import Logprob as ResponseLogprob
from openai_harmony import Message, Role
from transformers import AutoTokenizer

from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
from vllm.entrypoints.openai.responses.context import HarmonyContext
from vllm.entrypoints.openai.responses.harmony import harmony_to_response_output
from vllm.entrypoints.openai.responses.streaming_events import (
    StreamingState,
    emit_content_delta_events,
    emit_previous_item_done_events,
)
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.harmony import HarmonyParser, Segment

MODEL_NAME = "openai/gpt-oss-20b"

REASONING_TEXT = "Let me think."
PREAMBLE_TEXT = "Checking é now."
FINAL_TEXT = "Héllo, 你好 world!"
HARMONY_OUTPUT = (
    f"<|channel|>analysis<|message|>{REASONING_TEXT}<|end|>"
    f"<|start|>assistant<|channel|>commentary<|message|>{PREAMBLE_TEXT}<|end|>"
    f"<|start|>assistant<|channel|>final<|message|>{FINAL_TEXT}<|return|>"
)


@pytest.fixture(scope="module")
def gpt_oss_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(text: str) -> list[int]:
    return get_encoding().encode(text, allowed_special="all")


def fake_logprobs(token_ids: list[int]) -> list[dict[int, Logprob]]:
    # Use the position as the logprob so entries are easy to tell apart.
    return [
        {token_id: Logprob(logprob=-float(i + 1), rank=1)}
        for i, token_id in enumerate(token_ids)
    ]


def make_request_output(
    token_ids: list[int],
    logprobs: list[dict[int, Logprob]] | None,
    finished: bool,
) -> RequestOutput:
    return RequestOutput(
        request_id="test-id",
        prompt=None,
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=token_ids,
                cumulative_logprob=None,
                logprobs=logprobs,
                finish_reason="stop" if finished else None,
            )
        ],
        finished=finished,
    )


def run_context(
    tokenizer, token_ids: list[int], chunk_size: int, with_logprobs: bool = True
) -> tuple[HarmonyContext, list[tuple[int, dict[int, Logprob]]]]:
    """Feed token_ids to a HarmonyContext in chunks of chunk_size.

    Returns the context and every non-None entry of
    last_append_text_logprobs, in order (what streaming would emit).
    """
    context = HarmonyContext(
        messages=[],
        available_tools=[],
        function_tool_names=frozenset(),
        response_parser=HarmonyParser(tokenizer),
    )
    logprobs = fake_logprobs(token_ids) if with_logprobs else None
    streamed = []
    for start in range(0, len(token_ids), chunk_size):
        end = min(start + chunk_size, len(token_ids))
        context.append_output(
            make_request_output(
                token_ids[start:end],
                logprobs[start:end] if logprobs is not None else None,
                finished=end == len(token_ids),
            )
        )
        assert len(context.last_append_text_logprobs) == len(
            context.last_append_segments
        )
        streamed.extend(e for e in context.last_append_text_logprobs if e is not None)
    return context, streamed


def message_logprobs(context: HarmonyContext) -> list[list[int] | None]:
    """Token ids with logprobs for each output message, in order."""
    out = []
    for msg in context.messages[context.num_init_messages :]:
        entries = context.text_logprobs_by_message.get(id(msg))
        out.append(None if entries is None else [t for t, _ in entries])
    return out


def test_text_logprobs_cover_only_output_text(gpt_oss_tokenizer):
    context, _ = run_context(gpt_oss_tokenizer, encode(HARMONY_OUTPUT), 10**6)

    assert [m.channel for m in context.messages] == [
        "analysis",
        "commentary",
        "final",
    ]
    # Reasoning has no output_text, so no logprobs. Preamble and final
    # text cover exactly their content tokens (no Harmony control tokens).
    assert message_logprobs(context) == [
        None,
        encode(PREAMBLE_TEXT),
        encode(FINAL_TEXT),
    ]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7])
def test_streaming_matches_non_streaming(gpt_oss_tokenizer, chunk_size):
    token_ids = encode(HARMONY_OUTPUT)
    full_context, full_streamed = run_context(gpt_oss_tokenizer, token_ids, 10**6)
    context, streamed = run_context(gpt_oss_tokenizer, token_ids, chunk_size)

    assert message_logprobs(context) == message_logprobs(full_context)
    # Logprob values come from the right positions, not just the right ids.
    assert streamed == full_streamed
    # What streaming emits per segment is exactly what the final output
    # reports per message.
    per_message = [
        entry
        for msg in context.messages
        for entry in context.text_logprobs_by_message.get(id(msg), [])
    ]
    assert streamed == per_message


def test_no_logprobs_requested(gpt_oss_tokenizer):
    context, streamed = run_context(
        gpt_oss_tokenizer, encode(HARMONY_OUTPUT), 1, with_logprobs=False
    )
    assert streamed == []
    assert context.text_logprobs_by_message == {}


def make_response_logprobs(tokens: list[str]) -> list[ResponseLogprob]:
    return [
        ResponseLogprob(token=t, logprob=-1.0, bytes=list(t.encode()), top_logprobs=[])
        for t in tokens
    ]


def test_streaming_events_carry_logprobs():
    state = StreamingState()
    delta_logprobs = make_response_logprobs(["Hi"])
    events = emit_content_delta_events(
        Segment(channel="final", recipient=None, delta="Hi"),
        state,
        text_logprobs=delta_logprobs,
    )
    delta_event = events[-1]
    assert delta_event.type == "response.output_text.delta"
    assert [lp.token for lp in delta_event.logprobs] == ["Hi"]

    done_logprobs = make_response_logprobs(["Hi", "!"])
    message = Message.from_role_and_content(Role.ASSISTANT, "Hi!").with_channel("final")
    events = emit_previous_item_done_events(message, state, text_logprobs=done_logprobs)
    item_done = events[-1]
    assert item_done.type == "response.output_item.done"
    assert item_done.item.content[0].logprobs == done_logprobs


def test_reasoning_delta_ignores_logprobs():
    events = emit_content_delta_events(
        Segment(channel="analysis", recipient=None, delta="hmm"),
        StreamingState(),
        text_logprobs=make_response_logprobs(["hmm"]),
    )
    assert all(e.type != "response.output_text.delta" for e in events)


@pytest.mark.parametrize("channel", ["final", "commentary"])
def test_final_output_carries_logprobs(channel):
    message = Message.from_role_and_content(Role.ASSISTANT, "Hi!").with_channel(channel)
    logprobs = make_response_logprobs(["Hi", "!"])
    [item] = harmony_to_response_output(message, frozenset(), text_logprobs=logprobs)
    assert item.content[0].logprobs == logprobs

    [item] = harmony_to_response_output(message, frozenset())
    assert item.content[0].logprobs is None
