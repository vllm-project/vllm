# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.kimi_k3 import KimiK3Parser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.kimi_k3_reasoning_parser import KimiK3ReasoningParser

pytestmark = pytest.mark.skip_global_cleanup

OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"
THINK_OPEN = f"{OPEN}think{SEP}"
THINK_CLOSE = f"{CLOSE}think{SEP}"
RESPONSE_OPEN = f"{OPEN}response{SEP}"


class DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == THINK_OPEN:
            return [1, 2, 3]
        if text == THINK_CLOSE:
            return [4, 2, 3]
        return [ord(ch) for ch in text]


class ReasoningOnlyParser(KimiK3Parser):
    reasoning_parser_cls = KimiK3ReasoningParser


def test_parser_manager_selects_kimi_k3_parser_for_reasoning_only():
    parser_cls = ParserManager.get_parser(reasoning_parser_name="kimi_k3")

    assert parser_cls is not None
    assert issubclass(parser_cls, KimiK3Parser)
    assert parser_cls.reasoning_parser_cls is KimiK3ReasoningParser
    assert parser_cls.tool_parser_cls is None


def test_parser_selection_thinking_disabled():
    parser = KimiK3ReasoningParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )

    assert parser._thinking_enabled is False


def test_extract_reasoning_with_xtml_tags():
    parser = KimiK3ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content = parser.extract_reasoning_content(
        f"{THINK_OPEN}step{THINK_CLOSE}{RESPONSE_OPEN}answer",
        request,
    )

    assert reasoning == "step"
    assert content == "answer"


def test_extract_reasoning_with_generation_prefix_consumed():
    parser = KimiK3ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content = parser.extract_reasoning_content(
        f"step{THINK_CLOSE}{RESPONSE_OPEN}answer",
        request,
    )

    assert reasoning == "step"
    assert content == "answer"


def test_delegating_parser_strips_response_wrapper_without_tool_parser():
    parser = ReasoningOnlyParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content, tool_calls = parser.parse(
        f"{THINK_OPEN}step{THINK_CLOSE}{RESPONSE_OPEN}answer",
        request,
    )

    assert reasoning == "step"
    assert content == "answer"
    assert tool_calls == []


def test_is_reasoning_end_uses_full_input_ids():
    parser = KimiK3ReasoningParser(DummyTokenizer())

    assert not parser.is_reasoning_end([4, 2])
    assert parser.is_reasoning_end([4, 2, 3])


def test_is_reasoning_end_ignores_stale_close_from_prior_turn():
    # DummyTokenizer: THINK_OPEN -> [1, 2, 3], THINK_CLOSE -> [4, 2, 3].
    # Multi-turn / agent continuation: a prior turn's think channel (its close
    # marker) is kept in the prompt, then the current turn opens a new think
    # block that has not closed yet. Reasoning must read as NOT ended, otherwise
    # the structured-output gate constrains the current turn's reasoning.
    parser = KimiK3ReasoningParser(DummyTokenizer())

    stale_close = [4, 2, 3]
    new_open = [1, 2, 3]
    # prior close, then current-turn open still unclosed -> not ended
    assert not parser.is_reasoning_end([*stale_close, *new_open])
    # ...then the current turn emits its own close -> ended
    assert parser.is_reasoning_end([*stale_close, *new_open, *stale_close])
    # open with no close yet -> not ended
    assert not parser.is_reasoning_end([*new_open])


def test_streaming_split_open_marker_is_held_back():
    parser = KimiK3ReasoningParser(DummyTokenizer())

    first = parser.extract_reasoning_content_streaming(
        previous_text="",
        current_text=OPEN,
        delta_text=OPEN,
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
    )
    second = parser.extract_reasoning_content_streaming(
        previous_text=OPEN,
        current_text=f"{OPEN}think",
        delta_text="think",
        previous_token_ids=[1],
        current_token_ids=[1, 2],
        delta_token_ids=[2],
    )
    third = parser.extract_reasoning_content_streaming(
        previous_text=f"{OPEN}think",
        current_text=THINK_OPEN + "step",
        delta_text=f"{SEP}step",
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, 3, 9],
        delta_token_ids=[3, 9],
    )

    assert first is None
    assert second is None
    assert isinstance(third, DeltaMessage)
    assert third.reasoning == "step"


def test_streaming_split_close_marker_hands_content_downstream():
    parser = KimiK3ReasoningParser(DummyTokenizer())

    previous_text = f"{THINK_OPEN}step"
    partial_close = parser.extract_reasoning_content_streaming(
        previous_text=previous_text,
        current_text=previous_text + CLOSE,
        delta_text=CLOSE,
        previous_token_ids=[1, 2, 3, 9],
        current_token_ids=[1, 2, 3, 9, 4],
        delta_token_ids=[4],
    )
    closed = parser.extract_reasoning_content_streaming(
        previous_text=previous_text + CLOSE,
        current_text=previous_text + f"{THINK_CLOSE}{RESPONSE_OPEN}answer",
        delta_text=f"think{SEP}{RESPONSE_OPEN}answer",
        previous_token_ids=[1, 2, 3, 9, 4],
        current_token_ids=[1, 2, 3, 9, 4, 2, 3, 10],
        delta_token_ids=[2, 3, 10],
    )

    assert partial_close is None
    assert isinstance(closed, DeltaMessage)
    assert closed.reasoning is None
    assert closed.content == f"{RESPONSE_OPEN}answer"
    assert parser.extract_content_ids([2, 3, 10]) == [10]


def test_thinking_disabled_streams_content():
    parser = KimiK3ReasoningParser(
        DummyTokenizer(), chat_template_kwargs={"enable_thinking": False}
    )

    delta = parser.extract_reasoning_content_streaming(
        previous_text="",
        current_text=f"{RESPONSE_OPEN}answer",
        delta_text=f"{RESPONSE_OPEN}answer",
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
    )

    assert isinstance(delta, DeltaMessage)
    assert delta.content == f"{RESPONSE_OPEN}answer"
    assert delta.reasoning is None


def test_delegating_parser_thinking_false_streams_response_content():
    parser = ReasoningOnlyParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"thinking": False},
    )

    first = parser.parse_delta(
        delta_text="OK",
        delta_token_ids=[10],
        request=request,
        prompt_token_ids=[1],
        finished=False,
    )
    partial_close = parser.parse_delta(
        delta_text=CLOSE,
        delta_token_ids=[2],
        request=request,
        prompt_token_ids=[1],
        finished=False,
    )
    closed = parser.parse_delta(
        delta_text=f"response{SEP}",
        delta_token_ids=[3, 4],
        request=request,
        prompt_token_ids=[1],
        finished=False,
    )

    assert first is not None
    assert first.content == "OK"
    assert first.reasoning is None
    assert partial_close is None
    assert closed is None


def test_adjust_request_keeps_xtml_markers_contiguous():
    parser = KimiK3ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    adjusted = parser.adjust_request(request)

    assert adjusted.skip_special_tokens is False
    if hasattr(adjusted, "spaces_between_special_tokens"):
        assert adjusted.spaces_between_special_tokens is False


OPEN_IDS = [1, 2, 3]
CLOSE_IDS = [4, 2, 3]


def _reference_is_reasoning_end(input_ids: list[int]) -> bool:
    """Full-sequence reference for the streaming check to be measured against.

    Two independent last-occurrence scans, i.e. the straightforward reading of
    "reasoning ended iff the newest think marker is a close marker".
    """

    def last(needle: list[int]) -> int:
        for i in range(len(input_ids) - len(needle), -1, -1):
            if input_ids[i : i + len(needle)] == needle:
                return i
        return -1

    last_close, last_open = last(CLOSE_IDS), last(OPEN_IDS)
    if last_open == -1:
        return last_close != -1
    return last_close > last_open


def test_is_reasoning_end_streaming_only_scans_the_step_window():
    """The decode-step check must not re-derive the answer from the whole
    sequence: an already-closed think block earlier in the sequence is the
    engine's business (it latches `reasoning_ended`), not this call's."""
    parser = KimiK3ReasoningParser(DummyTokenizer())
    prompt = [*CLOSE_IDS, *OPEN_IDS, 9, 9, 9]

    assert not parser.is_reasoning_end_streaming([*prompt, 7], [7])
    assert parser.is_reasoning_end_streaming([*prompt, *CLOSE_IDS], CLOSE_IDS)


def test_is_reasoning_end_streaming_detects_marker_across_step_boundary():
    """A 3-token marker can be split over decode steps; the check carries the
    preceding len(marker)-1 tokens so the final token still completes it."""
    parser = KimiK3ReasoningParser(DummyTokenizer())
    history = [*OPEN_IDS, 5, *CLOSE_IDS[:2]]

    assert not parser.is_reasoning_end_streaming(history, [CLOSE_IDS[1]])
    assert parser.is_reasoning_end_streaming([*history, CLOSE_IDS[2]], [CLOSE_IDS[2]])


def test_is_reasoning_end_streaming_reopened_block_is_not_ended():
    """Kimi K3 may close and immediately reopen the think channel inside one
    window (speculative decoding lands several tokens per step). The newest
    marker wins, matching is_reasoning_end."""
    parser = KimiK3ReasoningParser(DummyTokenizer())
    window = [*CLOSE_IDS, *OPEN_IDS]

    assert not parser.is_reasoning_end_streaming([*OPEN_IDS, 5, *window], window)


def test_is_reasoning_end_streaming_accepts_an_iterator_delta():
    """`should_advance` may hand over an islice rather than a list."""
    parser = KimiK3ReasoningParser(DummyTokenizer())
    full = [*OPEN_IDS, 5, *CLOSE_IDS]

    assert parser.is_reasoning_end_streaming(full, iter(CLOSE_IDS))


def test_is_reasoning_end_streaming_thinking_disabled():
    parser = KimiK3ReasoningParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )

    assert parser.is_reasoning_end_streaming([1], [1])


@pytest.mark.parametrize("seed", range(24))
def test_reasoning_end_matches_reference_over_marker_dense_sequences(seed):
    """Both the full-sequence check and the per-step check must agree with the
    two-scan reference, on sequences where the markers (which share the suffix
    [2, 3]) collide and overlap constantly."""
    import random

    rnd = random.Random(seed)
    parser = KimiK3ReasoningParser(DummyTokenizer())

    for _ in range(200):
        head = [rnd.choice([1, 2, 3, 4]) for _ in range(rnd.randrange(0, 14))]
        delta = [rnd.choice([1, 2, 3, 4]) for _ in range(rnd.randrange(1, 5))]
        full = [*head, *delta]

        assert parser.is_reasoning_end(full) == _reference_is_reasoning_end(full)

        # The engine only calls the streaming check while reasoning is still
        # open, so that is the only case it has to agree on.
        if not _reference_is_reasoning_end(head):
            assert parser.is_reasoning_end_streaming(
                full, delta
            ) == _reference_is_reasoning_end(full), (head, delta)
