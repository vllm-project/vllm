# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.kimi_k3 import KimiK3Parser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.kimi_k3_reasoning_parser import KimiK3ReasoningParser
from vllm.tokenizers import TokenizerLike

pytestmark = pytest.mark.skip_global_cleanup

OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"
THINK_OPEN = f"{OPEN}think{SEP}"
THINK_CLOSE = f"{CLOSE}think{SEP}"
RESPONSE_OPEN = f"{OPEN}response{SEP}"
OPEN_IDS = [1, 2, 3]
CLOSE_IDS = [4, 2, 3]
RESPONSE_OPEN_IDS = [1, 5, 3]


class DummyTokenizer:
    def __len__(self) -> int:
        return 1

    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, *args, **kwargs) -> list[int]:
        if text == THINK_OPEN:
            return [1, 2, 3]
        if text == THINK_CLOSE:
            return [4, 2, 3]
        if text == RESPONSE_OPEN:
            return RESPONSE_OPEN_IDS
        return [ord(ch) for ch in text]


def _dummy_tokenizer() -> TokenizerLike:
    return cast(TokenizerLike, DummyTokenizer())


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
    parser = ReasoningOnlyParser(_dummy_tokenizer())
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
    assert not parser.is_reasoning_end([*RESPONSE_OPEN_IDS, *new_open])
    # A prior response channel alone does not establish the new turn's state.
    assert not parser.is_reasoning_end([*RESPONSE_OPEN_IDS, 9])
    # ...then the current turn emits its own close -> ended
    assert parser.is_reasoning_end([*stale_close, *new_open, *stale_close])
    # open with no close yet -> not ended
    assert not parser.is_reasoning_end([*new_open])


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        pytest.param(
            [*OPEN_IDS, 9, 10, *CLOSE_IDS, *RESPONSE_OPEN_IDS, 11],
            2,
            id="open_and_close_markers",
        ),
        pytest.param(
            [9, 10, *CLOSE_IDS, *RESPONSE_OPEN_IDS, 11],
            2,
            id="open_marker_consumed_as_generation_prefix",
        ),
        pytest.param([*RESPONSE_OPEN_IDS, 11, 12], 0, id="response_only"),
        pytest.param([*OPEN_IDS, 9, 10], 2, id="unterminated_after_open"),
        pytest.param([9, 10, 11], 3, id="unterminated_without_markers"),
        pytest.param([], 0, id="empty"),
        pytest.param([9, 4, 2, 10], 4, id="partial_close_marker_is_reasoning"),
        pytest.param(
            [*OPEN_IDS, 9, *RESPONSE_OPEN_IDS, 11],
            2 + len(RESPONSE_OPEN_IDS),
            id="response_open_inside_open_think_is_reasoning",
        ),
    ],
)
def test_count_reasoning_tokens_matches_think_channel(token_ids, expected):
    """reasoning_tokens must cover exactly what extract_reasoning labels as
    reasoning: marker tokens are excluded and a consumed generation prefix
    means the output starts inside the think channel."""
    parser = KimiK3ReasoningParser(DummyTokenizer())

    assert parser.count_reasoning_tokens(token_ids) == expected


def test_count_reasoning_tokens_is_zero_when_thinking_disabled():
    parser = KimiK3ReasoningParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )

    assert parser.count_reasoning_tokens([*OPEN_IDS, 9, 10, *CLOSE_IDS]) == 0


def test_count_reasoning_tokens_through_delegating_parser():
    parser = ReasoningOnlyParser(_dummy_tokenizer())

    assert parser.count_reasoning_tokens([*OPEN_IDS, 9, *CLOSE_IDS, 11]) == 1


@pytest.mark.parametrize(
    ("prefix", "prefix_ids"),
    [(OPEN, [1]), (OPEN + "response", [1, 5]), (RESPONSE_OPEN + "{", [1, 5, 3, 123])],
)
def test_response_only_transition_preserves_content(prefix, prefix_ids):
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    body = '{"ok":true}'
    text = RESPONSE_OPEN + body
    all_ids = [*RESPONSE_OPEN_IDS, *(ord(ch) for ch in body)]
    content = ""

    for chunk, delta_ids in (
        (prefix, prefix_ids),
        (text[len(prefix) :], all_ids[len(prefix_ids) :]),
    ):
        delta = parser.parse_delta(chunk, delta_ids, request, finished=False)
        if delta:
            assert not delta.reasoning
            content += delta.content or ""

    assert content == body
    assert parser._stream_state.reasoning_ended


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("split_opener", [False, True])
@pytest.mark.parametrize(
    "draft_suffix", [[123], [*RESPONSE_OPEN_IDS, 123]], ids=["json", "invalid_opener"]
)
def test_response_only_constraint_start_with_prompt(
    committed, split_opener, draft_suffix
):
    from types import SimpleNamespace

    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    tokenizer = DummyTokenizer()
    reasoner = KimiK3ReasoningParser(tokenizer)
    prompt = [9, *tokenizer.encode(f'{OPEN}message role="assistant"{SEP}'), *OPEN_IDS]
    request = Request(
        request_id="test",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(
            max_tokens=32,
            structured_outputs=StructuredOutputsParams(json_object=True),
        ),
        pooling_params=None,
    )
    prior = RESPONSE_OPEN_IDS[:-1] if split_opener else []
    delta = (
        RESPONSE_OPEN_IDS[-1:] if split_opener else RESPONSE_OPEN_IDS
    ) + draft_suffix
    request.append_output_token_ids(prior)
    if committed:
        request.append_output_token_ids(delta)
    manager = SimpleNamespace(
        enable_in_reasoning=False, _get_reasoner=lambda _: reasoner
    )

    bounds = StructuredOutputManager._get_constraint_bounds(
        manager, request, delta, spec_tokens_committed=committed
    )
    expected = 1 if split_opener else len(RESPONSE_OPEN_IDS)
    assert bounds.grammar_start == bounds.constraint_start == expected


def test_quoted_response_opener_does_not_end_reasoning():
    parser = KimiK3ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    text = THINK_OPEN + "step" + RESPONSE_OPEN + "answer"
    token_ids = [*OPEN_IDS, *(ord(ch) for ch in "step" + RESPONSE_OPEN + "answer")]

    assert parser.extract_reasoning(text, request) == (
        "step" + RESPONSE_OPEN + "answer",
        None,
    )
    delta = parser.extract_reasoning_streaming("", text, text, [], token_ids, token_ids)
    assert delta is not None
    assert delta.reasoning == "step" + RESPONSE_OPEN + "answer"
    assert delta.content is None
    assert parser.extract_content_ids(token_ids) == []
    assert not parser.is_reasoning_end_streaming(token_ids, token_ids)
    assert parser.count_reasoning_tokens(token_ids) == len(token_ids) - len(OPEN_IDS)


def test_quoted_response_prefix_with_consumed_think_opener():
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    quoted = RESPONSE_OPEN + " is an example"
    first = parser.parse_delta(
        quoted,
        [ord(ch) for ch in quoted],
        request,
        finished=False,
    )

    assert first is not None
    assert first.reasoning == quoted
    assert first.content is None
    assert not parser._stream_state.reasoning_ended

    second = parser.parse_delta(
        THINK_CLOSE + RESPONSE_OPEN + "{}",
        [*CLOSE_IDS, *RESPONSE_OPEN_IDS, ord("{"), ord("}")],
        request,
        finished=True,
    )

    assert second is not None
    assert second.content == "{}"
    assert not second.reasoning


def test_terminal_quoted_partial_response_marker_is_preserved():
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    quoted = "Quoted example: " + OPEN + "response"

    delta = parser.parse_delta(
        quoted, [ord(ch) for ch in quoted], request, finished=True
    )

    assert delta is not None
    assert delta.reasoning == quoted
    assert delta.content is None


@pytest.mark.parametrize("split_opener", [False, True])
@pytest.mark.parametrize("quoted_marker", [THINK_OPEN, RESPONSE_OPEN])
def test_response_only_json_can_quote_markers(split_opener, quoted_marker):
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    body = '{"marker":"' + quoted_marker + '"}'
    chunks = [(RESPONSE_OPEN + body, [*RESPONSE_OPEN_IDS, *map(ord, body)])]
    if split_opener:
        chunks = [(RESPONSE_OPEN, RESPONSE_OPEN_IDS), (body, list(map(ord, body)))]
    content = ""
    for index, (text, ids) in enumerate(chunks):
        delta = parser.parse_delta(
            text, ids, request, finished=index == len(chunks) - 1
        )
        if delta:
            assert not delta.reasoning
            content += delta.content or ""
    assert content == body


def test_response_control_does_not_close_explicit_think():
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    reasoner = KimiK3ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    text = THINK_OPEN + "step" + RESPONSE_OPEN + "{}"
    ids = [*OPEN_IDS, *map(ord, "step"), *RESPONSE_OPEN_IDS, *map(ord, "{}")]

    assert not reasoner.is_reasoning_end_streaming(ids, ids)
    first = parser.parse_delta(text, ids, request, finished=False)
    second = parser.parse_delta(
        " next", list(map(ord, " next")), request, finished=True
    )

    assert first is not None and first.reasoning == "step" + RESPONSE_OPEN + "{}"
    assert first.content is None
    assert second is not None and second.reasoning == " next"
    assert second.content is None


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
        _dummy_tokenizer(),
        chat_template_kwargs={"thinking": False},
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
