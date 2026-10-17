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

    def decode(self, token_ids, **kwargs) -> str:
        tokens = {
            1: OPEN,
            2: "think",
            3: SEP,
            4: CLOSE,
            5: "response",
            6: "<|end_of_msg|>",
        }
        return "".join(tokens.get(token_id, chr(token_id)) for token_id in token_ids)


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


def test_consumed_think_prefix_preserves_quoted_think_opener_in_answer():
    parser = KimiK3ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    body = f'{{"marker":"{THINK_OPEN}"}}'

    reasoning, content = parser.extract_reasoning_content(
        f"step{THINK_CLOSE}{RESPONSE_OPEN}{body}",
        request,
    )

    assert reasoning == "step"
    assert content == body


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


@pytest.mark.parametrize("marker", [THINK_OPEN, THINK_CLOSE])
def test_response_only_json_preserves_quoted_think_marker(marker):
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    body = f'{{"marker":"{marker}"}}'

    reasoning, content, tool_calls = parser.parse(
        RESPONSE_OPEN + body,
        request,
        model_output_token_ids=[*RESPONSE_OPEN_IDS, *map(ord, body)],
    )

    assert reasoning is None
    assert content == body
    assert tool_calls == []


@pytest.mark.parametrize("finished_reasoning", [False, True])
def test_consumed_think_prefix_preserves_literal_response_marker(finished_reasoning):
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    reasoning = "Consider the marker " + RESPONSE_OPEN + " here"
    body = '{"ok":true}'
    # The prompt already opened thinking. The generated opener below is ordinary
    # text; only the later response opener is made of control tokens.
    text = reasoning
    token_ids = list(map(ord, reasoning))
    if finished_reasoning:
        text += THINK_CLOSE + RESPONSE_OPEN + body
        token_ids += [*CLOSE_IDS, *RESPONSE_OPEN_IDS, *map(ord, body)]

    assert parser.parse(
        text,
        request,
        model_output_token_ids=token_ids,
    ) == (reasoning, body if finished_reasoning else None, [])


@pytest.mark.parametrize("prefix", ["\n", f"Quoted {RESPONSE_OPEN} here. "])
def test_response_control_after_reasoning_recovers_content(prefix):
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    body = '{"ok":true}'
    text = prefix + RESPONSE_OPEN + body
    assert parser.parse(
        text,
        request,
        model_output_token_ids=[*map(ord, prefix), *RESPONSE_OPEN_IDS, *map(ord, body)],
    ) == (prefix, body, [])


@pytest.mark.parametrize(
    ("prefix", "continued", "with_close"),
    [("x <", "", True), ("a <", " b", True), ("a <", " b", False)],
)
def test_streaming_transition_flushes_withheld_reasoning(prefix, continued, with_close):
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    boundary_text = (THINK_CLOSE if with_close else "") + RESPONSE_OPEN
    boundary_ids = (CLOSE_IDS if with_close else []) + RESPONSE_OPEN_IDS
    chunks = [
        (prefix, list(map(ord, prefix))),
        (
            continued + boundary_text + "ok",
            [*map(ord, continued), *boundary_ids, *map(ord, "ok")],
        ),
    ]
    reasoning, content = "", ""
    for index, (text, ids) in enumerate(chunks):
        delta = parser.parse_delta(text, ids, request, finished=index == 1)
        if delta is not None:
            reasoning += delta.reasoning or ""
            content += delta.content or ""

    assert reasoning == prefix + continued
    assert content == "ok"
    assert ReasoningOnlyParser(_dummy_tokenizer()).parse(
        prefix + continued + boundary_text + "ok",
        request,
        model_output_token_ids=[
            *map(ord, prefix + continued),
            *boundary_ids,
            *map(ord, "ok"),
        ],
    ) == (prefix + continued, "ok", [])


def test_streaming_boundary_search_only_scans_the_step_window(monkeypatch):
    parser = KimiK3ReasoningParser(DummyTokenizer())
    searched_lengths = []
    original_bounds = parser._reasoning_bounds

    def observe(token_ids):
        searched_lengths.append(len(token_ids))
        return original_bounds(token_ids)

    monkeypatch.setattr(parser, "_reasoning_bounds", observe)
    for length in (16, 16384):
        ids = [ord("x")] * length
        parser.extract_reasoning_streaming(
            previous_text="x" * (length - 1),
            current_text="x" * length,
            delta_text="x",
            previous_token_ids=ids[:-1],
            current_token_ids=ids,
            delta_token_ids=[ord("x")],
        )

    # One new token plus two carried tokens, regardless of reasoning length.
    assert searched_lengths == [3, 3]


@pytest.mark.parametrize("with_close", [False, True])
@pytest.mark.parametrize("partial_marker_text", [False, True])
def test_streaming_waits_for_buffered_delimiter_text(with_close, partial_marker_text):
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    boundary_text = (THINK_CLOSE if with_close else "") + RESPONSE_OPEN
    boundary_ids = (CLOSE_IDS if with_close else []) + RESPONSE_OPEN_IDS
    visible = "plan" + boundary_text[:8] if partial_marker_text else ""
    full_text = "plan" + boundary_text + "answer"
    chunks = [
        ("", list(map(ord, "plan"))),
        (visible, boundary_ids),
        (full_text[len(visible) :], list(map(ord, "answer"))),
    ]
    reasoning, content = "", ""
    for index, (text, ids) in enumerate(chunks):
        delta = parser.parse_delta(
            text, ids, request, finished=index == len(chunks) - 1
        )
        if delta is not None:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
        if index == 1:
            assert not parser._stream_state.reasoning_ended
            # Grammar can activate on IDs while frontend text is still buffered.
            assert parser._reasoning_parser is not None
            assert parser._reasoning_parser.is_reasoning_end_streaming(
                [*map(ord, "plan"), *boundary_ids], boundary_ids
            )

    assert reasoning == "plan"
    assert content == "answer"
    assert parser._stream_state.reasoning_ended


def test_streaming_reopen_does_not_repeat_prior_content():
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    chunks = [
        ("p", [ord("p")]),
        (THINK_CLOSE + THINK_OPEN + "q", [*CLOSE_IDS, *OPEN_IDS, ord("q")]),
        (" more", list(map(ord, " more"))),
        (" tail", list(map(ord, " tail"))),
    ]
    reasoning, content = "", ""
    for index, (text, ids) in enumerate(chunks):
        delta = parser.parse_delta(
            text, ids, request, finished=index == len(chunks) - 1
        )
        if delta is not None:
            reasoning += delta.reasoning or ""
            content += delta.content or ""

    # Preserve the existing close/reopen behavior, emitting each delta once.
    assert reasoning == "p"
    assert content == THINK_OPEN + "q more tail"


@pytest.mark.parametrize(
    ("visible", "token_ids", "expected"),
    [
        (
            RESPONSE_OPEN + '{"a":1}',
            [*RESPONSE_OPEN_IDS, *map(ord, '{"a":1}END tail')],
            (None, '{"a":1}', []),
        ),
        ("plan ", list(map(ord, "plan STOP more")), ("plan ", None, [])),
        (
            RESPONSE_OPEN + '{"a":1}',
            [*RESPONSE_OPEN_IDS, *map(ord, '{"a":1}'), 6],
            (None, '{"a":1}', []),
        ),
    ],
    ids=["stop-in-content", "stop-in-reasoning", "omitted-eos"],
)
def test_parse_preserves_engine_text_truncation(visible, token_ids, expected):
    # The engine retains stop/EOS IDs after removing their text from output.text.
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    assert parser.parse(visible, request, model_output_token_ids=token_ids) == expected


@pytest.mark.parametrize("prefix", ["", "plan", THINK_OPEN + "plan"])
def test_response_transition_preserves_reasoning_and_content(prefix):
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    body = '{"marker":"' + RESPONSE_OPEN + '"}'
    content = ""
    reasoning = ""
    prefix_ids = (
        [*OPEN_IDS, *map(ord, "plan")]
        if prefix.startswith(THINK_OPEN)
        else list(map(ord, prefix))
    )
    for text, ids in (
        (prefix, prefix_ids),
        (OPEN + "response", RESPONSE_OPEN_IDS[:-1]),
        (SEP + "{", [RESPONSE_OPEN_IDS[-1], ord("{")]),
        (body[1:], list(map(ord, body[1:]))),
    ):
        delta = parser.parse_delta(text, ids, request, finished=False)
        if delta:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
    assert reasoning == ("plan" if prefix else "")
    assert content == body
    assert parser._stream_state.reasoning_ended


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("prompt_ends_inside_close", [False, True])
def test_constraint_start_with_prompt(committed, prompt_ends_inside_close):
    from types import SimpleNamespace

    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    tokenizer = DummyTokenizer()
    reasoner = KimiK3ReasoningParser(tokenizer)
    # The generated-output boundary must not depend on a particular template.
    prompt = [
        9,
        *tokenizer.encode(f'{OPEN}message role="assistant" custom="x"{SEP}'),
        *OPEN_IDS,
    ]
    if prompt_ends_inside_close:
        prompt.extend([9, *CLOSE_IDS[:-1]])
    request = Request(
        request_id="test",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(
            max_tokens=32,
            structured_outputs=StructuredOutputsParams(json_object=True),
        ),
        pooling_params=None,
    )
    if prompt_ends_inside_close:
        delta = [CLOSE_IDS[-1], ord("{")]
    else:
        request.append_output_token_ids(
            [*OPEN_IDS, *map(ord, "plan"), *RESPONSE_OPEN_IDS[:-1]]
        )
        # A later invalid draft opener must not hide the first transition.
        delta = [RESPONSE_OPEN_IDS[-1], *OPEN_IDS, *RESPONSE_OPEN_IDS, ord("{")]
    if committed:
        request.append_output_token_ids(delta)
    manager = SimpleNamespace(
        enable_in_reasoning=False, _get_reasoner=lambda _: reasoner
    )
    bounds = StructuredOutputManager._get_constraint_bounds(
        manager,
        request,
        delta,
        spec_tokens_committed=committed,
    )
    assert bounds.grammar_start == bounds.constraint_start == 1


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
        pytest.param(
            [ord("\n"), *RESPONSE_OPEN_IDS, *map(ord, "{}")],
            1,
            id="response_open_after_consumed_prefix_ends_reasoning",
        ),
        pytest.param([*OPEN_IDS, 9, 10], 2, id="unterminated_after_open"),
        pytest.param([9, 10, 11], 3, id="unterminated_without_markers"),
        pytest.param([], 0, id="empty"),
        pytest.param([9, 4, 2, 10], 4, id="partial_close_marker_is_reasoning"),
        pytest.param(
            [*OPEN_IDS, 9, *RESPONSE_OPEN_IDS, 11],
            1,
            id="response_open_implicitly_closes_thinking",
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
    previous_ids = [*OPEN_IDS, *map(ord, "step")]
    partial_close = parser.extract_reasoning_content_streaming(
        previous_text=previous_text,
        current_text=previous_text + CLOSE,
        delta_text=CLOSE,
        previous_token_ids=previous_ids,
        current_token_ids=[*previous_ids, 4],
        delta_token_ids=[4],
    )
    closed = parser.extract_reasoning_content_streaming(
        previous_text=previous_text + CLOSE,
        current_text=previous_text + f"{THINK_CLOSE}{RESPONSE_OPEN}answer",
        delta_text=f"think{SEP}{RESPONSE_OPEN}answer",
        previous_token_ids=[*previous_ids, 4],
        current_token_ids=[
            *previous_ids,
            *CLOSE_IDS,
            *RESPONSE_OPEN_IDS,
            *map(ord, "answer"),
        ],
        delta_token_ids=[2, 3, *RESPONSE_OPEN_IDS, *map(ord, "answer")],
    )

    assert partial_close is None
    assert isinstance(closed, DeltaMessage)
    assert closed.reasoning is None
    assert closed.content == f"{RESPONSE_OPEN}answer"
    assert parser.extract_content_ids(
        [2, 3, *RESPONSE_OPEN_IDS, *map(ord, "answer")]
    ) == [*RESPONSE_OPEN_IDS, *map(ord, "answer")]


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


def test_extract_reasoning_with_generation_prefix_consumed():
    parser = KimiK3ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content = parser.extract_reasoning_content(
        f"step{THINK_CLOSE}{RESPONSE_OPEN}answer",
        request,
    )

    assert reasoning == "step"
    assert content == "answer"


def test_response_only_transition_preserves_content():
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    body = '{"marker":"' + RESPONSE_OPEN + '"}'
    content = ""
    for text, ids in (
        (OPEN + "response", RESPONSE_OPEN_IDS[:-1]),
        (SEP + "{", [RESPONSE_OPEN_IDS[-1], ord("{")]),
        (body[1:], list(map(ord, body[1:]))),
    ):
        delta = parser.parse_delta(text, ids, request, finished=False)
        if delta:
            assert not delta.reasoning
            content += delta.content or ""
    assert content == body
    assert parser._stream_state.reasoning_ended


def test_quoted_response_prefix_with_consumed_think_opener():
    parser = ReasoningOnlyParser(_dummy_tokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])
    quoted = RESPONSE_OPEN + " is an example"
    first = parser.parse_delta(
        quoted,
        list(map(ord, quoted)),
        request,
        finished=False,
    )
    assert first is not None and first.reasoning == quoted
    assert first.content is None
    assert not parser._stream_state.reasoning_ended

    second = parser.parse_delta(
        THINK_CLOSE + RESPONSE_OPEN + "{}",
        [*CLOSE_IDS, *RESPONSE_OPEN_IDS, ord("{"), ord("}")],
        request,
        finished=True,
    )
    assert second is not None and second.content == "{}"
    assert not second.reasoning
