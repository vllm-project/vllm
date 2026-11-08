# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
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
RESPONSE_OPEN_IDS = [ord(ch) for ch in RESPONSE_OPEN]


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


@pytest.mark.parametrize(
    "req",
    [
        ChatCompletionRequest(model="test-model", messages=[]),
        ResponsesRequest(model="test-model", input="hi"),
    ],
)
def test_adjust_request_keeps_xtml_markers_contiguous(req):
    parser = KimiK3ReasoningParser(DummyTokenizer())

    adjusted = parser.adjust_request(req)

    assert adjusted.skip_special_tokens is False
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


class _FullRescanParser(KimiK3ReasoningParser):
    """The streaming extraction as it was before it resumed its marker searches:
    every call re-reads both accumulated texts from the start. The incremental
    path has to reproduce it exactly, so it stays here as the reference."""

    def _reasoning_text_ready_to_emit(self, text: str) -> str:
        m_open = self._think_open_re.search(text)
        if m_open is not None:
            text = text[m_open.end() :]
        overlap = 0
        for marker in (self._think_open, self._think_close):
            max_check = min(len(marker) - 1, len(text))
            for n in range(max_check, 0, -1):
                if text.endswith(marker[:n]):
                    overlap = max(overlap, n)
                    break
        return text[:-overlap] if overlap else text

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        self._last_streaming_delta_token_ids = None
        self._last_streaming_content_token_ids = None
        if not self._thinking_enabled:
            return DeltaMessage(content=delta_text)

        if self._think_close_re.search(previous_text):
            return DeltaMessage(content=delta_text)

        m_close = self._think_close_re.search(current_text)
        if m_close is not None:
            self._last_streaming_delta_token_ids = tuple(delta_token_ids)
            self._last_streaming_content_token_ids = self._extract_content_ids(
                list(current_token_ids)
            )
            m_open = self._think_open_re.search(current_text)
            r_start = m_open.end() if m_open is not None else 0
            reasoning = current_text[r_start : m_close.start()]
            already_sent = self._reasoning_text_ready_to_emit(previous_text)
            if reasoning.startswith(already_sent):
                reasoning_delta = reasoning[len(already_sent) :]
            else:
                reasoning_delta = reasoning
            content = current_text[m_close.end() :]
            return DeltaMessage(
                reasoning=reasoning_delta or None,
                content=content or None,
            )

        current_reasoning = self._reasoning_text_ready_to_emit(current_text)
        previous_reasoning = self._reasoning_text_ready_to_emit(previous_text)
        if current_reasoning.startswith(previous_reasoning):
            reasoning_delta = current_reasoning[len(previous_reasoning) :]
        else:
            reasoning_delta = current_reasoning
        if not reasoning_delta:
            return None
        return DeltaMessage(reasoning=reasoning_delta)

    # The class-level alias would otherwise stay bound to the production method.
    extract_reasoning_content_streaming = extract_reasoning_streaming


# Marker pieces, their fragments, whitespace-tolerant spellings and near misses:
# the streams below keep splitting, completing and abandoning markers.
_STREAM_FRAGMENTS = [
    "a",
    "bc",
    " ",
    "  ",
    "\n",
    "x y",
    "<",
    "|",
    "<|",
    "|>",
    "open",
    "close",
    "think",
    "sep",
    "response",
    "<|open|>",
    "<|close|>",
    "<|sep|>",
    THINK_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    THINK_CLOSE,
    f"{OPEN} think {SEP}",
    f"{CLOSE}\nthink\n{SEP}",
    f"{CLOSE}response{SEP}",
    RESPONSE_OPEN,
    "<|clo",
    "se|>think<|se",
    "p|>",
    "<|open|>thin",
]


def _random_chunks(rnd) -> list[str]:
    """One random stream split into deltas of 0-21 characters, so a marker is cut
    at arbitrary points (empty deltas included)."""
    text = "".join(rnd.choice(_STREAM_FRAGMENTS) for _ in range(rnd.randrange(1, 16)))
    chunks, i = [], 0
    while i < len(text):
        size = rnd.choice([0, 1, 1, 2, 3, 5, 8, 21])
        chunks.append(text[i : i + size])
        i += size
    return chunks


def _delta_ids(chunk: str) -> list[int]:
    return [ord(ch) % 5 + 1 for ch in chunk]


def _stream_in_lockstep(chunks, *, copy_previous=False):
    """Feed one stream to the parser and to the reference, call by call, the way
    the serving layer does: each call hands back the previous call's text."""
    parser = KimiK3ReasoningParser(DummyTokenizer())
    reference = _FullRescanParser(DummyTokenizer())
    previous = ""
    previous_ids: list[int] = []
    for chunk in chunks:
        current = previous + chunk
        delta_ids = _delta_ids(chunk)
        current_ids = previous_ids + delta_ids
        # An equal but distinct string: trusted by content, so it must agree too.
        given = (previous + "x")[:-1] if copy_previous and previous else previous
        args = (given, current, chunk, previous_ids, current_ids, delta_ids)
        expected = reference.extract_reasoning_streaming(*args)
        assert parser.extract_reasoning_streaming(*args) == expected, (
            previous,
            chunk,
        )
        assert parser.extract_content_ids(current_ids) == reference.extract_content_ids(
            current_ids
        )
        previous, previous_ids = current, current_ids


@pytest.mark.parametrize("seed", range(24))
def test_streaming_matches_full_rescan_over_marker_dense_streams(seed):
    import random

    rnd = random.Random(seed)
    for _ in range(150):
        _stream_in_lockstep(_random_chunks(rnd))


@pytest.mark.parametrize("seed", range(4))
def test_streaming_matches_full_rescan_for_equal_but_distinct_previous_text(seed):
    import random

    rnd = random.Random(seed)
    for _ in range(100):
        _stream_in_lockstep(_random_chunks(rnd), copy_previous=True)


@pytest.mark.parametrize("seed", range(4))
def test_streaming_matches_full_rescan_when_the_parser_is_reused(seed):
    """A new stream starts from an empty previous text, and two streams may take
    turns on one parser: neither may inherit the other's scan."""
    import random

    rnd = random.Random(seed)
    parser = KimiK3ReasoningParser(DummyTokenizer())
    reference = _FullRescanParser(DummyTokenizer())

    for _ in range(100):
        streams = [_random_chunks(rnd), _random_chunks(rnd)]
        texts = ["", ""]
        ids: list[list[int]] = [[], []]
        for turn in range(2 * max(len(s) for s in streams)):
            which = turn % 2
            if turn // 2 >= len(streams[which]):
                continue
            chunk = streams[which][turn // 2]
            current = texts[which] + chunk
            delta_ids = _delta_ids(chunk)
            current_ids = ids[which] + delta_ids
            args = (texts[which], current, chunk, ids[which], current_ids, delta_ids)
            assert parser.extract_reasoning_streaming(
                *args
            ) == reference.extract_reasoning_streaming(*args), (texts[which], chunk)
            texts[which], ids[which] = current, current_ids


@pytest.mark.parametrize("seed", range(4))
def test_streaming_matches_full_rescan_for_calls_that_do_not_append(seed):
    import random

    rnd = random.Random(seed)
    parser = KimiK3ReasoningParser(DummyTokenizer())
    reference = _FullRescanParser(DummyTokenizer())

    def text() -> str:
        return "".join(rnd.choice(_STREAM_FRAGMENTS) for _ in range(rnd.randrange(8)))

    for _ in range(500):
        previous, current, delta = text(), text(), text()
        args = (previous, current, delta, [1], [1, 2], [2])
        assert parser.extract_reasoning_streaming(
            *args
        ) == reference.extract_reasoning_streaming(*args), args


@pytest.mark.parametrize(
    ("current", "delta"),
    [
        ("XYZWV", "qq"),  # right length, does not end with the delta
        ("XYZWqq", "qq"),  # ends with the delta, longer than previous + delta
        ("Xqq", "qq"),  # ends with the delta, shorter than previous + delta
        ("XYZW", "qq"),  # wrong length, does not end with the delta
        ("abcXY", "qq"),  # extends previous, does not end with the delta
    ],
)
def test_streaming_falls_back_when_a_call_does_not_append_to_the_scanned_text(
    current, delta
):
    """The scan is only valid for the text it was taken on. Handed that very
    string back with a `current_text` that is not `previous_text +
    delta_text`, the call must be answered from scratch, as before."""
    parser = KimiK3ReasoningParser(DummyTokenizer())
    reference = _FullRescanParser(DummyTokenizer())
    previous = "abc"
    for p in (parser, reference):
        # Two appends, so the scan has resumed and its offsets are tight.
        p.extract_reasoning_streaming("", "ab", "ab", [], [], [])
        p.extract_reasoning_streaming("ab", previous, "c", [], [], [])

    args = (previous, current, delta, [1], [1, 2], [2])
    assert parser.extract_reasoning_streaming(
        *args
    ) == reference.extract_reasoning_streaming(*args)


def test_streaming_releases_the_reasoning_text_when_the_think_block_closes():
    """The scan keeps the text it was taken on alive between calls. Once the
    think block closes the stream moves on to content, and the parser must not
    keep the whole reasoning text for the rest of the request."""
    import gc
    import weakref

    class Text(str):
        """A str that can be weakly referenced."""

    parser = KimiK3ReasoningParser(DummyTokenizer())
    reasoning = Text("step " * 1000)
    previous = parser.extract_reasoning_streaming("", reasoning, reasoning, [], [], [])
    assert previous is not None and previous.reasoning == reasoning

    closed = Text(reasoning + THINK_CLOSE + "answer")
    delta = THINK_CLOSE + "answer"
    message = parser.extract_reasoning_streaming(reasoning, closed, delta, [], [], [])
    assert message is not None and message.content == "answer"

    texts = [weakref.ref(reasoning), weakref.ref(closed)]
    del reasoning, closed
    gc.collect()
    assert all(text() is None for text in texts)


def test_delegating_parser_streaming_matches_full_rescan():
    """Through `DelegatingParser.parse_delta`, the entry point that owns the
    text it passes back as `previous_text`."""
    import random

    class ReferenceParser(KimiK3Parser):
        reasoning_parser_cls = _FullRescanParser

    request = ChatCompletionRequest(model="test-model", messages=[])
    rnd = random.Random(1234)
    for _ in range(300):
        chunks = _random_chunks(rnd)
        parsers = [
            ReasoningOnlyParser(_dummy_tokenizer()),
            ReferenceParser(_dummy_tokenizer()),
        ]
        for i, chunk in enumerate(chunks):
            messages = [
                parser.parse_delta(
                    chunk,
                    _delta_ids(chunk),
                    request,
                    prompt_token_ids=[7],
                    finished=i == len(chunks) - 1,
                )
                for parser in parsers
            ]
            assert messages[0] == messages[1], (chunks, i)


@pytest.mark.parametrize("copy_previous", [False, True])
def test_streaming_appends_resume_the_scan_instead_of_rescanning(copy_previous):
    """Deltas that complete no marker, including half-written ones, are answered
    from the resumed scan, also when the caller hands back an equal copy of the
    previous text rather than the very string it was given."""
    rescans = 0

    class CountingParser(KimiK3ReasoningParser):
        def _extract_reasoning_from_scratch(self, *args):
            nonlocal rescans
            rescans += 1
            return super()._extract_reasoning_from_scratch(*args)

    parser = CountingParser(DummyTokenizer())
    previous = ""
    for delta in ["reasoning step "] * 20 + ["<|clo", "se|>thin", "k step "]:
        current = previous + delta
        given = (previous + "x")[:-1] if copy_previous and previous else previous
        parser.extract_reasoning_streaming(given, current, delta, [], [], [])
        previous = current

    assert rescans == 0


def test_streaming_cost_does_not_grow_with_the_accumulated_text():
    """Each call used to re-read the whole accumulated text, so the cost of one
    delta grew with the length of the reasoning; the median over the last
    calls of a long stream must stay close to that of a short one."""
    import gc
    import statistics
    import time

    def tail_median(num_chars: int) -> float:
        parser = KimiK3ReasoningParser(DummyTokenizer())
        text = "reasoning step " * (num_chars // 15)
        deltas = [text[i : i + 4] for i in range(0, len(text), 4)]
        previous, samples = "", []
        for i, delta in enumerate(deltas):
            current = previous + delta
            start = time.perf_counter()
            parser.extract_reasoning_streaming(previous, current, delta, [], [], [])
            elapsed = time.perf_counter() - start
            if i >= len(deltas) - 300:
                samples.append(elapsed)
            previous = current
        return statistics.median(samples)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        short, long = tail_median(3_000), tail_median(300_000)
    finally:
        if gc_was_enabled:
            gc.enable()

    assert long < 6 * short, (short, long)
