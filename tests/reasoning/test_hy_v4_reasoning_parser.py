# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the HYV4 reasoning parser.

The extractor takes a plain ``{token: id}`` vocab, so these run offline against
a synthetic vocab instead of downloading a checkpoint.
"""

import pytest

from vllm.reasoning.hy_v4_reasoning_parser import (
    HYV4ReasoningExtractor,
    HYV4ReasoningParser,
    detect_token_suffix,
)

SUFFIX = ":6124c78e"
START = f"<think{SUFFIX}>"
END = f"</think{SUFFIX}>"
START_ID = 10
END_ID = 11
VOCAB = {START: START_ID, END: END_ID}


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None, init_kwargs=None):
        self._vocab = VOCAB if vocab is None else vocab
        self.init_kwargs = init_kwargs or {}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


@pytest.fixture
def thinking() -> HYV4ReasoningExtractor:
    return HYV4ReasoningExtractor(VOCAB, SUFFIX, thinking=True)


@pytest.fixture
def no_think() -> HYV4ReasoningExtractor:
    return HYV4ReasoningExtractor(VOCAB, SUFFIX, thinking=False)


class TestDetectTokenSuffix:
    """The suffix is per-checkpoint and read from the tokenizer vocab."""

    def test_suffixed_tokens(self):
        assert detect_token_suffix(FakeTokenizer({START: 1})) == SUFFIX

    def test_unsuffixed_tokens(self):
        assert detect_token_suffix(FakeTokenizer({"<think>": 1})) == ""

    @pytest.mark.parametrize(
        "token", ["<tool_calls:x>", "<tool_call:x>", "<arg_key:x>", "<arg_value:x>"]
    )
    def test_any_structural_token_carries_the_suffix(self, token):
        assert detect_token_suffix(FakeTokenizer({token: 1})) == ":x"

    def test_no_structural_tokens(self):
        assert detect_token_suffix(FakeTokenizer({"hello": 1})) == ""

    def test_rejects_special_token_declaration(self):
        """transformers 5 no longer round-trips these, so fail loudly."""
        tokenizer = FakeTokenizer(
            {"<think>": 1},
            {"model_specific_special_tokens": {"think_begin_token": START}},
        )
        pytest.importorskip("transformers")
        import transformers

        if int(transformers.__version__.split(".")[0]) < 5:
            pytest.skip("guard only applies to transformers>=5")
        with pytest.raises(RuntimeError, match="tokenizer_config.json"):
            detect_token_suffix(tokenizer)


class TestExtractorInit:
    def test_thinking_requires_think_tokens(self):
        with pytest.raises(RuntimeError, match="start/end tokens"):
            HYV4ReasoningExtractor({}, SUFFIX, thinking=True)

    def test_no_think_tolerates_missing_tokens(self):
        """no_think never inspects the ids, so a missing token is not fatal."""
        extractor = HYV4ReasoningExtractor({}, SUFFIX, thinking=False)
        assert extractor.start_token_id is None
        assert extractor.end_token_id is None


class TestExtractReasoning:
    def test_start_token_absent_from_output(self, thinking):
        """HYV4 injects <think> at the end of the prompt, not in the output."""
        assert thinking.extract_reasoning(f"pondering{END}answer") == (
            "pondering",
            "answer",
        )

    def test_start_token_present_is_stripped(self, thinking):
        """Older prompts echo <think>; it must not leak into reasoning."""
        assert thinking.extract_reasoning(f"{START}pondering{END}answer") == (
            "pondering",
            "answer",
        )

    def test_truncated_reasoning_has_no_content(self, thinking):
        assert thinking.extract_reasoning("pondering") == ("pondering", None)

    def test_empty_content_is_none(self, thinking):
        """An empty string would render as a stray empty message."""
        assert thinking.extract_reasoning(f"pondering{END}") == ("pondering", None)

    def test_no_think_is_all_content(self, no_think):
        assert no_think.extract_reasoning("answer") == (None, "answer")


class TestExtractReasoningStreaming:
    @staticmethod
    def run(extractor: HYV4ReasoningExtractor, chunks: list[tuple[str, list[int]]]):
        previous = ""
        previous_ids: list[int] = []
        reasoning, content = "", ""
        for text, ids in chunks:
            current = previous + text
            delta = extractor.extract_reasoning_streaming(
                previous, current, text, previous_ids, previous_ids + ids, ids
            )
            if delta is not None:
                reasoning += delta["reasoning"] or ""
                content += delta["content"] or ""
            previous, previous_ids = current, previous_ids + ids
        return reasoning, content

    def test_splits_on_end_token(self, thinking):
        chunks = [("pondering", [1]), (END, [END_ID]), ("answer", [2])]
        assert self.run(thinking, chunks) == ("pondering", "answer")

    def test_no_think_streams_content_only(self, no_think):
        assert self.run(no_think, [("answer", [2])]) == ("", "answer")

    @pytest.mark.parametrize("token_id", [START_ID, END_ID])
    def test_lone_special_token_delta_is_dropped(self, thinking, token_id):
        """A delta that is just <think>/</think> carries no user-visible text."""
        assert (
            thinking.extract_reasoning_streaming(
                "x", "x", "", [1], [1, token_id], [token_id]
            )
            is None
        )

    def test_end_token_inside_delta_splits_it(self, thinking):
        delta = thinking.extract_reasoning_streaming(
            "r", f"r{END}c", f"{END}c", [1], [1, END_ID, 2], [END_ID, 2]
        )
        assert delta == {"reasoning": "", "content": "c"}

    def test_after_end_everything_is_content(self, thinking):
        delta = thinking.extract_reasoning_streaming(
            f"r{END}", f"r{END}c", "c", [1, END_ID], [1, END_ID, 2], [2]
        )
        assert delta == {"reasoning": None, "content": "c"}

    def test_before_end_everything_is_reasoning(self, thinking):
        delta = thinking.extract_reasoning_streaming("r", "ra", "a", [1], [1, 2], [2])
        assert delta == {"reasoning": "a", "content": None}

    def test_echoed_start_token_in_delta(self, thinking):
        """Whole ``<think>r</think>c`` arriving at once."""
        text = f"{START}r{END}c"
        delta = thinking.extract_reasoning_streaming(
            "", text, text, [], [START_ID, 1, END_ID, 2], [START_ID, 1, END_ID, 2]
        )
        assert delta == {"reasoning": "r", "content": "c"}

    def test_reasoning_continues_after_echoed_start(self, thinking):
        delta = thinking.extract_reasoning_streaming(
            START, f"{START}a", "a", [START_ID], [START_ID, 1], [1]
        )
        assert delta == {"reasoning": "a", "content": None}

    def test_content_after_echoed_start_and_end(self, thinking):
        delta = thinking.extract_reasoning_streaming(
            f"{START}{END}",
            f"{START}{END}c",
            "c",
            [START_ID, END_ID],
            [START_ID, END_ID, 2],
            [2],
        )
        assert delta == {"reasoning": None, "content": "c"}

    def test_no_think_empty_delta_is_dropped(self, no_think):
        assert no_think.extract_reasoning_streaming("a", "a", "", [1], [1], []) is None


class TestReasoningState:
    def test_is_reasoning_end(self, thinking, no_think):
        assert not thinking.has_reasoning_ended([1, 2])
        assert thinking.has_reasoning_ended([1, END_ID, 2])
        assert no_think.has_reasoning_ended([1, 2])

    def test_is_reasoning_end_scans_backwards(self, thinking):
        """A later <think> reopens reasoning, so order matters."""
        assert thinking.has_reasoning_ended([START_ID, END_ID])
        assert not thinking.has_reasoning_ended([END_ID, START_ID])

    def test_is_reasoning_end_on_empty(self, thinking):
        assert not thinking.has_reasoning_ended([])

    def test_reasoning_ended_in_delta_is_an_edge_detector(self, thinking):
        """Only True on the step carrying </think>, unlike the state query."""
        assert thinking.reasoning_ended_in_delta([1, END_ID], [END_ID])
        assert not thinking.reasoning_ended_in_delta([1, END_ID], [2])

    def test_reasoning_ended_in_delta_no_think(self, no_think):
        assert no_think.reasoning_ended_in_delta([1], [2])

    def test_extract_content_ids(self, thinking):
        assert thinking.extract_content_ids([1, END_ID, 2, 3]) == [2, 3]
        assert thinking.extract_content_ids([1, 2]) == []

    def test_extract_content_ids_needs_a_token_after_end(self, thinking):
        """</think> as the final token means content has not started yet."""
        assert thinking.extract_content_ids([1, 2, END_ID]) == []
        assert thinking.extract_content_ids([1, 2, END_ID, 3]) == [3]

    def test_extract_content_ids_no_think_returns_everything(self, no_think):
        assert no_think.extract_content_ids([1, 2]) == [1, 2]

    def test_count_reasoning_tokens_ignores_injected_start(self, thinking, no_think):
        assert thinking.count_reasoning_tokens([1, 2, END_ID, 3]) == 2
        assert thinking.count_reasoning_tokens([START_ID, 1, 2, END_ID]) == 2
        # Truncated reasoning: everything counts.
        assert thinking.count_reasoning_tokens([1, 2, 3]) == 3
        assert no_think.count_reasoning_tokens([1, 2, 3]) == 0

    @pytest.mark.parametrize("token_ids", [[], [END_ID], [START_ID, END_ID]])
    def test_count_reasoning_tokens_empty_reasoning(self, thinking, token_ids):
        assert thinking.count_reasoning_tokens(token_ids) == 0


class TestReasoningDelimiters:
    """``ReasoningConfig.initialize_token_ids`` derives the thinking-budget
    token ids from these, and silently disables itself when they are None."""

    @pytest.mark.parametrize("reasoning_effort", [None, "no_think", "high"])
    def test_delimiters_reported_in_every_mode(self, reasoning_effort):
        kwargs = (
            {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
        )
        parser = HYV4ReasoningParser(FakeTokenizer(), **kwargs)
        assert parser.reasoning_start_str == START
        assert parser.reasoning_end_str == END


class TestThinkingMode:
    """The default must track the chat template, which uses 'high' when the
    request omits reasoning_effort. Defaulting to no_think would return the
    chain-of-thought and a stray ``</think:SUF>`` as content."""

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, True),
            ({"reasoning_effort": None}, True),
            ({"reasoning_effort": "high"}, True),
            ({"reasoning_effort": "low"}, True),
            ({"reasoning_effort": "no_think"}, False),
            ({"chat_template_kwargs": {"reasoning_effort": "no_think"}}, False),
            ({"chat_template_kwargs": {"reasoning_effort": "high"}}, True),
            ({"chat_template_kwargs": {}}, True),
            ({"chat_template_kwargs": None}, True),
        ],
    )
    def test_thinking_resolution(self, kwargs, expected):
        parser = HYV4ReasoningParser(FakeTokenizer(), **kwargs)
        assert parser._extractor.thinking is expected

    def test_chat_template_kwargs_wins(self):
        parser = HYV4ReasoningParser(
            FakeTokenizer(),
            reasoning_effort="high",
            chat_template_kwargs={"reasoning_effort": "no_think"},
        )
        assert parser._extractor.thinking is False


class TestParserDelegation:
    """The wrapper adapts the extractor to vLLM types; check the seams only."""

    @pytest.fixture
    def parser(self) -> HYV4ReasoningParser:
        return HYV4ReasoningParser(FakeTokenizer())

    def test_suffix_is_detected_from_the_tokenizer(self, parser):
        assert parser.reasoning_start_str == START

    def test_extract_reasoning(self, parser):
        assert parser.extract_reasoning(f"r{END}c", None) == ("r", "c")

    def test_state_queries(self, parser):
        assert parser.is_reasoning_end([1, END_ID])
        assert parser.is_reasoning_end_streaming([1], [END_ID])
        assert parser.extract_content_ids([1, END_ID, 5, 6]) == [5, 6]
        assert parser.count_reasoning_tokens([1, 2, END_ID, 3]) == 2

    def test_streaming_returns_delta_message(self, parser):
        delta = parser.extract_reasoning_streaming(
            "r", f"r{END}c", f"{END}c", [1], [1, END_ID, 2], [END_ID, 2]
        )
        assert delta is not None
        assert delta.content == "c"

    def test_streaming_passes_through_none(self, parser):
        """A dropped delta must stay dropped, not become an empty message."""
        assert (
            parser.extract_reasoning_streaming("x", "x", "", [1], [1, END_ID], [END_ID])
            is None
        )
