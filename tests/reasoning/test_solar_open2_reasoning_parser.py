# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning.solar_open2_reasoning_parser import (
    SolarOpen2ReasoningParser,
)
from vllm.tool_parsers.solar_open2_tool_parser import SolarOpen2ToolParser

pytestmark = pytest.mark.skip_global_cleanup


class MockSolarOpen2Tokenizer:
    """Offline stand-in that encodes each sentinel as one synthetic token.

    Atomicity is a property of this mock, not something it verifies about the
    real tokenizer; it is the precondition the parser's token-level checks
    (delimiter-id resolution for ``thinking_token_budget``) need in order to
    be exercised at all.
    """

    _SENTINELS = (
        "<|think:start|>",
        "<|think:end|>",
        "<|tool_call:start|>",
        "<|tool_call:end|>",
        "<|tool_arg:start|>",
        "<|tool_arg:end|>",
    )

    def __init__(self):
        self._vocab = {chr(i): i for i in range(128)}
        self._sentinel_ids = {}
        for offset, sentinel in enumerate(self._SENTINELS):
            token_id = 128 + offset
            self._vocab[sentinel] = token_id
            self._sentinel_ids[token_id] = sentinel

    def get_vocab(self):
        return self._vocab

    def tokenize(self, text: str):
        return self.decode_tokens(self.encode(text))

    def decode_tokens(self, token_ids):
        return [self._sentinel_ids.get(t, chr(t) if t < 128 else "") for t in token_ids]

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = []
        i = 0
        while i < len(text):
            for sentinel in self._SENTINELS:
                if text.startswith(sentinel, i):
                    ids.append(self._vocab[sentinel])
                    i += len(sentinel)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, token_ids):
        return "".join(self.decode_tokens(token_ids))


class MockPartialVocabTokenizer(MockSolarOpen2Tokenizer):
    """Encodes the sentinels atomically but hides one of them from
    ``get_vocab()``, which is where the parser resolves delimiter ids."""

    def __init__(self, hidden: str):
        super().__init__()
        self._hidden = hidden

    def get_vocab(self):
        return {k: v for k, v in super().get_vocab().items() if k != self._hidden}


@pytest.fixture(scope="module")
def mock_tokenizer():
    return MockSolarOpen2Tokenizer()


@pytest.fixture
def parser(mock_tokenizer):
    return SolarOpen2ReasoningParser(mock_tokenizer)


# Under the current chat template, the generation prompt always prefills
# ``<|think:start|>`` (medium/high effort) or ``<|think:start|><|think:end|>``
# (low effort). The model output therefore never contains
# ``<|think:start|>`` — only ``<|think:end|>`` (or nothing, for low effort).


class TestExtractReasoning:
    """Non-streaming reasoning extraction tests."""

    def test_standard_reasoning_and_content(self, parser):
        model_output = "I need to analyze<|think:end|>The answer is 42"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "I need to analyze"
        assert content == "The answer is 42"

    def test_reasoning_only_no_end_tag(self, parser):
        """Under an open-reasoning effort, no end tag means truncated mid-think."""
        model_output = "Still thinking..."
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == model_output
        assert content is None

    def test_reasoning_with_empty_content(self, parser):
        model_output = "My reasoning<|think:end|>"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "My reasoning"
        assert content is None

    @pytest.mark.parametrize("effort", ["none", "low"])
    def test_pure_content_no_tags(self, parser, effort):
        """A closed-pair effort on the request makes marker-less output content."""
        model_output = "Hello, world!"
        request = SimpleNamespace(
            chat_template_kwargs=None, reasoning_effort=effort, reasoning=None
        )
        reasoning, content = parser.extract_reasoning(model_output, request)
        assert reasoning is None
        assert content == "Hello, world!"

    def test_pure_content_no_tags_via_template_kwargs(self, parser):
        model_output = "Hello, world!"
        request = SimpleNamespace(
            chat_template_kwargs={"reasoning_effort": "low"},
            reasoning_effort=None,
            reasoning=None,
        )
        reasoning, content = parser.extract_reasoning(model_output, request)
        assert reasoning is None
        assert content == "Hello, world!"

    def test_top_level_effort_wins_over_template_kwargs(self, parser):
        """Top-level ``reasoning_effort`` overrides ``chat_template_kwargs``."""
        request = SimpleNamespace(
            chat_template_kwargs={"reasoning_effort": "low"},
            reasoning_effort="high",
            reasoning=None,
        )
        reasoning, content = parser.extract_reasoning("truncated think", request)
        assert reasoning == "truncated think"
        assert content is None

    def test_server_default_effort_used_when_request_unset(self, mock_tokenizer):
        """An unset request falls back to the server's default effort."""
        parser = SolarOpen2ReasoningParser(
            mock_tokenizer, chat_template_kwargs={"reasoning_effort": "low"}
        )
        request = SimpleNamespace(
            chat_template_kwargs=None, reasoning_effort=None, reasoning=None
        )
        reasoning, content = parser.extract_reasoning("Direct answer", request)
        assert reasoning is None
        assert content == "Direct answer"

    def test_responses_effort_wins_over_template_kwargs(self, parser):
        """Responses API ``request.reasoning.effort`` overrides
        ``chat_template_kwargs``."""
        request = SimpleNamespace(
            chat_template_kwargs={"reasoning_effort": "low"},
            reasoning=SimpleNamespace(effort="high"),
        )
        reasoning, content = parser.extract_reasoning("truncated think", request)
        assert reasoning == "truncated think"
        assert content is None

        request = SimpleNamespace(
            chat_template_kwargs={"reasoning_effort": "high"},
            reasoning=SimpleNamespace(effort="low"),
        )
        reasoning, content = parser.extract_reasoning("Direct answer", request)
        assert reasoning is None
        assert content == "Direct answer"

    def test_unknown_effort_treated_as_closed(self, parser):
        """An effort outside medium/high/xhigh renders a closed pair."""
        request = SimpleNamespace(
            chat_template_kwargs=None, reasoning_effort="minimal", reasoning=None
        )
        reasoning, content = parser.extract_reasoning("Direct answer", request)
        assert reasoning is None
        assert content == "Direct answer"

    def test_empty_reasoning(self, parser):
        """An immediately emitted end tag leaves the reasoning channel absent."""
        model_output = "<|think:end|>Direct answer"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning is None
        assert content == "Direct answer"

    def test_multiline_reasoning(self, parser):
        model_output = "Step 1: analyze\nStep 2: solve\n<|think:end|>Final answer"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "Step 1: analyze\nStep 2: solve\n"
        assert content == "Final answer"


class TestEmbeddedToolCallPromotion:
    """Non-streaming recovery of tool calls emitted *inside* the think block.

    The server extracts reasoning before the tool parser sees the content
    channel, so a complete block before ``<|think:end|>`` is promoted into
    content — *after* any existing content, since ``SolarOpen2ToolParser``
    keeps only the text preceding the first ``<|tool_call:start|>``.
    """

    TOOL_BLOCK = (
        "<|tool_call:start|>get_weather\n"
        "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    )
    TOOL_BLOCK_2 = (
        "<|tool_call:start|>get_time\n"
        "<|tool_arg:start|>tz<|tool_arg:value|>KST<|tool_arg:end|>\n"
        "<|tool_call:end|>"
    )

    def test_embedded_tool_call_promoted_to_content(self, parser):
        """A block inside think surfaces in content; the thinking text stays."""
        model_output = f"I should check the weather.\n{self.TOOL_BLOCK}\n<|think:end|>"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "I should check the weather.\n\n"
        assert content == self.TOOL_BLOCK

    def test_promoted_block_appended_after_existing_content(self, parser):
        """Real content after the end tag precedes the promoted block."""
        model_output = f"thinking\n{self.TOOL_BLOCK}\n<|think:end|>The answer is 42"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "thinking\n\n"
        assert content == f"The answer is 42\n{self.TOOL_BLOCK}"

    def test_embedded_plus_tool_call_after_think_end(self, parser):
        """An embedded call and one after the end tag both land in content."""
        model_output = (
            f"plan calls\n{self.TOOL_BLOCK}\n<|think:end|>{self.TOOL_BLOCK_2}"
        )
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "plan calls\n\n"
        assert content == f"{self.TOOL_BLOCK_2}\n{self.TOOL_BLOCK}"

    def test_multiple_embedded_blocks(self, parser):
        model_output = (
            f"first\n{self.TOOL_BLOCK}\nthen\n{self.TOOL_BLOCK_2}\n<|think:end|>"
        )
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "first\n\nthen\n\n"
        assert content == f"{self.TOOL_BLOCK}\n{self.TOOL_BLOCK_2}"

    def test_block_without_args_promoted(self, parser):
        block = "<|tool_call:start|>refresh\n<|tool_call:end|>"
        model_output = f"no args needed\n{block}\n<|think:end|>done"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "no args needed\n\n"
        assert content == f"done\n{block}"

    def test_whitespace_only_remainder_leaves_no_reasoning(self, parser):
        """Only whitespace surviving promotion leaves the channel absent."""
        model_output = f"\n{self.TOOL_BLOCK}\n<|think:end|>done"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning is None
        assert content == f"done\n{self.TOOL_BLOCK}"

    def test_mere_sentinel_mention_not_promoted(self, parser):
        """Mentioning a sentinel promotes nothing and does not restrip reasoning."""
        thinking = "Should I emit <|tool_call:start|> here? No.\n"
        model_output = f"{thinking}<|think:end|>Answer"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == thinking
        assert content == "Answer"

    def test_abandoned_call_not_promoted(self, parser):
        """A call left unclosed inside think fails the grammar and stays put."""
        thinking = (
            "let me call\n<|tool_call:start|>foo\n"
            "<|tool_arg:start|>a<|tool_arg:value|>1\n"
        )
        model_output = f"{thinking}<|think:end|>after"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == thinking
        assert content == "after"

    def test_truncated_mid_think_regression(self, parser):
        """Truncation mid-think still promotes a complete embedded block."""
        model_output = f"checking weather\n{self.TOOL_BLOCK}"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "checking weather\n"
        assert content == self.TOOL_BLOCK

    def test_truncated_mid_think_without_tool_block(self, parser):
        model_output = "checking weather in Se"
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == model_output
        assert content is None

    def test_promoted_blocks_parse_via_tool_parser(self, parser, mock_tokenizer):
        """The promoted content channel parses through ``SolarOpen2ToolParser``."""
        model_output = (
            f"Need the weather.\n{self.TOOL_BLOCK}\n<|think:end|>Checking now."
        )
        reasoning, content = run_reasoning_extraction(
            parser, [model_output], streaming=False
        )
        assert reasoning == "Need the weather.\n\n"

        tool_parser = SolarOpen2ToolParser(mock_tokenizer)
        info = tool_parser.extract_tool_calls(
            content, request=SimpleNamespace(tools=None)
        )
        assert info.tools_called
        assert [tc.function.name for tc in info.tool_calls] == ["get_weather"]
        assert json.loads(info.tool_calls[0].function.arguments) == {"city": "Seoul"}
        assert info.content == "Checking now.\n"


_ARG = "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"

# Reasoning-channel texts covering every shape the two patterns could disagree
# on: well-formed calls, bodies the argument grammar does not describe, values
# quoting the sentinels' shared prefix, each dropped sentinel, and text that is
# no call at all.
_PROMOTION_FIXTURES = {
    "no_sentinel": "just thinking about the weather\n",
    "mention_only": "Should I emit <|tool_call:start|> here? No.\n",
    "stray_end_sentinel": "<|tool_call:end|>\nand then some thinking\n",
    "single_call": (
        f"I should check.\n<|tool_call:start|>get_weather\n{_ARG}<|tool_call:end|>"
    ),
    "two_calls": (
        f"first\n<|tool_call:start|>get_weather\n{_ARG}<|tool_call:end|>\n"
        f"then\n<|tool_call:start|>get_time\n"
        f"<|tool_arg:start|>tz<|tool_arg:value|>KST<|tool_arg:end|>\n<|tool_call:end|>"
    ),
    "no_args": "<|tool_call:start|>refresh\n<|tool_call:end|>",
    # ``<|tool_call:start|>`` followed straight by a newline names an empty
    # function, which is what the streaming state machine reads too.
    "empty_function_name": f"<|tool_call:start|>\n{_ARG}<|tool_call:end|>",
    # A body holding no argument group at all: the tool parser reads it as a
    # call with no arguments.
    "unstructured_body": "<|tool_call:start|>refresh\nnote to self\n<|tool_call:end|>",
    # The same, without the newline before the end sentinel. Requiring the
    # body to be argument groups each followed by an optional newline made
    # this unmatchable, so the call was dropped from the response entirely.
    "body_without_trailing_newline": (
        "<|tool_call:start|>refresh\nnote<|tool_call:end|>"
    ),
    # A value quoting the sentinels' shared ``<|tool_call:`` prefix is not a
    # call boundary.
    "literal_prefix_in_value": (
        "<|tool_call:start|>search\n<|tool_arg:start|>q<|tool_arg:value|>"
        "what does <|tool_call: mean<|tool_arg:end|>\n<|tool_call:end|>"
    ),
    # Dropped ``<|tool_arg:end|>``: still one call, with a value that runs to
    # the end sentinel.
    "dropped_arg_end": (
        "<|tool_call:start|>get_weather\n"
        "<|tool_arg:start|>city<|tool_arg:value|>Seoul\n<|tool_call:end|>"
    ),
    # Never closed: a generation cut off mid-call, or a repetition loop.
    "unterminated": f"planning\n<|tool_call:start|>write_file\n{_ARG * 8}",
    # Dropped ``<|tool_call:end|>`` with another call behind it.
    "dropped_call_end": (
        f"<|tool_call:start|>first\n{_ARG}"
        f"<|tool_call:start|>second\n{_ARG}<|tool_call:end|>"
    ),
    # A name line running into a call boundary before its newline.
    "boundary_before_name": (
        "<|tool_call:start|>oops<|tool_call:end|>\n"
        f"<|tool_call:start|>get_weather\n{_ARG}<|tool_call:end|>"
    ),
    # The same, with a later end sentinel the name must not reach across.
    "end_sentinel_on_the_name_line": (
        "<|tool_call:start|>oops<|tool_call:end|>\nbody\n<|tool_call:end|>"
    ),
}


class TestEmbeddedToolCallPatternAgreement:
    """``embedded_tool_call_regex`` decides what leaves the reasoning channel
    for ``SolarOpen2ToolParser`` to parse, so the two patterns must not drift.

    It is the ``<|tool_call:end|>``-terminated subset of
    ``SolarOpen2ToolParser.tool_call_pattern``. The tool parser also ends a
    call at the next ``<|tool_call:start|>``, so a dropped end sentinel still
    yields a call there — but that recovery reads the *following* sentinel,
    and promotion excises blocks one at a time and rejoins them, so it cannot
    keep the two adjacent.
    """

    @pytest.fixture
    def tool_parser(self, mock_tokenizer):
        return SolarOpen2ToolParser(mock_tokenizer)

    @staticmethod
    def _blocks(parser, text: str) -> list[str]:
        return [m.group(0) for m in parser.embedded_tool_call_regex.finditer(text)]

    @pytest.mark.parametrize("fixture", sorted(_PROMOTION_FIXTURES))
    def test_promotes_exactly_the_terminated_tool_parser_calls(
        self, parser, tool_parser, fixture
    ):
        """Agreement pin, block for block: a future divergence in either
        pattern changes which blocks get promoted, and fails here."""
        text = _PROMOTION_FIXTURES[fixture]
        promoted = [m.span() for m in parser.embedded_tool_call_regex.finditer(text)]
        terminated = [
            m.span()
            for m in tool_parser.tool_call_pattern.finditer(text)
            if m.group(0).endswith(tool_parser.TOOL_CALL_END)
        ]
        assert promoted == terminated

    @pytest.mark.parametrize("fixture", sorted(_PROMOTION_FIXTURES))
    def test_each_promoted_block_is_one_call_on_its_own(
        self, parser, tool_parser, fixture
    ):
        """Promotion lifts each block out of its surroundings, so each one
        must parse as exactly one call in isolation — not rely on the text
        that happened to follow it."""
        request = SimpleNamespace(tools=None)
        for block in self._blocks(parser, _PROMOTION_FIXTURES[fixture]):
            assert tool_parser.tool_call_pattern.fullmatch(block) is not None
            info = tool_parser.extract_tool_calls(block, request)
            assert info.tools_called
            assert len(info.tool_calls) == 1
            assert info.content is None

    @pytest.mark.parametrize("fixture", sorted(_PROMOTION_FIXTURES))
    def test_promoted_content_parses_as_the_promoted_blocks(
        self, parser, tool_parser, fixture
    ):
        """End to end: the content channel ``extract_reasoning`` builds must
        yield one call per promoted block, in order."""
        text = _PROMOTION_FIXTURES[fixture]
        request = SimpleNamespace(tools=None)
        expected = [
            tool_parser.extract_tool_calls(block, request).tool_calls[0].function.name
            for block in self._blocks(parser, text)
        ]
        _, content = run_reasoning_extraction(
            parser, [f"{text}<|think:end|>"], streaming=False
        )
        if not expected:
            assert content is None
            return
        info = tool_parser.extract_tool_calls(content, request)
        assert [tc.function.name for tc in info.tool_calls] == expected

    def test_unterminated_call_stays_in_reasoning(self, parser):
        """A call the model never closed is not promoted: the tool parser
        recovers one only from the sentinel that follows it, so the block
        stays where the run around it is still intact."""
        text = _PROMOTION_FIXTURES["unterminated"]
        reasoning, content = run_reasoning_extraction(
            parser, [f"{text}<|think:end|>"], streaming=False
        )
        assert reasoning == text
        assert content is None

    def test_call_missing_its_end_before_the_next_call_is_not_promoted(self, parser):
        """Only the closed call moves; the one whose end sentinel the model
        dropped keeps its place in reasoning."""
        reasoning, content = run_reasoning_extraction(
            parser,
            [f"{_PROMOTION_FIXTURES['dropped_call_end']}<|think:end|>"],
            streaming=False,
        )
        assert reasoning == f"<|tool_call:start|>first\n{_ARG}"
        assert content == f"<|tool_call:start|>second\n{_ARG}<|tool_call:end|>"

    def test_unstructured_call_body_is_promoted(self, parser, tool_parser):
        """A body holding no argument group is a call with no arguments for
        the tool parser, so promotion must move it rather than leave it in
        reasoning for the response to drop. The fixture also omits the
        newline before the end sentinel, which the argument-group body made
        unmatchable."""
        block = _PROMOTION_FIXTURES["body_without_trailing_newline"]
        _, content = run_reasoning_extraction(
            parser, [f"hmm\n{block}<|think:end|>"], streaming=False
        )
        assert content == block
        info = tool_parser.extract_tool_calls(content, SimpleNamespace(tools=None))
        assert [tc.function.name for tc in info.tool_calls] == ["refresh"]
        assert json.loads(info.tool_calls[0].function.arguments) == {}

    def test_literal_call_prefix_in_a_value_is_promoted(self, parser, tool_parser):
        """Tempering is on the full sentinels, so a value quoting their
        shared ``<|tool_call:`` prefix is not read as a call boundary — the
        same allowance the tool parser and the streaming path make."""
        block = _PROMOTION_FIXTURES["literal_prefix_in_value"]
        _, content = run_reasoning_extraction(
            parser, [f"quoting markup\n{block}<|think:end|>"], streaming=False
        )
        assert content == block
        info = tool_parser.extract_tool_calls(content, SimpleNamespace(tools=None))
        assert json.loads(info.tool_calls[0].function.arguments) == {
            "q": "what does <|tool_call: mean"
        }

    def test_a_promoted_block_cannot_cross_a_call_boundary(self, parser):
        """The body is tempered on the full call sentinels, so a match can
        never span two calls. That is also what bounds each match attempt:
        an unterminated call is rejected in one pass over its own bytes
        rather than one pass per argument group it holds.
        """
        blocks = self._blocks(parser, _PROMOTION_FIXTURES["dropped_call_end"])
        assert len(blocks) == 1
        assert blocks[0].count("<|tool_call:start|>") == 1

    def test_the_function_name_is_not_read_across_a_call_boundary(self, parser):
        """The name is the remainder of the start sentinel's line, and the
        tempering keeps it inside its own block. Bounding it to that line is
        additionally what keeps the scan linear, but laziness makes the two
        forms behaviourally identical, so only the boundary is pinned here.
        """
        assert (
            parser.embedded_tool_call_regex.fullmatch(
                _PROMOTION_FIXTURES["empty_function_name"]
            )
            is not None
        )
        # A start sentinel whose line runs into a call boundary can never
        # become a call, and the name is not read across that boundary — a
        # later end sentinel does not rescue it.
        assert (
            parser.embedded_tool_call_regex.search(
                _PROMOTION_FIXTURES["end_sentinel_on_the_name_line"]
            )
            is None
        )


class TestDelimiterProperties:
    """Delimiters must be exposed so ``ReasoningConfig`` can resolve token ids."""

    def test_delimiter_strings(self, parser):
        assert parser.reasoning_start_str == "<|think:start|>"
        assert parser.reasoning_end_str == "<|think:end|>"

    def test_mock_tokenizer_encodes_delimiters_atomically(self, parser, mock_tokenizer):
        """Precondition for the token-id tests: one id per delimiter in the mock."""
        for text in (parser.reasoning_start_str, parser.reasoning_end_str):
            ids = mock_tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) == 1, (text, ids)

    def test_delimiter_ids_resolved_from_vocab(self, parser, mock_tokenizer):
        """The ids the parser scans for come from ``get_vocab()`` and match encoding."""
        vocab = mock_tokenizer.get_vocab()
        for text, token_id in (
            (parser.reasoning_start_str, parser.think_start_token_id),
            (parser.reasoning_end_str, parser.think_end_token_id),
        ):
            assert text in vocab, f"{text} is not listed in get_vocab()"
            assert token_id == vocab[text]
            assert mock_tokenizer.encode(text, add_special_tokens=False) == [token_id]


class TestIsReasoningEnd:
    """``is_reasoning_end`` over token ids."""

    def test_reasoning_ended(self, parser, mock_tokenizer):
        text = "some reasoning<|think:end|>content"
        token_ids = mock_tokenizer.encode(text, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is True

    def test_reasoning_not_ended(self, parser, mock_tokenizer):
        text = "still reasoning"
        token_ids = mock_tokenizer.encode(text, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is False

    def test_empty_ids(self, parser):
        assert parser.is_reasoning_end([]) is False

    def test_multi_turn_prompt_high_effort_not_ended(self, parser, mock_tokenizer):
        """Only the last think block counts: a prior closed pair is not an end."""
        prompt = (
            "<|im:start|>user<|im:content|>q1<|im:end|>\n"
            "<|im:start|>assistant<|im:content|>"
            "<|think:start|><|think:end|>a1<|im:end|>\n"
            "<|im:start|>user<|im:content|>q2<|im:end|>\n"
            "<|im:start|>assistant<|im:content|><|think:start|>"
        )
        token_ids = mock_tokenizer.encode(prompt, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is False

    def test_multi_turn_prompt_low_effort_ended(self, parser, mock_tokenizer):
        """A prefilled empty pair as the last block means reasoning has ended."""
        prompt = (
            "<|im:start|>user<|im:content|>q1<|im:end|>\n"
            "<|im:start|>assistant<|im:content|>"
            "<|think:start|><|think:end|>a1<|im:end|>\n"
            "<|im:start|>user<|im:content|>q2<|im:end|>\n"
            "<|im:start|>assistant<|im:content|><|think:start|><|think:end|>"
        )
        token_ids = mock_tokenizer.encode(prompt, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is True

    def test_single_turn_prompt_high_effort_not_ended(self, parser, mock_tokenizer):
        """A trailing open start tag with no end tag anywhere is not ended."""
        prompt = (
            "<|im:start|>user<|im:content|>q1<|im:end|>\n"
            "<|im:start|>assistant<|im:content|><|think:start|>"
        )
        token_ids = mock_tokenizer.encode(prompt, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is False

    def test_literal_start_tag_in_prior_text_anchors_on_last(
        self, parser, mock_tokenizer
    ):
        """A stray literal start tag in prior text must not move the anchor."""
        prompt = (
            "<|im:start|>assistant<|im:content|>"
            "<|think:start|>I'll mention <|think:start|> here<|think:end|>"
            "ok<|im:end|>\n"
            "<|im:start|>user<|im:content|>again?<|im:end|>\n"
            "<|im:start|>assistant<|im:content|><|think:start|>"
        )
        token_ids = mock_tokenizer.encode(prompt, add_special_tokens=False)
        assert parser.is_reasoning_end(token_ids) is False

    @pytest.mark.parametrize("hidden", ["<|think:start|>", "<|think:end|>"])
    def test_partially_exposed_vocab_falls_back_to_text(self, hidden):
        """A tokenizer exposing only one delimiter id must take the text path."""
        # The backwards id scan stops at the first delimiter it meets, so an
        # unresolved start id would let it run into a prior turn's end tag.
        half_vocab = MockPartialVocabTokenizer(hidden)
        parser = SolarOpen2ReasoningParser(half_vocab)
        prior_turn = "<|think:start|>a<|think:end|>b"
        open_block = half_vocab.encode(f"{prior_turn}<|think:start|>new")
        closed_block = half_vocab.encode(f"{prior_turn}<|think:start|>c<|think:end|>")
        assert parser.is_reasoning_end(open_block) is False
        assert parser.is_reasoning_end(closed_block) is True
        # The streaming override needs both ids too, so it must fall back to
        # the text path rather than decide from an end-token-free delta window.
        assert parser.is_reasoning_end_streaming(open_block, []) is False
        assert parser.is_reasoning_end_streaming(closed_block, []) is True
        # Raw model output has no start delimiter — the template prefills it
        # into the prompt — so the text path degrades to plain containment.
        assert parser.is_reasoning_end(half_vocab.encode("reasoning text")) is False
        assert (
            parser.is_reasoning_end(half_vocab.encode("reasoning<|think:end|>answer"))
            is True
        )

    @pytest.mark.parametrize("in_content", [False, True])
    def test_is_reasoning_end_streaming_follows_stream_state(
        self, parser, mock_tokenizer, in_content
    ):
        """While a stream is active the text-level state answers, not the delta."""
        parser._stream_active = True
        parser._stream_in_content = in_content
        ids = mock_tokenizer.encode("thinking hard", add_special_tokens=False)
        end_id = mock_tokenizer.encode("<|think:end|>", add_special_tokens=False)[0]
        assert parser.is_reasoning_end_streaming(ids, []) is in_content
        assert parser.is_reasoning_end_streaming(ids + [end_id], [end_id]) is in_content

    def test_is_reasoning_end_streaming_only_inspects_delta(
        self, parser, mock_tokenizer
    ):
        """Outside a stream the answer comes from the step's own tokens only."""
        prefix = mock_tokenizer.encode("thinking hard", add_special_tokens=False)
        end_id = mock_tokenizer.encode("<|think:end|>", add_special_tokens=False)[0]
        assert parser.is_reasoning_end_streaming(prefix, prefix[-2:]) is False
        assert parser.is_reasoning_end_streaming(prefix + [end_id], [end_id]) is True
        # The marker being anywhere in the prefix is not enough: the step
        # that carries it is the boundary.
        assert (
            parser.is_reasoning_end_streaming(prefix + [end_id] + prefix, prefix)
            is False
        )


def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _run_reasoning_stream(parser, deltas):
    """Streaming driver tolerating a delta that carries ``reasoning`` and
    ``content`` at once, which this parser emits when the end tag and trailing
    content share a chunk and
    ``tests.reasoning.utils.StreamingReasoningReconstructor`` rejects.
    """
    reasoning: str | None = None
    content: str | None = None
    previous_text = ""
    for delta in deltas:
        current_text = previous_text + delta
        msg = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        previous_text = current_text
        if msg is None:
            continue
        if msg.reasoning is not None:
            reasoning = (reasoning or "") + msg.reasoning
        if msg.content is not None:
            content = (content or "") + msg.content
    return reasoning, content


class TestExtractReasoningStreaming:
    """Streaming reasoning extraction over chunked output."""

    _TOOL_BLOCK = TestEmbeddedToolCallPromotion.TOOL_BLOCK

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 100])
    def test_standard_reasoning_and_content(self, parser, chunk_size):
        model_output = "I need to analyze<|think:end|>The answer is 42"
        deltas = _split_into_chunks(model_output, chunk_size)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        assert reasoning == "I need to analyze"
        assert content == "The answer is 42"

    def test_multiline_reasoning(self, parser):
        model_output = "Step 1: analyze\nStep 2: solve\n<|think:end|>Final answer"
        deltas = _split_into_chunks(model_output, 3)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        assert reasoning == "Step 1: analyze\nStep 2: solve\n"
        assert content == "Final answer"

    def test_empty_reasoning(self, parser):
        """Output opening on the end tag streams content only."""
        model_output = "<|think:end|>Direct answer"
        deltas = _split_into_chunks(model_output, 2)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        # Empty reasoning surfaces as None or "": only the content matters.
        assert reasoning in (None, "")
        assert content == "Direct answer"

    def test_reasoning_with_empty_content(self, parser):
        """Reasoning followed by nothing leaves content None."""
        model_output = "My reasoning<|think:end|>"
        deltas = _split_into_chunks(model_output, 4)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        assert reasoning == "My reasoning"
        assert content is None

    # Low-effort streaming has no end tag at all, and the streaming API does
    # not expose the request, so "low effort, all content" is indistinguishable
    # from "high effort, still thinking". The bias is reasoning-first, as in
    # every other ``<think>``-style parser here; the batch path is pinned by
    # ``test_pure_content_no_tags`` above.

    def test_sentinel_split_across_chunks(self, parser):
        """A sentinel split mid-token is buffered, not emitted as reasoning."""
        prefix_end = "<|think:"
        tail_end = "end|>"
        chunks = [
            "reasoning body ",
            prefix_end,
            tail_end + "after",
        ]
        reasoning, content = _run_reasoning_stream(parser, chunks)
        assert reasoning == "reasoning body "
        assert content == "after"

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 100])
    def test_embedded_tool_call_implicit_reasoning_end(self, parser, chunk_size):
        """A tool call inside think ends reasoning implicitly (Qwen3 convention)."""
        block = (
            "<|tool_call:start|>get_weather\n"
            "<|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>\n"
            "<|tool_call:end|>"
        )
        model_output = f"need the weather\n{block}"
        deltas = _split_into_chunks(model_output, chunk_size)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        assert reasoning == "need the weather\n"
        assert content == block

    def test_tool_sentinel_split_across_chunks(self, parser):
        """A tool sentinel split mid-token still triggers the implicit end."""
        chunks = ["thinking ", "<|tool_c", "all:start|>f\n<|tool_call:end|>"]
        reasoning, content = _run_reasoning_stream(parser, chunks)
        assert reasoning == "thinking "
        assert content == "<|tool_call:start|>f\n<|tool_call:end|>"

    def test_think_end_before_tool_call_keeps_normal_transition(self, parser):
        """A tool call after the end tag is plain content."""
        model_output = "plan<|think:end|>text <|tool_call:start|>f\n<|tool_call:end|>"
        deltas = _split_into_chunks(model_output, 5)
        reasoning, content = _run_reasoning_stream(parser, deltas)
        assert reasoning == "plan"
        assert content == "text <|tool_call:start|>f\n<|tool_call:end|>"

    def test_is_reasoning_end_defers_during_active_stream(self, parser, mock_tokenizer):
        """The ended signal waits until the end tag is observed textually."""
        # The server stops routing deltas through this parser once the flag
        # flips, so flipping mid-holdback would leak the held-back end-tag
        # bytes into the content channel as a spurious delta.
        end_text = "<|think:end|>"

        # Mid-holdback the flag stays False even though the token list already
        # decodes to text containing the end tag.
        parser.extract_reasoning_streaming(
            previous_text="",
            current_text="reasoning <|th",
            delta_text="reasoning <|th",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        end_token_ids = mock_tokenizer.encode(end_text, add_special_tokens=False)
        assert end_text in mock_tokenizer.decode(end_token_ids)
        assert parser.is_reasoning_end(end_token_ids) is False

        # Once the tail arrives textually the parser transitions.
        parser.extract_reasoning_streaming(
            previous_text="reasoning <|th",
            current_text="reasoning <|think:end|>content",
            delta_text="ink:end|>content",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        assert parser.is_reasoning_end(end_token_ids) is True

    def test_stream_state_reset_between_streams(self, parser):
        """Two streams on one parser instance must not share buffered state."""
        first = "A<|think:end|>alpha"
        second = "B<|think:end|>beta"
        r1, c1 = _run_reasoning_stream(parser, _split_into_chunks(first, 3))
        r2, c2 = _run_reasoning_stream(parser, _split_into_chunks(second, 3))
        assert (r1, c1) == ("A", "alpha")
        assert (r2, c2) == ("B", "beta")

    @pytest.mark.parametrize("chunk_size", [1, 1000])
    @pytest.mark.parametrize(
        "model_output",
        [
            "I need to analyze<|think:end|>The answer is 42",
            "My reasoning<|think:end|>",
            "<|think:end|>Direct answer",
            "Step 1: analyze\nStep 2: solve\n<|think:end|>Final answer",
            # Truncated mid-think, with and without an embedded tool call.
            "Still thinking...",
            f"checking weather\n{_TOOL_BLOCK}",
            # Tool call after the end tag: plain content on both paths.
            "plan<|think:end|>text <|tool_call:start|>f\n<|tool_call:end|>",
        ],
    )
    def test_streaming_parity_with_non_stream(self, parser, model_output, chunk_size):
        """Streaming matches the batch parser on every shape where both agree."""
        ns_r, ns_c = run_reasoning_extraction(parser, [model_output], streaming=False)
        deltas = _split_into_chunks(model_output, chunk_size)
        s_r, s_c = _run_reasoning_stream(parser, deltas)
        # Batch returns None for empty reasoning; streaming may flush "".
        assert (s_r or None) == ns_r
        assert s_c == ns_c

    def test_trailing_partial_sentinel_held_back_at_end_of_stream(self, parser):
        """Documented divergence: a trailing ``<`` stays held back at end of stream."""
        model_output = "thinking <"
        ns_r, ns_c = run_reasoning_extraction(parser, [model_output], streaming=False)
        s_r, s_c = _run_reasoning_stream(parser, _split_into_chunks(model_output, 1))
        assert (ns_r, ns_c) == ("thinking <", None)
        assert (s_r, s_c) == ("thinking ", None)

    def test_embedded_tool_call_ordering_diverges_from_non_stream(self, parser):
        """Documented divergence: streaming cannot reorder the promoted block
        after the answer, and hands the markup after the implicit end to the
        content channel verbatim for ``SolarOpen2ToolParser`` to own.
        """
        model_output = f"thinking\n{self._TOOL_BLOCK}\n<|think:end|>The answer is 42"
        ns_r, ns_c = run_reasoning_extraction(parser, [model_output], streaming=False)
        s_r, s_c = _run_reasoning_stream(parser, _split_into_chunks(model_output, 1))
        assert (ns_r, ns_c) == ("thinking\n\n", f"The answer is 42\n{self._TOOL_BLOCK}")
        assert (s_r, s_c) == (
            "thinking\n",
            f"{self._TOOL_BLOCK}\n<|think:end|>The answer is 42",
        )
