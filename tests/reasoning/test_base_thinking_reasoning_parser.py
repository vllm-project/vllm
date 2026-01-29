# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


# Create a concrete test implementation of BaseThinkingReasoningParser
class TestThinkingReasoningParser(BaseThinkingReasoningParser):
    """Test implementation of BaseThinkingReasoningParser."""

    @property
    def start_token(self) -> str:
        return "<test:think>"

    @property
    def end_token(self) -> str:
        return "</test:think>"


class TestThinkingReasoningParserAlt(BaseThinkingReasoningParser):
    """Alternative test implementation with different tokens."""

    @property
    def start_token(self) -> str:
        return "<alt:start>"

    @property
    def end_token(self) -> str:
        return "<alt:end>"


# Use a test model
REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="module")
def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)
    # Add custom test tokens
    test_tokens = ["<test:think>", "</test:think>", "<alt:start>", "<alt:end>"]
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in test_tokens if token not in existing_tokens]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
    return tokenizer


class TestBaseThinkingReasoningParserInit:
    """
    Test initialization and basic properties of
    BaseThinkingReasoningParser.
    """

    def test_successful_initialization(self, test_tokenizer):
        """Test successful initialization with valid tokens."""
        parser = TestThinkingReasoningParser(test_tokenizer)
        assert parser.start_token == "<test:think>"
        assert parser.end_token == "</test:think>"
        assert parser.start_token_id is not None
        assert parser.end_token_id is not None

    def test_initialization_with_missing_tokenizer(self):
        """Test that initialization fails without tokenizer."""
        with pytest.raises(ValueError, match="model tokenizer must be passed"):
            TestThinkingReasoningParser(None)

    def test_initialization_with_missing_tokens(self, test_tokenizer):
        """Test that initialization fails when tokens are not in vocabulary."""

        # Create a parser with tokens not in vocabulary
        class MissingTokenParser(BaseThinkingReasoningParser):
            @property
            def start_token(self) -> str:
                return "<missing:start>"

            @property
            def end_token(self) -> str:
                return "<missing:end>"

        with pytest.raises(
            RuntimeError, match="could not locate think start/end tokens"
        ):
            MissingTokenParser(test_tokenizer)

    def test_initialization_with_empty_tokens(self, test_tokenizer):
        """Test that initialization fails with empty token strings."""

        class EmptyTokenParser(BaseThinkingReasoningParser):
            @property
            def start_token(self) -> str:
                return ""

            @property
            def end_token(self) -> str:
                return ""

        with pytest.raises(
            ValueError, match="start_token and end_token must be defined"
        ):
            EmptyTokenParser(test_tokenizer)


class TestBaseThinkingReasoningParserMethods:
    """Test the methods of BaseThinkingReasoningParser."""

    def test_is_reasoning_end(self, test_tokenizer):
        """Test the is_reasoning_end method."""
        parser = TestThinkingReasoningParser(test_tokenizer)
        end_token_id = parser.end_token_id
        start_token_id = parser.start_token_id
        # Test with end token present
        assert parser.is_reasoning_end([1, 2, end_token_id, 4]) is True

        # Test without end token
        assert parser.is_reasoning_end([1, 2, 3, 4]) is False

        # Test with empty list
        assert parser.is_reasoning_end([]) is False

        # Test with interleaved thinking
        assert parser.is_reasoning_end([1, start_token_id, 2, end_token_id]) is True
        assert parser.is_reasoning_end([1, start_token_id, 2, 3]) is False
        assert (
            parser.is_reasoning_end(
                [1, start_token_id, 2, end_token_id, 2, 2, start_token_id]
            )
            is False
        )

    def test_is_reasoning_end_streaming(self, test_tokenizer):
        """Test the is_reasoning_end_streaming method."""
        parser = TestThinkingReasoningParser(test_tokenizer)
        end_token_id = parser.end_token_id
        start_token_id = parser.start_token_id

        assert (
            parser.is_reasoning_end_streaming([1, 2, end_token_id], [end_token_id])
            is True
        )
        assert parser.is_reasoning_end_streaming([1, 2, 3, 4], [4]) is False
        assert parser.is_reasoning_end_streaming([], []) is False
        assert (
            parser.is_reasoning_end_streaming(
                [1, start_token_id, 2, end_token_id], [end_token_id]
            )
            is True
        )
        assert (
            parser.is_reasoning_end_streaming([1, start_token_id, 2, 3], [3]) is False
        )
        assert (
            parser.is_reasoning_end_streaming(
                [1, start_token_id, 2, end_token_id, 2, start_token_id, 2],
                [2],
            )
            is False
        )
        assert (
            parser.is_reasoning_end_streaming(
                [1, start_token_id, 2, end_token_id, 2, 2], [2]
            )
            is False
        )

    def test_extract_content_ids(self, test_tokenizer):
        """Test the extract_content_ids method."""
        parser = TestThinkingReasoningParser(test_tokenizer)
        end_token_id = parser.end_token_id

        # Test with end token in the middle
        input_ids = [1, 2, end_token_id, 4, 5]
        content_ids = parser.extract_content_ids(input_ids)
        assert content_ids == [4, 5]

        # Test with end token at the end
        input_ids = [1, 2, 3, end_token_id]
        content_ids = parser.extract_content_ids(input_ids)
        assert content_ids == []

        # Test without end token
        input_ids = [1, 2, 3, 4]
        content_ids = parser.extract_content_ids(input_ids)
        assert content_ids == []

        # Test with end token as last element (should not extract)
        input_ids = [1, 2, 3, end_token_id]
        content_ids = parser.extract_content_ids(input_ids)
        assert content_ids == []


class TestBaseThinkingReasoningParserExtraction:
    """Test reasoning content extraction methods."""

    def test_extract_reasoning_with_both_tokens(self, test_tokenizer):
        """Test extraction when both start and end tokens are present."""
        parser = TestThinkingReasoningParser(test_tokenizer)
        request = ChatCompletionRequest(messages=[], model="test-model")

        model_output = "<test:think>This is reasoning</test:think>This is content"
        reasoning, content = parser.extract_reasoning(model_output, request)

        assert reasoning == "This is reasoning"
        assert content == "This is content"

    def test_extract_reasoning_only_end_token(self, test_tokenizer):
        """Test extraction when only end token is present."""
        parser = TestThinkingReasoningParser(test_tokenizer)
        request = ChatCompletionRequest(messages=[], model="test-model")

        model_output = "This is reasoning</test:think>This is content"
        reasoning, content = parser.extract_reasoning(model_output, request)

        assert reasoning == "This is reasoning"
        assert content == "This is content"

    def test_extract_reasoning_no_end_token(self, test_tokenizer):
        """Test extraction when no end token is present."""
        parser = TestThinkingReasoningParser(test_tokenizer)
        request = ChatCompletionRequest(messages=[], model="test-model")

        model_output = "This is just content"
        reasoning, content = parser.extract_reasoning(model_output, request)

        assert reasoning == "This is just content"
        assert content is None

    def test_extract_reasoning_empty_output(self, test_tokenizer):
        """Test extraction with empty output."""
        parser = TestThinkingReasoningParser(test_tokenizer)
        request = ChatCompletionRequest(messages=[], model="test-model")

        model_output = ""
        reasoning, content = parser.extract_reasoning(model_output, request)

        assert reasoning == ""
        assert content is None

    def test_extract_reasoning_only_tokens(self, test_tokenizer):
        """Test extraction with only tokens and no content."""
        parser = TestThinkingReasoningParser(test_tokenizer)
        request = ChatCompletionRequest(messages=[], model="test-model")

        model_output = "<test:think></test:think>"
        reasoning, content = parser.extract_reasoning(model_output, request)

        assert reasoning == ""
        assert content is None


class TestBaseThinkingReasoningParserStreaming:
    """Test streaming functionality of BaseThinkingReasoningParser."""

    @pytest.mark.parametrize("streaming", [True, False])
    def test_simple_reasoning_extraction(self, test_tokenizer, streaming):
        """
        Test basic reasoning extraction in both
        streaming and non-streaming modes.
        """
        parser = TestThinkingReasoningParser(test_tokenizer)

        model_output = [
            "<test:think>",
            "Some ",
            "reasoning ",
            "content",
            "</test:think>",
            "Final ",
            "answer",
        ]

        reasoning, content = run_reasoning_extraction(
            parser, model_output, streaming=streaming
        )

        assert reasoning == "Some reasoning content"
        assert content == "Final answer"

    def test_streaming_with_incremental_deltas(self, test_tokenizer):
        """Test streaming processing with small incremental deltas."""
        parser = TestThinkingReasoningParser(test_tokenizer)

        deltas = [
            "<test:think>",
            "Some ",
            "reasoning ",
            "content",
            "</test:think>",
            "Final ",
            "answer",
        ]

        reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)

        assert reasoning == "Some reasoning content"
        assert content == "Final answer"

    def test_streaming_with_start_token(self, test_tokenizer):
        """Test streaming with start token included."""
        parser = TestThinkingReasoningParser(test_tokenizer)

        deltas = [
            "<test:think>",
            "Some ",
            "reasoning",
            "</test:think>",
            "Answer",
        ]

        reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)

        assert reasoning == "Some reasoning"
        assert content == "Answer"

    def test_streaming_no_end_token(self, test_tokenizer):
        """Test streaming when no end token is encountered."""
        parser = TestThinkingReasoningParser(test_tokenizer)

        deltas = [
            "<test:think>",
            "Some ",
            "reasoning ",
            "without ",
            "end",
        ]

        reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)

        assert reasoning == "Some reasoning without end"
        assert content is None

    def test_streaming_only_end_token(self, test_tokenizer):
        """Test streaming when only end token appears."""
        parser = TestThinkingReasoningParser(test_tokenizer)

        deltas = [
            "<test:think>",
            "Reasoning ",
            "content",
            "</test:think>",
            "Final",
        ]

        reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)

        assert reasoning == "Reasoning content"
        assert content == "Final"


class TestBaseThinkingReasoningParserMultipleImplementations:
    """
    Test that multiple implementations of
    BaseThinkingReasoningParser work correctly.
    """

    def test_different_token_implementations(self, test_tokenizer):
        """
        Test that different implementations
        with different tokens work independently.
        """
        parser1 = TestThinkingReasoningParser(test_tokenizer)
        parser2 = TestThinkingReasoningParserAlt(test_tokenizer)

        # Test parser1
        model_output1 = "Reasoning1</test:think>Content1"
        reasoning1, content1 = run_reasoning_extraction(parser1, [model_output1])
        assert reasoning1 == "Reasoning1"
        assert content1 == "Content1"

        # Test parser2
        model_output2 = "Reasoning2<alt:end>Content2"
        reasoning2, content2 = run_reasoning_extraction(parser2, [model_output2])
        assert reasoning2 == "Reasoning2"
        assert content2 == "Content2"

        # Verify tokens are different
        assert parser1.start_token != parser2.start_token
        assert parser1.end_token != parser2.end_token
        assert parser1.start_token_id != parser2.start_token_id
        assert parser1.end_token_id != parser2.end_token_id


class TestBaseThinkingReasoningParserEdgeCases:
    """Test edge cases and error conditions."""

    def test_multiple_end_tokens(self, test_tokenizer):
        """Test behavior with multiple end tokens."""
        parser = TestThinkingReasoningParser(test_tokenizer)

        model_output = "First</test:think>Middle</test:think>Last"
        reasoning, content = run_reasoning_extraction(parser, [model_output])

        # Should stop at first end token
        assert reasoning == "First"
        assert content == "Middle</test:think>Last"

    def test_nested_tokens(self, test_tokenizer):
        """Test behavior with nested-like token patterns."""
        parser = TestThinkingReasoningParser(test_tokenizer)

        model_output = "<test:think>Outer<test:think>Inner</test:think>Content"
        reasoning, content = run_reasoning_extraction(parser, [model_output])

        # Should process normally, start from first start token
        assert reasoning == "Outer<test:think>Inner"
        assert content == "Content"

    def test_malformed_tokens(self, test_tokenizer):
        """Test behavior with malformed token-like strings."""
        parser = TestThinkingReasoningParser(test_tokenizer)

        model_output = "<test:thinking>Not a real token</test:thinking>Content"
        reasoning, content = run_reasoning_extraction(parser, [model_output])

        # Should treat as regular content since tokens don't match exactly
        assert reasoning == ("<test:thinking>Not a real token</test:thinking>Content")
        assert content is None
