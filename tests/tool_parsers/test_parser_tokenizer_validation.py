# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate that ParserManager.get_parser() catches incompatible
tool-parser / tokenizer combinations at startup rather than failing
per-request with HTTP 500.

See: https://github.com/vllm-project/vllm/issues/59087
"""

from unittest.mock import MagicMock

import pytest

from vllm.parser.parser_manager import ParserManager

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def bare_tokenizer():
    """A minimal mock tokenizer with no special tool tokens."""
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {"hello": 0, "world": 1, "test": 2}
    return tokenizer


class TestIncompatibleParserTokenizer:
    """Parsers that require specific tokens should fail at startup."""

    def test_mistral_parser_incompatible_tokenizer(self, bare_tokenizer):
        """gpt2-like vocab has no [TOOL_CALLS] token."""
        with pytest.raises(TypeError, match="incompatible"):
            ParserManager.get_parser(
                tool_parser_name="mistral",
                enable_auto_tools=True,
                tokenizer=bare_tokenizer,
            )

    def test_jamba_parser_incompatible_tokenizer(self, bare_tokenizer):
        """Vocab has no <tool_calls> token."""
        with pytest.raises(TypeError, match="incompatible"):
            ParserManager.get_parser(
                tool_parser_name="jamba",
                enable_auto_tools=True,
                tokenizer=bare_tokenizer,
            )

    def test_llama3_json_parser_incompatible_tokenizer(self, bare_tokenizer):
        """Vocab has no <|python_tag|> token."""
        with pytest.raises(TypeError, match="incompatible"):
            ParserManager.get_parser(
                tool_parser_name="llama3_json",
                enable_auto_tools=True,
                tokenizer=bare_tokenizer,
            )

    def test_lfm2_parser_incompatible_tokenizer(self, bare_tokenizer):
        """Vocab has no <|tool_call_start|> token."""
        with pytest.raises(TypeError, match="incompatible"):
            ParserManager.get_parser(
                tool_parser_name="lfm2",
                enable_auto_tools=True,
                tokenizer=bare_tokenizer,
            )


class TestCompatibleParserTokenizer:
    """Parsers that degrade gracefully should succeed."""

    def test_hermes_parser_compatible(self, bare_tokenizer):
        """Hermes degrades gracefully — no required-token check."""
        result = ParserManager.get_parser(
            tool_parser_name="hermes",
            enable_auto_tools=True,
            tokenizer=bare_tokenizer,
        )
        assert result is not None

    def test_no_tool_parser(self, bare_tokenizer):
        """No tool parser → no validation needed."""
        result = ParserManager.get_parser(
            tool_parser_name=None,
            enable_auto_tools=False,
            tokenizer=bare_tokenizer,
        )
        assert result is None

    def test_no_tokenizer_skips_validation(self):
        """When tokenizer is None, validation is skipped."""
        result = ParserManager.get_parser(
            tool_parser_name="mistral",
            enable_auto_tools=True,
            tokenizer=None,
        )
        assert result is not None
