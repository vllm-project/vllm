# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for max_whitespace_cnt parameter passing to xgrammar."""

from unittest.mock import Mock, patch

import pytest

from vllm.config import ModelConfig, SchedulerConfig, StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

pytestmark = pytest.mark.cpu_test


class TestMaxWhitespaceCnt:
    """Test that max_whitespace_cnt is passed correctly to xgrammar."""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create a mock VllmConfig with custom structured outputs config."""
        model_config = Mock(spec=ModelConfig)
        model_config.skip_tokenizer_init = True
        model_config.get_vocab_size = Mock(return_value=50000)
        model_config.runner_type = "generate"
        model_config.tokenizer = "test-tokenizer"
        model_config.tokenizer_mode = "auto"
        model_config.trust_remote_code = False
        model_config.tokenizer_revision = None

        scheduler_config = Mock(spec=SchedulerConfig)
        scheduler_config.max_num_seqs = 128

        config = Mock(spec=VllmConfig)
        config.model_config = model_config
        config.scheduler_config = scheduler_config
        config.speculative_config = None
        return config

    def _make_backend(self, mock_vllm_config, max_whitespace_cnt):
        """Helper to create an XgrammarBackend with given max_whitespace_cnt."""
        mock_vllm_config.structured_outputs_config = StructuredOutputsConfig(
            backend="xgrammar",
            max_whitespace_cnt=max_whitespace_cnt,
        )
        return XgrammarBackend(
            vllm_config=mock_vllm_config,
            tokenizer=Mock(),
            vocab_size=50000,
        )

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_max_whitespace_cnt_default(
        self, mock_xgr, mock_vllm_config
    ):
        """Verify default max_whitespace_cnt=2 is passed to compile_json_schema."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        backend = self._make_backend(mock_vllm_config, max_whitespace_cnt=2)

        backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        kwargs = call.call_args.kwargs
        assert kwargs.get("max_whitespace_cnt") == 2
        assert kwargs.get("any_whitespace") is True

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_max_whitespace_cnt_one(
        self, mock_xgr, mock_vllm_config
    ):
        """Verify max_whitespace_cnt=1 is passed correctly."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        backend = self._make_backend(mock_vllm_config, max_whitespace_cnt=1)

        backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        assert call.call_args.kwargs.get("max_whitespace_cnt") == 1

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_max_whitespace_cnt_none(
        self, mock_xgr, mock_vllm_config
    ):
        """Verify max_whitespace_cnt=None (unbounded) is passed correctly."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        backend = self._make_backend(mock_vllm_config, max_whitespace_cnt=None)

        backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        assert call.call_args.kwargs.get("max_whitespace_cnt") is None

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_max_whitespace_cnt_json_object(
        self, mock_xgr, mock_vllm_config
    ):
        """Verify max_whitespace_cnt is passed for JSON object (no schema) case."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        backend = self._make_backend(mock_vllm_config, max_whitespace_cnt=2)

        backend.compile_grammar(StructuredOutputOptions.JSON_OBJECT, "")

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        assert call.call_args.kwargs.get("max_whitespace_cnt") == 2

    @patch("vllm.v1.structured_output.backend_xgrammar.xgr")
    def test_disable_any_whitespace_overrides(
        self, mock_xgr, mock_vllm_config
    ):
        """Verify disable_any_whitespace=True sets any_whitespace=False."""
        mock_compiled = Mock()
        mock_xgr.GrammarCompiler.return_value.compile_json_schema.return_value = (
            mock_compiled
        )
        mock_xgr.TokenizerInfo.from_huggingface.return_value = Mock()

        mock_vllm_config.structured_outputs_config = StructuredOutputsConfig(
            backend="xgrammar",
            disable_any_whitespace=True,
            max_whitespace_cnt=2,
        )
        backend = XgrammarBackend(
            vllm_config=mock_vllm_config,
            tokenizer=Mock(),
            vocab_size=50000,
        )

        backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')

        call = mock_xgr.GrammarCompiler.return_value.compile_json_schema
        call.assert_called_once()
        assert call.call_args.kwargs.get("any_whitespace") is False
        assert call.call_args.kwargs.get("max_whitespace_cnt") == 2
