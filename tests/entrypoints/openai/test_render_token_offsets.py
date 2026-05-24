# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for token offsets surfacing via render endpoints."""

from unittest.mock import Mock

from vllm.config import ModelConfig
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


class TestCompletionRequestField:
    def test_default_is_false(self):
        """return_token_offsets must default to False so existing
        callers see zero behavioral change."""
        req = CompletionRequest(model="m", prompt="hi")
        assert req.return_token_offsets is False

    def test_accepts_true(self):
        req = CompletionRequest(model="m", prompt="hi", return_token_offsets=True)
        assert req.return_token_offsets is True

    def test_none_coerces_to_false_in_tok_params(self):
        """JSON null must coerce to False when forwarded into TokenizeParams."""
        req = CompletionRequest(model="m", prompt="hi", return_token_offsets=None)
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False

    def test_build_tok_params_forwards_true(self):
        req = CompletionRequest(model="m", prompt="hi", return_token_offsets=True)
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is True

    def test_build_tok_params_default_is_false(self):
        req = CompletionRequest(model="m", prompt="hi")
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False
