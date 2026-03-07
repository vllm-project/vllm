# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for chat template availability checking.

This module tests the _check_chat_template_available() function which
determines whether the /v1/chat/completions endpoint should be enabled
based on chat template availability.
"""

from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.openai.generate.api_router import _check_chat_template_available


class MockModelConfig:
    """Mock model configuration for testing."""

    def __init__(self, model="gpt2"):
        self.model = model
        self.trust_remote_code = False
        self.hf_config = MagicMock()


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, has_template=False):
        self.chat_template = (
            "{% for msg in messages %}{{ msg.content }}{% endfor %}"
            if has_template
            else None
        )
        self.name_or_path = "test-tokenizer"


class MockRenderer:
    """Mock renderer for testing."""

    def __init__(self, has_template=False):
        self.tokenizer = MockTokenizer(has_template)


class MockEngineClient:
    """Mock engine client for testing."""

    def __init__(self, model="gpt2", has_template=False):
        self.model_config = MockModelConfig(model)
        self.renderer = MockRenderer(has_template)


@pytest.fixture
def base_model_engine():
    """Fixture for base model (GPT2) without chat template."""
    return MockEngineClient(model="gpt2", has_template=False)


@pytest.fixture
def instruct_model_engine():
    """Fixture for instruct model with chat template."""
    return MockEngineClient(model="meta-llama/Llama-3.2-1B-Instruct", has_template=True)


def test_check_chat_template_available_with_user_template(base_model_engine):
    """Test that user-provided template always returns True."""
    user_template = "{% for message in messages %}{{ message.content }}{% endfor %}"

    with patch("vllm.renderers.hf.resolve_chat_template") as mock_resolve:
        # User template is priority #1, so it always succeeds
        mock_resolve.return_value = user_template

        result = _check_chat_template_available(base_model_engine, user_template)

    assert result is True
    # Should call resolve_chat_template with user's template
    mock_resolve.assert_called_once()


def test_check_chat_template_available_with_model_template(instruct_model_engine):
    """Test that model with built-in template returns True."""
    with patch("vllm.renderers.hf.resolve_chat_template") as mock_resolve:
        # Simulate model having a chat template
        mock_resolve.return_value = "{% for msg in messages %}..."

        result = _check_chat_template_available(instruct_model_engine, None)

    assert result is True


def test_check_chat_template_unavailable(base_model_engine):
    """Test that base model without template returns False."""
    with patch("vllm.renderers.hf.resolve_chat_template") as mock_resolve:
        # Simulate no template found (base model)
        mock_resolve.return_value = None

        result = _check_chat_template_available(base_model_engine, None)

    assert result is False


def test_check_chat_template_handles_errors(base_model_engine):
    """Test that exceptions during resolution return False."""
    with patch("vllm.renderers.hf.resolve_chat_template") as mock_resolve:
        mock_resolve.side_effect = RuntimeError("Test error")

        result = _check_chat_template_available(base_model_engine, None)

    assert result is False


def test_check_chat_template_tokenizer_not_initialized(base_model_engine):
    """Test that skip_tokenizer_init=True disables chat endpoint."""
    base_model_engine.renderer.tokenizer = None  # Simulate skip_tokenizer_init=True

    # Should return False even with user-provided template
    # because tokenizer is needed to convert formatted text to token IDs
    result = _check_chat_template_available(base_model_engine, None)
    assert result is False

    # Even with user template, still need tokenizer for tokenization
    user_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    result = _check_chat_template_available(base_model_engine, user_template)
    assert result is False


@pytest.mark.parametrize(
    "model_name,has_template,user_template,expected",
    [
        # Base model, no user template -> False
        ("gpt2", False, None, False),
        # Instruct model, no user template -> True
        ("meta-llama/Llama-3.2-1B-Instruct", True, None, True),
        # Base model, user provides template -> True
        ("gpt2", False, "{% for msg in messages %}...", True),
        # Instruct model, user overrides template -> True
        ("meta-llama/Llama-3.2-1B-Instruct", True, "{% custom %}", True),
    ],
)
def test_check_chat_template_various_scenarios(
    model_name, has_template, user_template, expected
):
    """Test various combinations of model types and user templates."""
    engine = MockEngineClient(model=model_name, has_template=has_template)

    with patch("vllm.renderers.hf.resolve_chat_template") as mock_resolve:
        # Simulate resolve_chat_template behavior
        if user_template is not None:
            # User template (priority #1)
            mock_resolve.return_value = user_template
        elif has_template:
            # Model has template (priority #2-4)
            mock_resolve.return_value = "{% model template %}"
        else:
            # No template available
            mock_resolve.return_value = None

        result = _check_chat_template_available(engine, user_template)

    assert result == expected
