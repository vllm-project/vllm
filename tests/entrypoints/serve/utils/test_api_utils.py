# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

import pytest

from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.serve.utils import api_utils
from vllm.entrypoints.serve.utils.api_utils import (
    _redact_sensitive_args,
    get_max_tokens,
    should_include_usage,
)


@pytest.mark.parametrize(
    ("stream_options", "expected"),
    [
        (None, (True, True)),
        (StreamOptions(include_usage=False), (True, True)),
        (
            StreamOptions(include_usage=False, continuous_usage_stats=False),
            (True, True),
        ),
        (
            StreamOptions(include_usage=True, continuous_usage_stats=False),
            (True, True),
        ),
    ],
)
def test_should_include_usage_force_enables_continuous_usage(stream_options, expected):
    assert should_include_usage(stream_options, True) == expected


class TestGetMaxTokens:
    """Tests for get_max_tokens() to ensure generation_config's max_tokens
    acts as a default when from model author, and as a ceiling when
    explicitly set by the user."""

    def test_default_sampling_params_used_when_no_request_max_tokens(self):
        """When user doesn't specify max_tokens, generation_config default
        should apply."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2048

    def test_request_max_tokens_not_capped_by_default_sampling_params(self):
        """When user specifies max_tokens in request, model author's
        generation_config max_tokens must NOT cap it (fixes #34005)."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 5000

    def test_override_max_tokens_caps_request(self):
        """When user explicitly sets max_tokens, it acts as a ceiling."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_override_max_tokens_used_as_default(self):
        """When no request max_tokens, override still applies as default."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_max_model_len_still_caps_output(self):
        """max_model_len - input_length is always the hard ceiling."""
        result = get_max_tokens(
            max_model_len=3000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2900  # 3000 - 100

    def test_request_max_tokens_smaller_than_default(self):
        """When user explicitly requests fewer tokens than gen_config default,
        that should be respected."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=512,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 512

    def test_input_length_exceeds_max_model_len(self):
        with pytest.raises(
            ValueError,
            match="Input length .* exceeds model's maximum context length .*",
        ):
            get_max_tokens(
                max_model_len=100,
                max_tokens=50,
                input_length=150,
                default_sampling_params={"max_tokens": 2048},
            )


class TestRedactSensitiveArgs:
    API_KEY = "sk-test-secret-12345"

    def test_redact_replaces_sensitive_values_only(self):
        args = {"api_key": self.API_KEY, "other": "visible"}
        redacted = _redact_sensitive_args(args)
        assert redacted == {"api_key": "***", "other": "visible"}
        # original dict must not be mutated
        assert args == {"api_key": self.API_KEY, "other": "visible"}

    def test_no_sensitive_fields_returns_original(self):
        args = {"model_tag": "org/model", "other": "visible"}
        assert _redact_sensitive_args(args) is args

    def test_api_key_not_in_log(self, monkeypatch, caplog):
        non_default = {
            "model_tag": "org/model",
            "default_chat_template_kwargs": {"enable_thinking": False},
            "api_key": self.API_KEY,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "qwen3_coder",
        }
        monkeypatch.setattr(api_utils, "get_non_default_args", lambda args: non_default)
        with caplog.at_level("INFO", logger="vllm.entrypoints.serve.utils.api_utils"):
            api_utils.log_non_default_args(args=Namespace())
        message = caplog.text
        assert self.API_KEY not in message
        assert "'api_key': '***'" in message
        # non-sensitive args are still logged
        assert "org/model" in message
        assert "qwen3_coder" in message
