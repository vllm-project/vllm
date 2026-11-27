# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

from vllm.entrypoints.common import startup
from vllm.entrypoints.common.startup import redact_sensitive_args


class TestRedactSensitiveArgs:
    API_KEY = "sk-test-secret-12345"

    def test_redact_replaces_sensitive_values_only(self):
        args = {
            "api_key": self.API_KEY,
            "hf_token": "hf_secret",
            "watermark_config": {"algorithm": "gumbel", "key": 42},
            "other": "visible",
        }
        redacted = redact_sensitive_args(args)
        assert redacted == {
            "api_key": "***",
            "hf_token": "***",
            "watermark_config": "***",
            "other": "visible",
        }
        # original dict must not be mutated
        assert args == {
            "api_key": self.API_KEY,
            "hf_token": "hf_secret",
            "watermark_config": {"algorithm": "gumbel", "key": 42},
            "other": "visible",
        }

    def test_no_sensitive_fields_returns_original(self):
        args = {"model_tag": "org/model", "other": "visible"}
        assert redact_sensitive_args(args) is args

    def test_api_key_not_in_log(self, monkeypatch, caplog):
        non_default = {
            "model_tag": "org/model",
            "default_chat_template_kwargs": {"enable_thinking": False},
            "api_key": self.API_KEY,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "qwen3_coder",
        }
        monkeypatch.setattr(startup, "get_non_default_args", lambda args: non_default)
        with caplog.at_level("INFO", logger="vllm.entrypoints.common.startup"):
            startup.log_non_default_args(args=Namespace())
        message = caplog.text
        assert self.API_KEY not in message
        assert "'api_key': '***'" in message
        # non-sensitive args are still logged
        assert "org/model" in message
        assert "qwen3_coder" in message

    def test_hf_token_not_in_log(self, monkeypatch, caplog):
        hf_token = "hf_TESTTOKENVALUE123"
        non_default = {
            "model_tag": "org/model",
            "hf_token": hf_token,
            "tool_call_parser": "qwen3_coder",
        }
        monkeypatch.setattr(startup, "get_non_default_args", lambda args: non_default)
        with caplog.at_level("INFO", logger="vllm.entrypoints.common.startup"):
            startup.log_non_default_args(args=Namespace())
        message = caplog.text
        assert hf_token not in message
        assert "'hf_token': '***'" in message
        assert "org/model" in message
