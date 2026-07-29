# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named tool choice for Kimi K3: allowed when the XTML structural tag is
attached (strict tool calling), rejected otherwise."""

import pytest

from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.exceptions import VLLMValidationError
from vllm.parser.kimi_k3 import KimiK3Parser
from vllm.sampling_params import StructuredOutputsParams

pytestmark = pytest.mark.skip_global_cleanup


class _DummyTokenizer:
    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _request(with_tag: bool) -> ChatCompletionRequest:
    req = ChatCompletionRequest(
        model="k3",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    if with_tag:
        req.structured_outputs = StructuredOutputsParams(
            structural_tag='{"type": "structural_tag", "format": {}}'
        )
    return req


def _parser():
    return KimiK3Parser(_DummyTokenizer(), chat_template_kwargs={"thinking": False})


def test_named_choice_allowed_with_structural_tag():
    req = _parser().adjust_request(_request(with_tag=True))
    assert req.skip_special_tokens is False


def test_named_choice_rejected_without_structural_tag(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", False)
    with pytest.raises(VLLMValidationError):
        _parser().adjust_request(_request(with_tag=False))
