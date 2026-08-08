# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for presence_penalty / frequency_penalty resolution from
default_sampling_params in ChatCompletionRequest and CompletionRequest.

Regression test for https://github.com/vllm-project/vllm/issues/50767:
these two penalties defaulted to 0.0 (not None) and to_sampling_params()
forwarded them straight through, so server-side defaults coming from
--override-generation-config / generation_config.json were silently
discarded on the /v1/chat/completions and /v1/completions endpoints.
"""

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)

_DEFAULTS = {"presence_penalty": 1.5, "frequency_penalty": 0.5}


def _chat(**kwargs):
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )


class TestChatCompletionPenaltyDefaults:
    def test_defaults_applied_when_client_omits(self):
        """Server-default penalties are applied when the client sends none."""
        sp = _chat().to_sampling_params(100, _DEFAULTS)
        assert sp.presence_penalty == 1.5
        assert sp.frequency_penalty == 0.5

    def test_client_value_overrides_default(self):
        """An explicit client penalty wins over the server default."""
        sp = _chat(presence_penalty=0.2).to_sampling_params(100, _DEFAULTS)
        assert sp.presence_penalty == 0.2
        assert sp.frequency_penalty == 0.5

    def test_falls_back_to_zero_without_default(self):
        """Without a server default the neutral 0.0 is used (no regression)."""
        sp = _chat().to_sampling_params(100, {})
        assert sp.presence_penalty == 0.0
        assert sp.frequency_penalty == 0.0


class TestCompletionPenaltyDefaults:
    def test_defaults_applied_when_client_omits(self):
        sp = CompletionRequest(model="test-model", prompt="hi").to_sampling_params(
            16, _DEFAULTS
        )
        assert sp.presence_penalty == 1.5
        assert sp.frequency_penalty == 0.5

    def test_client_value_overrides_default(self):
        sp = CompletionRequest(
            model="test-model", prompt="hi", frequency_penalty=0.9
        ).to_sampling_params(16, _DEFAULTS)
        assert sp.presence_penalty == 1.5
        assert sp.frequency_penalty == 0.9
