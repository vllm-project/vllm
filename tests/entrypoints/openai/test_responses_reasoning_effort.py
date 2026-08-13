# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-effort handling on /v1/responses.

Two endpoints serve the same model with the same tiers, so they must accept the
same vocabulary. They did not: `ChatCompletionRequest.reasoning_effort` has
always allowed DeepSeek's `max`, while `ResponsesRequest.reasoning` took its
type from the OpenAI SDK, whose `ReasoningEffort` stops at `xhigh`. A request
for `max` was rejected by schema validation before reaching a model that ships
a prompt for exactly that tier.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokenizers.deepseek_v4_encoding import REASONING_EFFORT_PROMPTS

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

# Every spelling either endpoint should accept: the OpenAI vocabulary plus
# DeepSeek's `max`.
ACCEPTED_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


@pytest.mark.parametrize("effort", ACCEPTED_EFFORTS)
def test_responses_accepts_every_effort_chat_accepts(effort):
    """The two endpoints accept the same vocabulary."""
    ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=effort,
    )
    request = ResponsesRequest(
        model=MODEL, input="hi", reasoning={"effort": effort}
    )
    assert request.reasoning is not None
    assert request.reasoning.effort == effort


def test_responses_accepts_max_specifically():
    """`max` is the tier this widening exists for.

    Guarding it on its own so the intent survives a future edit to the
    parametrised list above.
    """
    request = ResponsesRequest(model=MODEL, input="hi", reasoning={"effort": "max"})
    assert request.reasoning.effort == "max"
    # And it is a real tier, not merely an accepted string.
    assert "max" in REASONING_EFFORT_PROMPTS


def test_responses_still_rejects_nonsense():
    """Widening the schema must not turn it into a free-text field."""
    with pytest.raises(ValidationError):
        ResponsesRequest(model=MODEL, input="hi", reasoning={"effort": "turbo"})


def test_effort_survives_into_chat_params():
    """The effort must reach the renderer, not be dropped in translation."""
    request = ResponsesRequest(model=MODEL, input="hi", reasoning={"effort": "max"})
    params = request.build_chat_params(
        default_template=None, default_template_content_format="string"
    )
    assert params.chat_template_kwargs["reasoning_effort"] == "max"
    # `max` means think hard, so thinking must not be switched off on the way.
    assert params.chat_template_kwargs.get("enable_thinking") is not False


def test_none_effort_disables_thinking():
    """`none` is the one value that means "do not think", not "think less"."""
    request = ResponsesRequest(model=MODEL, input="hi", reasoning={"effort": "none"})
    params = request.build_chat_params(
        default_template=None, default_template_content_format="string"
    )
    assert params.chat_template_kwargs["enable_thinking"] is False
