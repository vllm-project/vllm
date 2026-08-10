# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`/v1/chat/completions` and `/v1/responses` must agree about DeepSeek-V4.

The defect these cover: with no thinking kwarg,
`DeepSeekV4Tokenizer.apply_chat_template` defaults thinking **on** while
`DeepSeekV4ReasoningParser` defaults it **off** and selects
`IdentityReasoningParser`. The model reasoned and the reasoning, with a bare
``</think>``, came back inside ``output_text`` looking like the answer.

`/v1/chat/completions` was unaffected only because it normalised thinking into
the chat-template kwargs at the protocol boundary; `/v1/responses` did not.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.deepseek_v4_chat_kwargs import (
    apply_deepseek_v4_chat_kwargs,
    is_deepseek_v4_model,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

V4 = "deepseek-ai/DeepSeek-V4-Flash-0731"
V4_BASE = "deepseek-ai/DeepSeek-V4-Flash"
NOT_V4 = "meta-llama/Llama-3.1-8B-Instruct"


class _HFConfig:
    def __init__(self, model_type=None, architectures=()):
        self.model_type = model_type
        self.architectures = list(architectures)


class _ModelConfig:
    def __init__(self, hf_config):
        self.hf_config = hf_config


@pytest.mark.parametrize(
    "model_name,expected",
    [
        (V4, True),
        (V4_BASE, True),
        ("deepseek-ai/deepseek_v4_flash", True),  # underscores normalise
        (NOT_V4, False),
        (None, False),
    ],
)
def test_detects_v4_by_name(model_name, expected):
    assert is_deepseek_v4_model(model_name, None) is expected


def test_detects_v4_by_config_even_when_name_is_unhelpful():
    cfg = _ModelConfig(_HFConfig(model_type="deepseek_v4"))
    assert is_deepseek_v4_model("some/local-path", cfg) is True

    cfg = _ModelConfig(_HFConfig(architectures=["DeepseekV4ForCausalLM"]))
    assert is_deepseek_v4_model("some/local-path", cfg) is True


def test_thinking_defaults_on_for_v4_when_request_is_silent():
    """The bug: silence meant 'off' to the parser and 'on' to the tokenizer."""
    out = apply_deepseek_v4_chat_kwargs({}, model_name=V4, model_config=None)
    assert out["thinking"] is True
    assert out["enable_thinking"] is True


def test_both_keys_are_set_not_just_one():
    """The parser reads both with a False default; one alone reintroduces drift."""
    out = apply_deepseek_v4_chat_kwargs({}, model_name=V4, model_config=None)
    assert set(("thinking", "enable_thinking")).issubset(out)


@pytest.mark.parametrize("key", ["thinking", "enable_thinking"])
@pytest.mark.parametrize("value", [True, False])
def test_an_explicit_kwarg_is_never_overridden(key, value):
    out = apply_deepseek_v4_chat_kwargs(
        {key: value}, model_name=V4, model_config=None
    )
    assert out[key] is value
    # The untouched sibling must not be invented, or an explicit False would be
    # silently re-enabled by the other key.
    assert "thinking" not in out or out["thinking"] is value or key != "thinking"


def test_explicit_request_field_wins_over_everything():
    out = apply_deepseek_v4_chat_kwargs(
        {"thinking": True},
        model_name=V4,
        model_config=None,
        thinking_enabled=False,
    )
    assert out["thinking"] is False
    assert out["enable_thinking"] is False


def test_non_v4_models_are_left_alone():
    out = apply_deepseek_v4_chat_kwargs({}, model_name=NOT_V4, model_config=None)
    assert out == {}


def test_input_is_not_mutated():
    original = {"foo": "bar"}
    apply_deepseek_v4_chat_kwargs(original, model_name=V4, model_config=None)
    assert original == {"foo": "bar"}


# --- the two endpoints must agree -----------------------------------------


@pytest.mark.parametrize("model_name", [V4, V4_BASE])
def test_both_endpoints_derive_the_same_thinking_state(model_name):
    """Same model, same silence, same answer — this is the regression."""
    chat = ChatCompletionRequest(
        model=model_name, messages=[{"role": "user", "content": "hi"}]
    )
    responses = ResponsesRequest(model=model_name, input="hi")

    chat_kwargs = chat.apply_chat_template_kwargs({})
    resp_kwargs = responses.apply_chat_template_kwargs({})

    assert chat_kwargs.get("thinking") == resp_kwargs.get("thinking") is True
    assert (
        chat_kwargs.get("enable_thinking") == resp_kwargs.get("enable_thinking") is True
    )


@pytest.mark.parametrize("model_name", [V4, V4_BASE])
def test_both_endpoints_honour_an_explicit_off(model_name):
    chat = ChatCompletionRequest(
        model=model_name, messages=[{"role": "user", "content": "hi"}]
    )
    responses = ResponsesRequest(model=model_name, input="hi")

    chat_kwargs = chat.apply_chat_template_kwargs({"enable_thinking": False})
    resp_kwargs = responses.apply_chat_template_kwargs({"enable_thinking": False})

    assert chat_kwargs["enable_thinking"] is False
    assert resp_kwargs["enable_thinking"] is False


def test_responses_effort_none_still_disables_thinking():
    """`reasoning.effort: "none"` is folded into enable_thinking upstream of us;
    normalisation must not resurrect thinking afterwards."""
    request = ResponsesRequest(
        model=V4, input="hi", reasoning={"effort": "none"}
    )
    params = request.build_chat_params(None, "string")
    assert params.chat_template_kwargs.get("enable_thinking") is False

    out = request.apply_chat_template_kwargs(params.chat_template_kwargs)
    assert out["enable_thinking"] is False


@pytest.mark.parametrize("effort", ["low", "minimal", "medium", "high", "xhigh", "max"])
def test_responses_accepts_every_effort_spelling(effort):
    request = ResponsesRequest(model=V4, input="hi", reasoning={"effort": effort})
    assert request.reasoning is not None
    assert request.reasoning.effort == effort
    # Thinking stays on for every tier except "none".
    out = request.apply_chat_template_kwargs(
        request.build_chat_params(None, "string").chat_template_kwargs
    )
    assert out.get("enable_thinking") is not False
