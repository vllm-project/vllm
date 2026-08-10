# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared DeepSeek-V4 chat-template-kwarg normalisation.

Both `/v1/chat/completions` and `/v1/responses` must hand the tokenizer and the
reasoning parser the *same* view of whether thinking is on. The two endpoints
previously derived it separately, and drifted:

* `ChatCompletionRequest` normalised it and defaulted thinking on for V4.
* `ResponsesRequest` did not, so with no thinking kwarg the tokenizer defaulted
  thinking **on** (`DeepSeekV4Tokenizer.apply_chat_template`) while
  `DeepSeekV4ReasoningParser` defaulted it **off** and selected
  `IdentityReasoningParser`. The model thought, and its reasoning plus a bare
  ``</think>`` were returned inside ``output_text`` as if they were the answer.

Keeping the derivation here means a third endpoint cannot repeat it.
"""

from typing import Any

from vllm.config import ModelConfig


def is_deepseek_v4_model(
    model_name: str | None,
    model_config: ModelConfig | None = None,
) -> bool:
    """Whether the served model is a DeepSeek-V4 variant.

    Checks the loaded config first and falls back to the requested model name,
    so it also works before a config is available.
    """
    hf_config = getattr(model_config, "hf_config", None)
    if getattr(hf_config, "model_type", None) == "deepseek_v4":
        return True

    architectures = getattr(hf_config, "architectures", None) or ()
    if any(
        "deepseekv4" in str(arch).replace("_", "").lower() for arch in architectures
    ):
        return True

    model = (model_name or "").lower().replace("_", "-")
    return "deepseek-v4" in model


def apply_deepseek_v4_chat_kwargs(
    chat_template_kwargs: dict[str, Any],
    *,
    model_name: str | None,
    model_config: ModelConfig | None = None,
    thinking_enabled: bool | None = None,
) -> dict[str, Any]:
    """Normalise DeepSeek-V4 thinking state into chat-template kwargs.

    Args:
        chat_template_kwargs: kwargs resolved so far, request over server default.
        model_name: the requested model id, used when no config is available.
        model_config: the loaded model config, when there is one.
        thinking_enabled: an explicit decision from a request-level field, such
            as `ChatCompletionRequest.thinking`. `None` means the request did
            not say, in which case an existing `thinking`/`enable_thinking`
            kwarg wins and thinking defaults **on** for V4 — matching
            `DeepSeekV4Tokenizer.apply_chat_template`.

    Returns a new dict; the input is not mutated.
    """
    chat_template_kwargs = dict(chat_template_kwargs)
    if not is_deepseek_v4_model(model_name, model_config):
        return chat_template_kwargs

    if thinking_enabled is not None:
        chat_template_kwargs["thinking"] = thinking_enabled
        chat_template_kwargs["enable_thinking"] = thinking_enabled
    elif (
        "thinking" not in chat_template_kwargs
        and "enable_thinking" not in chat_template_kwargs
    ):
        # Both keys are set, not just one: the tokenizer reads either, and
        # DeepSeekV4ReasoningParser reads both with a `False` default, so
        # leaving one unset is what let the two disagree in the first place.
        chat_template_kwargs["thinking"] = True
        chat_template_kwargs["enable_thinking"] = True

    return chat_template_kwargs
