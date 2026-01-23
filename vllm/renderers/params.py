# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt

if TYPE_CHECKING:
    from vllm.config import ModelConfig


_S = TypeVar("_S", bound=list[int] | torch.Tensor)


def merge_kwargs(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
    /,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, Any]:
    if a is None:
        a = {}
    if b is None:
        b = {}

    return a | {k: v for k, v in b.items() if v not in unset_values}


@dataclass(frozen=True)
class ChatParams:
    """Configuration to control how to parse chat messages."""

    chat_template: str | None = None
    """The chat template to apply."""

    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    """The format of the chat template."""

    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    """The kwargs to pass to the chat template."""

    def with_defaults(self, default_chat_template_kwargs: dict[str, Any] | None):
        if not default_chat_template_kwargs:
            return self

        return ChatParams(
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                default_chat_template_kwargs,
                self.chat_template_kwargs,
            ),
        )

    def get_apply_chat_template_kwargs(self) -> dict[str, Any]:
        return merge_kwargs(
            self.chat_template_kwargs,
            dict(chat_template=self.chat_template),
        )


@dataclass(frozen=True)
class TokenizeParams:
    """Configuration to control how prompts are tokenized."""

    max_length: int | None = None
    """Maximum allowable total input token length. If provided,
    token inputs longer than this raise `ValueError`."""

    add_special_tokens: bool = True
    """Whether to add special tokens."""

    truncate_prompt_tokens: int | None = None
    """Number of tokens to keep. `None` means no truncation.
    `0` yields an empty list (and skips embeds).
    `-1` maps to `model_config.max_model_len`."""

    needs_detokenization: bool = False
    """Whether the output prompt needs to contain text."""

    @staticmethod
    def from_config(
        model_config: "ModelConfig",
        max_length: int | None = None,
        truncate_prompt_tokens: int | None = None,
        add_special_tokens: bool = True,
        needs_detokenization: bool = False,
    ) -> "TokenizeParams":
        if truncate_prompt_tokens is not None and truncate_prompt_tokens < 0:
            truncate_prompt_tokens = model_config.max_model_len

        return TokenizeParams(
            max_length=max_length,
            truncate_prompt_tokens=truncate_prompt_tokens,
            add_special_tokens=add_special_tokens,
            needs_detokenization=needs_detokenization,
        )

    def __post_init__(self) -> None:
        truncate_prompt_tokens = self.truncate_prompt_tokens
        max_length = self.max_length

        if (
            max_length is not None
            and truncate_prompt_tokens is not None
            and truncate_prompt_tokens > max_length
        ):
            raise VLLMValidationError(
                f"{truncate_prompt_tokens=} cannot be greater than "
                f"{max_length=}. Please select a smaller truncation size.",
                parameter="truncate_prompt_tokens",
                value=truncate_prompt_tokens,
            )

    def get_encode_kwargs(self) -> dict[str, Any]:
        return dict(
            truncation=self.truncate_prompt_tokens is not None,
            max_length=self.truncate_prompt_tokens,
            add_special_tokens=self.add_special_tokens,
        )

    def apply_pre_tokenization(self, prompt: TextPrompt) -> TextPrompt:
        """
        Ensure that the prompt meets the requirements set out by this config.
        If that is not possible, raise a `VLLMValidationError`.

        This method is run before tokenization occurs.
        """
        # The place to implement https://github.com/vllm-project/vllm/pull/31366
        return prompt

    def _apply_truncation(self, tokens: _S) -> _S:
        """Apply truncation to a token sequence."""
        truncate_prompt_tokens = self.truncate_prompt_tokens

        if truncate_prompt_tokens is None or truncate_prompt_tokens >= len(tokens):
            return tokens

        return tokens[-truncate_prompt_tokens:]  # type: ignore[return-value]

    def _apply_length_check(self, tokens: _S) -> _S:
        """Apply length checks to a token sequence."""
        max_length = self.max_length

        if max_length is not None and len(tokens) > max_length:
            raise VLLMValidationError(
                f"This model's maximum context length is {max_length} tokens. "
                f"However, your request has {len(tokens)} input tokens. "
                "Please reduce the length of the input messages.",
                parameter="input_tokens",
                value=len(tokens),
            )

        return tokens

    def _validate_tokens(self, tokens: _S) -> _S:
        """Apply all validators to a token sequence."""
        for validator in (self._apply_truncation, self._apply_length_check):
            tokens = validator(tokens)

        return tokens

    def apply_post_tokenization(
        self,
        prompt: TokensPrompt | EmbedsPrompt,
    ) -> TokensPrompt | EmbedsPrompt:
        """
        Ensure that the prompt meets the requirements set out by this config.
        If that is not possible, raise a `VLLMValidationError`.

        This method is run after tokenization occurs.
        """
        if "prompt_token_ids" in prompt:
            prompt["prompt_token_ids"] = self._validate_tokens(  # type: ignore[typeddict-unknown-key]
                prompt["prompt_token_ids"]  # type: ignore[typeddict-item]
            )
        if "prompt_embeds" in prompt:
            prompt["prompt_embeds"] = self._validate_tokens(  # type: ignore[typeddict-unknown-key]
                prompt["prompt_embeds"]  # type: ignore[typeddict-item]
            )

        return prompt
