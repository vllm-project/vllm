# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch

from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


_S = TypeVar("_S", bound=list[int] | torch.Tensor)


def merge_kwargs(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    /,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, Any]:
    if defaults is None:
        defaults = {}
    if overrides is None:
        overrides = {}

    return defaults | {k: v for k, v in overrides.items() if v not in unset_values}


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
        """The arguments to pass to `tokenizer.apply_chat_template`."""
        return merge_kwargs(
            self.chat_template_kwargs,
            dict(chat_template=self.chat_template),
        )


@dataclass(frozen=True)
class TokenizeParams:
    """Configuration to control how prompts are tokenized."""

    max_total_tokens: int | None
    """
    Maximum allowed number of input + output tokens.
    
    Usually, this refers to the model's context length.
    """

    max_output_tokens: int = 0
    """Maximum requested number of output tokens."""

    truncate_prompt_tokens: int | None = None
    """
    Number of tokens to keep:
    - `None` means no truncation.
    - `-1` maps to `max_input_tokens`.
    """

    do_lower_case: bool | None = None
    """Whether to normalize text to lower case before tokenization."""

    add_special_tokens: bool = True
    """Whether to add special tokens."""

    needs_detokenization: bool = False
    """Whether the output prompt needs to contain text."""

    max_total_tokens_param: str = "max_total_tokens"
    """Override this to edit the message for validation errors."""

    max_output_tokens_param: str = "max_output_tokens"
    """Override this to edit the message for validation errors."""

    truncate_prompt_tokens_param: str = "truncate_prompt_tokens"
    """Override this to edit the message for validation errors."""

    @property
    def max_input_tokens(self) -> int | None:
        """Maximum allowed number of input tokens."""
        if self.max_total_tokens is None:
            return None

        return self.max_total_tokens - self.max_output_tokens

    def __post_init__(self) -> None:
        max_total_tokens = self.max_total_tokens
        max_output_tokens = self.max_output_tokens
        max_input_tokens = self.max_input_tokens
        truncate_prompt_tokens = self.truncate_prompt_tokens

        if (
            max_output_tokens is not None
            and max_total_tokens is not None
            and max_output_tokens > max_total_tokens
        ):
            raise VLLMValidationError(
                f"{self.max_output_tokens_param}={max_output_tokens}"
                f"cannot be greater than "
                f"{self.max_total_tokens_param}={max_total_tokens=}. "
                f"Please request fewer output tokens.",
                parameter=self.max_output_tokens_param,
                value=max_output_tokens,
            )

        if (
            max_input_tokens is not None
            and truncate_prompt_tokens is not None
            and truncate_prompt_tokens > max_input_tokens
        ):
            raise VLLMValidationError(
                f"{self.truncate_prompt_tokens_param}={truncate_prompt_tokens} "
                f"cannot be greater than {self.max_total_tokens_param} - "
                f"{self.max_output_tokens_param} = {max_input_tokens}. "
                f"Please request a smaller truncation size.",
                parameter=self.truncate_prompt_tokens_param,
                value=truncate_prompt_tokens,
            )

    def with_kwargs(self, tokenization_kwargs: dict[str, Any] | None):
        if tokenization_kwargs is None:
            tokenization_kwargs = {}

        max_length = tokenization_kwargs.pop("max_length", self.max_input_tokens)
        truncate_prompt_tokens = tokenization_kwargs.pop(
            "truncate_prompt_tokens", self.truncate_prompt_tokens
        )
        do_lower_case = tokenization_kwargs.pop("do_lower_case", self.do_lower_case)
        add_special_tokens = tokenization_kwargs.pop(
            "add_special_tokens", self.add_special_tokens
        )
        needs_detokenization = tokenization_kwargs.pop(
            "needs_detokenization", self.needs_detokenization
        )

        # https://huggingface.co/docs/transformers/en/pad_truncation
        if truncation := tokenization_kwargs.pop("truncation", None):
            if truncation in (True, "longest_first"):
                truncate_prompt_tokens = max_length
            elif truncation in (False, "do_not_truncate"):
                truncate_prompt_tokens = None
            else:
                # To emit the below warning
                tokenization_kwargs["truncation"] = truncation

        if tokenization_kwargs:
            logger.warning(
                "The following tokenization arguments are not supported "
                "by vLLM Renderer and will be ignored: %s",
                tokenization_kwargs,
            )

        max_total_tokens = self.max_total_tokens

        return TokenizeParams(
            max_total_tokens=max_total_tokens,
            max_output_tokens=(
                0
                if max_total_tokens is None or max_length is None
                else max_total_tokens - max_length
            ),
            truncate_prompt_tokens=truncate_prompt_tokens,
            do_lower_case=do_lower_case,
            add_special_tokens=add_special_tokens,
            needs_detokenization=needs_detokenization,
        )

    def get_encode_kwargs(self) -> dict[str, Any]:
        """The arguments to pass to `tokenizer.encode`."""
        max_length = self.truncate_prompt_tokens
        if max_length is None or max_length < 0:
            max_length = self.max_input_tokens

        return dict(
            truncation=self.truncate_prompt_tokens is not None,
            max_length=max_length,
            add_special_tokens=self.add_special_tokens,
        )

    def _apply_lowercase(self, tokenizer: TokenizerLike | None, text: str) -> str:
        do_lower_case = self.do_lower_case
        if do_lower_case is None:
            do_lower_case = getattr(tokenizer, "do_lower_case", False)

        if do_lower_case:
            text = text.lower()

        return text

    def _validate_text(self, tokenizer: TokenizerLike | None, text: str) -> str:
        """Apply all validators to prompt text."""
        # TODO: Implement https://github.com/vllm-project/vllm/pull/31366
        for validator in (self._apply_lowercase,):
            text = validator(tokenizer, text)

        return text

    def apply_pre_tokenization(
        self,
        tokenizer: TokenizerLike | None,
        prompt: TextPrompt,
    ) -> TextPrompt:
        """
        Ensure that the prompt meets the requirements set out by this config.
        If that is not possible, raise a `VLLMValidationError`.

        This method is run before tokenization occurs.
        """
        prompt["prompt"] = self._validate_text(tokenizer, prompt["prompt"])

        return prompt

    def _apply_truncation(self, tokenizer: TokenizerLike | None, tokens: _S) -> _S:
        """Apply truncation to a token sequence."""
        max_length = self.truncate_prompt_tokens
        # NOTE: We don't want to set `max_length` if it is None
        # Otherwise no error is raised by `self._apply_length_check`
        if max_length is not None and max_length < 0:
            max_length = self.max_input_tokens

        if max_length is None or max_length >= len(tokens):
            return tokens
        if max_length == 0:
            return tokens[:0]  # type: ignore[return-value]

        if getattr(tokenizer, "truncation_side", "left") == "left":
            return tokens[-max_length:]  # type: ignore[return-value]

        return tokens[:max_length]  # type: ignore[return-value]

    def _apply_length_check(self, tokenizer: TokenizerLike | None, tokens: _S) -> _S:
        """Apply length checks to a token sequence."""
        max_input_tokens = self.max_input_tokens

        if max_input_tokens is not None and len(tokens) > max_input_tokens:
            raise VLLMValidationError(
                f"You passed {len(tokens)} input tokens and "
                f"requested {self.max_output_tokens} output tokens. "
                f"However, the model's context length is only "
                f"{self.max_total_tokens}, resulting in a maximum "
                f"input length of {max_input_tokens}. "
                f"Please reduce the length of the input messages.",
                parameter="input_tokens",
                value=len(tokens),
            )

        return tokens

    def _validate_tokens(self, tokenizer: TokenizerLike | None, tokens: _S) -> _S:
        """Apply all validators to a token sequence."""
        for validator in (
            self._apply_truncation,
            self._apply_length_check,
        ):
            tokens = validator(tokenizer, tokens)

        return tokens

    def apply_post_tokenization(
        self,
        tokenizer: TokenizerLike | None,
        prompt: TokensPrompt | EmbedsPrompt,
    ) -> TokensPrompt | EmbedsPrompt:
        """
        Ensure that the prompt meets the requirements set out by this config.
        If that is not possible, raise a `VLLMValidationError`.

        This method is run after tokenization occurs.
        """
        if "prompt_token_ids" in prompt:
            prompt["prompt_token_ids"] = self._validate_tokens(  # type: ignore[typeddict-unknown-key]
                tokenizer,
                prompt["prompt_token_ids"],  # type: ignore[typeddict-item]
            )
        if "prompt_embeds" in prompt:
            prompt["prompt_embeds"] = self._validate_tokens(  # type: ignore[typeddict-unknown-key]
                tokenizer,
                prompt["prompt_embeds"],  # type: ignore[typeddict-item]
            )

        return prompt
