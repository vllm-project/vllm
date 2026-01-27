# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Any, TypeAlias

from pydantic import ConfigDict, Field, model_validator

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel,
)
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs


class TokenizeCompletionRequest(OpenAIBaseModel):
    model: str | None = None
    prompt: str

    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."
        ),
    )
    return_token_strs: bool | None = Field(
        default=False,
        description=(
            "If true, also return the token strings corresponding to the token ids."
        ),
    )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
        )


class TokenizeChatRequest(OpenAIBaseModel):
    model: str | None = None
    messages: list[ChatCompletionMessageParam]

    add_generation_prompt: bool = Field(
        default=True,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )
    return_token_strs: bool | None = Field(
        default=False,
        description=(
            "If true, also return the token strings corresponding to the token ids."
        ),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            'This allows you to "prefill" part of the model\'s response for it. '
            "Cannot be used at the same time as `add_generation_prompt`."
        ),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Additional kwargs to pass to the HF processor.",
    )
    tools: list[ChatCompletionToolsParam] | None = Field(
        default=None,
        description="A list of tools the model may call.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return data

    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        return ChatParams(
            chat_template=self.chat_template or default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=merge_kwargs(
                self.chat_template_kwargs,
                dict(
                    add_generation_prompt=self.add_generation_prompt,
                    continue_final_message=self.continue_final_message,
                ),
            ),
        )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
        )


TokenizeRequest: TypeAlias = TokenizeCompletionRequest | TokenizeChatRequest


class TokenizeResponse(OpenAIBaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: list[str] | None = None


class DetokenizeRequest(OpenAIBaseModel):
    model: str | None = None
    tokens: list[int]

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            needs_detokenization=True,
            max_total_tokens_param="max_model_len",
        )


class DetokenizeResponse(OpenAIBaseModel):
    prompt: str


class TokenizerInfoResponse(OpenAIBaseModel):
    """
    Response containing tokenizer configuration
    equivalent to tokenizer_config.json
    """

    model_config = ConfigDict(extra="allow")
    tokenizer_class: str
