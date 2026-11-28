# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Annotated, Any, TypeAlias

from pydantic import ConfigDict, Field, model_validator

from vllm.config import ModelConfig
from vllm.entrypoints.generate.base.protocol import AnyResponseFormat
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel
from vllm.entrypoints.serve.utils.request_validation import FailFastRequestMixin
from vllm.exceptions import VLLMValidationError
from vllm.renderers import (
    ChatParams,
    ReasoningEffort,
    TokenizeParams,
    map_reasoning_effort,
    merge_kwargs,
)
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)


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
            max_total_tokens=None,
            max_output_tokens=0,
            add_special_tokens=self.add_special_tokens,
        )


class TokenizeChatRequest(FailFastRequestMixin, OpenAIBaseModel):
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
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description=(
            "Reasoning effort used to render the chat template, with the same "
            "meaning as in the Chat Completions API."
        ),
    )
    documents: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect."
        ),
    )
    response_format: AnyResponseFormat | None = Field(
        default=None,
        description="Output format to include in model-specific chat templating.",
    )
    media_io_kwargs: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Additional kwargs to pass to the media IO connectors, "
            "keyed by modality. Merged with engine-level media_io_kwargs."
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
        if not isinstance(data, dict):
            return data
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise VLLMValidationError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True.",
            )
        return data

    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        reasoning_effort_map: dict[str, str] | None = None,
    ) -> ChatParams:
        reasoning_effort = map_reasoning_effort(
            self.reasoning_effort, reasoning_effort_map
        )

        extra_kwargs: dict[str, Any] = dict(
            add_generation_prompt=self.add_generation_prompt,
            continue_final_message=self.continue_final_message,
            documents=self.documents,
            reasoning_effort=reasoning_effort,
        )

        # Same as the chat completions request, so that /tokenize renders the
        # prompt the chat route would generate from.
        user_kwargs = self.chat_template_kwargs or {}
        if reasoning_effort is not None and "enable_thinking" not in user_kwargs:
            extra_kwargs["enable_thinking"] = reasoning_effort != "none"

        return ChatParams(
            chat_template=self.chat_template or default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=merge_kwargs(
                self.chat_template_kwargs,
                extra_kwargs,
            ),
            media_io_kwargs=self.media_io_kwargs,
            response_format=self.response_format,
        )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=None,
            max_output_tokens=0,
            add_special_tokens=self.add_special_tokens,
        )

    def to_chat_completion_request(self) -> ChatCompletionRequest:
        """Build the chat completion request with this request's chat fields."""
        data = self.model_dump(
            include={
                "model",
                "messages",
                "tools",
                "add_generation_prompt",
                "continue_final_message",
                "add_special_tokens",
                "chat_template",
                "chat_template_kwargs",
                "reasoning_effort",
                "documents",
                "media_io_kwargs",
                "mm_processor_kwargs",
                "response_format",
            },
            exclude_none=True,
        )
        return ChatCompletionRequest.model_validate(data)


TokenizeRequest: TypeAlias = TokenizeCompletionRequest | TokenizeChatRequest


class TokenizeResponse(OpenAIBaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: list[str] | None = None


class DetokenizeRequest(FailFastRequestMixin, OpenAIBaseModel):
    model: str | None = None
    # TODO: Factor `torch.iinfo` out. `torch.iinfo` pulls torch into a
    # Pydantic protocol file that currently has no torch dependency.
    # See: https://github.com/vllm-project/vllm/pull/34468#discussion_r2801173630
    tokens: list[Annotated[int, Field(ge=0, le=2**63 - 1)]]

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=None,
            max_output_tokens=0,
            needs_detokenization=True,
        )


class DetokenizeResponse(OpenAIBaseModel):
    prompt: str


class TokenizerInfoResponse(OpenAIBaseModel):
    """Response containing tokenizer configuration
    equivalent to tokenizer_config.json
    """

    model_config = ConfigDict(extra="allow")
    tokenizer_class: str
