# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Annotated, Any

from pydantic import (
    Field,
    model_validator,
)

from vllm import PoolingParams
from vllm.config.pooler import get_use_activation
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.protocol import OpenAIBaseModel
from vllm.utils import random_uuid
from vllm.utils.serial_utils import EmbedDType, EncodingFormat, Endianness


class PoolingBasicRequestMixin(OpenAIBaseModel):
    model: str | None = None
    user: str | None = None
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None

    request_id: str = Field(
        default_factory=random_uuid,
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )

    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )


class ChatRequestMixin(OpenAIBaseModel):
    messages: list[ChatCompletionMessageParam]

    add_generation_prompt: bool = Field(
        default=False,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
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

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return data


class CompletionRequestMixin(OpenAIBaseModel):
    input: list[int] | list[list[int]] | str | list[str]

    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."
        ),
    )


class MMProcessorRequestMixin(OpenAIBaseModel):
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )


class EncodingRequestMixin(OpenAIBaseModel):
    encoding_format: EncodingFormat = "float"
    embed_dtype: EmbedDType = Field(
        default="float32",
        description=(
            "What dtype to use for encoding. Default to using float32 for base64 "
            "encoding to match the OpenAI python client behavior. "
            "This parameter will affect base64 and binary_response."
        ),
    )
    endianness: Endianness = Field(
        default="native",
        description=(
            "What endianness to use for encoding. Default to using native for "
            "base64 encoding to match the OpenAI python client behavior."
            "This parameter will affect base64 and binary_response."
        ),
    )


class ClassificationRequestMixin(OpenAIBaseModel):
    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )

    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )

    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "Default is True.",
    )

    def to_pooling_params(self):
        return PoolingParams(
            use_activation=get_use_activation(self),
            truncate_prompt_tokens=self.truncate_prompt_tokens,
        )


class EmbeddingRequestMixin(EncodingRequestMixin):
    dimensions: int | None = None
    normalize: bool | None = Field(
        default=None,
        description="Whether to normalize the embeddings outputs. Default is True.",
    )
