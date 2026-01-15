# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Annotated, Any, TypeAlias

from pydantic import (
    Field,
    model_validator,
)

from vllm import PoolingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.utils import random_uuid
from vllm.utils.serial_utils import EmbedDType, EncodingFormat, Endianness


class EmbeddingCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings
    model: str | None = None
    input: list[int] | list[list[int]] | str | list[str]
    encoding_format: EncodingFormat = "float"
    dimensions: int | None = None
    user: str | None = None
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None

    # --8<-- [start:embedding-extra-params]
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."
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
    request_id: str = Field(
        default_factory=random_uuid,
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    normalize: bool | None = Field(
        default=None,
        description="Whether to normalize the embeddings outputs. Default is True.",
    )
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
    # --8<-- [end:embedding-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            use_activation=self.normalize,
        )


class EmbeddingChatRequest(OpenAIBaseModel):
    model: str | None = None
    messages: list[ChatCompletionMessageParam]

    encoding_format: EncodingFormat = "float"
    dimensions: int | None = None
    user: str | None = None
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None

    # --8<-- [start:chat-embedding-extra-params]
    add_generation_prompt: bool = Field(
        default=False,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
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
        description=("Additional kwargs to pass to the HF processor."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    request_id: str = Field(
        default_factory=random_uuid,
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    normalize: bool | None = Field(
        default=None,
        description="Whether to normalize the embeddings outputs. Default is True.",
    )
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
    # --8<-- [end:chat-embedding-extra-params]

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return data

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            use_activation=self.normalize,
        )


EmbeddingRequest: TypeAlias = EmbeddingCompletionRequest | EmbeddingChatRequest


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: list[float] | str


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo


class EmbeddingBytesResponse(OpenAIBaseModel):
    content: list[bytes]
    headers: dict[str, str] | None = None
    media_type: str = "application/octet-stream"
