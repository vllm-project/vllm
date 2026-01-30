# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

from vllm import PoolingParams
from vllm.config.pooler import get_use_activation
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
from vllm.utils import random_uuid
from vllm.utils.serial_utils import EmbedDType, EncodingFormat, Endianness


class MultimodalEmbeddingInput(BaseModel):
    """Input type for multimodal embeddings.
    
    Supports text, image, and instruction inputs for embedding generation.
    At least one of text or image must be provided.
    
    Examples:
        >>> # Text only
        >>> MultimodalEmbeddingInput(text="Hello world")
        >>> # Image only
        >>> MultimodalEmbeddingInput(image="https://example.com/image.jpg")
        >>> # Text and image with instruction
        >>> MultimodalEmbeddingInput(
        ...     instruction="Represent this image for retrieval",
        ...     text="A cat",
        ...     image="https://example.com/cat.jpg"
        ... )
    """
    instruction: str | None = None
    text: str | None = None
    image: str | None = None


class PoolingBasicRequestMixin(OpenAIBaseModel):
    # --8<-- [start:pooling-common-params]
    model: str | None = None
    user: str | None = None
    # --8<-- [end:pooling-common-params]

    # --8<-- [start:pooling-common-extra-params]
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
    # --8<-- [end:pooling-common-extra-params]


class CompletionRequestMixin(OpenAIBaseModel):
    # --8<-- [start:completion-params]
    input: (
        list[int]
        | list[list[int]]
        | str
        | list[str]
        | MultimodalEmbeddingInput
        | list[MultimodalEmbeddingInput]
    )
    # --8<-- [end:completion-params]

    # --8<-- [start:completion-extra-params]
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."
        ),
    )
    # --8<-- [end:completion-extra-params]


class ChatRequestMixin(OpenAIBaseModel):
    # --8<-- [start:chat-params]
    messages: list[ChatCompletionMessageParam]
    # --8<-- [end:chat-params]

    # --8<-- [start:chat-extra-params]
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
    # --8<-- [end:chat-extra-params]

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return data


class EncodingRequestMixin(OpenAIBaseModel):
    # --8<-- [start:encoding-params]
    encoding_format: EncodingFormat = "float"
    # --8<-- [end:encoding-params]

    # --8<-- [start:encoding-extra-params]
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
    # --8<-- [end:encoding-extra-params]


class EmbedRequestMixin(EncodingRequestMixin):
    # --8<-- [start:embed-params]
    dimensions: int | None = None
    # --8<-- [end:embed-params]

    # --8<-- [start:embed-extra-params]
    normalize: bool | None = Field(
        default=None,
        description="Whether to normalize the embeddings outputs. Default is True.",
    )
    # --8<-- [end:embed-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            dimensions=self.dimensions,
            use_activation=self.normalize,
            truncate_prompt_tokens=getattr(self, "truncate_prompt_tokens", None),
        )


class ClassifyRequestMixin(OpenAIBaseModel):
    # --8<-- [start:classify-extra-params]
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
    # --8<-- [end:classify-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            use_activation=get_use_activation(self),
            truncate_prompt_tokens=getattr(self, "truncate_prompt_tokens", None),
        )
