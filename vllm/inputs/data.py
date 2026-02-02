# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, cast

import torch
from typing_extensions import NotRequired, TypedDict, TypeIs, TypeVar

from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.multimodal.inputs import (
        MultiModalDataDict,
        MultiModalInputs,
        MultiModalUUIDDict,
    )
else:
    MultiModalDataDict = object
    MultiModalInputs = object
    MultiModalUUIDDict = object


class _CommonKeys(TypedDict):
    multi_modal_data: NotRequired[MultiModalDataDict | None]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[dict[str, Any] | None]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """

    multi_modal_uuids: NotRequired[MultiModalUUIDDict]
    """
    Optional user-specified UUIDs for multimodal items, mapped by modality.
    Lists must match the number of items per modality and may contain `None`.
    For `None` entries, the hasher will compute IDs automatically; non-None
    entries override the default hashes for caching, and MUST be unique per
    multimodal item.
    """

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


class TextPrompt(_CommonKeys):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""


class TokensPrompt(_CommonKeys):
    """Schema for a tokenized prompt."""

    prompt_token_ids: list[int]
    """A list of token IDs to pass to the model."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    token_type_ids: NotRequired[list[int]]
    """A list of token type IDs to pass to the cross encoder model."""


class EmbedsPrompt(_CommonKeys):
    """Schema for a prompt provided via token embeddings."""

    prompt_embeds: torch.Tensor
    """The embeddings of the prompt."""


class DataPrompt(_CommonKeys):
    """Represents generic inputs handled by IO processor plugins."""

    data: Any
    """The input data"""

    data_format: str
    """The input data format"""


SingletonPrompt: TypeAlias = str | TextPrompt | TokensPrompt | EmbedsPrompt
"""
Set of possible schemas for a single prompt:

- A text prompt ([`str`][] or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt ([`TokensPrompt`][vllm.inputs.data.TokensPrompt])
- An embeddings prompt ([`EmbedsPrompt`][vllm.inputs.data.EmbedsPrompt])

Note that "singleton" is as opposed to a data structure
which encapsulates multiple prompts, i.e. of the sort
which may be utilized for encoder/decoder models when
the user desires to express both the encoder & decoder
prompts explicitly, i.e. 
[`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt]

A prompt of type [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt] may be 
employed as (1) input to a decoder-only model, (2) input to
the encoder of an encoder/decoder model, in the scenario
where the decoder-prompt is not specified explicitly, or
(3) as a member of a larger data structure encapsulating
more than one prompt, i.e. 
[`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt]
"""


def is_tokens_prompt(prompt: SingletonPrompt) -> TypeIs[TokensPrompt]:
    return (
        isinstance(prompt, dict)
        and "prompt_token_ids" in prompt
        and "prompt_embeds" not in prompt
    )


def is_embeds_prompt(prompt: SingletonPrompt) -> TypeIs[EmbedsPrompt]:
    return (
        isinstance(prompt, dict)
        and "prompt_token_ids" not in prompt
        and "prompt_embeds" in prompt
    )


_T1_co = TypeVar(
    "_T1_co", bound=SingletonPrompt, default=SingletonPrompt, covariant=True
)
_T2_co = TypeVar(
    "_T2_co", bound=SingletonPrompt, default=SingletonPrompt, covariant=True
)


# TODO: Make fields ReadOnly once mypy supports it
class ExplicitEncoderDecoderPrompt(TypedDict, Generic[_T1_co, _T2_co]):
    """
    Represents an encoder/decoder model input prompt,
    comprising an explicit encoder prompt and a decoder prompt.

    The encoder and decoder prompts, respectively, may be formatted
    according to any of the
    [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt] schemas,
    and are not required to have the same schema.

    Only the encoder prompt may have multi-modal data. mm_processor_kwargs
    should be at the top-level, and should not be set in the encoder/decoder
    prompts, since they are agnostic to the encoder/decoder.

    Note that an
    [`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt]
    may not be used as an input to a decoder-only model,
    and that the `encoder_prompt` and `decoder_prompt`
    fields of this data structure themselves must be
    [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt] instances.
    """

    encoder_prompt: _T1_co

    decoder_prompt: _T2_co | None

    mm_processor_kwargs: NotRequired[dict[str, Any]]


PromptType: TypeAlias = SingletonPrompt | ExplicitEncoderDecoderPrompt[Any, Any]
"""
Set of possible schemas for an LLM input, including
both decoder-only and encoder/decoder input types:

- A text prompt ([`str`][] or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt ([`TokensPrompt`][vllm.inputs.data.TokensPrompt])
- An embeddings prompt ([`EmbedsPrompt`][vllm.inputs.data.EmbedsPrompt])
- A single data structure containing both an encoder and a decoder prompt
  ([`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt])
"""


class TokenInputs(TypedDict):
    """Represents token-based inputs."""

    type: Literal["token"]
    """The type of inputs."""

    prompt_token_ids: list[int]
    """The token IDs of the prompt."""

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


def token_inputs(
    prompt_token_ids: list[int],
    cache_salt: str | None = None,
) -> TokenInputs:
    """Construct [`TokenInputs`][vllm.inputs.data.TokenInputs] from optional
    values."""
    inputs = TokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


class EmbedsInputs(TypedDict):
    """Represents embeddings-based inputs."""

    type: Literal["embeds"]
    """The type of inputs."""

    prompt_embeds: torch.Tensor
    """The embeddings of the prompt."""

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


def embeds_inputs(
    prompt_embeds: torch.Tensor,
    cache_salt: str | None = None,
) -> EmbedsInputs:
    """Construct [`EmbedsInputs`][vllm.inputs.data.EmbedsInputs] from optional
    values."""
    inputs = EmbedsInputs(type="embeds", prompt_embeds=prompt_embeds)

    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


DecoderOnlyInputs: TypeAlias = TokenInputs | EmbedsInputs | MultiModalInputs
"""
The inputs in [`LLMEngine`][vllm.engine.llm_engine.LLMEngine] before they are
passed to the model executor.
This specifies the data required for decoder-only models.
"""


class EncoderDecoderInputs(TypedDict):
    """
    The inputs in [`LLMEngine`][vllm.engine.llm_engine.LLMEngine] before they
    are passed to the model executor.

    This specifies the required data for encoder-decoder models.
    """

    encoder: TokenInputs | MultiModalInputs
    """The inputs for the encoder portion."""

    decoder: TokenInputs | MultiModalInputs
    """The inputs for the decoder portion."""


SingletonInputs: TypeAlias = TokenInputs | EmbedsInputs | MultiModalInputs
"""
A processed [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt] which can be
passed to [`Sequence`][collections.abc.Sequence].
"""

ProcessorInputs: TypeAlias = DecoderOnlyInputs | EncoderDecoderInputs
"""
The outputs from [`vllm.inputs.preprocess.InputPreprocessor`][].
"""

_T1 = TypeVar("_T1", bound=SingletonPrompt, default=SingletonPrompt)
_T2 = TypeVar("_T2", bound=SingletonPrompt, default=SingletonPrompt)


def build_explicit_enc_dec_prompt(
    encoder_prompt: _T1,
    decoder_prompt: _T2 | None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> ExplicitEncoderDecoderPrompt[_T1, _T2]:
    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}
    return ExplicitEncoderDecoderPrompt(
        encoder_prompt=encoder_prompt,
        decoder_prompt=decoder_prompt,
        mm_processor_kwargs=mm_processor_kwargs,
    )


def zip_enc_dec_prompts(
    enc_prompts: Iterable[_T1],
    dec_prompts: Iterable[_T2 | None],
    mm_processor_kwargs: Iterable[dict[str, Any]] | dict[str, Any] | None = None,
) -> list[ExplicitEncoderDecoderPrompt[_T1, _T2]]:
    """
    Zip encoder and decoder prompts together into a list of
    [`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt]
    instances.

    `mm_processor_kwargs` may also be provided; if a dict is passed, the same
    dictionary will be used for every encoder/decoder prompt. If an iterable is
    provided, it will be zipped with the encoder/decoder prompts.
    """
    if mm_processor_kwargs is None:
        mm_processor_kwargs = cast(dict[str, Any], {})
    if isinstance(mm_processor_kwargs, dict):
        return [
            build_explicit_enc_dec_prompt(
                encoder_prompt,
                decoder_prompt,
                cast(dict[str, Any], mm_processor_kwargs),
            )
            for (encoder_prompt, decoder_prompt) in zip(enc_prompts, dec_prompts)
        ]
    return [
        build_explicit_enc_dec_prompt(encoder_prompt, decoder_prompt, mm_proc_kwargs)
        for (encoder_prompt, decoder_prompt, mm_proc_kwargs) in zip(
            enc_prompts, dec_prompts, mm_processor_kwargs
        )
    ]


def to_enc_dec_tuple_list(
    enc_dec_prompts: Iterable[ExplicitEncoderDecoderPrompt[_T1, _T2]],
) -> list[tuple[_T1, _T2 | None]]:
    return [
        (enc_dec_prompt["encoder_prompt"], enc_dec_prompt["decoder_prompt"])
        for enc_dec_prompt in enc_dec_prompts
    ]


@dataclass
class StreamingInput:
    """Input data for a streaming generation request.

    This is used with generate() to support multi-turn streaming sessions
    where inputs are provided via an async generator.
    """

    prompt: PromptType
    sampling_params: SamplingParams | None = None
