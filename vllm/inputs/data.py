# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, Literal, Optional, Union, cast

import torch
from typing_extensions import NotRequired, TypedDict, TypeIs, TypeVar

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalDataDict, MultiModalInputs


class TextPrompt(TypedDict):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: list[int]
    """A list of token IDs to pass to the model."""

    token_type_ids: NotRequired[list[int]]
    """A list of token type IDs to pass to the cross encoder model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


class EmbedsPrompt(TypedDict):
    """Schema for a prompt provided via token embeddings."""

    prompt_embeds: torch.Tensor
    """The embeddings of the prompt."""

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


SingletonPrompt = Union[str, TextPrompt, TokensPrompt, EmbedsPrompt]
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
    return (isinstance(prompt, dict) and "prompt_token_ids" in prompt
            and "prompt_embeds" not in prompt)


def is_embeds_prompt(prompt: SingletonPrompt) -> TypeIs[EmbedsPrompt]:
    return (isinstance(prompt, dict) and "prompt_token_ids" not in prompt
            and "prompt_embeds" in prompt)


_T1_co = TypeVar("_T1_co",
                 bound=SingletonPrompt,
                 default=SingletonPrompt,
                 covariant=True)
_T2_co = TypeVar("_T2_co",
                 bound=SingletonPrompt,
                 default=SingletonPrompt,
                 covariant=True)


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

    decoder_prompt: Optional[_T2_co]

    mm_processor_kwargs: NotRequired[dict[str, Any]]


PromptType = Union[SingletonPrompt, ExplicitEncoderDecoderPrompt]
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

    token_type_ids: NotRequired[list[int]]
    """The token type IDs of the prompt."""

    prompt: NotRequired[str]
    """
    The original prompt text corresponding to the token IDs, if available.
    """

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


def token_inputs(
    prompt_token_ids: list[int],
    token_type_ids: Optional[list[int]] = None,
    prompt: Optional[str] = None,
    cache_salt: Optional[str] = None,
) -> TokenInputs:
    """Construct [`TokenInputs`][vllm.inputs.data.TokenInputs] from optional
    values."""
    inputs = TokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if token_type_ids is not None:
        inputs["token_type_ids"] = token_type_ids
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
    cache_salt: Optional[str] = None,
) -> EmbedsInputs:
    """Construct [`EmbedsInputs`][vllm.inputs.data.EmbedsInputs] from optional
    values."""
    inputs = EmbedsInputs(type="embeds", prompt_embeds=prompt_embeds)

    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


DecoderOnlyInputs = Union[TokenInputs, EmbedsInputs, "MultiModalInputs"]
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

    encoder: Union[TokenInputs, "MultiModalInputs"]
    """The inputs for the encoder portion."""

    decoder: Union[TokenInputs, "MultiModalInputs"]
    """The inputs for the decoder portion."""


SingletonInputs = Union[TokenInputs, EmbedsInputs, "MultiModalInputs"]
"""
A processed [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt] which can be 
passed to [`vllm.sequence.Sequence`][].
"""

ProcessorInputs = Union[DecoderOnlyInputs, EncoderDecoderInputs]
"""
The outputs from [`vllm.inputs.preprocess.InputPreprocessor`][].
"""

_T1 = TypeVar("_T1", bound=SingletonPrompt, default=SingletonPrompt)
_T2 = TypeVar("_T2", bound=SingletonPrompt, default=SingletonPrompt)


def build_explicit_enc_dec_prompt(
    encoder_prompt: _T1,
    decoder_prompt: Optional[_T2],
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
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
    dec_prompts: Iterable[Optional[_T2]],
    mm_processor_kwargs: Optional[Union[Iterable[dict[str, Any]],
                                        dict[str, Any]]] = None,
) -> list[ExplicitEncoderDecoderPrompt[_T1, _T2]]:
    """
    Zip encoder and decoder prompts together into a list of
    [`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt]
    instances.

    ``mm_processor_kwargs`` may also be provided; if a dict is passed, the same
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
            ) for (encoder_prompt,
                   decoder_prompt) in zip(enc_prompts, dec_prompts)
        ]
    return [
        build_explicit_enc_dec_prompt(encoder_prompt, decoder_prompt,
                                      mm_proc_kwargs)
        for (encoder_prompt, decoder_prompt, mm_proc_kwargs
             ) in zip(enc_prompts, dec_prompts, mm_processor_kwargs)
    ]


def to_enc_dec_tuple_list(
    enc_dec_prompts: Iterable[ExplicitEncoderDecoderPrompt[_T1, _T2]],
) -> list[tuple[_T1, Optional[_T2]]]:
    return [(enc_dec_prompt["encoder_prompt"],
             enc_dec_prompt["decoder_prompt"])
            for enc_dec_prompt in enc_dec_prompts]
