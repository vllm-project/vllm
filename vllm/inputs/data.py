# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch
from typing_extensions import NotRequired, TypedDict, assert_never

if TYPE_CHECKING:
    from vllm.multimodal.inputs import (
        MultiModalDataDict,
        MultiModalEncDecInputs,
        MultiModalInputs,
        MultiModalUUIDDict,
    )
else:
    MultiModalDataDict = object
    MultiModalEncDecInputs = object
    MultiModalInputs = object
    MultiModalUUIDDict = object


# Inputs to LLM API
class _PromptOptions(TypedDict):
    """
    Additional options available to all
    [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt].
    """

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


class TextPrompt(_PromptOptions):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""


class TokensPrompt(_PromptOptions):
    """Schema for a tokenized prompt."""

    prompt_token_ids: list[int]
    """A list of token IDs to pass to the model."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    token_type_ids: NotRequired[list[int]]
    """A list of token type IDs to pass to the cross encoder model."""


class EmbedsPrompt(_PromptOptions):
    """Schema for a prompt provided via token embeddings."""

    prompt_embeds: torch.Tensor
    """The embeddings of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token embeddings, if available."""


DecoderOnlyPrompt: TypeAlias = (
    str | TextPrompt | list[int] | TokensPrompt | EmbedsPrompt
)
"""
Schema of a prompt for a decoder-only model:

- A text prompt (string or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt (list of token IDs, or
  [`TokensPrompt`][vllm.inputs.data.TokensPrompt])
- An embeddings prompt ([`EmbedsPrompt`][vllm.inputs.data.EmbedsPrompt])

For encoder-decoder models, passing a singleton prompt is shorthand for passing
`ExplicitEncoderDecoderPrompt(encoder_prompt=prompt, decoder_prompt=None)`.
"""


EncoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt
"""
Schema of a prompt for the encoder part of a encoder-decoder model:

- A text prompt (string or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt (list of token IDs, or
  [`TokensPrompt`][vllm.inputs.data.TokensPrompt])
"""


DecoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt
"""
Schema of a prompt for the decoder part of an encoder-decoder model:

- A text prompt (string or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt (list of token IDs, or
  [`TokensPrompt`][vllm.inputs.data.TokensPrompt])

Note:
    Multi-modal inputs are not supported for decoder prompts.
"""


class ExplicitEncoderDecoderPrompt(TypedDict):
    """
    Schema for a pair of encoder and decoder singleton prompts.

    Note:
        This schema is not valid for decoder-only models.
    """

    encoder_prompt: EncoderPrompt
    """The prompt for the encoder part of the model."""

    decoder_prompt: DecoderPrompt | None
    """
    The prompt for the decoder part of the model.

    Passing `None` will cause the prompt to be inferred automatically.
    """


EncoderDecoderPrompt: TypeAlias = EncoderPrompt | ExplicitEncoderDecoderPrompt
"""
Schema for a prompt for an encoder-decoder model.

You can pass a singleton encoder prompt, in which case the decoder prompt is
considered to be `None` (i.e., infer automatically).
"""


SingletonPrompt: TypeAlias = DecoderOnlyPrompt | EncoderPrompt | DecoderPrompt
"""
Schema for a single prompt. This is as opposed to a data structure
which encapsulates multiple prompts, such as
[`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt].
"""


PromptType: TypeAlias = DecoderOnlyPrompt | EncoderDecoderPrompt
"""
Schema for any prompt, regardless of model type.

This is the input format accepted by most [`LLM`][vllm.entrypoints.llm.LLM] APIs.
"""


class DataPrompt(_PromptOptions):
    """
    Represents generic inputs that are converted to
    [`PromptType`][vllm.inputs.data.PromptType] by IO processor plugins.
    """

    data: Any
    """The input data."""

    data_format: str
    """The input data format."""


# Outputs of processor
class _InputOptions(TypedDict):
    """
    Additional options available to all input types.
    """

    arrival_time: NotRequired[float]
    """The time when the input was received (before rendering)."""

    cache_salt: NotRequired[str]
    """Optional cache salt to be used for prefix caching."""


class TokenInputs(_InputOptions):
    """Represents token-based inputs."""

    type: Literal["token"]
    """The type of inputs."""

    prompt_token_ids: list[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""


def token_inputs(
    prompt_token_ids: list[int],
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> TokenInputs:
    """Construct [`TokenInputs`][vllm.inputs.data.TokenInputs] from optional
    values."""
    inputs = TokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


class EmbedsInputs(_InputOptions):
    """Represents embeddings-based inputs."""

    type: Literal["embeds"]
    """The type of inputs."""

    prompt_embeds: torch.Tensor
    """The embeddings of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""


def embeds_inputs(
    prompt_embeds: torch.Tensor,
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> EmbedsInputs:
    """Construct [`EmbedsInputs`][vllm.inputs.data.EmbedsInputs] from optional
    values."""
    inputs = EmbedsInputs(type="embeds", prompt_embeds=prompt_embeds)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


DecoderOnlyInputs: TypeAlias = TokenInputs | EmbedsInputs | MultiModalInputs
"""
A processed prompt from
[`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
which can be passed to
[`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor]
for decoder-only models.
"""


EncoderInputs: TypeAlias = TokenInputs | MultiModalEncDecInputs
"""
A processed encoder prompt from
[`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
which can be passed to
[`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor]
for encoder-decoder models.
"""


DecoderInputs: TypeAlias = TokenInputs | MultiModalInputs
"""
A processed decoder prompt from
[`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
which can be passed to
[`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor]
for encoder-decoder models.
"""


class EncoderDecoderInputs(TypedDict):
    """
    A processed pair of encoder and decoder singleton prompts.
    [`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
    which can be passed to
    [`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor]
    for encoder-decoder models.
    """

    type: Literal["enc_dec"]

    encoder_prompt: EncoderInputs
    """The inputs for the encoder portion."""

    decoder_prompt: DecoderInputs
    """The inputs for the decoder portion."""

    arrival_time: NotRequired[float]
    """The time when the input was received (before rendering)."""


ProcessorInputs: TypeAlias = DecoderOnlyInputs | EncoderDecoderInputs
"""
A processed prompt from
[`InputPreprocessor`][vllm.inputs.preprocess.InputPreprocessor]
which can be passed to
[`InputProcessor`][vllm.v1.engine.input_processor.InputProcessor].
"""


SingletonInputs: TypeAlias = DecoderOnlyInputs | MultiModalEncDecInputs
"""The inputs for a single encoder/decoder prompt."""


def _validate_enc_inputs(inputs: SingletonInputs) -> EncoderInputs:
    if inputs["type"] == "embeds":
        raise ValueError(
            "Embedding inputs are not supported for encoder-decoder models"
        )

    if inputs["type"] == "multimodal" and "encoder_prompt_token_ids" not in inputs:
        raise RuntimeError(
            "You should register an encoder-decoder multi-modal processor "
            "for encoder-decoder models."
        )

    return inputs  # type: ignore[return-value]


def _validate_dec_inputs(inputs: SingletonInputs) -> DecoderInputs:
    if inputs["type"] == "embeds":
        raise ValueError(
            "Embedding inputs are not supported for encoder-decoder models"
        )

    return inputs


def _prepare_decoder_input_ids_for_generation(
    decoder_input_ids: list[int],
    decoder_start_token_id: int,
) -> list[int]:
    """
    Prepare `decoder_input_ids` for generation with encoder-decoder models,
    according to `GenerationMixin._prepare_decoder_input_ids_for_generation()`.

    Source:
    https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/generation/utils.py
    """
    if len(decoder_input_ids) == 0 or decoder_input_ids[0] != decoder_start_token_id:
        decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

    return decoder_input_ids


def build_enc_dec_inputs(
    encoder_inputs: SingletonInputs,
    decoder_inputs: SingletonInputs | None,
    decoder_start_token_id: int,
) -> EncoderDecoderInputs:
    enc_inputs = _validate_enc_inputs(encoder_inputs)

    if decoder_inputs is None:
        dec_inputs: DecoderInputs = enc_inputs
    else:
        dec_inputs = _validate_dec_inputs(decoder_inputs)

    enc_inputs_new: EncoderInputs
    dec_inputs_new: DecoderInputs

    if enc_inputs["type"] == "multimodal":
        from vllm.multimodal.inputs import mm_inputs

        enc_inputs_new = token_inputs(
            enc_inputs["encoder_prompt_token_ids"],
            prompt=enc_inputs.get("encoder_prompt"),
        )
        dec_inputs_new = mm_inputs(
            prompt_token_ids=dec_inputs["prompt_token_ids"],
            prompt=dec_inputs.get("prompt"),
            mm_kwargs=enc_inputs["mm_kwargs"],
            mm_hashes=enc_inputs["mm_hashes"],
            mm_placeholders=enc_inputs["mm_placeholders"],
        )
    elif enc_inputs["type"] == "token":
        enc_inputs_new = token_inputs(prompt_token_ids=[])
        dec_inputs_new = dec_inputs
    else:
        assert_never(enc_inputs)

    dec_inputs_new["prompt_token_ids"] = _prepare_decoder_input_ids_for_generation(
        dec_inputs_new["prompt_token_ids"],
        decoder_start_token_id,
    )

    if cache_salt := enc_inputs.get("cache_salt"):
        dec_inputs_new["cache_salt"] = cache_salt

    return EncoderDecoderInputs(
        type="enc_dec",
        encoder_prompt=enc_inputs_new,
        decoder_prompt=dec_inputs_new,
    )
