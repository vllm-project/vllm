"""Schema and utilities for inputs to the engine client (`LLMEngine`/`AsyncLLM`)."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

from typing_extensions import NotRequired, TypedDict, assert_never

if TYPE_CHECKING:
    import torch

    from vllm.multimodal.inputs import MultiModalKwargsOptionalItems, PlaceholderRange


class _InputOptions(TypedDict):
    """
    Additional options available to all
    [`SingletonInput`][vllm.inputs.engine.SingletonInput] types.
    """

    arrival_time: NotRequired[float]
    """The time when the input was received (before rendering)."""

    cache_salt: NotRequired[str]
    """Optional cache salt to be used for prefix caching."""


class TokensInput(_InputOptions):
    """Represents token-based input to the engine."""

    type: Literal["token"]
    """The type of input."""

    prompt_token_ids: list[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""


def tokens_input(
    prompt_token_ids: list[int],
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> TokensInput:
    """
    Construct [`TokensInput`][vllm.inputs.engine.TokensInput]
    from optional values.
    """
    inputs = TokensInput(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


class EmbedsInput(_InputOptions):
    """Represents embeddings-based input to the engine."""

    type: Literal["embeds"]
    """The type of input."""

    prompt_embeds: "torch.Tensor"
    """The embeddings of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""


def embeds_input(
    prompt_embeds: "torch.Tensor",
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> EmbedsInput:
    """
    Construct [`EmbedsInput`][vllm.inputs.engine.EmbedsInput]
    from optional values.
    """
    inputs = EmbedsInput(type="embeds", prompt_embeds=prompt_embeds)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


MultiModalHashes: TypeAlias = Mapping[str, list[str]]
"""
A dictionary containing per-item hashes for each modality.
"""


MultiModalPlaceholders: TypeAlias = Mapping[str, Sequence["PlaceholderRange"]]
"""
A dictionary containing per-item placeholder ranges for each modality.
"""


class MultiModalInput(_InputOptions):
    """Represents multi-modal input to the engine."""

    type: Literal["multimodal"]
    """The type of input."""

    prompt_token_ids: list[int]
    """The processed token IDs which includes placeholder tokens."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    mm_kwargs: "MultiModalKwargsOptionalItems"
    """Keyword arguments to be directly passed to the model after batching."""

    mm_hashes: MultiModalHashes
    """The hashes of the multi-modal data."""

    mm_placeholders: MultiModalPlaceholders
    """
    For each modality, information about the placeholder tokens in
    `prompt_token_ids`.
    """


def mm_input(
    prompt_token_ids: list[int],
    mm_kwargs: "MultiModalKwargsOptionalItems",
    mm_hashes: MultiModalHashes,
    mm_placeholders: MultiModalPlaceholders,
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> MultiModalInput:
    inputs = MultiModalInput(
        type="multimodal",
        prompt_token_ids=prompt_token_ids,
        mm_kwargs=mm_kwargs,
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholders,
    )

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


class MultiModalEncDecInput(MultiModalInput):
    """
    Represents multi-modal input to the engine for encoder-decoder models.

    Note:
        Even text-only encoder-decoder models are currently implemented
        as multi-modal models for convenience.
        (Example: https://github.com/vllm-project/bart-plugin)
    """

    encoder_prompt_token_ids: list[int]
    """The processed token IDs of the encoder prompt."""

    encoder_prompt: NotRequired[str]
    """The prompt text corresponding to the encoder token IDs, if available."""


def mm_enc_dec_input(
    encoder_inputs: MultiModalInput,
    decoder_prompt_token_ids: list[int],
    *,
    decoder_prompt: str | None = None,
) -> MultiModalEncDecInput:
    inputs = MultiModalEncDecInput(
        type="multimodal",
        prompt_token_ids=decoder_prompt_token_ids,
        encoder_prompt_token_ids=encoder_inputs["prompt_token_ids"],
        mm_kwargs=encoder_inputs["mm_kwargs"],
        mm_hashes=encoder_inputs["mm_hashes"],
        mm_placeholders=encoder_inputs["mm_placeholders"],
    )

    if decoder_prompt is not None:
        inputs["prompt"] = decoder_prompt
    if "prompt" in encoder_inputs:
        inputs["encoder_prompt"] = encoder_inputs["prompt"]
    if "cache_salt" in encoder_inputs:
        inputs["cache_salt"] = encoder_inputs["cache_salt"]

    return inputs


DecoderOnlyEngineInput: TypeAlias = TokensInput | EmbedsInput | MultiModalInput
"""
A rendered [`DecoderOnlyPrompt`][vllm.inputs.llm.DecoderOnlyPrompt]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


EncoderInput: TypeAlias = TokensInput | MultiModalEncDecInput
"""
A rendered [`EncoderPrompt`][vllm.inputs.llm.EncoderPrompt]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


DecoderEngineInput: TypeAlias = TokensInput | MultiModalInput
"""
A rendered [`DecoderPrompt`][vllm.inputs.llm.DecoderPrompt]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


class EncoderDecoderInput(TypedDict):
    """
    A rendered [`EncoderDecoderPrompt`][vllm.inputs.llm.EncoderDecoderPrompt]
    which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
    """

    type: Literal["enc_dec"]

    encoder_prompt: EncoderInput
    """The inputs for the encoder portion."""

    decoder_prompt: DecoderEngineInput
    """The inputs for the decoder portion."""

    arrival_time: NotRequired[float]
    """The time when the input was received (before rendering)."""


SingletonInput: TypeAlias = DecoderOnlyEngineInput | MultiModalEncDecInput
"""
A rendered [`SingletonPrompt`][vllm.inputs.llm.SingletonPrompt]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


EngineInput: TypeAlias = DecoderOnlyEngineInput | EncoderDecoderInput
"""
A rendered [`PromptType`][vllm.inputs.llm.PromptType]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


def _validate_enc_input(enc_input: SingletonInput) -> EncoderInput:
    if enc_input["type"] == "embeds":
        raise ValueError(
            "Embedding inputs are not supported for encoder-decoder models"
        )

    if (
        enc_input["type"] == "multimodal"
        and "encoder_prompt_token_ids" not in enc_input
    ):
        raise RuntimeError(
            "You should register an encoder-decoder multi-modal processor "
            "for encoder-decoder models."
        )

    return enc_input  # type: ignore[return-value]


def _validate_dec_input(dec_input: SingletonInput) -> DecoderEngineInput:
    if dec_input["type"] == "embeds":
        raise ValueError(
            "Embedding inputs are not supported for encoder-decoder models"
        )

    return dec_input


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


def build_enc_dec_input(
    encoder_input: SingletonInput,
    decoder_input: SingletonInput | None,
    decoder_start_token_id: int,
) -> EncoderDecoderInput:
    enc_input = _validate_enc_input(encoder_input)

    if decoder_input is None:
        dec_input: DecoderEngineInput = enc_input
    else:
        dec_input = _validate_dec_input(decoder_input)

    enc_input_new: EncoderInput
    dec_input_new: DecoderEngineInput

    if enc_input["type"] == "multimodal":
        enc_input_new = tokens_input(
            enc_input["encoder_prompt_token_ids"],
            prompt=enc_input.get("encoder_prompt"),
        )
        dec_input_new = mm_input(
            prompt_token_ids=dec_input["prompt_token_ids"],
            prompt=dec_input.get("prompt"),
            mm_kwargs=enc_input["mm_kwargs"],
            mm_hashes=enc_input["mm_hashes"],
            mm_placeholders=enc_input["mm_placeholders"],
        )
    elif enc_input["type"] == "token":
        enc_input_new = tokens_input(prompt_token_ids=[])
        dec_input_new = dec_input
    else:
        assert_never(enc_input)

    dec_input_new["prompt_token_ids"] = _prepare_decoder_input_ids_for_generation(
        dec_input_new["prompt_token_ids"],
        decoder_start_token_id,
    )

    if cache_salt := enc_input.get("cache_salt"):
        dec_input_new["cache_salt"] = cache_salt

    return EncoderDecoderInput(
        type="enc_dec",
        encoder_prompt=enc_input_new,
        decoder_prompt=dec_input_new,
    )


def split_enc_dec_input(
    inputs: EngineInput,
) -> tuple[SingletonInput | None, SingletonInput]:
    if inputs["type"] == "enc_dec":
        return inputs["encoder_prompt"], inputs["decoder_prompt"]

    return None, inputs
