# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, TypedDict

from vllm.utils import length_from_prompt_token_ids_or_embeds

from .data import (
    EmbedsPrompt,
    ProcessorInputs,
    PromptType,
    SingletonInputs,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
)

if TYPE_CHECKING:
    import torch


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.utils.collection_utils import is_list_of

DecoderOnlyDictPrompt: TypeAlias = TextPrompt | TokensPrompt | EmbedsPrompt
"""
A [`DecoderOnlyPrompt`][vllm.inputs.data.DecoderOnlyPrompt]
that has been normalized to a dictionary.
"""


EncoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt
"""
A [`EncoderPrompt`][vllm.inputs.data.EncoderPrompt]
that has been normalized to a dictionary.
"""


DecoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt
"""
A [`DecoderPrompt`][vllm.inputs.data.DecoderPrompt]
that has been normalized to a dictionary.
"""


class EncoderDecoderDictPrompt(TypedDict):
    """
    A [`EncoderDecoderPrompt`][vllm.inputs.data.EncoderDecoderPrompt]
    that has been normalized to a dictionary.
    """

    encoder_prompt: EncoderDictPrompt

    decoder_prompt: DecoderDictPrompt | None


SingletonDictPrompt: TypeAlias = (
    DecoderOnlyDictPrompt | EncoderDictPrompt | DecoderDictPrompt
)
"""
A [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt]
that has been normalized to a dictionary.
"""


DictPromptType: TypeAlias = DecoderOnlyDictPrompt | EncoderDecoderDictPrompt
"""
A [`PromptType`][vllm.inputs.data.PromptType]
that has been normalized to a dictionary.
"""


def parse_dec_only_prompt(prompt: PromptType | DictPromptType) -> DecoderOnlyDictPrompt:
    """
    Parse a prompt for a decoder-only model and normalize it to a dictionary.
    """
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)

    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")

        return TokensPrompt(prompt_token_ids=prompt)

    if isinstance(prompt, dict):
        if "encoder_prompt" in prompt:
            raise TypeError("Cannot pass encoder-decoder prompt to decoder-only models")

        return prompt

    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def _parse_enc_prompt(prompt: SingletonPrompt) -> EncoderDictPrompt:
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)

    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")

        return TokensPrompt(prompt_token_ids=prompt)

    if isinstance(prompt, dict):
        if "prompt_embeds" in prompt:
            raise TypeError("Cannot pass embeddings prompt to encoder-decoder models")

        return prompt

    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def _parse_dec_prompt(prompt: SingletonPrompt) -> DecoderDictPrompt:
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)

    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")

        return TokensPrompt(prompt_token_ids=prompt)

    if isinstance(prompt, dict):
        if "prompt_embeds" in prompt:
            raise TypeError("Cannot pass embeddings prompt to encoder-decoder models")

        if (
            "multi_modal_data" in prompt
            or "mm_processor_kwargs" in prompt
            or "multi_modal_uuids" in prompt
        ):
            raise TypeError("Cannot pass multi-modal inputs to decoder prompt")

        return prompt

    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def parse_enc_dec_prompt(
    prompt: PromptType | DictPromptType,
) -> EncoderDecoderDictPrompt:
    """
    Parse a prompt for an encoder-decoder model and normalize it to a dictionary.
    """
    if isinstance(prompt, dict) and "encoder_prompt" in prompt:
        enc_prompt: SingletonPrompt = prompt["encoder_prompt"]  # type: ignore[typeddict-item]
        dec_prompt: SingletonPrompt | None = prompt["decoder_prompt"]  # type: ignore[typeddict-item]
    else:
        enc_prompt = prompt
        dec_prompt = None

    return EncoderDecoderDictPrompt(
        encoder_prompt=_parse_enc_prompt(enc_prompt),
        decoder_prompt=None if dec_prompt is None else _parse_dec_prompt(dec_prompt),
    )


def split_enc_dec_inputs(
    inputs: ProcessorInputs,
) -> tuple[SingletonInputs | None, SingletonInputs]:
    if "encoder" in inputs and "decoder" in inputs:
        # NOTE: This passes pyright but not mypy
        return (
            inputs["encoder"],  # type: ignore[typeddict-item]
            inputs["decoder"],  # type: ignore[typeddict-item]
        )

    return None, inputs


class PromptComponents(NamedTuple):
    text: str | None = None
    token_ids: list[int] | None = None
    embeds: "torch.Tensor | None" = None


def get_prompt_components(prompt: PromptType | DictPromptType) -> PromptComponents:
    # TODO: Remove the non-dict cases once we finish updating all APIs to use Renderer
    if isinstance(prompt, str):
        return PromptComponents(text=prompt)
    if isinstance(prompt, list):
        return PromptComponents(token_ids=prompt)

    if encoder_prompt := prompt.get("encoder_prompt"):
        return get_prompt_components(encoder_prompt)  # type: ignore[arg-type]

    return PromptComponents(
        text=prompt.get("prompt"),  # type: ignore[arg-type]
        token_ids=prompt.get("prompt_token_ids"),  # type: ignore[arg-type]
        embeds=prompt.get("prompt_embeds"),
    )


def get_prompt_len(prompt: TokensPrompt | EmbedsPrompt):
    return length_from_prompt_token_ids_or_embeds(
        prompt.get("prompt_token_ids"),  # type: ignore[arg-type]
        prompt.get("prompt_embeds"),  # type: ignore[arg-type]
    )
