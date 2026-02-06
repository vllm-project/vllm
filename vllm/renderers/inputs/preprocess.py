"""
Schemas and utilites for preprocessing inputs.
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, TypedDict, overload

from vllm.inputs import (
    EmbedsPrompt,
    PromptType,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
)
from vllm.utils.collection_utils import is_list_of

if TYPE_CHECKING:
    import torch

    from vllm.config import ModelConfig
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


@overload
def prompt_to_seq(
    prompt_or_prompts: SingletonPrompt | Sequence[SingletonPrompt],
) -> Sequence[SingletonPrompt]: ...


@overload
def prompt_to_seq(  # type: ignore[misc]
    prompt_or_prompts: PromptType | Sequence[PromptType],
) -> Sequence[PromptType]: ...


def prompt_to_seq(
    prompt_or_prompts: PromptType | bytes | Sequence[PromptType | bytes],
) -> Sequence[PromptType | bytes]:
    if isinstance(prompt_or_prompts, (dict, str, bytes)) or (
        len(prompt_or_prompts) > 0 and is_list_of(prompt_or_prompts, int)
    ):
        return [prompt_or_prompts]  # type: ignore[list-item]

    return prompt_or_prompts  # type: ignore[return-value]


def conversation_to_seq(
    conversation_or_conversations: Sequence["ChatCompletionMessageParam"]
    | Sequence[list["ChatCompletionMessageParam"]],
) -> Sequence[list["ChatCompletionMessageParam"]]:
    if len(conversation_or_conversations) > 0 and is_list_of(
        conversation_or_conversations, dict
    ):
        return [conversation_or_conversations]  # type: ignore[list-item]

    return conversation_or_conversations  # type: ignore[return-value]


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


DictPrompt: TypeAlias = DecoderOnlyDictPrompt | EncoderDecoderDictPrompt
"""
A [`PromptType`][vllm.inputs.data.PromptType]
that has been normalized to a dictionary.
"""


def parse_dec_only_prompt(prompt: PromptType | DictPrompt) -> DecoderOnlyDictPrompt:
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
    prompt: PromptType | DictPrompt,
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


def parse_model_prompt(model_config: "ModelConfig", prompt: PromptType):
    if model_config.is_encoder_decoder:
        return parse_enc_dec_prompt(prompt)

    return parse_dec_only_prompt(prompt)


class PromptComponents(NamedTuple):
    text: str | None = None
    token_ids: list[int] | None = None
    embeds: "torch.Tensor | None" = None


def get_prompt_components(prompt: "PromptType | DictPrompt") -> PromptComponents:
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
