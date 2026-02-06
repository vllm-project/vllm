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
from vllm.utils import length_from_prompt_token_ids_or_embeds
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
that has been standardized into a dictionary.
"""


EncoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt
"""
A [`EncoderPrompt`][vllm.inputs.data.EncoderPrompt]
that has been standardized into a dictionary.
"""


DecoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt
"""
A [`DecoderPrompt`][vllm.inputs.data.DecoderPrompt]
that has been standardized into a dictionary.
"""


class EncoderDecoderDictPrompt(TypedDict):
    """
    A [`EncoderDecoderPrompt`][vllm.inputs.data.EncoderDecoderPrompt]
    that has been standardized into a dictionary.
    """

    encoder_prompt: EncoderDictPrompt

    decoder_prompt: DecoderDictPrompt | None


SingletonDictPrompt: TypeAlias = (
    DecoderOnlyDictPrompt | EncoderDictPrompt | DecoderDictPrompt
)
"""
A [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt]
that has been standardized into a dictionary.
"""


DictPrompt: TypeAlias = DecoderOnlyDictPrompt | EncoderDecoderDictPrompt
"""
A [`PromptType`][vllm.inputs.data.PromptType]
that has been standardized into a dictionary.
"""


def parse_dec_only_prompt(prompt: object) -> DecoderOnlyDictPrompt:
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

        if (
            "prompt" in prompt
            or "prompt_token_ids" in prompt
            or "prompt_embeds" in prompt
        ):
            return prompt  # type: ignore[return-value]

        raise TypeError("Prompt dictionary must contain text, tokens, or embeddings")

    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def _parse_enc_prompt(prompt: object) -> EncoderDictPrompt:
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)

    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")

        return TokensPrompt(prompt_token_ids=prompt)

    if isinstance(prompt, dict):
        if "prompt_embeds" in prompt:
            raise TypeError("Cannot pass embeddings prompt to encoder-decoder models")

        if "prompt" in prompt or "prompt_token_ids" in prompt:
            return prompt  # type: ignore[return-value]

        raise TypeError("Prompt dictionary must contain text or tokens")

    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def _parse_dec_prompt(prompt: object) -> DecoderDictPrompt:
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

        if "prompt" in prompt or "prompt_token_ids" in prompt:
            return prompt  # type: ignore[return-value]

        raise TypeError("Prompt dictionary must contain text or tokens")

    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def parse_enc_dec_prompt(prompt: object) -> EncoderDecoderDictPrompt:
    """
    Parse a prompt for an encoder-decoder model and normalize it to a dictionary.
    """
    if isinstance(prompt, dict) and "encoder_prompt" in prompt:
        enc_prompt: object = prompt["encoder_prompt"]  # type: ignore[typeddict-item]
        dec_prompt: object | None = prompt["decoder_prompt"]  # type: ignore[typeddict-item]
    else:
        enc_prompt = prompt
        dec_prompt = None

    return EncoderDecoderDictPrompt(
        encoder_prompt=_parse_enc_prompt(enc_prompt),
        decoder_prompt=None if dec_prompt is None else _parse_dec_prompt(dec_prompt),
    )


def parse_model_prompt(model_config: "ModelConfig", prompt: object):
    if model_config.is_encoder_decoder:
        return parse_enc_dec_prompt(prompt)

    return parse_dec_only_prompt(prompt)


class PromptComponents(NamedTuple):
    text: str | None = None
    token_ids: list[int] | None = None
    embeds: "torch.Tensor | None" = None


def extract_prompt_components(
    model_config: "ModelConfig",
    prompt: object,
) -> PromptComponents:
    target_prompt = (
        parse_enc_dec_prompt(prompt)["encoder_prompt"]
        if model_config.is_encoder_decoder
        else parse_dec_only_prompt(prompt)
    )

    return PromptComponents(
        text=target_prompt.get("prompt"),
        token_ids=target_prompt.get("prompt_token_ids"),  # type: ignore[arg-type]
        embeds=target_prompt.get("prompt_embeds"),
    )


def extract_prompt_len(model_config: "ModelConfig", prompt: object):
    target_prompt = (
        parse_enc_dec_prompt(prompt)["encoder_prompt"]
        if model_config.is_encoder_decoder
        else parse_dec_only_prompt(prompt)
    )

    return length_from_prompt_token_ids_or_embeds(
        target_prompt.get("prompt_token_ids"),  # type: ignore[arg-type]
        target_prompt.get("prompt_embeds"),
    )
