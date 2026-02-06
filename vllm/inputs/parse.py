# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias, TypedDict

from typing_extensions import TypeIs

from vllm.utils import length_from_prompt_token_ids_or_embeds

from .data import (
    EmbedsPrompt,
    ExplicitEncoderDecoderPrompt,
    ProcessorInputs,
    PromptType,
    SingletonInputs,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
)

if TYPE_CHECKING:
    import torch


class ParsedStrPrompt(TypedDict):
    type: Literal["str"]
    content: str


class ParsedTextPrompt(TypedDict):
    type: Literal["text"]
    content: TextPrompt


class ParsedTokensPrompt(TypedDict):
    type: Literal["tokens"]
    content: TokensPrompt


class ParsedEmbedsPrompt(TypedDict):
    type: Literal["embeds"]
    content: EmbedsPrompt


ParsedSingletonPrompt: TypeAlias = (
    ParsedStrPrompt | ParsedTextPrompt | ParsedTokensPrompt | ParsedEmbedsPrompt
)


def parse_singleton_prompt(prompt: SingletonPrompt) -> ParsedSingletonPrompt:
    if isinstance(prompt, str):
        return ParsedStrPrompt(type="str", content=prompt)
    elif isinstance(prompt, dict):
        # Type ignores are because mypy does not correctly infer the TypedDicts
        # Pyright does succeed.
        if "prompt_embeds" in prompt:
            return ParsedEmbedsPrompt(type="embeds", content=prompt)  # type: ignore[typeddict-item]
        elif "prompt_token_ids" in prompt:
            return ParsedTokensPrompt(type="tokens", content=prompt)  # type: ignore[typeddict-item]
        elif "prompt" in prompt:
            return ParsedTextPrompt(type="text", content=prompt)
    raise TypeError(
        "inputs must be a string, TextPrompt, TokensPrompt, or EmbedsPrompt"
    )


def is_explicit_encoder_decoder_prompt(
    prompt: PromptType,
) -> TypeIs[ExplicitEncoderDecoderPrompt]:
    return isinstance(prompt, dict) and "encoder_prompt" in prompt


def split_enc_dec_prompt(
    prompt: PromptType,
) -> tuple[SingletonPrompt, SingletonPrompt | None]:
    if isinstance(prompt, str):
        return prompt, None

    if "encoder_prompt" in prompt and "decoder_prompt" in prompt:
        # NOTE: This passes pyright but not mypy
        return (
            prompt["encoder_prompt"],  # type: ignore[typeddict-item]
            prompt["decoder_prompt"],  # type: ignore[typeddict-item]
        )

    return prompt, None


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


def get_prompt_components(prompt: PromptType) -> PromptComponents:
    if isinstance(prompt, str):
        return PromptComponents(text=prompt)

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
