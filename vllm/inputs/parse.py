# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias, TypedDict, cast

from typing_extensions import TypeIs

from vllm.utils.collection_utils import is_list_of

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


def parse_raw_prompts(
    prompt: str | list[str] | list[int] | list[list[int]],
) -> Sequence[TextPrompt] | Sequence[TokensPrompt]:
    if isinstance(prompt, str):
        # case 1: a string
        return [TextPrompt(prompt=prompt)]

    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")

        # case 2: array of strings
        if is_list_of(prompt, str):
            prompt = cast(list[str], prompt)
            return [TextPrompt(prompt=elem) for elem in prompt]

        # case 3: array of tokens
        if is_list_of(prompt, int):
            prompt = cast(list[int], prompt)
            return [TokensPrompt(prompt_token_ids=prompt)]

        # case 4: array of token arrays
        if is_list_of(prompt, list):
            first = prompt[0]
            if not isinstance(first, list):
                raise ValueError("prompt expected to be a list of lists")

            if len(first) == 0:
                raise ValueError("Please provide at least one prompt")

            # strict validation: every nested list must be list[int]
            if not all(is_list_of(elem, int) for elem in prompt):
                raise TypeError("Nested lists must contain only integers")

            prompt = cast(list[list[int]], prompt)
            return [TokensPrompt(prompt_token_ids=elem) for elem in prompt]

    raise TypeError(
        "prompt must be a string, array of strings, "
        "array of tokens, or array of token arrays"
    )


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
