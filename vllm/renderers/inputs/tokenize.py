"""Schemas and utilities for tokenization inputs."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeAlias, TypedDict

from vllm.inputs import EmbedsPrompt, TokensPrompt

DecoderOnlyTokPrompt: TypeAlias = TokensPrompt | EmbedsPrompt
"""
A [`DecoderOnlyDictPrompt`][vllm.renderers.inputs.preprocess.DecoderOnlyDictPrompt]
that has been tokenized.
"""


EncoderTokPrompt: TypeAlias = TokensPrompt
"""
A [`EncoderDictPrompt`][vllm.renderers.inputs.preprocess.EncoderDictPrompt]
that has been tokenized.
"""


DecoderTokPrompt: TypeAlias = TokensPrompt
"""
A [`DecoderDictPrompt`][vllm.renderers.inputs.preprocess.DecoderDictPrompt]
that has been tokenized.
"""


class EncoderDecoderTokPrompt(TypedDict):
    """A
    [`EncoderDecoderDictPrompt`][vllm.renderers.inputs.preprocess.EncoderDecoderDictPrompt]
    that has been tokenized.
    """

    encoder_prompt: EncoderTokPrompt

    decoder_prompt: DecoderTokPrompt | None


SingletonTokPrompt: TypeAlias = (
    DecoderOnlyTokPrompt | EncoderTokPrompt | DecoderTokPrompt
)
"""
A [`SingletonDictPrompt`][vllm.renderers.inputs.preprocess.SingletonDictPrompt]
that has been tokenized.
"""


TokPrompt: TypeAlias = DecoderOnlyTokPrompt | EncoderDecoderTokPrompt
"""
A [`DictPrompt`][vllm.renderers.inputs.preprocess.DictPrompt]
that has been tokenized.
"""


def get_detokenization_tokens(prompt: TokensPrompt) -> list[int]:
    """Consume the special-token mask and return IDs for prompt detokenization."""
    token_ids = prompt["prompt_token_ids"]
    special_tokens_mask = prompt.pop("prompt_special_tokens_mask", None)
    if special_tokens_mask is None:
        return token_ids
    # TokenizeParams truncates both token_ids and the mask, but pads only token_ids.
    # IDs beyond the mask are therefore padding tokens.
    return [
        token_id
        for index, token_id in enumerate(token_ids)
        if index >= len(special_tokens_mask) or not special_tokens_mask[index]
    ]
