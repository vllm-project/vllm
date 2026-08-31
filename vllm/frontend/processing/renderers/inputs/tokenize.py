"""
Schemas and utilities for tokenization inputs.
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeAlias, TypedDict

from vllm.frontend.processing.inputs import EmbedsPrompt, TokensPrompt

DecoderOnlyTokPrompt: TypeAlias = TokensPrompt | EmbedsPrompt
"""
A [`DecoderOnlyDictPrompt`][]
that has been tokenized.
"""


EncoderTokPrompt: TypeAlias = TokensPrompt
"""
A [`EncoderDictPrompt`][]
that has been tokenized.
"""


DecoderTokPrompt: TypeAlias = TokensPrompt
"""
A [`DecoderDictPrompt`][]
that has been tokenized.
"""


class EncoderDecoderTokPrompt(TypedDict):
    """
    A
    [`EncoderDecoderDictPrompt`][vllm.frontend.processing.renderers.inputs.preprocess.EncoderDecoderDictPrompt]
    that has been tokenized.
    """

    encoder_prompt: EncoderTokPrompt

    decoder_prompt: DecoderTokPrompt | None


SingletonTokPrompt: TypeAlias = (
    DecoderOnlyTokPrompt | EncoderTokPrompt | DecoderTokPrompt
)
"""
A [`SingletonDictPrompt`][]
that has been tokenized.
"""


TokPrompt: TypeAlias = DecoderOnlyTokPrompt | EncoderDecoderTokPrompt
"""
A [`DictPrompt`][vllm.frontend.processing.renderers.inputs.preprocess.DictPrompt]
that has been tokenized.
"""
