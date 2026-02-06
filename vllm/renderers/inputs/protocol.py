# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeAlias, TypedDict

from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt

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
