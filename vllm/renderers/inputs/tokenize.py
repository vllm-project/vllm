"""
Schemas and utilities for tokenization inputs.
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypedDict

from vllm.inputs import TokensPrompt


class EncoderDecoderTokPrompt(TypedDict):
    """
    A
    [`EncoderDecoderDictPrompt`][vllm.renderers.inputs.preprocess.EncoderDecoderDictPrompt]
    that has been tokenized.
    """

    encoder_prompt: TokensPrompt

    decoder_prompt: TokensPrompt | None
