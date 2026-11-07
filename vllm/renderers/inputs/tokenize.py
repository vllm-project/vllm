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


def get_truncation_indices(
    truncation: slice, special_tokens_mask: list[int]
) -> list[int] | None:
    """The indices of the prompt tokens to keep so that `truncation` spares the
    special tokens.

    `special_tokens_mask` flags the tokens that the tokenizer added, such as
    `[CLS]` and `[SEP]`. These are all kept and only the text is truncated, as
    the tokenizer's own truncation does.

    `None` means `truncation` applies as it is: there are no special tokens to
    keep, or no room for them.
    """
    length = len(special_tokens_mask)
    special = [i for i in range(length) if special_tokens_mask[i]]
    text = [i for i in range(length) if not special_tokens_mask[i]]
    start, stop, _ = truncation.indices(length)
    num_text = stop - start - len(special)
    if not special or num_text < 0:
        return None

    # Only the text is truncated, from the same side as `truncation`.
    text = text[:num_text] if start == 0 else text[len(text) - num_text :]
    return sorted(special + text)
