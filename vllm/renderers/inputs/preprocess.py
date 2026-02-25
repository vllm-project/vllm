"""
Schemas and utilites for preprocessing inputs.
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, TypedDict, overload
import numpy as np
import math
from vllm.inputs import (
    EmbedsPrompt,
    ExplicitEncoderDecoderPrompt,
    ProcessorInputs,
    PromptType,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.collection_utils import is_list_of
from vllm.transformers_utils.processor import get_feature_extractor
if TYPE_CHECKING:
    import torch

    from vllm.config import ModelConfig
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


@overload
def prompt_to_seq(
    prompt_or_prompts: SingletonPrompt | bytes | Sequence[SingletonPrompt | bytes],
) -> Sequence[SingletonPrompt]: ...


@overload
def prompt_to_seq(  # type: ignore[misc]
    prompt_or_prompts: ExplicitEncoderDecoderPrompt
    | Sequence[ExplicitEncoderDecoderPrompt],
) -> Sequence[ExplicitEncoderDecoderPrompt]: ...


@overload
def prompt_to_seq(  # type: ignore[misc]
    prompt_or_prompts: PromptType | Sequence[PromptType],
) -> Sequence[PromptType]: ...


def prompt_to_seq(
    prompt_or_prompts: PromptType | bytes | Sequence[PromptType | bytes],
) -> Sequence[PromptType]:
    if isinstance(prompt_or_prompts, (dict, str, bytes)) or (
        len(prompt_or_prompts) > 0 and is_list_of(prompt_or_prompts, int)
    ):
        return [prompt_or_prompts]  # type: ignore[list-item]

    return prompt_or_prompts  # type: ignore[return-value]


def conversation_to_seq(
    conversation_or_conversations: list["ChatCompletionMessageParam"]
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


def parse_dec_only_prompt(prompt: PromptType | object) -> DecoderOnlyDictPrompt:
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


def _parse_enc_prompt(prompt: PromptType | object) -> EncoderDictPrompt:
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


def _parse_dec_prompt(prompt: PromptType | object) -> DecoderDictPrompt:
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


def parse_enc_dec_prompt(prompt: PromptType | object) -> EncoderDecoderDictPrompt:
    """
    Parse a prompt for an encoder-decoder model and normalize it to a dictionary.
    """
    if isinstance(prompt, dict) and "encoder_prompt" in prompt:
        enc_prompt = prompt["encoder_prompt"]  # type: ignore[typeddict-item]
        dec_prompt = prompt["decoder_prompt"]  # type: ignore[typeddict-item]
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


def extract_target_prompt(model_config: "ModelConfig", prompt: object):
    return (
        parse_enc_dec_prompt(prompt)["encoder_prompt"]
        if model_config.is_encoder_decoder
        else parse_dec_only_prompt(prompt)
    )


def extract_prompt_components(
    model_config: "ModelConfig",
    prompt: PromptType | ProcessorInputs,
) -> PromptComponents:
    target_prompt = extract_target_prompt(model_config, prompt)

    return PromptComponents(
        text=target_prompt.get("prompt"),
        token_ids=target_prompt.get("prompt_token_ids"),
        embeds=target_prompt.get("prompt_embeds"),
    )


def extract_prompt_len(
    model_config: "ModelConfig", prompt: PromptType | ProcessorInputs
):
    target_prompt = extract_target_prompt(model_config, prompt)

    return length_from_prompt_token_ids_or_embeds(
        target_prompt.get("prompt_token_ids"),
        target_prompt.get("prompt_embeds"),
    )


def chunk_audio_data(
    audio_data: np.ndarray, 
    sample_rate: int,
    model_name: str,
) -> list[np.ndarray]:
    
    processor = get_feature_extractor(model_name)
    MAX_AUDIO_LEN = processor.chunk_length 
    chunk_size = sample_rate * MAX_AUDIO_LEN
    overlap_size = sample_rate
    chunks = []
    i = 0
    while i < audio_data.shape[-1]:
        if i + chunk_size >= audio_data.shape[-1]:
            # handle last chunk
            chunks.append(audio_data[..., i:])
            break

        # Find the best split point in the overlap region
        search_start = i + chunk_size - overlap_size
        search_end = min(i + chunk_size, audio_data.shape[-1])
        split_point = _find_split_point(audio_data, search_start, search_end, sample_rate)
        # Extract chunk up to the split point
        chunks.append(audio_data[..., i:split_point])
        i = split_point

    return chunks

def _find_split_point( 
    wav: np.ndarray, 
    start_idx: int, 
    end_idx: int, 
    samplerate: int
) -> int:
    """Find the best point to split audio by
    looking for silence or low amplitude.
    Args:
        wav: Audio tensor [1, T]
        start_idx: Start index of search region
        end_idx: End index of search region
    Returns:
        Index of best splitting point
    """
    segment = wav[start_idx:end_idx]

    # Calculate RMS energy in small windows
    min_energy = math.inf
    quietest_idx = 0
    min_energy_window = samplerate // 10
    assert min_energy_window is not None
    for i in range(0, len(segment) - min_energy_window, min_energy_window):
        window = segment[i : i + min_energy_window]
        energy = (window**2).mean() ** 0.5
        if energy < min_energy:
            quietest_idx = i + start_idx
            min_energy = energy
    return quietest_idx