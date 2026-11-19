# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Whisper's multimodal preprocessing."""

import asyncio

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.model_executor.models.whisper import (
    WhisperForConditionalGeneration,
    WhisperProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.renderers.hf import HfRenderer
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.tokenizers.registry import cached_tokenizer_from_config

from ...utils import build_model_context

SOT_SEQUENCE = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"


@pytest.mark.parametrize("model_id", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("audio_duration_s", [5, 30, 35])
def test_audio_features_fit_encoder_window(
    model_id: str,
    audio_duration_s: int,
) -> None:
    """The encoder only has positions for one 30 s window, so a longer clip
    has to be cut to that window like the HF feature extractor does by default.
    """
    ctx = build_model_context(model_id, limit_mm_per_prompt={"audio": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert isinstance(processor.info, WhisperProcessingInfo)
    feature_extractor = processor.info.get_feature_extractor()

    sampling_rate = feature_extractor.sampling_rate
    rng = np.random.RandomState(0)
    audio = rng.rand(sampling_rate * audio_duration_s).astype(np.float32)

    processed_inputs = processor(
        "<|startoftranscript|>",
        mm_items=processor.info.parse_mm_data({"audio": [(audio, sampling_rate)]}),
        hf_processor_mm_kwargs={},
    )

    input_features = processed_inputs["mm_kwargs"].get_data()["input_features"]
    assert isinstance(input_features, torch.Tensor)
    assert input_features.shape[-1] == feature_extractor.nb_max_frames


def _render_decoder_prompt_token_ids(
    model_id: str,
    prompt: object,
    use_async: bool,
) -> list[int]:
    model_config = ModelConfig(model=model_id, max_model_len=448)
    renderer = HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )
    parsed_prompt = parse_model_prompt(model_config, prompt)

    if use_async:
        [engine_input] = asyncio.run(renderer.render_cmpl_async([parsed_prompt]))
    else:
        [engine_input] = renderer.render_cmpl([parsed_prompt])

    assert engine_input["type"] == "enc_dec"
    return engine_input["decoder_prompt"]["prompt_token_ids"]


def _audio() -> tuple[np.ndarray, int]:
    return np.zeros(16_000, dtype=np.float32), 16_000


def _with_audio(decoder_prompt: object | None = None) -> dict:
    encoder_prompt = {"prompt": "", "multi_modal_data": {"audio": _audio()}}
    if decoder_prompt is None:
        return encoder_prompt
    return {"encoder_prompt": encoder_prompt, "decoder_prompt": decoder_prompt}


@pytest.mark.parametrize("model_id", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("decoder_prompt", "expected_decoder_text"),
    [
        pytest.param(None, "<|startoftranscript|>", id="implicit-empty"),
        pytest.param(
            {"prompt_token_ids": []}, "<|startoftranscript|>", id="explicit-empty"
        ),
        pytest.param(
            "<|en|><|transcribe|><|notimestamps|>",
            SOT_SEQUENCE,
            id="explicit-without-sot",
        ),
        pytest.param(SOT_SEQUENCE, SOT_SEQUENCE, id="explicit-with-sot"),
    ],
)
def test_decoder_start_token(
    model_id: str,
    use_async: bool,
    decoder_prompt: object | None,
    expected_decoder_text: str,
) -> None:
    """A decoder prompt that is not prompted with `<|startofprev|>` gets
    `<|startoftranscript|>` prepended when it lacks it."""
    decoder_prompt_token_ids = _render_decoder_prompt_token_ids(
        model_id, _with_audio(decoder_prompt), use_async
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    expected = tokenizer.encode(expected_decoder_text, add_special_tokens=False)
    assert decoder_prompt_token_ids == expected


@pytest.mark.parametrize("model_id", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("use_async", [False, True])
def test_generation_prompt_decoder_layout(model_id: str, use_async: bool) -> None:
    """A transcription prompt reaches the decoder in the reference layout:
    `get_prompt_ids(prompt)` followed by the SOT sequence, nothing in front."""
    request_prompt = "Hello. See you on Wednesday."
    prompt = WhisperForConditionalGeneration.get_generation_prompt(
        SpeechToTextParams(
            audio=_audio()[0],
            stt_config=SpeechToTextConfig(),
            model_config=ModelConfig(model=model_id, max_model_len=448),
            language="en",
            task_type="transcribe",
            request_prompt=request_prompt,
        )
    )

    decoder_prompt_token_ids = _render_decoder_prompt_token_ids(
        model_id, prompt, use_async
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    expected = tokenizer.get_prompt_ids(request_prompt).tolist() + tokenizer.encode(
        SOT_SEQUENCE, add_special_tokens=False
    )
    assert decoder_prompt_token_ids == expected
