# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native Ming frontend must preserve the released ASR request contract."""

import os
from io import BytesIO

import numpy as np
import pytest
import torch
from starlette.datastructures import UploadFile
from transformers import AutoProcessor

from vllm.config import ModelConfig
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.model_executor.models.ming_audio import (
    MingAudioProjector,
    get_audio_token_count,
)
from vllm.model_executor.models.ming_v3_asr import (
    ASR_INSTRUCTION,
    MingASRStreamingPostProcessor,
    MingV3ASRForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.transformers_utils.configs.ming_v3_asr import MingAudioConfig

from .test_common import _assert_inputs_equal

MODEL = os.environ.get("VLLM_TEST_MING_ASR_MODEL", "inclusionAI/Ming-3.0-asr-tiny")


@pytest.fixture(scope="module")
def asr_processors():
    config = ModelConfig(
        MODEL, dtype="bfloat16", mm_processor_cache_gb=1, trust_remote_code=True
    )
    return (
        config,
        MULTIMODAL_REGISTRY.create_processor(config),
        AutoProcessor.from_pretrained(MODEL, trust_remote_code=True),
    )


@pytest.mark.parametrize(
    "samples,language,frames",
    [
        (201, None, 1),
        (321, None, 3000),
        (16001, "en", 3000),
        (480001, "zh", 3000),
        (480161, "en", 1001),
        (960000, "zh", 6000),
        (960161, "zh", 6000),
    ],
)
def test_transcription_request_matches_native(
    asr_processors, samples, language, frames
):
    """Compare features and token IDs, including language seed placement and tails."""
    config, processor, native = asr_processors
    audio = np.random.RandomState(42).uniform(-0.5, 0.5, samples).astype(np.float32)
    prompt = ASR_INSTRUCTION + "This is an audio about 同囚."
    request = TranscriptionRequest(
        file=UploadFile(BytesIO()),
        prompt=prompt,
        language=language,
        vllm_xargs={"audio_chunk_frames": frames},
    )
    stt_params = request.build_stt_params(
        audio,
        MingV3ASRForConditionalGeneration.get_speech_to_text_config(
            config, "transcribe"
        ),
        config,
        "transcribe",
    )
    inputs = MingV3ASRForConditionalGeneration.get_generation_prompt(stt_params)
    actual = processor(
        inputs["prompt_token_ids"],
        mm_items=processor.info.parse_mm_data(inputs["multi_modal_data"]),
        hf_processor_mm_kwargs=inputs["mm_processor_kwargs"],
    )
    expected = native.apply_transcription_request(
        (audio, 16000),
        prompt=prompt,
        language_hint=language,
        audio_chunk_frames=frames,
    )
    assert actual["prompt_token_ids"] == expected.input_ids[0].tolist()
    data = actual["mm_kwargs"].get_data()
    torch.testing.assert_close(
        data["audio_feats"][0],
        expected.audio_feats[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "kernels,strides,paddings", [([3], [2], [1]), ([5, 3], [3, 2], [2, 0])]
)
def test_audio_token_count_matches_chunked_convolutions(kernels, strides, paddings):
    """Count each chunk's rounding separately, including a nonempty tail."""
    config = MingAudioConfig(
        n_audio_state=4,
        downsampler_kernel_size_list=kernels,
        downsampler_stride_list=strides,
        downsampler_padding_list=paddings,
    )
    encoder_conv = torch.nn.Conv1d(1, config.n_audio_state, 3, stride=2, padding=1)
    projector = MingAudioProjector(config, hidden_size=4)
    num_frames, chunk_frames = 3103, 1001
    with torch.inference_mode():
        actual = sum(
            projector(encoder_conv(chunk).transpose(1, 2)).shape[1]
            for chunk in torch.zeros(1, 1, num_frames).split(chunk_frames, dim=-1)
        )
    assert get_audio_token_count(config, num_frames, chunk_frames) == actual


@pytest.mark.parametrize("scoped", [False, True])
def test_chunk_policy_has_separate_cache_entries(asr_processors, scoped):
    """The same waveform encoded with different chunks must not share cached data."""
    config, processor, _ = asr_processors
    cache = MultiModalProcessorOnlyCache(config)
    audio = np.random.RandomState(1).randn(480161).astype(np.float32)
    results = []
    for frames in (3000, 6000, 3000):
        options: dict[str, object] = {"audio_chunk_frames": frames}
        if scoped:
            options = {"audio_kwargs": options}
        kwargs = dict(
            mm_items=processor.info.parse_mm_data({"audio": audio}),
            hf_processor_mm_kwargs=options,
        )
        result = processor("<audio><audioPatch></audio>", **kwargs, cache=cache)
        _assert_inputs_equal(result, processor("<audio><audioPatch></audio>", **kwargs))
        results.append(result)
    assert results[0]["mm_hashes"] != results[1]["mm_hashes"]
    _assert_inputs_equal(results[0], results[2])
    assert config.hf_config.audio_config.chunk_frames == 3000
    for result, frames in zip(results, (3000, 6000, 3000)):
        data = result["mm_kwargs"].get_data()
        assert torch.as_tensor(data["audio_chunk_frames"]).item() == frames


@pytest.mark.parametrize(
    "options,samples,expected_frames",
    [
        ({}, 321, 6000),
        ({"audio_chunk_frames": 3000}, 321, 3000),
        ({"audio_kwargs": {"audio_chunk_frames": 4500}}, 321, 4500),
        (
            {"audio_kwargs": {"audio_chunk_frames": None, "context_chunks": True}},
            480161,
            6000,
        ),
    ],
)
def test_processor_defaults_and_request_audio_options(
    options, samples, expected_frames
):
    """Configured defaults and both request encodings must select the chunk policy."""
    config = ModelConfig(
        MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        mm_processor_kwargs={"audio_kwargs": {"audio_chunk_frames": 6000}},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    actual = processor(
        "<audio><audioPatch></audio>",
        mm_items=processor.info.parse_mm_data(
            {"audio": np.zeros(samples, dtype=np.float32)}
        ),
        hf_processor_mm_kwargs=options,
    )
    frames = actual["mm_kwargs"].get_data()["audio_chunk_frames"]
    assert torch.as_tensor(frames).item() == expected_frames


@pytest.mark.parametrize(
    "options,max_model_len",
    [({"audio_chunk_frames": 1}, 512), ({"context_chunks": True}, 2048)],
)
def test_dummy_audio_respects_processor_defaults(options, max_model_len):
    """Configured chunking must keep profiling within token and duration limits."""
    config = ModelConfig(
        MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        mm_processor_kwargs={"audio_kwargs": options},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    inputs = processor.get_dummy_mm_inputs({"audio": 1})
    assert len(inputs["prompt_token_ids"]) == max_model_len


@pytest.mark.parametrize(
    "chunks,expected",
    [
        (["Chi", "nese", "\t", " 你好", " 世界 "], "你好 世界"),
        (["same ", "way ", "you did"], "same way you did"),
        (["Eng", "lish"], ""),
        (["Canton", "ese", "\t", "你好"], "你好"),
        (["你", "好"], "你好"),
    ],
)
def test_streaming_envelope_matches_final_text(chunks, expected):
    processor = MingASRStreamingPostProcessor()
    deltas = [processor.process_delta(chunk, False) for chunk in chunks]
    deltas.append(processor.process_delta("", True))
    assert "".join(deltas) == expected
    assert (
        MingV3ASRForConditionalGeneration.post_process_output("".join(chunks))
        == expected
    )
