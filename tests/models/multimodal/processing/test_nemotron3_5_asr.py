# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from transformers.models.nemotron_asr_streaming import (
    NemotronAsrStreamingFeatureExtractor as HFNemotronAsrStreamingFeatureExtractor,
)

from vllm.model_executor.models.nemotron3_5_asr import (
    Nemotron3_5AsrDummyInputsBuilder,
    Nemotron3_5AsrMultiModalProcessor,
    _get_subsampling_output_lengths,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalSharedField
from vllm.transformers_utils.processors import NemotronAsrStreamingFeatureExtractor

from ...utils import build_model_context


@pytest.fixture
def processor() -> Nemotron3_5AsrMultiModalProcessor:
    ctx = build_model_context(
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
        model_config_kwargs={"max_model_len": 512},
        limit_mm_per_prompt={"audio": 1},
    )
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


@pytest.mark.parametrize(
    ("num_samples", "valid_mel_frames", "expected_encoder_frames"),
    [(4040, 25, 4), (4160, 26, 5)],
)
@pytest.mark.parametrize(("language", "prompt_id"), [("en-US", 0), ("auto", 101)])
def test_nemotron_processor_builds_encoder_decoder_contract(
    processor: Nemotron3_5AsrMultiModalProcessor,
    num_samples: int,
    valid_mel_frames: int,
    expected_encoder_frames: int,
    language: str,
    prompt_id: int,
) -> None:
    mm_items = processor.info.parse_mm_data(
        {"audio": np.zeros(num_samples, dtype=np.float32)}
    )

    processed = processor(
        "",
        mm_items=mm_items,
        hf_processor_mm_kwargs={"language": language, "sampling_rate": 16000},
    )

    assert processed["prompt_token_ids"] == [13087]
    assert processed["encoder_prompt_token_ids"] == [0] * expected_encoder_frames
    assert processed["mm_placeholders"]["audio"][0].length == expected_encoder_frames

    audio_item = processed["mm_kwargs"]["audio"][0]
    assert audio_item["input_features"].data.shape == (
        valid_mel_frames + 1,
        128,
    )
    assert audio_item["attention_mask"].data.shape == (valid_mel_frames + 1,)
    assert audio_item["attention_mask"].data.sum().item() == valid_mel_frames
    assert not audio_item["attention_mask"].data[-1]
    assert torch.count_nonzero(audio_item["input_features"].data[-1]).item() == 0
    assert audio_item["prompt_ids"].data.item() == prompt_id
    assert audio_item["prompt_ids"].field.keep_on_cpu
    assert audio_item["num_lookahead_tokens"].data == 3
    assert isinstance(
        audio_item["num_lookahead_tokens"].field,
        MultiModalSharedField,
    )
    assert audio_item["num_lookahead_tokens"].field.keep_on_cpu


@pytest.mark.parametrize(
    (
        "max_encoder_frames",
        "max_model_len",
        "expected_audio_samples",
        "expected_encoder_frames",
    ),
    [(4, 4096, 3999, 4), (5000, 6000, 6398879, 5000), (5000, 512, 654239, 512)],
)
def test_nemotron_dummy_audio_is_bounded(
    processor: Nemotron3_5AsrMultiModalProcessor,
    max_encoder_frames: int,
    max_model_len: int,
    expected_audio_samples: int,
    expected_encoder_frames: int,
) -> None:
    info = processor.info
    info.ctx.model_config.hf_config.encoder_config.max_position_embeddings = (
        max_encoder_frames
    )
    info.ctx.model_config.max_model_len = max_model_len
    builder = Nemotron3_5AsrDummyInputsBuilder(info)

    mm_data = builder.get_dummy_mm_data(
        seq_len=4096,
        mm_counts={"audio": 1},
        mm_options={},
    )

    (audio,) = mm_data["audio"]
    feature_extractor_info = info.get_feature_extractor()
    assert len(audio) == expected_audio_samples

    feature_extractor = NemotronAsrStreamingFeatureExtractor(feature_size=128)
    features = feature_extractor(
        audio,
        sampling_rate=feature_extractor_info.sampling_rate,
        return_tensors="pt",
    )
    encoder_config = info.get_hf_config().encoder_config
    physical_frames = _get_subsampling_output_lengths(
        torch.tensor([features["input_features"].shape[1]]),
        subsampling_factor=encoder_config.subsampling_factor,
        subsampling_conv_kernel_size=encoder_config.subsampling_conv_kernel_size,
        subsampling_conv_stride=encoder_config.subsampling_conv_stride,
    )
    valid_frames = _get_subsampling_output_lengths(
        features["attention_mask"].sum(-1),
        subsampling_factor=encoder_config.subsampling_factor,
        subsampling_conv_kernel_size=encoder_config.subsampling_conv_kernel_size,
        subsampling_conv_stride=encoder_config.subsampling_conv_stride,
    )

    assert physical_frames.item() == expected_encoder_frames
    assert valid_frames.item() == expected_encoder_frames


@pytest.mark.parametrize(
    ("num_samples", "kwargs", "error"),
    [
        (159, {}, "audio samples"),
        (4000, {}, "audio samples"),
        (1600, {"is_streaming": True}, "complete audio clip"),
        (
            1600,
            {"audio_kwargs": {"padding": "max_length", "max_length": 4800}},
            "Padded audio",
        ),
    ],
)
def test_nemotron_processor_rejects_unsupported_audio(
    processor: Nemotron3_5AsrMultiModalProcessor, num_samples, kwargs, error
) -> None:
    processor.info.get_hf_config().encoder_config.max_position_embeddings = 4
    mm_items = processor.info.parse_mm_data(
        {"audio": np.zeros(num_samples, dtype=np.float32)}
    )
    with pytest.raises(ValueError, match=error):
        processor("", mm_items=mm_items, hf_processor_mm_kwargs=kwargs)


def test_nemotron_feature_extractor_matches_transformers() -> None:
    torch.manual_seed(0)
    audios = [torch.randn(1600).numpy(), torch.randn(4040).numpy()]

    hf_feature_extractor = HFNemotronAsrStreamingFeatureExtractor(feature_size=128)
    feature_extractor = NemotronAsrStreamingFeatureExtractor(feature_size=128)

    expected = hf_feature_extractor(
        audios,
        sampling_rate=16000,
        return_tensors="pt",
    )
    actual = feature_extractor(
        audios,
        sampling_rate=16000,
        return_tensors="pt",
    )

    torch.testing.assert_close(actual["attention_mask"], expected["attention_mask"])
    torch.testing.assert_close(
        actual["input_features"],
        expected["input_features"],
        rtol=1e-5,
        atol=1e-6,
    )
