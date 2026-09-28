# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import (
    BatchFeature,
    Nemotron3_5AsrProcessor,
    PreTrainedTokenizerBase,
)
from transformers.models.nemotron_asr_streaming import (
    NemotronAsrStreamingFeatureExtractor as HFNemotronAsrStreamingFeatureExtractor,
)

from vllm.model_executor.models.nemotron3_5_asr import (
    Nemotron3_5AsrDummyInputsBuilder,
    Nemotron3_5AsrMultiModalProcessor,
    Nemotron3_5AsrProcessingInfo,
    _get_subsampling_output_lengths,
)
from vllm.multimodal.inputs import MultiModalSharedField
from vllm.transformers_utils.processors import NemotronAsrStreamingFeatureExtractor


class _Tokenizer(PreTrainedTokenizerBase):
    def convert_tokens_to_ids(self, token):
        assert token == "<blank>"
        return 13087

    def encode(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


class _MMConfig:
    enable_mm_embeds = False
    mm_hasher_algorithm = "blake3"

    def merge_mm_processor_kwargs(self, kwargs):
        return dict(kwargs)

    def get_limit_per_prompt(self, modality):
        assert modality == "audio"
        return 1


class _HFProcessor:
    def __init__(self, valid_mel_frames: int):
        self.valid_mel_frames = valid_mel_frames
        self.feature_extractor = SimpleNamespace(sampling_rate=16000, hop_length=160)

    def __call__(self, audio, text=None, language="auto", **kwargs):
        assert text is None
        del kwargs
        batch_size = len(audio)
        num_mel_frames = self.valid_mel_frames + 1
        attention_mask = torch.zeros(batch_size, num_mel_frames, dtype=torch.bool)
        attention_mask[:, : self.valid_mel_frames] = True

        return BatchFeature(
            {
                "input_features": torch.zeros(batch_size, num_mel_frames, 128),
                "attention_mask": attention_mask,
                "prompt_ids": torch.full(
                    (batch_size,), 10 if language == "en-US" else 101
                ),
                "num_lookahead_tokens": torch.tensor(3),
            }
        )


class _ProcessingContext:
    def __init__(self, valid_mel_frames: int, max_encoder_frames: int = 4):
        encoder_config = SimpleNamespace(
            max_position_embeddings=max_encoder_frames,
            subsampling_factor=8,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=2,
        )
        self.model_config = SimpleNamespace(
            model="nvidia/nemotron-3.5-asr-streaming-0.6b",
            revision=None,
            max_model_len=4096,
            encoder_config={},
            hf_config=SimpleNamespace(
                blank_token_id=13087,
                encoder_config=encoder_config,
            ),
        )
        self.processor = _HFProcessor(valid_mel_frames)
        self.tokenizer = _Tokenizer()
        self.mm_config = _MMConfig()

    def get_tokenizer(self):
        return self.tokenizer

    def get_hf_config(self, typ=None):
        if typ is not None:
            assert isinstance(self.model_config.hf_config, typ)
        return self.model_config.hf_config

    def get_hf_processor(self, **kwargs):
        del kwargs
        return self.processor

    def get_mm_config(self):
        return self.mm_config

    def get_merged_mm_kwargs(self, kwargs):
        return self.mm_config.merge_mm_processor_kwargs(kwargs)

    def call_hf_processor(self, hf_processor, data, kwargs):
        merged_kwargs = self.get_merged_mm_kwargs(kwargs)
        merged_kwargs.setdefault("return_tensors", "pt")
        return hf_processor(**data, **merged_kwargs)


def _build_processor(valid_mel_frames: int):
    class _TestProcessingInfo(Nemotron3_5AsrProcessingInfo):
        def get_hf_config(self):
            return self.ctx.get_hf_config()

        def get_hf_processor(self, **kwargs):
            return self.ctx.get_hf_processor(**kwargs)

    info = _TestProcessingInfo(_ProcessingContext(valid_mel_frames))
    return Nemotron3_5AsrMultiModalProcessor(
        info,
        Nemotron3_5AsrDummyInputsBuilder(info),
    )


@pytest.mark.parametrize(
    ("valid_mel_frames", "expected_encoder_frames"),
    [(25, 4), (26, 5)],
)
def test_nemotron_processor_builds_encoder_decoder_contract(
    valid_mel_frames: int,
    expected_encoder_frames: int,
) -> None:
    processor = _build_processor(valid_mel_frames)
    processor.info.ctx.model_config.hf_config.encoder_config.max_position_embeddings = (
        32
    )
    mm_items = processor.info.parse_mm_data({"audio": np.zeros(1600, dtype=np.float32)})

    processed = processor(
        "",
        mm_items=mm_items,
        hf_processor_mm_kwargs={"language": "en-US", "sampling_rate": 16000},
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
    assert audio_item["prompt_ids"].data.item() == 10
    assert audio_item["prompt_ids"].field.keep_on_cpu
    assert audio_item["num_lookahead_tokens"].data.item() == 3
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
    max_encoder_frames: int,
    max_model_len: int,
    expected_audio_samples: int,
    expected_encoder_frames: int,
) -> None:
    info = _build_processor(valid_mel_frames=25).info
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
        (1600, {"is_streaming": True}, "streaming audio chunks"),
        (
            1600,
            {"audio_kwargs": {"padding": "max_length", "max_length": 4800}},
            "Padded audio",
        ),
    ],
)
def test_nemotron_processor_rejects_unsupported_audio(
    num_samples, kwargs, error
) -> None:
    processor = _build_processor(valid_mel_frames=25)
    processor.info.ctx.processor = Nemotron3_5AsrProcessor(
        NemotronAsrStreamingFeatureExtractor(feature_size=128), _Tokenizer()
    )
    mm_items = processor.info.parse_mm_data(
        {"audio": np.zeros(num_samples, dtype=np.float32)}
    )
    with pytest.raises(ValueError, match=error):
        processor("", mm_items=mm_items, hf_processor_mm_kwargs=kwargs)


def test_nemotron_centered_feature_extractor_masks_extra_stft_frame() -> None:
    feature_extractor = NemotronAsrStreamingFeatureExtractor(feature_size=128)

    output = feature_extractor(
        np.zeros(4040, dtype=np.float32),
        sampling_rate=16000,
        return_tensors="pt",
    )

    assert output["input_features"].shape == (1, 26, 128)
    assert output["attention_mask"].shape == (1, 26)
    assert output["attention_mask"].sum().item() == 25
    assert not output["attention_mask"][0, -1]
    assert torch.count_nonzero(output["input_features"][0, -1]).item() == 0


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


@pytest.mark.parametrize(("language", "prompt_id"), [("en-US", 0), ("auto", 101)])
def test_nemotron_feature_extractor_works_with_hf_processor(
    language: str, prompt_id: int
) -> None:
    processor = Nemotron3_5AsrProcessor(
        NemotronAsrStreamingFeatureExtractor(feature_size=128),
        _Tokenizer(),
        supported_num_lookahead_tokens=[3, 0, 6, 13],
        default_num_lookahead_tokens=3,
    )

    output = processor(
        audio=np.zeros(1600, dtype=np.float32),
        sampling_rate=16000,
        language=language,
    )

    assert output["input_features"].shape == (1, 11, 128)
    assert output["attention_mask"].sum().item() == 10
    assert output["prompt_ids"].tolist() == [prompt_id]
    assert output["num_lookahead_tokens"] == 3
