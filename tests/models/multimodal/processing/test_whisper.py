# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Whisper's multimodal preprocessing."""

import numpy as np
import pytest
import torch
from transformers import BatchFeature, WhisperFeatureExtractor

from vllm.model_executor.models.whisper import (
    WHISPER_AUDIO_IPC_SCHEMA,
    WHISPER_AUDIO_IPC_SCHEMA_KEY,
    WhisperForConditionalGeneration,
    WhisperMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import ProcessorInputs

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("audio_duration_s", [1, 30, 35])
def test_audio_pcm_keeps_encoder_prompt_length(
    model_id: str,
    audio_duration_s: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PCM stays float32 and unpadded. Prompt length is max_source_positions."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"audio": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    feature_extractor = processor.info.get_feature_extractor()

    sampling_rate = feature_extractor.sampling_rate
    rng = np.random.RandomState(0)
    audio = rng.rand(sampling_rate * audio_duration_s).astype(np.float32)
    mm_items = processor.info.parse_mm_data({"audio": [(audio, sampling_rate)]})

    def _forbid_extractor_call(*args, **kwargs):
        raise AssertionError("WhisperFeatureExtractor.__call__ should not run")

    monkeypatch.setattr(WhisperFeatureExtractor, "__call__", _forbid_extractor_call)

    processed_inputs = processor(
        "<|startoftranscript|>",
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )

    mm_data = processed_inputs["mm_kwargs"].get_data()
    assert "input_features" not in mm_data
    audio_pcm = mm_data["audio_pcm"]
    assert isinstance(audio_pcm, torch.Tensor)
    assert audio_pcm.dtype == torch.float32
    assert audio_pcm.device.type == "cpu"
    assert audio_pcm.shape[-1] == audio.shape[0]
    torch.testing.assert_close(audio_pcm.reshape(-1), torch.from_numpy(audio))

    num_tokens = processor.info.get_num_audio_tokens()
    assert num_tokens == processor.info.get_hf_config().max_source_positions
    assert len(processed_inputs["encoder_prompt_token_ids"]) == num_tokens


def test_whisper_mm_field_publishes_pcm() -> None:
    processor = object.__new__(WhisperMultiModalProcessor)
    fields = processor._get_mm_fields_config(BatchFeature(), {})
    assert set(fields) == {"audio_pcm"}
    assert fields["audio_pcm"].field.keep_on_cpu is False


class _HashCtx:
    def get_mm_config(self):
        class _Cfg:
            mm_hasher_algorithm = "sha256"

        return _Cfg()


class _HashInfo:
    model_id = "openai/whisper-tiny"
    ctx = _HashCtx()


def _processor_for_hash() -> WhisperMultiModalProcessor:
    processor = object.__new__(WhisperMultiModalProcessor)
    processor.info = _HashInfo()
    return processor


def _hash_inputs(
    *,
    uuids: dict | None = None,
    kwargs: dict | None = None,
) -> ProcessorInputs:
    audio = np.array([0.0, 0.25, -0.5, 0.125], dtype=np.float32)
    items = MultiModalDataItems()
    items["audio"] = AudioProcessorItems([audio])
    return ProcessorInputs(
        prompt=[0],
        mm_data_items=items,
        mm_uuid_items=uuids,
        hf_processor_mm_kwargs={} if kwargs is None else dict(kwargs),
    )


def test_audio_ipc_schema_salts_whisper_mm_hash() -> None:
    processor = _processor_for_hash()
    plain = _hash_inputs()
    unsalted = plain.get_mm_hashes(_HashInfo.model_id, "sha256")
    salted = processor._get_mm_hashes(plain)
    assert plain.hf_processor_mm_kwargs == {}
    assert unsalted["audio"] != salted["audio"]

    explicit = _hash_inputs(
        kwargs={WHISPER_AUDIO_IPC_SCHEMA_KEY: WHISPER_AUDIO_IPC_SCHEMA}
    )
    assert processor._get_mm_hashes(explicit)["audio"] == salted["audio"]
    assert explicit.hf_processor_mm_kwargs == {
        WHISPER_AUDIO_IPC_SCHEMA_KEY: WHISPER_AUDIO_IPC_SCHEMA
    }

    uuid = "11111111-1111-1111-1111-111111111111"
    with_uuid = _hash_inputs(uuids={"audio": [uuid]})
    bare = with_uuid.get_mm_hashes(_HashInfo.model_id, "sha256")
    assert bare["audio"] == [uuid]
    salted_uuid = processor._get_mm_hashes(with_uuid)
    assert salted_uuid["audio"] != [uuid]
    assert with_uuid.hf_processor_mm_kwargs == {}

    manual = _hash_inputs(
        uuids={"audio": [uuid]},
        kwargs={WHISPER_AUDIO_IPC_SCHEMA_KEY: WHISPER_AUDIO_IPC_SCHEMA},
    )
    assert (
        manual.get_mm_hashes(_HashInfo.model_id, "sha256")["audio"]
        == salted_uuid["audio"]
    )


def test_embed_multimodal_rejects_logmel_without_pcm() -> None:
    model = object.__new__(WhisperForConditionalGeneration)
    mel = torch.zeros(1, 80, 3000)
    with pytest.raises(ValueError, match="input_features without audio_pcm"):
        model.embed_multimodal(input_features=mel)
    with pytest.raises(ValueError, match="input_features without audio_pcm"):
        model.embed_multimodal(input_features=[mel[0]])
    with pytest.raises(ValueError, match="input_features without audio_pcm"):
        model.embed_multimodal(input_features=mel, audio_pcm=[])
