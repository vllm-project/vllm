# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from unittest.mock import MagicMock

import httpx
import numpy as np
import pybase64
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend, WhisperFeatureExtractor

from tests.utils import RemoteLaunchRenderServer
from vllm import envs
from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.funaudiochat import (
    FunAudioChatDummyInputsBuilder,
    FunAudioChatMultiModalProcessor,
    FunAudioChatProcessingInfo,
)
from vllm.multimodal.processing import InputProcessingContext
from vllm.multimodal.processing.context import profiling_media
from vllm.multimodal.utils import encode_audio_url
from vllm.transformers_utils.configs.funaudiochat import FunAudioChatConfig


class RecordingFeatureExtractor:
    sampling_rate = 16000
    n_fft = 400
    hop_length = 160
    n_samples = 300 * sampling_rate
    nb_max_frames = n_samples // hop_length
    feature_size = 128

    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def __call__(self, wavs, **kwargs):
        self.last_kwargs = kwargs
        n_frames = [max(1, int(w.shape[0]) // self.hop_length) for w in wavs]
        max_frames = max(n_frames)
        features = torch.zeros((len(wavs), 128, max_frames))
        mask = torch.zeros((len(wavs), max_frames), dtype=torch.long)
        for i, frames in enumerate(n_frames):
            mask[i, :frames] = 1
        return {"input_features": features, "attention_mask": mask}


class MockSpeechTokenizer:
    pad_token = "<|audio_pad|>"

    def __call__(self, speech_strs, **kwargs):
        token_len = max(1, len(self.pad_token))
        lengths = [len(s) // token_len for s in speech_strs]
        max_len = max(lengths)
        pad_to = kwargs.get("pad_to_multiple_of")
        if pad_to:
            max_len = ((max_len + pad_to - 1) // pad_to) * pad_to
        ids = torch.zeros((len(speech_strs), max_len), dtype=torch.long)
        mask = torch.zeros((len(speech_strs), max_len), dtype=torch.long)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


class MockTokenizer:
    def encode(self, text, **kwargs):
        return [1, 2, 3]


@pytest.fixture
def funaudiochat_processor(monkeypatch):
    feature_extractor = RecordingFeatureExtractor()
    speech_tokenizer = MockSpeechTokenizer()
    tokenizer = MockTokenizer()

    monkeypatch.setattr(
        "vllm.model_executor.models.funaudiochat.WhisperFeatureExtractor.from_pretrained",
        lambda *args, **kwargs: feature_extractor,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.funaudiochat.TokenizersBackend.from_pretrained",
        lambda *args, **kwargs: speech_tokenizer,
    )

    ctx = MagicMock()
    ctx.get_hf_config.return_value = FunAudioChatConfig()
    ctx.model_config.hf_config = FunAudioChatConfig()
    ctx.model_config.model = "funaudiochat"
    ctx.model_config.revision = None
    ctx.model_config.tokenizer_revision = None
    mm_config = MagicMock()
    mm_config.enable_mm_embeds = False
    mm_config.allow_missing_mm_embeddings = False
    ctx.model_config.multimodal_config = mm_config
    ctx.model_config.get_multimodal_config.return_value = mm_config
    ctx.tokenizer = tokenizer
    ctx.get_tokenizer.return_value = tokenizer

    info = FunAudioChatProcessingInfo(ctx)
    processor = FunAudioChatMultiModalProcessor(
        info, FunAudioChatDummyInputsBuilder(info)
    )
    return processor, feature_extractor


def test_max_audio_samples_match_profiled_300s_budget(funaudiochat_processor):
    processor, _ = funaudiochat_processor
    sampling_rate = 16000

    assert processor.info.get_max_speech_frames() == 7500
    assert processor.info.get_max_audio_samples(sampling_rate) == 4_800_000
    assert processor.info.get_max_audio_samples(sampling_rate) / sampling_rate == 300.0


def test_over_budget_audio_is_rejected(funaudiochat_processor):
    processor, feature_extractor = funaudiochat_processor
    sampling_rate = 16000
    over_budget = np.zeros(
        processor.info.get_max_audio_samples(sampling_rate) + 1, dtype=np.float32
    )
    mm_items = processor.info.parse_mm_data({"audio": [over_budget]}, validate=False)

    with pytest.raises(ValueError, match="too long for FunAudioChat"):
        processor._apply_hf_processor_main(mm_items, {})

    assert feature_extractor.last_kwargs is None


def test_profiled_max_audio_is_accepted(funaudiochat_processor):
    processor, feature_extractor = funaudiochat_processor
    sampling_rate = 16000
    max_samples = processor.info.get_max_audio_samples(sampling_rate)
    audio = np.zeros(max_samples, dtype=np.float32)
    mm_items = processor.info.parse_mm_data({"audio": [audio]}, validate=False)

    processed = processor._apply_hf_processor_main(mm_items, {})

    assert processed["speech_attention_mask"].sum().item() == 7500
    assert feature_extractor.last_kwargs is not None
    assert feature_extractor.last_kwargs.get("padding") == "max_length"


@pytest.fixture
def local_funaudiochat_model(tmp_path):
    FunAudioChatConfig(
        architectures=["FunAudioChatForConditionalGeneration"],
        audio_config={"max_source_positions": 5},
    ).save_pretrained(tmp_path)
    WhisperFeatureExtractor(chunk_length=1).save_pretrained(tmp_path)
    audio_tokens = ["<|audio_bos|>", "<|AUDIO|>", "<|audio_eos|>"]
    vocab = {
        token: i for i, token in enumerate(["[UNK]", "<|audio_pad|>", *audio_tokens])
    }
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="<|audio_pad|>",
        additional_special_tokens=audio_tokens,
        chat_template="{% for msg in messages %}{{ msg.content }}{% endfor %}",
    )
    tokenizer.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path / "speech_tokenizer")
    return str(tmp_path), tokenizer


@pytest.mark.parametrize("validate", [True, False], ids=["admission", "processor"])
def test_audio_items_fit_the_preprocessing_budget(
    local_funaudiochat_model, monkeypatch, validate
):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 0)
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_PREPROCESSING_BYTES", 800_000)
    model, tokenizer = local_funaudiochat_model
    model_config = ModelConfig(model=model, dtype="float32", max_model_len=128)
    info = FunAudioChatProcessingInfo(InputProcessingContext(model_config, tokenizer))
    processor = FunAudioChatMultiModalProcessor(
        info, FunAudioChatDummyInputsBuilder(info)
    )
    audio = np.zeros(1600, dtype=np.float32)
    accepted = info.parse_mm_data({"audio": [audio]}, validate=validate)
    assert processor._apply_hf_processor_main(accepted, {})["input_features"].shape == (
        1,
        80,
        100,
    )

    with pytest.raises(VLLMValidationError, match="At most 1 audio"):
        items = info.parse_mm_data({"audio": [audio, audio]}, validate=validate)
        processor._apply_hf_processor_main(items, {})

    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_PREPROCESSING_BYTES", 500_000)
    with pytest.raises(VLLMValidationError, match="At most 0 audio"):
        items = info.parse_mm_data({"audio": [audio]}, validate=validate)
        processor._apply_hf_processor_main(items, {})

    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_PREPROCESSING_BYTES", 1_200_000)
    accepted = info.parse_mm_data({"audio": [audio, audio]}, validate=validate)
    assert processor._apply_hf_processor_main(accepted, {})["input_features"].shape == (
        2,
        80,
        100,
    )

    long_audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(VLLMValidationError, match="waveform preprocessing exceeds"):
        items = info.parse_mm_data({"audio": [audio, long_audio]}, validate=validate)
        processor._apply_hf_processor_main(items, {})


def test_precomputed_audio_embeds_ignore_waveform_budget(
    local_funaudiochat_model, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_PREPROCESSING_BYTES", 800_000)
    model, tokenizer = local_funaudiochat_model
    model_config = ModelConfig(
        model=model,
        dtype="float32",
        max_model_len=128,
        limit_mm_per_prompt={"audio": 2},
        enable_mm_embeds=True,
    )
    info = FunAudioChatProcessingInfo(InputProcessingContext(model_config, tokenizer))
    processor = FunAudioChatMultiModalProcessor(
        info, FunAudioChatDummyInputsBuilder(info)
    )
    embeds = torch.zeros((2, 1, model_config.get_inputs_embeds_size()))
    prompt = processor.dummy_inputs.get_dummy_text({"audio": 2})

    result = processor(
        prompt,
        mm_items=info.parse_mm_data({"audio": embeds}),
        hf_processor_mm_kwargs={},
    )

    processed_embeds = result["mm_kwargs"].get_data()["audio_embeds"]
    assert isinstance(processed_embeds, torch.Tensor)
    assert processed_embeds.shape == embeds.shape

    audio = np.zeros(1600, dtype=np.float32)
    with pytest.raises(VLLMValidationError, match="At most 1 audio"):
        processor._apply_hf_processor_main(
            info.parse_mm_data({"audio": [audio, audio]}), {}
        )
    with profiling_media():
        profiled_items = info.parse_mm_data({"audio": [audio, audio]})
        assert (
            processor._apply_hf_processor_main(profiled_items, {})[
                "input_features"
            ].shape[0]
            == 2
        )


@pytest.mark.parametrize(
    ("budget", "accepted", "rejected", "error_message"),
    [
        (500_000, [], [1600], "At most 0 audio"),
        (800_000, [[1600]], [1600, 1600], "At most 1 audio"),
        (
            1_200_000,
            [[1600, 1600], [16000]],
            [1600, 16000],
            "waveform preprocessing exceeds",
        ),
    ],
    ids=["none-fit", "one-fits", "depends-on-waveform-length"],
)
def test_render_respects_the_audio_preprocessing_budget(
    local_funaudiochat_model, budget, accepted, rejected, error_message
):
    model, _ = local_funaudiochat_model
    with RemoteLaunchRenderServer(
        model,
        [
            "--host",
            "127.0.0.1",
            "--dtype",
            "float32",
            "--max-model-len",
            "128",
            "--limit-mm-per-prompt",
            '{"audio":2}',
        ],
        env_dict={"VLLM_MAX_MEDIA_PREPROCESSING_BYTES": str(budget)},
    ) as server:
        url = server.url_for("v1/chat/completions/render")
        for lengths, expected_status in [
            *((lengths, 200) for lengths in accepted),
            (rejected, 400),
        ]:
            content = [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": encode_audio_url(
                            np.zeros(samples, dtype=np.float32), 16000
                        )
                    },
                }
                for samples in lengths
            ]
            response = httpx.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                },
            )
            assert response.status_code == expected_status, response.text[:512]
        assert error_message in response.json()["error"]["message"]


def test_render_audio_embeds_ignore_waveform_budget(local_funaudiochat_model):
    model, _ = local_funaudiochat_model
    hidden_size = ModelConfig(model=model).get_inputs_embeds_size()
    buffer = io.BytesIO()
    torch.save(torch.zeros((1, hidden_size)), buffer)
    encoded = pybase64.b64encode(buffer.getvalue()).decode("ascii")
    content = [{"type": "audio_embeds", "audio_embeds": encoded}] * 2

    with RemoteLaunchRenderServer(
        model,
        [
            "--host",
            "127.0.0.1",
            "--dtype",
            "float32",
            "--max-model-len",
            "128",
            "--enable-mm-embeds",
            "--limit-mm-per-prompt",
            '{"audio":2}',
        ],
        env_dict={"VLLM_MAX_MEDIA_PREPROCESSING_BYTES": "500000"},
    ) as server:
        response = httpx.post(
            server.url_for("v1/chat/completions/render"),
            json={"model": model, "messages": [{"role": "user", "content": content}]},
        )
        assert response.status_code == 200, response.text[:512]
        assert len(response.json()["features"]["mm_placeholders"]["audio"]) == 2
