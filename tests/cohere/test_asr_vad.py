# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import pytest_asyncio
import torch
from fastapi import UploadFile

from tests.entrypoints.openai.correctness.test_transcription_api_correctness import (
    LONGFORM_NUM_SAMPLES,
    load_longform_dataset,
    run_longform_evaluation,
)
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import RemoteOpenAIServer
from vllm.assets.audio import AudioAsset
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest,
    TranslationRequest,
)
from vllm.entrypoints.openai.speech_to_text.speech_to_text import OpenAISpeechToText
from vllm.multimodal.audio import split_audio_with_vad

HF_MODEL_NAME = "openai/whisper-large-v3"
CI_MODEL_NAME = "whisper-large-v3"


def _get_server_model(ci_model_name: str, hf_model_name: str) -> str:
    # CI pre-downloads the checkpoint under ENGINES_DIR; use it when available
    # so the test can run offline, otherwise fall back to the HF model id.
    # engines_dir = os.environ.get("ENGINES_DIR", "/root/engines")
    # local_model_dir = Path(engines_dir) / ci_model_name
    # if local_model_dir.is_dir():
    #     return str(local_model_dir)
    return hf_model_name


# we test Whisper with VAD instead of Cohere ASR because the default
# VAD setting are picked from faster_whisper repo
# https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py#L41-L48
# and testing Cohere model showed that the optimal VAD setting is sensitive to
# downstream applications for Cohere. This test only tests if VAD works as we
# dont know the optimal VAD setting for CohereASR.
def test_long_audio_vad_wer_correctness():
    server_model = _get_server_model(CI_MODEL_NAME, HF_MODEL_NAME)
    model_info = HF_EXAMPLE_MODELS.find_hf_info(HF_MODEL_NAME)
    server_args = [
        f"--served-model-name={HF_MODEL_NAME}",
    ]

    # using VAD improves the WER from 9.5 to 8.22
    extra_body = {"vad_config.enabled": True}

    if model_info.trust_remote_code:
        server_args.append("--trust-remote-code")

    with RemoteOpenAIServer(
        server_model,
        server_args,
    ) as remote_server:
        dataset = load_longform_dataset()
        client = remote_server.get_async_client()
        wer = run_longform_evaluation(
            HF_MODEL_NAME,
            client,
            dataset,
            max_concurrent_reqs=LONGFORM_NUM_SAMPLES,
            extra_body=extra_body,
        )

    expected_wer = 8.22
    print(f"Expected WER: {expected_wer}, Actual WER: {wer}")
    torch.testing.assert_close(wer, expected_wer, atol=1e-1, rtol=1e-2)


# ============================================================
# Tests for VAD Config Building
# ============================================================


def test_transcription_request_builds_vad_config_from_flat_form_fields():
    request = TranscriptionRequest.model_validate(
        {
            "file": UploadFile(io.BytesIO(b"audio"), filename="audio.wav"),
            "model": "stub-model",
            "vad_config.enabled": True,
            "vad_config.threshold": 0.7,
            "vad_config.speech_pad_ms": 450,
        }
    )

    vad_config = request.build_vad_config()

    assert vad_config.enabled is True
    assert vad_config.threshold == 0.7
    assert vad_config.speech_pad_ms == 450
    assert vad_config.min_silence_duration_ms == 2000


def test_translation_request_builds_vad_config_from_flat_form_fields():
    request = TranslationRequest.model_validate(
        {
            "file": UploadFile(io.BytesIO(b"audio"), filename="audio.wav"),
            "model": "stub-model",
            "vad_config.enabled": True,
            "vad_config.min_silence_duration_ms": 1500,
        }
    )

    vad_config = request.build_vad_config()

    assert vad_config.enabled is True
    assert vad_config.min_silence_duration_ms == 1500
    assert vad_config.speech_pad_ms == 600


def test_decode_and_chunk_speech_maps_all_vad_request_fields_to_splitter():
    expected_vad_config = {
        "enabled": True,
        "threshold": 0.7,
        "neg_threshold": 0.2,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 12.5,
        "min_silence_duration_ms": 1500,
        "speech_pad_ms": 450,
        "min_silence_at_max_speech_ms": 120,
        "use_max_poss_sil_at_max_speech": False,
    }
    request = TranscriptionRequest.model_validate(
        {
            "file": UploadFile(io.BytesIO(b"audio"), filename="audio.wav"),
            "model": "stub-model",
            **{
                f"vad_config.{field_name}": value
                for field_name, value in expected_vad_config.items()
            },
        }
    )

    server = OpenAISpeechToText.__new__(OpenAISpeechToText)
    server.asr_config = SpeechToTextConfig(sample_rate=16_000)
    thread_vad = object()
    server._vad_provider = Mock()
    server._vad_provider.get.return_value = thread_vad

    audio = np.arange(32, dtype=np.float32)
    expected_chunks = [audio.copy()]

    with (
        patch(
            "vllm.entrypoints.openai.speech_to_text.speech_to_text.load_audio",
            return_value=(audio, 16_000),
        ),
        patch(
            "vllm.entrypoints.openai.speech_to_text.speech_to_text.get_audio_duration",
            return_value=2.0,
        ),
        patch(
            "vllm.entrypoints.openai.speech_to_text.speech_to_text.split_audio_with_vad",
            return_value=expected_chunks,
        ) as mock_split_audio_with_vad,
    ):
        chunks, duration = OpenAISpeechToText._decode_and_chunk_speech(
            server,
            b"audio-bytes",
            request,
        )

    assert chunks == expected_chunks
    assert duration == 2.0
    server._vad_provider.get.assert_called_once_with()
    mock_split_audio_with_vad.assert_called_once()

    kwargs = mock_split_audio_with_vad.call_args.kwargs
    assert kwargs["duration"] == 2.0
    assert kwargs["asr_config"] is server.asr_config
    assert kwargs["vad"] is thread_vad
    np.testing.assert_array_equal(kwargs["audio_data"], audio)
    assert kwargs["sample_rate"] == 16_000

    vad_config = kwargs["vad_config"]
    for field_name, expected_value in expected_vad_config.items():
        assert getattr(vad_config, field_name) == expected_value


# ============================================================
# Tests for Transcription validation with VAD
# ============================================================


@pytest.fixture(scope="module")
def server():
    server_model = _get_server_model(CI_MODEL_NAME, HF_MODEL_NAME)
    model_info = HF_EXAMPLE_MODELS.find_hf_info(HF_MODEL_NAME)
    server_args = [
        f"--served-model-name={HF_MODEL_NAME}",
    ]
    if model_info.trust_remote_code:
        server_args.append("--trust-remote-code")

    with RemoteOpenAIServer(server_model, server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def whisper_client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture
def mary_had_lamb():
    path = AudioAsset("mary_had_lamb").get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.mark.asyncio
async def test_basic_audio_with_per_request_vad(whisper_client, mary_had_lamb):
    transcription = await whisper_client.audio.transcriptions.create(
        model=HF_MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        temperature=0.0,
        extra_body={
            "vad_config.enabled": True,
            "vad_config.speech_pad_ms": 450,
        },
    )
    out = json.loads(transcription)
    out_text = out["text"]
    assert "Mary had a little lamb," in out_text


# ============================================================
# Tests for Audio VAD Chunking Utilities
# ============================================================


class TestAudioVADChunking:
    """Tests for the split_audio_with_vad branch behavior."""

    @staticmethod
    def _make_vad_config(*, enabled: bool):
        return SimpleNamespace(
            enabled=enabled,
            threshold=0.5,
            neg_threshold=None,
            min_speech_duration_ms=0,
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=2000,
            speech_pad_ms=600,
            min_silence_at_max_speech_ms=98,
            use_max_poss_sil_at_max_speech=True,
        )

    @staticmethod
    def _make_stub_vad(
        *,
        expected_audio: np.ndarray | None = None,
        expected_sample_rate: int | None = None,
        speech_timestamps: list[dict[str, int]] | None = None,
    ):
        class _StubVAD:
            def get_speech_timestamps(
                self,
                audio_data: np.ndarray,
                sample_rate: int,
                vad_config,
            ) -> list[dict[str, int]]:
                if expected_audio is not None:
                    np.testing.assert_array_equal(audio_data, expected_audio)
                if expected_sample_rate is not None:
                    assert sample_rate == expected_sample_rate
                assert vad_config.enabled
                return speech_timestamps or []

        return _StubVAD()

    def test_split_audio_with_vad_and_rms_split(self):
        """Only oversized VAD spans should be re-split with RMS chunking."""

        audio = np.arange(12, dtype=np.float32)
        speech_timestamps = [
            {"start": 0, "end": 6},
            {"start": 10, "end": 12},
        ]
        asr_config = SpeechToTextConfig(
            max_audio_clip_s=4,
            overlap_chunk_second=0,
            min_energy_split_window_size=1,
        )
        vad_config = self._make_vad_config(enabled=True)
        vad = self._make_stub_vad(
            expected_audio=audio,
            expected_sample_rate=1,
            speech_timestamps=speech_timestamps,
        )

        with patch("vllm.multimodal.audio.split_audio") as mock_split_audio:
            mock_split_audio.side_effect = lambda partial_audio, *_args: [
                partial_audio[:4].copy(),
                partial_audio[4:].copy(),
            ]

            result = split_audio_with_vad(
                duration=12.0,
                asr_config=asr_config,
                vad=vad,
                vad_config=vad_config,
                audio_data=audio,
                sample_rate=1,
            )

        mock_split_audio.assert_called_once()
        call_args = mock_split_audio.call_args.args
        np.testing.assert_array_equal(call_args[0], audio[0:6])
        assert call_args[1:] == (1, 4, 0, 1)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], audio[0:4])
        np.testing.assert_array_equal(result[1], audio[4:6])
        np.testing.assert_array_equal(result[2], audio[10:12])

    def test_split_audio_with_rms_split_only(self):
        """Long audio without VAD should use RMS splitting directly."""

        audio = np.arange(10, dtype=np.float32)
        expected_chunks = [audio[:5], audio[5:]]
        asr_config = SpeechToTextConfig(
            max_audio_clip_s=4,
            overlap_chunk_second=1,
            min_energy_split_window_size=2,
        )
        vad_config = self._make_vad_config(enabled=False)
        vad = self._make_stub_vad()

        with patch("vllm.multimodal.audio.split_audio") as mock_split_audio:
            mock_split_audio.return_value = expected_chunks

            result = split_audio_with_vad(
                duration=10.0,
                asr_config=asr_config,
                vad=vad,
                vad_config=vad_config,
                audio_data=audio,
                sample_rate=16000,
            )

        mock_split_audio.assert_called_once_with(audio, 16000, 4, 1, 2)
        assert result == expected_chunks

    def test_split_audio_with_vad_only_trims_nonspeech(self):
        """Short audio with VAD should return one concatenated speech-only chunk."""

        audio = np.arange(12, dtype=np.float32)
        speech_timestamps = [
            {"start": 1, "end": 4},
            {"start": 6, "end": 9},
        ]
        expected_trimmed_audio = np.concatenate([audio[1:4], audio[6:9]])
        asr_config = SpeechToTextConfig(
            max_audio_clip_s=None,
            overlap_chunk_second=1,
            min_energy_split_window_size=None,
        )
        vad_config = self._make_vad_config(enabled=True)
        vad = self._make_stub_vad(
            expected_audio=audio,
            expected_sample_rate=16000,
            speech_timestamps=speech_timestamps,
        )

        with patch("vllm.multimodal.audio.split_audio") as mock_split_audio:
            result = split_audio_with_vad(
                duration=2.0,
                asr_config=asr_config,
                vad=vad,
                vad_config=vad_config,
                audio_data=audio,
                sample_rate=16000,
            )

        mock_split_audio.assert_not_called()
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected_trimmed_audio)

    def test_split_audio_without_vad_or_rms_split_returns_original_audio(self):
        """Short audio without VAD should be returned unchanged."""

        audio = np.arange(12, dtype=np.float32)
        asr_config = SpeechToTextConfig(
            max_audio_clip_s=None,
            overlap_chunk_second=1,
            min_energy_split_window_size=None,
        )
        vad_config = self._make_vad_config(enabled=False)
        vad = self._make_stub_vad()

        with patch("vllm.multimodal.audio.split_audio") as mock_split_audio:
            result = split_audio_with_vad(
                duration=2.0,
                asr_config=asr_config,
                vad=vad,
                vad_config=vad_config,
                audio_data=audio,
                sample_rate=16000,
            )

        mock_split_audio.assert_not_called()
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], audio)
