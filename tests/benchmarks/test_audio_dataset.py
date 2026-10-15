#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import io
import json
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pybase64 as base64
import pytest
import soundfile as sf

import vllm.benchmarks.datasets.datasets as datasets_module
import vllm.benchmarks.lib.endpoint_request_func as request_func_module
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput
from vllm.tokenizers import TokenizerLike

pytestmark = pytest.mark.skip_global_cleanup


class _ReadableBinary(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _TokenizedPrompt:
    def __init__(self, prompt: str) -> None:
        self.input_ids = prompt.split()


class _Tokenizer:
    def __init__(self, name_or_path: str = "openai/whisper-large-v3") -> None:
        self.name_or_path = name_or_path

    def __call__(self, prompt: str) -> _TokenizedPrompt:
        return _TokenizedPrompt(prompt)


class CohereAsrTokenizer(_Tokenizer):
    def __init__(self, name_or_path: str = "/models/cohere-transcribe") -> None:
        super().__init__(name_or_path)


class _CohereNameOnlyTokenizer(_Tokenizer):
    def __init__(self) -> None:
        super().__init__("cohere/some-local-checkpoint")


def _as_tokenizer(tokenizer: _Tokenizer) -> TokenizerLike:
    return cast(TokenizerLike, tokenizer)


def _write_wav(path: Path, duration_s: float = 0.1, sample_rate: int = 16_000) -> None:
    num_samples = int(duration_s * sample_rate)
    sf.write(path, np.zeros(num_samples, dtype=np.float32), sample_rate)


class _FakeFormData:
    def __init__(self) -> None:
        self.fields: list[tuple[str, object, dict[str, str]]] = []

    def add_field(self, name: str, value: object, **kwargs: str) -> None:
        self.fields.append((name, value, kwargs))


class _FakeContent:
    async def iter_any(self):
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        yield b'data: {"usage":{"completion_tokens":1}}\n\n'
        yield b"data: [DONE]\n\n"


class _FakeResponse:
    def __init__(self) -> None:
        self.status = 200
        self.reason = "OK"
        self.content = _FakeContent()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self) -> None:
        self.uploaded_bytes: bytes | None = None
        self.upload_filename: str | None = None
        self.fields: list[tuple[str, object, dict[str, str]]] | None = None

    def post(self, *, url: str, data: _FakeFormData, headers: dict[str, str]):
        del url, headers
        self.fields = list(data.fields)
        _, file_obj, file_kwargs = self.fields[0]
        file_obj = cast(_ReadableBinary, file_obj)
        self.uploaded_bytes = file_obj.read()
        self.upload_filename = file_kwargs.get("filename")
        return _FakeResponse()


class _FakeProfileSession:
    def __init__(self) -> None:
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    def post(self, *, url: str, headers: dict[str, str]):
        self.url = url
        self.headers = headers
        return _FakeResponse()


def test_asr_dataset_sample_handles_local_audio_paths(tmp_path: Path) -> None:
    audio_path = tmp_path / "earnings.wav"
    _write_wav(audio_path, duration_s=0.1)

    dataset = object.__new__(datasets_module.ASRDataset)
    dataset.data = [
        {
            "audio": {
                "path": str(audio_path),
                "bytes": None,
            },
            "text": "quarterly earnings call",
        }
    ]

    samples = dataset.sample(
        tokenizer=_as_tokenizer(_Tokenizer()),
        num_requests=1,
        output_len=32,
        asr_min_audio_len_sec=0.0,
        asr_max_audio_len_sec=1.0,
    )

    assert len(samples) == 1
    assert samples[0].multi_modal_data == {"audio_path": str(audio_path)}
    assert (
        samples[0].prompt == "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    )


@pytest.mark.parametrize("has_filepath", [True, False])
def test_asr_dataset_sample_handles_embedded_audio_bytes(
    tmp_path: Path, has_filepath: bool
) -> None:
    audio_path = tmp_path / "earnings.wav"
    _write_wav(audio_path, duration_s=0.1)

    test_path = None
    if has_filepath:
        test_path = audio_path

    dataset = object.__new__(datasets_module.ASRDataset)
    dataset.data = [
        {
            "audio": {
                "path": test_path,
                "bytes": audio_path.read_bytes(),
            },
            "text": "quarterly earnings call",
        }
    ]

    samples = dataset.sample(
        tokenizer=_as_tokenizer(_Tokenizer()),
        num_requests=1,
        output_len=32,
        asr_min_audio_len_sec=0.0,
        asr_max_audio_len_sec=1.0,
    )

    assert len(samples) == 1
    assert isinstance(samples[0].multi_modal_data, dict)
    audio, sample_rate = samples[0].multi_modal_data["audio"]
    assert sample_rate == 16_000
    assert isinstance(audio, np.ndarray)
    assert audio.size > 0


def test_async_request_openai_audio_handles_local_audio_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "earnings.wav"
    _write_wav(audio_path, duration_s=0.25)

    monkeypatch.setattr(request_func_module.aiohttp, "FormData", _FakeFormData)
    session = _FakeSession()
    request_input = RequestFuncInput(
        prompt="",
        api_url="http://localhost:8000/v1/audio/transcriptions",
        prompt_len=1,
        output_len=32,
        model="openai/whisper-large-v3",
        multi_modal_content={"audio_path": str(audio_path)},
    )

    output = asyncio.run(
        request_func_module.async_request_openai_audio(request_input, session)
    )

    assert session.upload_filename == audio_path.name
    assert session.uploaded_bytes == audio_path.read_bytes()
    assert output.success is True
    assert output.generated_text == "hello"
    assert output.output_tokens == 1
    assert output.input_audio_duration == pytest.approx(0.25, abs=1e-2)


def test_async_request_openai_audio_handles_decoded_audio_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(request_func_module.aiohttp, "FormData", _FakeFormData)
    session = _FakeSession()
    request_input = RequestFuncInput(
        prompt="",
        api_url="http://localhost:8000/v1/audio/transcriptions",
        prompt_len=1,
        output_len=32,
        model="openai/whisper-large-v3",
        multi_modal_content={
            "audio": (np.zeros(1_600, dtype=np.float32), 16_000),
        },
    )

    output = asyncio.run(
        request_func_module.async_request_openai_audio(request_input, session)
    )

    assert session.upload_filename == "audio.wav"
    assert session.uploaded_bytes is not None
    assert output.success is True
    assert output.generated_text == "hello"


def test_audio_request_rejects_profile_endpoint() -> None:
    request_input = RequestFuncInput(
        prompt="",
        api_url="http://localhost:8000/start_profile",
        prompt_len=1,
        output_len=32,
        model="openai/whisper-large-v3",
    )

    with pytest.raises(ValueError, match="OpenAI Audio API"):
        asyncio.run(
            request_func_module.async_request_openai_audio(
                request_input, _FakeProfileSession()
            )
        )


@pytest.mark.parametrize("endpoint", ["start_profile", "stop_profile"])
def test_audio_benchmark_profile_request_has_no_inference_payload(
    endpoint: str,
) -> None:
    session = _FakeProfileSession()

    output = asyncio.run(
        request_func_module.async_request_profile(
            f"http://localhost:8000/{endpoint}",
            session,
            extra_headers={"X-Test-Header": "profile"},
        )
    )

    assert output.success is True
    assert session.url == f"http://localhost:8000/{endpoint}"
    assert session.headers == {"X-Test-Header": "profile"}


_COHERE_ASR_PROMPT = (
    "<|startofcontext|><|startoftranscript|>"
    "<|emo:undefined|><|en|><|en|><|pnc|><|noitn|>"
    "<|notimestamp|><|nodiarize|>"
)


def _make_asr_dataset(tmp_path: Path) -> datasets_module.ASRDataset:
    audio_path = tmp_path / "sample.wav"
    _write_wav(audio_path, duration_s=0.1)
    dataset = object.__new__(datasets_module.ASRDataset)
    dataset.data = [
        {
            "audio": {"path": str(audio_path), "bytes": None},
            "text": "hello world",
        }
    ]
    return dataset


def test_asr_dataset_cohere_class_name_gets_decoder_prompt(tmp_path: Path) -> None:
    dataset = _make_asr_dataset(tmp_path)
    samples = dataset.sample(
        tokenizer=_as_tokenizer(CohereAsrTokenizer()),
        num_requests=1,
        output_len=32,
        asr_min_audio_len_sec=0.0,
        asr_max_audio_len_sec=1.0,
    )
    assert len(samples) == 1
    assert samples[0].prompt == _COHERE_ASR_PROMPT


def test_asr_dataset_cohere_name_or_path_fallback_gets_decoder_prompt(
    tmp_path: Path,
) -> None:
    dataset = _make_asr_dataset(tmp_path)
    samples = dataset.sample(
        tokenizer=_as_tokenizer(_CohereNameOnlyTokenizer()),
        num_requests=1,
        output_len=32,
        asr_min_audio_len_sec=0.0,
        asr_max_audio_len_sec=1.0,
    )
    assert len(samples) == 1
    assert samples[0].prompt == _COHERE_ASR_PROMPT


def test_asr_dataset_unknown_tokenizer_gets_empty_prompt(tmp_path: Path) -> None:
    dataset = _make_asr_dataset(tmp_path)
    samples = dataset.sample(
        tokenizer=_as_tokenizer(_Tokenizer(name_or_path="some-other/asr-model")),
        num_requests=1,
        output_len=32,
        asr_min_audio_len_sec=0.0,
        asr_max_audio_len_sec=1.0,
    )
    assert len(samples) == 1
    assert samples[0].prompt == ""


@pytest.mark.parametrize("with_tokenizer", [False, True])
@pytest.mark.parametrize("enable_multimodal_chat", [False, True])
@pytest.mark.parametrize("output_len", [None, -1, 0, 64])
def test_custom_audio_get_samples_preserves_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_tokenizer: bool,
    enable_multimodal_chat: bool,
    output_len: int | None,
) -> None:
    """Skipping tokenizer init must not drop audio on either request path."""
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.linspace(-0.5, 0.5, 1600), 16_000)
    dataset_path = tmp_path / "audio.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "prompt": "Transcribe this audio.",
                "audio": str(audio_path),
                "output_tokens": 17,
            }
        )
        + "\n"
    )
    args = argparse.Namespace(
        dataset_name="custom_audio",
        dataset_path=str(dataset_path),
        disable_shuffle=True,
        seed=0,
        num_prompts=1,
        custom_output_len=output_len,
        enable_multimodal_chat=enable_multimodal_chat,
        request_id_prefix="audio-",
        no_oversample=False,
    )
    tokenizer = _as_tokenizer(_Tokenizer()) if with_tokenizer else None
    samples = datasets_module.get_samples(args, tokenizer)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.prompt == "Transcribe this audio."
    assert sample.prompt_len == (3 if with_tokenizer else 1)
    assert sample.request_id == "audio-0"
    expected_len = (
        (17 if with_tokenizer else 256) if output_len in (None, -1) else output_len
    )
    assert sample.expected_output_len == expected_len
    assert isinstance(sample.multi_modal_data, dict)
    request = RequestFuncInput(
        prompt=sample.prompt,
        api_url="http://localhost:8000/v1/audio/transcriptions",
        prompt_len=sample.prompt_len,
        output_len=sample.expected_output_len,
        model="test-audio-model",
        multi_modal_content=sample.multi_modal_data,
    )
    if enable_multimodal_chat:
        messages = request_func_module._get_chat_messages(request)
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample.prompt},
                    sample.multi_modal_data,
                ],
            }
        ]
        assert sample.multi_modal_data["type"] == "input_audio"
        audio_content = sample.multi_modal_data["input_audio"]
        assert audio_content["format"] == "wav"
        audio, sample_rate = sf.read(
            io.BytesIO(base64.b64decode(audio_content["data"]))
        )
    else:
        audio, sample_rate = sample.multi_modal_data["audio"]
        monkeypatch.setattr(request_func_module.aiohttp, "FormData", _FakeFormData)
        session = _FakeSession()
        output = asyncio.run(
            request_func_module.async_request_openai_audio(request, session)
        )
        assert output.success
        assert session.uploaded_bytes is not None
        uploaded_audio, uploaded_rate = sf.read(io.BytesIO(session.uploaded_bytes))
        np.testing.assert_array_equal(uploaded_audio, audio)
        assert uploaded_rate == sample_rate
    expected_audio, expected_rate = sf.read(audio_path)
    np.testing.assert_array_equal(audio, expected_audio)
    assert sample_rate == expected_rate


@pytest.mark.parametrize("enable_multimodal_chat", [False, True])
@pytest.mark.parametrize("skip_chat_template", [False, True])
def test_custom_audio_preserves_chat_template_behavior(
    tmp_path: Path, enable_multimodal_chat: bool, skip_chat_template: bool
) -> None:
    class TemplateTokenizer(_Tokenizer):
        chat_template = "test-template"

        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is False
            return "templated " + messages[0]["content"]

    audio_path = tmp_path / "sample.wav"
    _write_wav(audio_path)
    dataset = object.__new__(datasets_module.CustomAudioDataset)
    dataset.data = [{"prompt": "Transcribe.", "audio": str(audio_path)}]
    sample = dataset.sample(
        tokenizer=_as_tokenizer(TemplateTokenizer()),
        num_requests=1,
        output_len=32,
        enable_multimodal_chat=enable_multimodal_chat,
        skip_chat_template=skip_chat_template,
    )[0]
    expected = "Transcribe."
    if not enable_multimodal_chat and not skip_chat_template:
        expected = "templated " + expected
    assert sample.prompt == expected
    assert sample.prompt_len == len(expected.split())


@pytest.mark.parametrize("with_tokenizer", [False, True])
@pytest.mark.parametrize("output_len", [None, -1])
def test_custom_audio_missing_output_tokens(
    tmp_path: Path, with_tokenizer: bool, output_len: int | None
) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_wav(audio_path)
    dataset = object.__new__(datasets_module.CustomAudioDataset)
    dataset.data = [{"prompt": "Transcribe.", "audio": str(audio_path)}]
    if with_tokenizer:
        with pytest.raises(ValueError, match="'output_tokens' field"):
            dataset.sample(
                tokenizer=_as_tokenizer(_Tokenizer()),
                num_requests=1,
                output_len=output_len,
            )
    else:
        sample = dataset.sample(tokenizer=None, num_requests=1, output_len=output_len)[
            0
        ]
        assert sample.expected_output_len == 256
        assert sample.multi_modal_data is not None
