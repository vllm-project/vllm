#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
import soundfile as sf

import vllm.benchmarks.datasets.datasets as datasets_module
import vllm.benchmarks.lib.endpoint_request_func as request_func_module
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

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
        tokenizer=_Tokenizer(),
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
        tokenizer=_Tokenizer(),
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
