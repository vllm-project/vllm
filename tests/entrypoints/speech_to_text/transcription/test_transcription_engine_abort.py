# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A transcription aborted inside the engine fails with 503 instead of
returning the chunks finished so far as a 200 transcript."""

import io
import threading
import time
from http import HTTPStatus
from typing import Any

import numpy as np
import pytest
import requests
import soundfile as sf
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer
from vllm.multimodal.media.audio import load_audio

MODEL_NAME = "openai/whisper-large-v3-turbo"

# Whisper splits audio into 30 s chunks, one engine request each. With one
# sequence at a time, the chunks queue behind each other, so a pause that
# lands while several are still waiting must abort the request mid-way.
NUM_REPEATS = 30
MIN_WAITING_CHUNKS = 8
POLL_TIMEOUT_S = 120


@pytest.fixture(scope="module")
def server():
    # One sequence at a time queues the chunks. Whisper's 448-token context would
    # clamp the default batch budget below one 1500-token audio item.
    args = ["--max-num-seqs", "1", "--max-num-batched-tokens", "1500"]
    env_dict = {"VLLM_SERVER_DEV_MODE": "1"}
    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as remote_server:
        yield remote_server


def _long_audio(mary_had_lamb) -> bytes:
    # 16 kHz keeps the upload well under the 25 MB default limit.
    audio, sr = load_audio(mary_had_lamb, sr=16000)
    # Silence between repeats keeps the chunk split points stable.
    audio = np.tile(np.pad(audio, (0, 1600)), NUM_REPEATS)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return buffer.getvalue()


def _num_requests_waiting(server: RemoteOpenAIServer) -> float:
    text = requests.get(server.url_for("metrics"), timeout=10).text
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == "vllm:num_requests_waiting"
    )


def _transcribe(server: RemoteOpenAIServer, audio: bytes) -> requests.Response:
    return requests.post(
        server.url_for("v1/audio/transcriptions"),
        files={"file": ("audio.wav", audio, "audio/wav")},
        data={"model": MODEL_NAME, "language": "en", "temperature": "0"},
        timeout=POLL_TIMEOUT_S,
    )


def test_engine_abort_mid_transcription_returns_503(server, mary_had_lamb):
    audio = _long_audio(mary_had_lamb)
    result: dict[str, Any] = {}
    done = threading.Event()

    def transcribe():
        try:
            result["response"] = _transcribe(server, audio)
        finally:
            done.set()

    thread = threading.Thread(target=transcribe)
    thread.start()
    try:
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while _num_requests_waiting(server) < MIN_WAITING_CHUNKS:
            assert not done.is_set(), "transcription finished before the pause"
            assert time.monotonic() < deadline, "chunks never queued"
            time.sleep(0.05)

        pause = requests.post(
            server.url_for("pause"),
            params={"mode": "abort", "clear_cache": "false"},
            timeout=60,
        )
        assert pause.status_code == HTTPStatus.OK
        assert done.wait(timeout=60), "transcription did not return after the abort"
    finally:
        assert (
            requests.post(server.url_for("resume"), timeout=10).status_code
            == HTTPStatus.OK
        )
        thread.join(timeout=60)

    response = result["response"]
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE, response.text
    assert response.json()["error"]["type"] == "ServiceUnavailableError"

    mary_had_lamb.seek(0)
    retry = _transcribe(server, mary_had_lamb.read())
    assert retry.status_code == HTTPStatus.OK, retry.text
    assert "Mary had a little lamb" in retry.json()["text"]
