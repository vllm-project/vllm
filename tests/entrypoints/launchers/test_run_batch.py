# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import tempfile
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import vllm.envs as envs
from vllm.assets.audio import AudioAsset
from vllm.connections import HTTPConnection
from vllm.entrypoints.launchers.run_batch import (
    BatchRequestOutput,
    BatchTranscriptionRequest,
    download_bytes_from_url,
    make_transcription_wrapper,
    upload_data,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.exceptions import VLLMValidationError
from vllm.utils.mem_constants import MiB_bytes

CHAT_MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
REASONING_MODEL_NAME = "Qwen/Qwen3-0.6B"
SPEECH_LARGE_MODEL_NAME = "openai/whisper-large-v3"
SPEECH_SMALL_MODEL_NAME = "openai/whisper-small"

INPUT_BATCH = "\n".join(
    json.dumps(req)
    for req in [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": CHAT_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 1000,
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": CHAT_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an unhelpful assistant.",
                    },
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 1000,
            },
        },
        {
            "custom_id": "request-3",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "NonExistModel",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an unhelpful assistant.",
                    },
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 1000,
            },
        },
        {
            "custom_id": "request-4",
            "method": "POST",
            "url": "/bad_url",
            "body": {
                "model": CHAT_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an unhelpful assistant.",
                    },
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 1000,
            },
        },
        {
            "custom_id": "request-5",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "stream": "True",
                "model": CHAT_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an unhelpful assistant.",
                    },
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 1000,
            },
        },
    ]
)

INVALID_INPUT_BATCH = "\n".join(
    json.dumps(req)
    for req in [
        {
            "invalid_field": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": CHAT_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 1000,
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": CHAT_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are an unhelpful assistant."},
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 1000,
            },
        },
    ]
)

INPUT_EMBEDDING_BATCH = "\n".join(
    json.dumps(req)
    for req in [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": EMBEDDING_MODEL_NAME,
                "input": "You are a helpful assistant.",
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": EMBEDDING_MODEL_NAME,
                "input": "You are an unhelpful assistant.",
            },
        },
        {
            "custom_id": "request-3",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": EMBEDDING_MODEL_NAME,
                "input": "Hello world!",
            },
        },
        {
            "custom_id": "request-4",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "NonExistModel",
                "input": "Hello world!",
            },
        },
    ]
)

_SCORE_RERANK_DOCUMENTS = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
]

INPUT_SCORE_BATCH = "\n".join(
    json.dumps(req)
    for req in [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/score",
            "body": {
                "model": RERANKER_MODEL_NAME,
                "queries": "What is the capital of France?",
                "documents": _SCORE_RERANK_DOCUMENTS,
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/score",
            "body": {
                "model": RERANKER_MODEL_NAME,
                "queries": "What is the capital of France?",
                "documents": _SCORE_RERANK_DOCUMENTS,
            },
        },
    ]
)

INPUT_RERANK_BATCH = "\n".join(
    json.dumps(req)
    for req in [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/rerank",
            "body": {
                "model": RERANKER_MODEL_NAME,
                "query": "What is the capital of France?",
                "documents": _SCORE_RERANK_DOCUMENTS,
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/rerank",
            "body": {
                "model": RERANKER_MODEL_NAME,
                "query": "What is the capital of France?",
                "documents": _SCORE_RERANK_DOCUMENTS,
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v2/rerank",
            "body": {
                "model": RERANKER_MODEL_NAME,
                "query": "What is the capital of France?",
                "documents": _SCORE_RERANK_DOCUMENTS,
            },
        },
    ]
)

INPUT_REASONING_BATCH = "\n".join(
    json.dumps(req)
    for req in [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": REASONING_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Solve this math problem: 2+2=?"},
                ],
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": REASONING_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
            },
        },
    ]
)

MINIMAL_WAV_BASE64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQQAAAAAAP9/"
_EXACT_LIMIT_AUDIO = b"a" * MiB_bytes
_OVERSIZED_BASE64_AUDIO = "A" * (4 * ((MiB_bytes + 2) // 3))
_OVERSIZED_HTTP_AUDIO = b"b" * (MiB_bytes + 1)


class _BatchHTTPServer(ThreadingHTTPServer):
    request_paths: list[str]


class _BatchHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, _BatchHTTPServer)
        server.request_paths.append(self.path)

        if self.path == "/exact-limit":
            self._send_body(_EXACT_LIMIT_AUDIO)
            return

        if self.path == "/oversized-chunked":
            self.send_response(200)
            self.end_headers()
            self._write_body(_OVERSIZED_HTTP_AUDIO)
            return

        self.send_error(404, "Unknown batch HTTP test path")

    def _send_body(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_body(body)

    def _write_body(self, body: bytes) -> None:
        try:
            for offset in range(0, len(body), 64 * 1024):
                self.wfile.write(body[offset : offset + 64 * 1024])
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, fmt: str, *args: object) -> None:
        return


@pytest.fixture
def batch_http_server() -> Generator[_BatchHTTPServer, None, None]:
    server = _BatchHTTPServer(("127.0.0.1", 0), _BatchHTTPHandler)
    server.request_paths = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _batch_http_url(server: _BatchHTTPServer, path: str) -> str:
    return f"http://localhost:{server.server_port}{path}"


async def _close_async_connection(connection: HTTPConnection) -> None:
    if connection._async_client is not None:
        await connection._async_client.close()


INPUT_TRANSCRIPTION_BATCH = (
    json.dumps(
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/audio/transcriptions",
            "body": {
                "model": SPEECH_LARGE_MODEL_NAME,
                "file_url": f"data:audio/wav;base64,{MINIMAL_WAV_BASE64}",
                "response_format": "json",
            },
        }
    )
    + "\n"
)

INPUT_TRANSCRIPTION_HTTP_BATCH = (
    json.dumps(
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/audio/transcriptions",
            "body": {
                "model": SPEECH_LARGE_MODEL_NAME,
                "file_url": AudioAsset("mary_had_lamb").url,
                "language": "en",
                "response_format": "json",
            },
        }
    )
    + "\n"
)

INPUT_TRANSLATION_BATCH = (
    json.dumps(
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/audio/translations",
            "body": {
                "model": SPEECH_SMALL_MODEL_NAME,
                "file_url": AudioAsset("mary_had_lamb").url,
                "response_format": "text",
                "language": "it",
                "to_language": "en",
                "temperature": 0.0,
            },
        }
    )
    + "\n"
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    },
}

INPUT_TOOL_CALLING_BATCH = json.dumps(
    {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": REASONING_MODEL_NAME,
            "messages": [
                {"role": "user", "content": "What is the weather in San Francisco?"},
            ],
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
            "max_tokens": 1000,
        },
    }
)


def test_empty_file():
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write("")
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                EMBEDDING_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        assert contents.strip() == ""


def test_completions():
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                CHAT_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)


def test_completions_invalid_input():
    """
    Ensure that we fail when the input doesn't conform to the openai api.
    """
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INVALID_INPUT_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                CHAT_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode != 0, f"{proc=}"


def test_embeddings():
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_EMBEDDING_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                EMBEDDING_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)


@pytest.mark.parametrize("input_batch", [INPUT_SCORE_BATCH, INPUT_RERANK_BATCH])
def test_score(input_batch):
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(input_batch)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                RERANKER_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)

            # Ensure that there is no error in the response.
            line_dict = json.loads(line)
            assert isinstance(line_dict, dict)
            assert line_dict["error"] is None


def test_reasoning_parser():
    """
    Test that reasoning_parser parameter works correctly in run_batch.
    """
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_REASONING_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                REASONING_MODEL_NAME,
                "--reasoning-parser",
                "qwen3",
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)

            # Ensure that there is no error in the response.
            line_dict = json.loads(line)
            assert isinstance(line_dict, dict)
            assert line_dict["error"] is None

            # Check that reasoning is present and not empty
            reasoning = line_dict["response"]["body"]["choices"][0]["message"][
                "reasoning"
            ]
            assert reasoning is not None
            assert len(reasoning) > 0


def test_transcription():
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_TRANSCRIPTION_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                SPEECH_LARGE_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        print(f"\n\ncontents: {contents}\n\n")
        for line in contents.strip().split("\n"):
            BatchRequestOutput.model_validate_json(line)

            line_dict = json.loads(line)
            assert isinstance(line_dict, dict)
            assert line_dict["error"] is None

            response_body = line_dict["response"]["body"]
            assert response_body is not None
            assert "text" in response_body
            assert "usage" in response_body


def test_transcription_http_url():
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_TRANSCRIPTION_HTTP_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                SPEECH_LARGE_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            BatchRequestOutput.model_validate_json(line)

            line_dict = json.loads(line)
            assert isinstance(line_dict, dict)
            assert line_dict["error"] is None

            response_body = line_dict["response"]["body"]
            assert response_body is not None
            assert "text" in response_body
            assert "usage" in response_body

            transcription_text = response_body["text"]
            assert "Mary had a little lamb" in transcription_text


def test_translation():
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_TRANSLATION_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                SPEECH_SMALL_MODEL_NAME,
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            BatchRequestOutput.model_validate_json(line)

            line_dict = json.loads(line)
            assert isinstance(line_dict, dict)
            assert line_dict["error"] is None

            response_body = line_dict["response"]["body"]
            assert response_body is not None
            assert "text" in response_body

            translation_text = response_body["text"]
            translation_text_lower = str(translation_text).strip().lower()
            assert "mary" in translation_text_lower or "lamb" in translation_text_lower


def test_tool_calling():
    """
    Test that tool calling works correctly in run_batch.
    Verifies that requests with tools return tool_calls in the response.
    """
    with (
        tempfile.NamedTemporaryFile("w") as input_file,
        tempfile.NamedTemporaryFile("r") as output_file,
    ):
        input_file.write(INPUT_TOOL_CALLING_BATCH)
        input_file.flush()
        proc = subprocess.Popen(
            [
                "vllm",
                "run-batch",
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                REASONING_MODEL_NAME,
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
            ],
        )
        proc.communicate()
        proc.wait()
        assert proc.returncode == 0, f"{proc=}"

        contents = output_file.read()
        for line in contents.strip().split("\n"):
            if not line.strip():  # Skip empty lines
                continue
            # Ensure that the output format conforms to the openai api.
            # Validation should throw if the schema is wrong.
            BatchRequestOutput.model_validate_json(line)

            # Ensure that there is no error in the response.
            line_dict = json.loads(line)
            assert isinstance(line_dict, dict)
            assert line_dict["error"] is None

            # Check that tool_calls are present in the response
            # With tool_choice="required", the model must call a tool
            response_body = line_dict["response"]["body"]
            assert response_body is not None
            message = response_body["choices"][0]["message"]
            assert "tool_calls" in message
            tool_calls = message.get("tool_calls")
            # With tool_choice="required", tool_calls must be present and non-empty
            assert tool_calls is not None
            assert isinstance(tool_calls, list)
            assert len(tool_calls) > 0
            # Verify tool_calls have the expected structure
            for tool_call in tool_calls:
                assert "id" in tool_call
                assert "type" in tool_call
                assert tool_call["type"] == "function"
                assert "function" in tool_call
                assert "name" in tool_call["function"]
                assert "arguments" in tool_call["function"]
                # Verify the tool name matches our tool definition
                assert tool_call["function"]["name"] == "get_current_weather"


# ---------------------------------------------------------------------------
# Unit tests for download_bytes_from_url SSRF protection
# ---------------------------------------------------------------------------


def _make_aiohttp_mocks(response_data: bytes = b"fake-data", status: int = 200):
    """Create mock objects that simulate aiohttp.ClientSession context managers."""

    async def iter_chunked(chunk_size: int):
        del chunk_size
        yield response_data

    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.content_length = len(response_data)
    mock_resp.content.iter_chunked = iter_chunked
    mock_resp.read = AsyncMock(return_value=response_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


@pytest.mark.asyncio
async def test_download_bytes_data_url_bypasses_domain_check():
    """data: URLs must work regardless of the domain allowlist."""
    data_url = f"data:audio/wav;base64,{MINIMAL_WAV_BASE64}"
    result = await download_bytes_from_url(
        data_url, allowed_media_domains=["example.com"]
    )
    assert isinstance(result, bytes)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_download_bytes_rejects_disallowed_domain():
    """HTTP URLs whose hostname is not in the allowlist must be rejected."""
    url = "https://evil.internal/secret"
    with pytest.raises(VLLMValidationError, match="allowed domains") as exc_info:
        await download_bytes_from_url(url, allowed_media_domains=["example.com"])
    # URL validation failures carry structured metadata for the frontend.
    assert exc_info.value.parameter == "url"
    assert exc_info.value.value == "evil.internal"


@pytest.mark.asyncio
async def test_download_bytes_rejects_unsupported_scheme():
    """Unsupported URL schemes are rejected with structured metadata."""
    with pytest.raises(VLLMValidationError, match="Unsupported URL scheme") as exc_info:
        await download_bytes_from_url("ftp://example.com/file")
    assert exc_info.value.parameter == "url"
    assert exc_info.value.value == "ftp"


@pytest.mark.asyncio
async def test_download_bytes_rejects_cloud_metadata_ip():
    """Cloud metadata endpoints must be blocked when an allowlist is set."""
    url = "http://169.254.169.254/latest/meta-data/"
    with pytest.raises(VLLMValidationError, match="allowed domains"):
        await download_bytes_from_url(url, allowed_media_domains=["example.com"])


@pytest.mark.asyncio
async def test_download_bytes_rejects_internal_ip():
    """Private-range IPs must be blocked when an allowlist is set."""
    for internal_url in [
        "http://10.0.0.1/secret",
        "http://192.168.1.1/admin",
        "http://127.0.0.1:8080/internal",
    ]:
        with pytest.raises(VLLMValidationError, match="allowed domains"):
            await download_bytes_from_url(
                internal_url, allowed_media_domains=["example.com"]
            )


@pytest.mark.asyncio
async def test_download_bytes_allows_permitted_domain():
    """HTTP URLs whose hostname IS in the allowlist must be fetched."""
    url = "https://example.com/audio.wav"
    expected = b"audio-bytes"
    mock_session = _make_aiohttp_mocks(expected)

    with patch(
        "vllm.entrypoints.launchers.run_batch.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        result = await download_bytes_from_url(
            url, allowed_media_domains=["example.com"]
        )
    assert result == expected


@pytest.mark.asyncio
async def test_download_bytes_no_allowlist_permits_any_domain():
    """Without an allowlist all HTTP URLs must be attempted (backward compat)."""
    url = "https://any-domain.example.org/file.wav"
    expected = b"some-data"
    mock_session = _make_aiohttp_mocks(expected)

    with patch(
        "vllm.entrypoints.launchers.run_batch.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        result = await download_bytes_from_url(url, allowed_media_domains=None)
    assert result == expected


@pytest.mark.asyncio
async def test_download_bytes_empty_allowlist_denies_all():
    """An empty allowlist must deny all HTTP URLs (least privilege)."""
    url = "https://any-domain.example.org/file.wav"
    with pytest.raises(VLLMValidationError, match="allowed domains"):
        await download_bytes_from_url(url, allowed_media_domains=[])


@pytest.mark.asyncio
async def test_download_bytes_unsupported_scheme():
    """Unsupported URL schemes must be rejected regardless of allowlist."""
    with pytest.raises(VLLMValidationError, match="Unsupported URL scheme"):
        await download_bytes_from_url("ftp://example.com/file.wav")

    with pytest.raises(VLLMValidationError, match="Unsupported URL scheme"):
        await download_bytes_from_url(
            "ftp://example.com/file.wav",
            allowed_media_domains=["example.com"],
        )


@pytest.mark.asyncio
async def test_download_bytes_backslash_bypass():
    """Backslash-@ URL confusion must not bypass the allowlist.

    urllib3.parse_url() and aiohttp/yarl disagree on backslash-before-@.
    The fix normalizes through urllib3 before handing to aiohttp.
    """
    bypass_url = "http://allowed.example.com\\@evil.internal/secret"
    with pytest.raises(VLLMValidationError, match="allowed domains"):
        await download_bytes_from_url(
            bypass_url, allowed_media_domains=["evil.internal"]
        )


@pytest.mark.asyncio
async def test_transcription_wrapper_rejects_oversized_data_url_before_decode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)
    handler = AsyncMock()
    wrapped_handler = make_transcription_wrapper(is_translation=False)(handler)
    request = BatchTranscriptionRequest.model_validate(
        {
            "model": SPEECH_LARGE_MODEL_NAME,
            "file_url": f"data:audio/wav;base64,{_OVERSIZED_BASE64_AUDIO}",
            "response_format": "json",
        }
    )

    with patch("vllm.entrypoints.launchers.run_batch.base64.b64decode") as decode:
        response = await wrapped_handler(request)

    assert isinstance(response, ErrorResponse)
    assert "Maximum file size exceeded" in response.error.message
    decode.assert_not_called()
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_download_bytes_allows_http_body_at_audio_limit(
    monkeypatch: pytest.MonkeyPatch,
    batch_http_server: _BatchHTTPServer,
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)
    connection = HTTPConnection()
    monkeypatch.setattr(
        "vllm.entrypoints.launchers.run_batch.global_http_connection",
        connection,
    )

    try:
        result = await download_bytes_from_url(
            _batch_http_url(batch_http_server, "/exact-limit")
        )
        assert result == _EXACT_LIMIT_AUDIO
    finally:
        await _close_async_connection(connection)


@pytest.mark.asyncio
async def test_transcription_wrapper_rejects_oversized_http_before_handler(
    monkeypatch: pytest.MonkeyPatch,
    batch_http_server: _BatchHTTPServer,
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)
    connection = HTTPConnection()
    monkeypatch.setattr(
        "vllm.entrypoints.launchers.run_batch.global_http_connection",
        connection,
    )
    handler = AsyncMock()
    wrapped_handler = make_transcription_wrapper(is_translation=False)(handler)
    request = BatchTranscriptionRequest.model_validate(
        {
            "model": SPEECH_LARGE_MODEL_NAME,
            "file_url": _batch_http_url(batch_http_server, "/oversized-chunked"),
            "response_format": "json",
        }
    )

    try:
        response = await wrapped_handler(request)
    finally:
        await _close_async_connection(connection)

    assert isinstance(response, ErrorResponse)
    assert "Maximum file size exceeded" in response.error.message
    handler.assert_not_awaited()


# ---------------------------------------------------------------------------
# Unit tests for upload_data retry behavior
# ---------------------------------------------------------------------------


def _make_aiohttp_put_session(status: int = 200, body_text: str = ""):
    """Mock an aiohttp.ClientSession whose PUT returns the given status."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=body_text)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.put = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


@pytest.mark.asyncio
async def test_upload_data_uploads_once_on_success():
    """A successful upload must not be retried (regression guard)."""
    session = _make_aiohttp_put_session(status=200)
    with patch(
        "vllm.entrypoints.launchers.run_batch.aiohttp.ClientSession",
        return_value=session,
    ):
        await upload_data(
            "https://example.com/output.jsonl", "payload", from_file=False
        )
    assert session.put.call_count == 1


@pytest.mark.asyncio
async def test_upload_data_error_includes_awaited_response_body():
    """A failed upload must surface the awaited response body, not a coroutine."""
    session = _make_aiohttp_put_session(status=500, body_text="server-error-detail")
    with (
        patch(
            "vllm.entrypoints.launchers.run_batch.aiohttp.ClientSession",
            return_value=session,
        ),
        patch(
            "vllm.entrypoints.launchers.run_batch.asyncio.sleep",
            AsyncMock(),
        ),
        pytest.raises(Exception) as exc_info,
    ):
        await upload_data(
            "https://example.com/output.jsonl", "payload", from_file=False
        )
    message = str(exc_info.value)
    assert "server-error-detail" in message
    assert "coroutine" not in message
