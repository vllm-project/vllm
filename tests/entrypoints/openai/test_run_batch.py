# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest

from vllm.assets.audio import AudioAsset
from vllm.entrypoints.openai import run_batch as run_batch_module
from vllm.entrypoints.openai.run_batch import (
    BatchProgressTracker,
    BatchRequestOutput,
    batch_output_writer,
    dispatch_batch,
    download_bytes_from_url,
    is_descriptor_alias,
    is_same_local_file,
    local_input_path,
    needs_staging,
    url_matches,
    validate_batch,
    validate_run_batch_args,
    write_finished,
)
from vllm.exceptions import VLLMValidationError

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
    mock_resp = MagicMock()
    mock_resp.status = status
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
        "vllm.entrypoints.openai.run_batch.aiohttp.ClientSession",
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
        "vllm.entrypoints.openai.run_batch.aiohttp.ClientSession",
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


# ---------------------------------------------------------------------------
# Unit tests for streaming batch execution
# ---------------------------------------------------------------------------


def test_validate_batch_counts_requests(tmp_path):
    """Blank lines are skipped and do not count towards the request total."""
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(INPUT_BATCH + "\n\n")

    assert validate_batch(str(input_path)) == len(INPUT_BATCH.strip().split("\n"))


def test_validate_batch_rejects_malformed_request(tmp_path):
    """A malformed request is rejected before any of the batch is run.

    Requests are parsed up front so that an invalid line fails the whole batch
    without leaving partial output behind.
    """
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(INPUT_BATCH + "\n" + INVALID_INPUT_BATCH + "\n")

    with pytest.raises(pydantic.ValidationError):
        validate_batch(str(input_path))


@pytest.mark.asyncio
async def test_dispatch_batch_writes_a_response_per_request(tmp_path):
    """Every request must produce exactly one response line.

    Responses finish out of order and are written in groups as they complete,
    so a missed drain would silently truncate the output.
    """
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(INPUT_BATCH + "\n")
    output_path = tmp_path / "output.jsonl"

    requests = [json.loads(line) for line in INPUT_BATCH.strip().split("\n")]

    with open(output_path, "w", encoding="utf-8") as output_file:
        # An empty registry routes every request to an error response, which
        # exercises the dispatch loop without starting an engine.
        await dispatch_batch(
            str(input_path),
            output_file,
            {},
            BatchProgressTracker(),
            max_inflight=2,
        )

    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == len(requests)
    assert {
        BatchRequestOutput.model_validate_json(line).custom_id for line in lines
    } == {request["custom_id"] for request in requests}


@pytest.mark.asyncio
async def test_batch_output_writer_persists_before_completion(tmp_path):
    """Responses reach disk while the batch is still running.

    This is what makes an interrupted run recoverable rather than a total loss.
    """
    output_path = tmp_path / "output.jsonl"

    async with batch_output_writer(str(output_path), None) as output_file:
        print("first", file=output_file)
        output_file.flush()
        assert output_path.read_text() == "first\n"
        print("second", file=output_file)

    assert output_path.read_text() == "first\nsecond\n"


def test_is_same_local_file_detects_aliases(tmp_path):
    """Aliases of one file must be recognised.

    Responses are written while the batch runs, so an output path that aliases
    the input would truncate it before it has been read.
    """
    target = tmp_path / "batch.jsonl"
    target.write_text("{}\n")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n")

    assert is_same_local_file(str(target), str(target))
    assert is_same_local_file(str(target), str(link))
    assert is_same_local_file(str(target), f"{tmp_path}/./batch.jsonl")

    assert not is_same_local_file(str(target), str(other))
    assert not is_same_local_file(str(target), "https://example.com/output.jsonl")


@pytest.mark.asyncio
async def test_dispatch_batch_counts_error_responses_as_completed(tmp_path):
    """Synthesized error responses advance progress like any other response.

    They never reach run_request, so counting only dispatched requests leaves
    the bar short of the batch size for inputs containing an unsupported URL.
    """
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(INPUT_BATCH + "\n")
    output_path = tmp_path / "output.jsonl"
    num_requests = len(INPUT_BATCH.strip().split("\n"))

    tracker = BatchProgressTracker()
    with open(output_path, "w", encoding="utf-8") as output_file:
        with tracker.pbar(total=num_requests) as pbar:
            # An empty registry makes every response a synthesized error.
            await dispatch_batch(
                str(input_path), output_file, {}, tracker, max_inflight=2
            )
        assert pbar.n == num_requests


@pytest.mark.asyncio
async def test_local_input_path_stages_a_stream(tmp_path):
    """A non-seekable input must be staged so the batch can be read twice.

    The batch is validated in one pass and run in another; a pipe would be
    drained by the first, leaving the second with nothing.
    """
    payload = '{"custom_id": "request-1"}\n'
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload.encode())
    os.close(write_fd)

    async with local_input_path(f"/dev/fd/{read_fd}", str(tmp_path)) as path:
        with open(path, encoding="utf-8") as f:
            assert f.read() == payload
        with open(path, encoding="utf-8") as f:
            assert f.read() == payload


def test_max_inflight_must_be_positive():
    """A non-positive bound would silently serialise the batch."""
    with pytest.raises(ValueError, match="max-inflight"):
        validate_run_batch_args(SimpleNamespace(max_inflight=0))


@pytest.mark.asyncio
async def test_dispatch_batch_respects_max_inflight(tmp_path, monkeypatch):
    """No more than max_inflight requests may be alive at once.

    This bound is what keeps frontend memory flat; without it the whole input
    file is submitted at once and memory grows with the batch.
    """
    requests = [
        {"custom_id": f"request-{i}", "method": "POST", "url": "/v1/chat/completions"}
        for i in range(50)
    ]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(r) for r in requests) + "\n")
    output_path = tmp_path / "output.jsonl"

    live = 0
    peak = 0

    async def fake_run_one_request(request_json, endpoint_registry):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return BatchRequestOutput(
            id="vllm-test",
            custom_id=json.loads(request_json)["custom_id"],
            response=None,
            error=None,
        )

    monkeypatch.setattr(run_batch_module, "run_one_request", fake_run_one_request)

    max_inflight = 8
    with open(output_path, "w", encoding="utf-8") as output_file:
        await dispatch_batch(
            str(input_path),
            output_file,
            {},
            BatchProgressTracker(),
            max_inflight,
        )

    assert peak <= max_inflight
    assert len(output_path.read_text().strip().split("\n")) == len(requests)


def _make_streaming_aiohttp_mocks(body: bytes, chunk_size: int = 8):
    """Mock a ClientSession whose GET body is read with content.iter_chunked."""

    async def iter_chunked(_size):
        for start in range(0, len(body), chunk_size):
            yield body[start : start + chunk_size]

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content.iter_chunked = iter_chunked
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


@pytest.mark.asyncio
async def test_local_input_path_downloads_a_url(tmp_path):
    """A URL body is staged to a local file, not held in memory."""
    payload = ("\n".join(INPUT_BATCH.strip().split("\n")[:2]) + "\n").encode()
    session = _make_streaming_aiohttp_mocks(payload)

    with patch(
        "vllm.entrypoints.openai.run_batch.aiohttp.ClientSession",
        return_value=session,
    ):
        async with local_input_path(
            "https://example.com/batch.jsonl", str(tmp_path)
        ) as path:
            assert path != "https://example.com/batch.jsonl"
            with open(path, "rb") as f:
                assert f.read() == payload
            assert validate_batch(path) == 2


@pytest.mark.asyncio
async def test_batch_output_writer_uploads_url_output_once(tmp_path, monkeypatch):
    """A URL destination is staged locally and uploaded after the batch."""
    uploaded = {}

    async def fake_upload_data(output_url, data_or_file, from_file):
        uploaded["url"] = output_url
        uploaded["from_file"] = from_file
        with open(data_or_file, encoding="utf-8") as f:
            uploaded["content"] = f.read()

    monkeypatch.setattr(run_batch_module, "upload_data", fake_upload_data)

    async with batch_output_writer(
        "https://example.com/output.jsonl", str(tmp_path)
    ) as output_file:
        print("first", file=output_file)
        print("second", file=output_file)
        assert not uploaded, "upload must wait until the batch finishes"

    assert uploaded == {
        "url": "https://example.com/output.jsonl",
        "from_file": True,
        "content": "first\nsecond\n",
    }


@pytest.mark.asyncio
async def test_batch_output_writer_url_output_uploads_nothing_on_failure(
    tmp_path, monkeypatch
):
    """A failed batch uploads nothing, so a URL output is all or nothing.

    Incremental writes only make a local output recoverable; this pins the
    documented limitation so it cannot change silently.
    """
    upload = AsyncMock()
    monkeypatch.setattr(run_batch_module, "upload_data", upload)

    with pytest.raises(RuntimeError):
        async with batch_output_writer(
            "https://example.com/output.jsonl", str(tmp_path)
        ) as output_file:
            print("first", file=output_file)
            raise RuntimeError("batch failed")

    upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_batch_cancels_inflight_on_failure(tmp_path, monkeypatch):
    """An aborted batch must not leave requests running behind it."""
    requests = [
        {"custom_id": f"request-{i}", "method": "POST", "url": "/v1/chat/completions"}
        for i in range(8)
    ]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(r) for r in requests) + "\n")

    started: list[asyncio.Task | None] = []

    async def fake_run_one_request(request_json, endpoint_registry):
        started.append(asyncio.current_task())
        if len(started) == 1:
            raise RuntimeError("request blew up")
        await asyncio.Event().wait()  # never finishes on its own

    monkeypatch.setattr(run_batch_module, "run_one_request", fake_run_one_request)

    with (
        open(tmp_path / "output.jsonl", "w", encoding="utf-8") as output_file,
        pytest.raises(RuntimeError, match="request blew up"),
    ):
        await dispatch_batch(
            str(input_path),
            output_file,
            {},
            BatchProgressTracker(),
            max_inflight=4,
        )

    await asyncio.sleep(0)
    assert all(task is not None and task.done() for task in started), (
        "requests were left in flight"
    )


@pytest.mark.asyncio
async def test_local_input_path_stages_a_descriptor_alias(tmp_path):
    """A descriptor alias is staged even though it stats as a regular file.

    Reopening /dev/fd/N shares the descriptor's offset on some platforms, so
    the second pass would start at EOF and the batch would run nothing.
    """
    source = tmp_path / "batch.jsonl"
    source.write_text(INPUT_BATCH + "\n")
    expected = len(INPUT_BATCH.strip().split("\n"))

    with open(source, encoding="utf-8") as handle:
        alias = f"/dev/fd/{handle.fileno()}"
        async with local_input_path(alias, str(tmp_path)) as path:
            assert path != alias
            # Readable twice, which is what the two passes need.
            assert validate_batch(path) == expected
            assert validate_batch(path) == expected


@pytest.mark.asyncio
async def test_unsupported_url_error_lists_registry_endpoints(tmp_path):
    """The error names the endpoints the registry actually serves.

    Built from the registry rather than a literal, so the two cannot drift.
    """
    registry = {
        "widgets": {
            "url": "/v1/widgets",
            "handler_getter": lambda: None,
            "wrapper_fn": None,
        }
    }
    request = json.loads(INPUT_BATCH.strip().split("\n")[0])
    request["url"] = "/v1/unsupported"
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(request) + "\n")
    output_path = tmp_path / "output.jsonl"

    with open(output_path, "w", encoding="utf-8") as output_file:
        await dispatch_batch(
            str(input_path), output_file, registry, BatchProgressTracker(), 4
        )

    error = BatchRequestOutput.model_validate_json(
        output_path.read_text().strip()
    ).error
    assert "/v1/unsupported" in error
    assert "/v1/widgets" in error


def test_only_descriptor_aliases_need_staging(tmp_path):
    """An ordinary file is read in place; only descriptor aliases are copied.

    Staging every path under /dev would copy a batch held on tmpfs, which is
    re-readable and may be large.
    """
    ordinary = tmp_path / "batch.jsonl"
    ordinary.write_text("{}\n")

    assert not needs_staging(str(ordinary))
    assert not is_descriptor_alias("/dev/shm/batch.jsonl")
    assert not is_descriptor_alias("/dev/null")

    assert is_descriptor_alias("/dev/stdin")
    assert is_descriptor_alias("/dev/fd/3")
    assert is_descriptor_alias("/proc/self/fd/12")
    assert is_descriptor_alias("/proc/451/fd/7")


@pytest.mark.asyncio
async def test_write_finished_keeps_responses_that_completed_with_a_failure(tmp_path):
    """A failure must not discard responses that finished alongside it.

    Requests complete in groups, so raising on the first failure would throw
    away work that had already succeeded.
    """

    async def succeed(custom_id):
        return BatchRequestOutput(
            id="vllm-test", custom_id=custom_id, response=None, error=None
        )

    async def fail():
        raise RuntimeError("request blew up")

    # The failure is first, so an implementation that raises immediately would
    # write nothing.
    tasks = [asyncio.create_task(fail())] + [
        asyncio.create_task(succeed(f"request-{i}")) for i in range(3)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    output_path = tmp_path / "output.jsonl"
    with open(output_path, "w", encoding="utf-8") as output_file:
        failure = write_finished(tasks, output_file, BatchProgressTracker())

    assert isinstance(failure, RuntimeError)
    written = [
        BatchRequestOutput.model_validate_json(line).custom_id
        for line in output_path.read_text().strip().split("\n")
    ]
    assert written == ["request-0", "request-1", "request-2"]


def test_url_matches_accepts_versioned_endpoints():
    """An endpoint with no version of its own is also served under one.

    Score and rerank are reachable as /score and /v1/score. The previous suffix
    match also accepted unrelated paths ending in the same segment.
    """
    for url in ("/score", "/v1/score", "/v2/score", "/v10/score"):
        assert url_matches("/score", url), url
    for url in ("/foo/score", "/a/b/score", "/scorecard", "/vX/score"):
        assert not url_matches("/score", url), url

    # An endpoint that already names its version matches only that version.
    assert url_matches("/v1/embeddings", "/v1/embeddings")
    for url in ("/v2/embeddings", "/v1/v1/embeddings"):
        assert not url_matches("/v1/embeddings", url), url
