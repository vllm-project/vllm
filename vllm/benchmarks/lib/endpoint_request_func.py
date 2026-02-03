# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The request function for API endpoints."""

import io
import json
import os
import sys
import time
import traceback
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import aiohttp
import regex as re
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


class StreamedResponseHandler:
    """Handles streaming HTTP responses by accumulating chunks until complete
    messages are available."""

    def __init__(self):
        self.buffer = ""

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """Add a chunk of bytes to the buffer and return any complete
        messages."""
        messages = []
        if not chunk_bytes:
            return messages

        chunk_str = chunk_bytes.decode("utf-8")
        self.buffer += chunk_str

        # Split by double newlines (SSE message separator)
        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            # Filter out non-data channel in SSE, e.g. event:<xxx>
            for line in message.splitlines():
                if line.startswith("data:"):
                    messages.append(line)

        # if self.buffer is not empty, check if it is a complete message
        # by removing data: prefix and check if it is a valid JSON
        if self.buffer.startswith("data:"):
            message_content = self.buffer.removeprefix("data:").strip()
            if message_content == "[DONE]":
                messages.append(self.buffer.strip())
                self.buffer = ""
            elif message_content:
                try:
                    json.loads(message_content)
                    messages.append(self.buffer.strip())
                    self.buffer = ""
                except json.JSONDecodeError:
                    # Incomplete JSON, wait for more chunks.
                    pass

        return messages


@dataclass
class RequestFuncInput:
    """The input for the request function."""

    prompt: str | list[str]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: str | None = None
    logprobs: int | None = None
    extra_headers: dict | None = None
    extra_body: dict | None = None
    multi_modal_content: dict | list[dict] | None = None
    ignore_eos: bool = False
    language: str | None = None
    request_id: str | None = None


@dataclass
class RequestFuncOutput:
    """The output of the request function including metrics."""

    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    start_time: float = 0.0


class RequestFunc(Protocol):
    def __call__(
        self,
        request_func_input: RequestFuncInput,
        session: aiohttp.ClientSession,
        pbar: tqdm | None = None,
    ) -> Awaitable[RequestFuncOutput]: ...


def _validate_api_url(
    api_url: str,
    api_name: str,
    expected_suffixes: str | set[str],
) -> None:
    if isinstance(expected_suffixes, str):
        expected_suffixes = {expected_suffixes}

    expected_suffixes = {*expected_suffixes, "profile"}

    if not api_url.endswith(tuple(expected_suffixes)):
        raise ValueError(f"{api_name} URL must end with one of: {expected_suffixes}.")


def _update_payload_common(
    payload: dict[str, Any],
    request_func_input: RequestFuncInput,
) -> None:
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)


def _update_headers_common(
    headers: dict[str, Any],
    request_func_input: RequestFuncInput,
) -> None:
    if request_func_input.extra_headers:
        headers |= request_func_input.extra_headers
    if request_func_input.request_id:
        headers["x-request-id"] = request_func_input.request_id


def _get_headers(content_type: str | None = None) -> dict[str, str]:
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """The async request function for the OpenAI Completions API.

    Args:
        request_func_input: The input for the request function.
        pbar: The progress bar to display the progress.

    Returns:
        The output of the request function.
    """
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Completions API", "completions")

    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "prompt": request_func_input.prompt,
        "repetition_penalty": 1.0,
        "max_tokens": request_func_input.output_len,
        "logprobs": request_func_input.logprobs,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    _update_payload_common(payload, request_func_input)

    headers = _get_headers()
    _update_headers_common(headers, request_func_input)

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                first_chunk_received = False
                handler = StreamedResponseHandler()

                async for chunk_bytes in response.content.iter_any():
                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if message.startswith(":"):
                            continue

                        # NOTE: SSE packet may come with data:<payload> or data:<space><payload>
                        chunk = message.removeprefix("data:").strip()

                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                if first_chunk_received:
                    output.success = True
                else:
                    output.success = False
                    output.error = (
                        "Never received a valid chunk to calculate TTFT."
                        "This response will be marked as failed!"
                    )
                output.generated_text = generated_text
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def _get_chat_content(
    request_func_input: RequestFuncInput,
    mm_position: Literal["first", "last"] = "last",
) -> list[dict[str, Any]]:
    text_contents = [{"type": "text", "text": request_func_input.prompt}]

    mm_contents = []
    if request_func_input.multi_modal_content:
        mm_content = request_func_input.multi_modal_content
        if isinstance(mm_content, list):
            mm_contents.extend(request_func_input.multi_modal_content)
        elif isinstance(mm_content, dict):
            mm_contents.append(request_func_input.multi_modal_content)
        else:
            raise TypeError(
                "multi_modal_content must be a dict or list[dict] for openai-chat"
            )

    if mm_position == "first":
        return mm_contents + text_contents

    return text_contents + mm_contents


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

    content = _get_chat_content(request_func_input, mm_position=mm_position)

    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "messages": [
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    _update_payload_common(payload, request_func_input)

    headers = _get_headers("application/json")
    _update_headers_common(headers, request_func_input)

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if message.startswith(":"):
                            continue

                        # NOTE: SSE packet may come with data:<payload> or data:<space><payload>
                        chunk = message.removeprefix("data:").strip()

                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                # Count the CoT token as an output token
                                content = choices[0]["delta"].get("content") or choices[0]["delta"].get("reasoning_content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")

                            most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_audio(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    # Lazy import without PlaceholderModule to avoid vllm dep.
    import soundfile

    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Audio API", {"transcriptions", "translations"})

    content = [{"type": "text", "text": request_func_input.prompt}]
    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "max_completion_tokens": request_func_input.output_len,
        "stream": True,
        "language": "en",
        # Flattened due to multipart/form-data
        "stream_include_usage": True,
        "stream_continuous_usage_stats": True,
    }
    _update_payload_common(payload, request_func_input)

    headers = _get_headers()
    _update_headers_common(headers, request_func_input)

    # Send audio file
    def to_bytes(y, sr):
        buffer = io.BytesIO()
        soundfile.write(buffer, y, sr, format="WAV")
        buffer.seek(0)
        return buffer

    mm_audio = request_func_input.multi_modal_content
    if not isinstance(mm_audio, dict) or "audio" not in mm_audio:
        raise TypeError("multi_modal_content must be a dict containing 'audio'")
    with to_bytes(*mm_audio["audio"]) as f:
        form = aiohttp.FormData()
        form.add_field("file", f, content_type="audio/wav")
        for key, value in payload.items():
            form.add_field(key, str(value))

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, data=form, headers=headers
            ) as response:
                if response.status == 200:
                    handler = StreamedResponseHandler()

                    async for chunk_bytes in response.content.iter_any():
                        messages = handler.add_chunk(chunk_bytes)
                        for message in messages:
                            chunk = message.removeprefix("data:").strip()
                            if chunk != "[DONE]":
                                timestamp = time.perf_counter()
                                data = json.loads(chunk)

                                if choices := data.get("choices"):
                                    content = choices[0]["delta"].get("content")
                                    # First token
                                    if ttft == 0.0:
                                        ttft = timestamp - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

                                    generated_text += content or ""
                                elif usage := data.get("usage"):
                                    output.output_tokens = usage.get(
                                        "completion_tokens"
                                    )

                                most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def _run_pooling_request(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: dict[str, Any],
    headers: dict[str, Any],
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    st = time.perf_counter()
    output.start_time = st
    try:
        async with session.post(url=api_url, headers=headers, json=payload) as response:
            if response.status == 200:
                output.ttft = output.latency = time.perf_counter() - st

                if payload.get("encoding_format", "float") == "bytes":
                    metadata = json.loads(response.headers["metadata"])
                    usage = metadata.get("usage", {})
                else:
                    data = await response.json()
                    usage = data.get("usage", {})

                output.success = True
                output.generated_text = ""
                output.prompt_len = usage.get("prompt_tokens", 0)
            else:
                output.success = False
                output.error = response.reason or ""
    except Exception as e:
        output.success = False
        output.error = str(e)

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_embeddings(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Embeddings API", "embeddings")

    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "input": request_func_input.prompt,
        # Many embedding models have short context length,
        # this is to avoid dropping some of the requests.
        "truncate_prompt_tokens": -1,
    }
    _update_payload_common(payload, request_func_input)

    headers = _get_headers("application/json")
    _update_headers_common(headers, request_func_input)

    return await _run_pooling_request(
        session,
        api_url,
        payload=payload,
        headers=headers,
        pbar=pbar,
    )


async def async_request_vllm_rerank(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "vLLM score API", "rerank")

    assert (
        isinstance(request_func_input.prompt, list)
        and len(request_func_input.prompt) > 1
    )

    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "query": request_func_input.prompt[0],
        "documents": request_func_input.prompt[1:],
        # Many reranker models have short context length,
        # this is to avoid dropping some of the requests.
        "truncate_prompt_tokens": -1,
    }

    headers = _get_headers("application/json")
    _update_headers_common(headers, request_func_input)

    return await _run_pooling_request(
        session,
        api_url,
        payload=payload,
        headers=headers,
        pbar=pbar,
    )


async def async_request_openai_embeddings_chat(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Embeddings API", "embeddings")

    content = _get_chat_content(request_func_input, mm_position=mm_position)

    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "messages": [
            {"role": "user", "content": content},
        ],
        # Many embedding models have short context length,
        # this is to avoid dropping some of the requests.
        "truncate_prompt_tokens": -1,
    }
    _update_payload_common(payload, request_func_input)

    headers = _get_headers("application/json")
    _update_headers_common(headers, request_func_input)

    return await _run_pooling_request(
        session,
        api_url,
        payload=payload,
        headers=headers,
        pbar=pbar,
    )


def _try_extract_request_idx(request_func_input: RequestFuncInput):
    if request_func_input.request_id:
        match = re.search(r"(\d+)$", request_func_input.request_id)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

    return None


def _preprocess_clip(request_func_input: RequestFuncInput):
    if request_func_input.multi_modal_content:
        # Image input
        request_func_input.prompt = ""


def _preprocess_vlm2vec(request_func_input: RequestFuncInput):
    if request_func_input.multi_modal_content:
        request_idx = _try_extract_request_idx(request_func_input)

        # Adjust the ratio manually if needed.
        use_image_only_prompt = request_idx is None or request_idx % 2 == 0

        if use_image_only_prompt:
            # Image input
            request_func_input.prompt = "Represent the given image."
        else:
            # Text+Image input
            request_func_input.prompt = (
                f"Represent the given image with the following question: "
                f"{request_func_input.prompt}"
            )


async def async_request_openai_embeddings_clip(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    _preprocess_clip(request_func_input)

    return await async_request_openai_embeddings_chat(
        request_func_input,
        session,
        pbar=pbar,
    )


async def async_request_openai_embeddings_vlm2vec(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    _preprocess_vlm2vec(request_func_input)

    return await async_request_openai_embeddings_chat(
        request_func_input,
        session,
        pbar=pbar,
        mm_position="first",
    )


async def async_request_infinity_embeddings(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "Infinity Embeddings API", "embeddings")

    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
    }

    if request_func_input.prompt:
        payload["input"] = request_func_input.prompt
    else:
        mm_content = request_func_input.multi_modal_content
        assert isinstance(mm_content, dict)

        mm_type = mm_content["type"]
        payload["input"] = mm_content[mm_type]["url"]
        payload["modality"] = mm_type.split("_", 1)[0]

    _update_payload_common(payload, request_func_input)

    headers = _get_headers("application/json")
    _update_headers_common(headers, request_func_input)

    return await _run_pooling_request(
        session,
        api_url,
        payload=payload,
        headers=headers,
        pbar=pbar,
    )


async def async_request_infinity_embeddings_clip(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    _preprocess_clip(request_func_input)

    return await async_request_infinity_embeddings(
        request_func_input,
        session,
        pbar=pbar,
    )


# TODO: Add more request functions for different API protocols.
ASYNC_REQUEST_FUNCS: dict[str, RequestFunc] = {
    "vllm": async_request_openai_completions,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "openai-audio": async_request_openai_audio,
    "openai-embeddings": async_request_openai_embeddings,
    "openai-embeddings-chat": async_request_openai_embeddings_chat,
    "openai-embeddings-clip": async_request_openai_embeddings_clip,
    "openai-embeddings-vlm2vec": async_request_openai_embeddings_vlm2vec,
    # Infinity embedding server: https://github.com/michaelfeil/infinity
    "infinity-embeddings": async_request_infinity_embeddings,
    "infinity-embeddings-clip": async_request_infinity_embeddings_clip,
    # (Infinity embedding server does not support vlm2vec)
    "vllm-rerank": async_request_vllm_rerank,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k
    for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions, async_request_openai_chat_completions)
]
