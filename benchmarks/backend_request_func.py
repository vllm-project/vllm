# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

import aiohttp
import huggingface_hub.constants
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# NOTE(simon): do not import vLLM here so the benchmark script
# can run without vLLM installed.

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: str | None = None
    logprobs: int | None = None
    extra_body: dict | None = None
    multi_modal_content: dict | list[dict] | None = None
    ignore_eos: bool = False
    language: str | None = None
    request_id: str | None = None
    api_key: str | None = None
    debug: bool | None = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        params = {
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
            "truncate": request_func_input.prompt_len,
            "ignore_eos_token": request_func_input.ignore_eos,
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        headers = None
        if request_func_input.request_id:
            headers = {"x-request-id": request_func_input.request_id}
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        if request_func_input.ignore_eos:
            output.output_tokens = request_func_input.output_len
        else:
            output.output_tokens = None

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")

                        # NOTE: Sometimes TGI returns a ping response without
                        # any data, we should skip it.
                        if chunk_bytes.startswith(":"):
                            continue
                        chunk = chunk_bytes.removeprefix("data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
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


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "completions", "profile")), (
        "TensorRT-LLM server must expose OpenAI-style endpoints: 'chat/completions' or 'completions'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        is_chat = api_url.endswith("chat/completions")

        # Build payload according to server's OpenAPI spec
        if is_chat:
            payload = {
                "model": request_func_input.model,
                "messages": [
                    {"role": "user", "content": request_func_input.prompt},
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": request_func_input.output_len,
                "stream": True,
            }
            if request_func_input.ignore_eos:
                payload["ignore_eos"] = True
        else:
            # /v1/completions
            payload = {
                "model": request_func_input.model,
                "prompt": request_func_input.prompt,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": request_func_input.output_len,
                "stream": True,
            }
            if request_func_input.ignore_eos:
                payload["ignore_eos"] = True

        # Merge extra body (e.g., response_format)
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)

        headers = {"Content-Type": "application/json"}
        if request_func_input.api_key:
            headers["Authorization"] = f"Bearer {request_func_input.api_key}"
        if request_func_input.request_id:
            headers["x-request-id"] = request_func_input.request_id

        if request_func_input.debug:
            print("[TRT-LLM] POST", api_url)
            print("[TRT-LLM] Headers:", {k: ("<hidden>" if k.lower()=="authorization" else v) for k, v in headers.items()})
            # Avoid dumping entire schema if huge
            debug_payload = dict(payload)
            if "response_format" in debug_payload:
                rf = debug_payload["response_format"]
                if isinstance(rf, dict) and "schema" in rf:
                    # Summarize schema
                    schema_type = rf.get("type")
                    debug_payload["response_format"] = {"type": schema_type, "schema_keys": list(rf["schema"].keys()) if isinstance(rf["schema"], dict) else type(rf["schema"]).__name__}
            print("[TRT-LLM] Payload:", debug_payload)

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        ttft_recorded = False
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    if request_func_input.debug:
                        print("[TRT-LLM] Response status:", response.status)
                        print("[TRT-LLM] Response content-type:", response.headers.get("content-type"))
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
                        if request_func_input.debug:
                            print("[TRT-LLM] Chunk:", (chunk_str[:200] + ("..." if len(chunk_str) > 200 else "")))
                        # Skip SSE comments
                        if chunk_str.startswith(":"):
                            continue
                        # Remove optional 'data:' prefix (with or without space)
                        if chunk_str.startswith("data:"):
                            chunk = chunk_str[5:].lstrip()
                        else:
                            chunk = chunk_str
                        if chunk == "[DONE]":
                            break

                        # Attempt JSON parse; skip invalid fragments
                        try:
                            data = json.loads(chunk)
                        except Exception:
                            if request_func_input.debug:
                                print("[TRT-LLM] Non-JSON chunk, skipping")
                            continue
                        timestamp = time.perf_counter()

                        if is_chat:
                            if choices := data.get("choices"):
                                content_piece = ""
                                # OpenAI-style streaming delta
                                if isinstance(choices[0], dict):
                                    if (delta := choices[0].get("delta")) and isinstance(delta, dict):
                                        content_piece = delta.get("content") or ""
                                    # Some servers send full message objects when not streaming
                                    elif (msg := choices[0].get("message")) and isinstance(msg, dict):
                                        content_piece = msg.get("content") or ""
                                    elif "text" in choices[0]:
                                        content_piece = choices[0].get("text") or ""
                                if content_piece:
                                    if not ttft_recorded:
                                        ttft_recorded = True
                                        output.ttft = timestamp - st
                                    else:
                                        output.itl.append(timestamp - most_recent_timestamp)
                                    generated_text += content_piece
                                    most_recent_timestamp = timestamp
                            if usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                                if request_func_input.debug:
                                    print("[TRT-LLM] Usage chunk:", usage)
                        else:
                            if choices := data.get("choices"):
                                text_piece = ""
                                if isinstance(choices[0], dict):
                                    text_piece = choices[0].get("text") or ""
                                    # Some servers might embed message.content for completions too
                                    if not text_piece and (msg := choices[0].get("message")):
                                        if isinstance(msg, dict):
                                            text_piece = msg.get("content") or ""
                                if text_piece:
                                    if not ttft_recorded:
                                        ttft_recorded = True
                                        output.ttft = timestamp - st
                                    else:
                                        output.itl.append(timestamp - most_recent_timestamp)
                                    generated_text += text_piece
                                    most_recent_timestamp = timestamp
                            if usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                                if request_func_input.debug:
                                    print("[TRT-LLM] Usage chunk:", usage)

                    output.generated_text = generated_text
                    output.success = True if ttft_recorded or bool(generated_text) else False
                    if not output.success:
                        output.error = (
                            "Never received a valid chunk to calculate TTFT. "
                            "This response will be marked as failed!"
                        )
                    output.latency = most_recent_timestamp - st
                    if request_func_input.debug:
                        preview = (generated_text[:500] + ("..." if len(generated_text) > 500 else ""))
                        print(f"[TRT-LLM] Final generated_text (len={len(generated_text)}):", preview)
                else:
                    try:
                        err_text = await response.text()
                    except Exception:
                        err_text = response.reason or ""
                    output.error = err_text or (response.reason or "")
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        # Fallback: if streaming returned no content but response_format is present,
        # retry once with non-streaming to fetch the final message.
        if (
            not output.success
            and isinstance(payload, dict)
            and payload.get("stream") is True
            and isinstance(payload.get("response_format"), dict)
        ):
            try:
                payload_no_stream = dict(payload)
                payload_no_stream["stream"] = False
                if request_func_input.debug:
                    print("[TRT-LLM] Fallback non-streaming POST", api_url)
                st_ns = time.perf_counter()
                async with session.post(url=api_url, json=payload_no_stream, headers=headers) as resp2:
                    if request_func_input.debug:
                        print("[TRT-LLM] Fallback status:", resp2.status)
                    if resp2.status == 200:
                        data2 = await resp2.json(content_type=None)
                        if request_func_input.debug:
                            # Print a truncated snapshot of raw JSON
                            raw_preview = json.dumps(data2)[:800]
                            print("[TRT-LLM] Fallback response preview:", raw_preview + ("..." if len(raw_preview) == 800 else ""))
                        if is_chat:
                            # Expect choices[0].message.content
                            choices = data2.get("choices") or []
                            if choices and isinstance(choices[0], dict):
                                message = choices[0].get("message") or {}
                                content = message.get("content") or ""
                                if content:
                                    output.generated_text = content
                                    output.success = True
                        else:
                            # Expect choices[0].text
                            choices = data2.get("choices") or []
                            if choices and isinstance(choices[0], dict):
                                text = choices[0].get("text") or ""
                                if text:
                                    output.generated_text = text
                                    output.success = True
                        if usage := data2.get("usage"):
                            output.output_tokens = usage.get("completion_tokens")
                        output.latency = time.perf_counter() - st_ns
                        if request_func_input.debug:
                            preview = (output.generated_text[:500] + ("..." if len(output.generated_text) > 500 else ""))
                            print(f"[TRT-LLM] Fallback generated_text (len={len(output.generated_text)}):", preview)
                    else:
                        # Keep previous error; optionally append
                        try:
                            err2 = await resp2.text()
                        except Exception:
                            err2 = resp2.reason or ""
                        if request_func_input.debug:
                            print("[TRT-LLM] Fallback error:", err2)
            except Exception as e:
                if request_func_input.debug:
                    print("[TRT-LLM] Fallback exception:", repr(e))

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        if request_func_input.request_id:
            headers["x-request-id"] = request_func_input.request_id

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    if "choices" in parsed_resp:
                        output.generated_text = parsed_resp["choices"][0]["text"]
                    elif "text" in parsed_resp:
                        output.generated_text = parsed_resp["text"][0]
                    else:
                        output.error = (
                            "Unexpected response format: "
                            "neither 'choices' nor 'text' found"
                        )
                        output.success = False
                    output.success = True
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


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        if request_func_input.request_id:
            headers["x-request-id"] = request_func_input.request_id

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
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
                            if usage := data.get("usage"):
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


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "profile")), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            mm_content = request_func_input.multi_modal_content
            if isinstance(mm_content, list):
                content.extend(mm_content)
            elif isinstance(mm_content, dict):
                content.append(mm_content)
            else:
                raise TypeError(
                    "multi_modal_content must be a dict or list[dict] for openai-chat"
                )
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        if request_func_input.request_id:
            headers["x-request-id"] = request_func_input.request_id

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")
                        # NOTE: SSE comments (often used as pings) start with a colon.
                        # These are not JSON data payload and should be skipped.
                        if chunk_bytes.startswith(":"):
                            continue

                        chunk = chunk_bytes.removeprefix("data: ")

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
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    # Lazy import without PlaceholderModule to avoid vllm dep.
    import soundfile

    api_url = request_func_input.api_url
    assert api_url.endswith(("transcriptions", "translations")), (
        "OpenAI Chat Completions API URL must end with 'transcriptions' "
    )
    "or `translations`."

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "language": "en",
            # Flattened due to multipart/form-data
            "stream_include_usage": True,
            "stream_continuous_usage_stats": True,
        }
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        if request_func_input.request_id:
            headers["x-request-id"] = request_func_input.request_id

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
            most_recent_timestamp = st
            try:
                async with session.post(
                    url=api_url, data=form, headers=headers
                ) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
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


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv("VLLM_USE_MODELSCOPE", "False").lower() == "true":
        from modelscope import snapshot_download

        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(pretrained_model_name_or_path):
            model_path = snapshot_download(
                model_id=pretrained_model_name_or_path,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
            )

            return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    if tokenizer_mode == "mistral":
        try:
            from vllm.tokenizers import MistralTokenizer
        except ImportError as e:
            raise ImportError(
                "MistralTokenizer requires vllm package.\n"
                "Please install it with `pip install vllm` "
                "to use mistral tokenizer mode."
            ) from e
        return MistralTokenizer.from_pretrained(str(pretrained_model_name_or_path))
    else:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "openai-audio": async_request_openai_audio,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
    "llama.cpp": async_request_openai_completions,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k
    for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions, async_request_openai_chat_completions)
]
