import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

import aiohttp
import huggingface_hub.constants
from tqdm.asyncio import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    logprobs: Optional[int] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        params = {
            "best_of": request_func_input.best_of,
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
            # TGI does not accept ignore_eos flag.
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")

                        #NOTE: Sometimes TGI returns a ping response without
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
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

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
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert request_func_input.best_of == 1
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        if request_func_input.ignore_eos:
            payload["min_length"] = request_func_input.output_len
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
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


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert request_func_input.best_of == 1

        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=request_func_input.api_url,
                                    json=payload) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    output.generated_text = parsed_resp["text"][0]
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
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!")
                    output.generated_text = generated_text
                    output.latency = latency
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
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.append(request_func_input.multi_modal_content)
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                },
            ],
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
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
    if os.getenv('VLLM_USE_MODELSCOPE', 'False').lower() == 'true':
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str, trust_remote_code: bool
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
            pretrained_model_name_or_path):
        pretrained_model_name_or_path = get_model(
            pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                         trust_remote_code=trust_remote_code)


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
}
