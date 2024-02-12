import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0
    ttft: float = 0
    prompt_len: int = 0


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        params = {
            "best_of": request_func_input.best_of,
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                    output.latency = time.perf_counter() - st

                    body = data.decode("utf-8").lstrip("data:")
                    output.generated_text = json.loads(body)["generated_text"]
                    output.success = True
                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

        if pbar:
            pbar.update(1)
        return output


async def async_request_vllm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": request_func_input.prompt,
            "n": 1,
            "best_of": request_func_input.best_of,
            "use_beam_search": request_func_input.use_beam_search,
            "temperature": 0.0 if request_func_input.use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "ignore_eos": True,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                    output.latency = time.perf_counter() - st

                    # When streaming, '\0' is appended to the end of the response.
                    body = data.decode("utf-8").strip("\0")
                    output.generated_text = json.loads(
                        body)["text"][0][len(request_func_input.prompt):]
                    output.success = True

                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

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
        assert not request_func_input.use_beam_search
        assert request_func_input.best_of == 1
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as resp:
                if resp.status == 200:
                    async for data in resp.content.iter_any():
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                    output.latency = time.perf_counter() - st

                    body = data.decode("utf-8").lstrip("data:")
                    output.generated_text = json.loads(body)["text_output"]
                    output.success = True

                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert request_func_input.best_of == 1
        assert not request_func_input.use_beam_search

        payload = {
            "prompts": request_func_input.prompt,
            "max_new_tokens": request_func_input.output_len,
            "ignore_eos": True,
            "do_sample": True,
            "temperature":
            0.01,  # deepspeed-mii does not accept 0.0 temperature.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # DeepSpeed-MII doesn't support streaming as of Jan 28 2024, will use 0 as placeholder.
        # https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=request_func_input.api_url,
                                    json=payload) as resp:
                if resp.status == 200:
                    parsed_resp = await resp.json()
                    output.latency = time.perf_counter() - st
                    output.generated_text = parsed_resp[0]["generated_text"]
                    output.success = True
                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("v1/completions")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = chunk.decode("utf-8").lstrip("data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            body = json.loads(chunk)
                            generated_text += body["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_vllm,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "tensorrt-llm": async_request_trt_llm,
}
