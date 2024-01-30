import json
import os
import time
from typing import Dict, Union

import aiohttp

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


async def async_request_tgi(
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    **kwargs,
) -> Dict[str, Union[str, bool, float]]:
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
        }
        payload = {
            "inputs": prompt,
            "parameters": params,
        }
        output = {}
        output["prompt_len"] = prompt_len

        ttft = 0
        st = time.perf_counter()
        async with session.post(url=api_url, json=payload) as response:
            if response.status == 200:
                async for data in response.content.iter_any():
                    if ttft == 0:
                        ttft = time.perf_counter() - st
                        output["ttft"] = ttft
                latency = time.perf_counter() - st

                body = data.decode("utf-8").lstrip("data:")
                generated_text = json.loads(body)["generated_text"]
                output["generated_text"] = generated_text
                output["success"] = True
                output["latency"] = latency
            else:
                output["generated_text"] = ""
                output["success"] = False

        return output


async def async_request_vllm(
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    **kwargs,
) -> Dict[str, Union[str, bool, float]]:
    assert api_url.endswith("generate")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
        }
        output = {}
        output["prompt_len"] = prompt_len

        ttft = 0
        st = time.perf_counter()
        async with session.post(url=api_url, json=payload) as response:
            if response.status == 200:
                async for data in response.content.iter_any():
                    if ttft == 0:
                        ttft = time.perf_counter() - st
                        output["ttft"] = ttft
                latency = time.perf_counter() - st

                # When streaming, '\0' is appended to the end of the response.
                body = data.decode("utf-8").strip("\0")
                generated_text = json.loads(body)["text"][0][len(prompt) :]
                output["generated_text"] = generated_text
                output["success"] = True
                output["latency"] = latency
            else:
                output["generated_text"] = ""
                output["success"] = False

        return output


async def async_request_trt_llm(
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    **kwargs,
) -> Dict[str, Union[str, bool, float]]:
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not use_beam_search
        assert best_of == 1
        payload = {
            "accumulate_tokens": True,
            "text_input": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "stream": True,
        }
        output = {}
        output["prompt_len"] = prompt_len

        st = time.perf_counter()
        async with session.post(url=api_url, json=payload) as resp:
            if resp.status == 200:
                async for data in resp.content.iter_any():
                    if ttft == 0:
                        ttft = time.perf_counter() - st
                        output["ttft"] = ttft
                latency = time.perf_counter() - st

                body = data.decode("utf-8").lstrip("data:")
                generated_text = json.loads(body)["text_output"]
                output["generated_text"] = generated_text
                output["success"] = True
                output["latency"] = latency
            else:
                output["generated_text"] = ""
                output["success"] = False

        return output


async def async_request_deepspeed_mii(
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    **kwargs,
) -> Dict[str, Union[str, bool, float]]:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert best_of == 1
        assert not use_beam_search

        payload = {
            "prompt": prompt,
            "max_new_tokens": output_len,
            "ignore_eos": True,
            "do_sample": True,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temperature.
            "top_p": 1.0,
        }
        output = {}
        output["prompt_len"] = prompt_len

        # DeepSpeed-MII doesn't support streaming as of Jan 28 2024, will use 0 as placeholder.
        # https://github.com/microsoft/DeepSpeed-MII/pull/311
        output["ttft"] = 0

        st = time.perf_counter()
        async with session.post(url=api_url, json=payload) as resp:
            if resp.status == 200:
                parsed_resp = await resp.json()
                latency = time.perf_counter() - st
                output["generated_text"] = parsed_resp[0]["generated_text"]
                output["success"] = True
                output["latency"] = latency
            else:
                output["generated_text"] = ""
                output["success"] = False

        return output


async def async_request_openai_completions(
    model: str,
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    **kwargs,
) -> Dict[str, Union[str, bool, float]]:
    assert api_url.endswith("v1/completions")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not use_beam_search
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": best_of,
            "max_tokens": output_len,
            "stream": True,
        }
        headers = {
            f"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = {}
        output["prompt_len"] = prompt_len

        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        async with session.post(
            url=api_url, json=payload, headers=headers
        ) as response:
            if response.status == 200:
                async for chunk in response.content:
                    if ttft == 0:
                        ttft = time.perf_counter() - st
                        output["ttft"] = ttft

                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    chunk = chunk.decode("utf-8").lstrip("data: ")
                    if chunk == "[DONE]":
                        latency = time.perf_counter() - st
                    else:
                        body = json.loads(chunk)
                        generated_text += body["choices"][0]["text"]

                output["generated_text"] = generated_text
                output["success"] = True
                output["latency"] = latency
            else:
                output["generated_text"] = ""
                output["success"] = False

    return output


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_vllm,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "tensorrt-llm": async_request_trt_llm,
}
