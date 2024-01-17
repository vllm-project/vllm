import os
import time
from typing import Dict, Union

import aiohttp
from openai import AsyncOpenAI


async def async_request_tgi(
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    **kwargs,
) -> Dict[str, Union[str, bool, float]]:
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
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

        st = time.perf_counter()
        async with session.post(url=api_url, json=payload) as resp:
            if resp.status == 200:
                parsed_resp = await resp.json()
                latency = time.perf_counter() - st
                output["generated_text"] = parsed_resp["generated_text"]
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
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
        output = {}
        output["prompt_len"] = prompt_len

        st = time.perf_counter()
        async with session.post(url=api_url, json=payload) as resp:
            if resp.status == 200:
                parsed_resp = await resp.json()
                latency = time.perf_counter() - st
                output["generated_text"] = parsed_resp["text"]
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
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        assert not use_beam_search
        assert best_of == 1
        payload = {
            "text_input": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "stream": False,
        }
        output = {}
        output["prompt_len"] = prompt_len

        st = time.perf_counter()
        async with session.post(url=api_url, json=payload) as resp:
            if resp.status == 200:
                parsed_resp = await resp.json()
                latency = time.perf_counter() - st
                output["generated_text"] = parsed_resp["text_output"]
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
):
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
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
):
    output = {}
    output["prompt_len"] = prompt_len
    oai_client = AsyncOpenAI(
        base_url=api_url, api_key=os.environ.get("OPENAI_API_KEY")
    )

    assert not use_beam_search

    try:
        st = time.perf_counter()
        resp = await oai_client.completions.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=output_len,
            best_of=best_of,
        )
        latency = time.perf_counter() - st
        output["generated_text"] = resp.choices[0].text
        output["success"] = True
        output["latency"] = latency
    except:
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
