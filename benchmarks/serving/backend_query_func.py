import os
import time
from typing import Dict, Union

import aiohttp
from openai import AsyncOpenAI


async def async_query_tgi(
    model: str,
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> Dict[str, Union[str, bool, float]]:
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    if not api_url.endswith("/generate"):
        api_url += "/generate"

    async with aiohttp.ClientSession(timeout=timeout) as session:
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 1.0,
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


async def async_query_vllm(
    model: str,
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> Dict[str, Union[str, bool, float]]:
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    if not api_url.endswith("/generate"):
        api_url += "/generate"

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


async def async_query_deepspeed_mii(
    model: str,
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
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


async def async_query_openai_completions(
    model: str,
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
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


ASYNC_QUERY_FUNCS = {
    "tgi": async_query_tgi,
    "vllm": async_query_vllm,
    "deepspeed-mii": async_query_deepspeed_mii,
    "openai": async_query_openai_completions,
}
