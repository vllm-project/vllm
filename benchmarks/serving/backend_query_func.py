import os
import time
from typing import Dict, Union

import aiohttp
import openai


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

    async with aiohttp.ClientSession(timeout=timeout) as session:
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        payload = {
            "inputs": prompt,
            "parameters": params,
        }
        output = dict()
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
        output = dict()
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


async def async_query_openai_completions(
    model: str,
    prompt: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
):
    output = dict()
    output["prompt_len"] = prompt_len
    oai_client = openai.AsyncOpenAI(
        base_url=api_url, api_key=os.environ.get("OPENAI_API_KEY")
    )

    st = time.perf_counter()
    try:
        resp = await oai_client.completions.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=output_len,
        )
        latency = time.perf_counter() - st
        output["generated_text"] = resp.choices[0].text
        output["success"] = True
        output["latency"] = latency
    except Exception as e:
        output["generated_text"] = ""
        output["success"] = False

    return output


ASYNC_QUERY_FUNCS = {
    "tgi": async_query_tgi,
    "vllm": async_query_vllm,
    "openai": async_query_openai_completions,
}
