# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from dataclasses import dataclass
from typing import Any, Union


@dataclass
class RequestConfig:
    """Configuration for a single request in a batch."""
    prompt: str
    max_tokens: int = 10
    temperature: float = 1.0
    top_k: Union[int, None] = None
    top_p: float = 1.0
    seed: Union[int, None] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logprobs: Union[int, None] = None
    min_p: float = 0.0
    bad_words: Union[list[str], None] = None
    logit_bias: Union[dict[int, float], None] = None
    allowed_token_ids: Union[list[int], None] = None
    min_tokens: int = 0
    return_tokens_as_token_ids: bool = False


async def send_request(async_client,
                       model: str,
                       config: RequestConfig,
                       return_full_response: bool = False):
    """Send a single async legacy completion request (old API)."""
    extra_body: dict[str, Any] = {}
    if config.top_k is not None:
        extra_body["top_k"] = config.top_k
    if config.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = config.repetition_penalty
    if config.min_p != 0.0:
        extra_body["min_p"] = config.min_p
    if config.min_tokens != 0:
        extra_body["min_tokens"] = config.min_tokens
    if config.logit_bias is not None:
        extra_body["logit_bias"] = config.logit_bias
    if config.allowed_token_ids is not None:
        extra_body["allowed_token_ids"] = config.allowed_token_ids
    if config.return_tokens_as_token_ids:
        extra_body["return_tokens_as_token_ids"] = True
    if config.bad_words is not None:
        raise ValueError(
            "bad_words is not supported in legacy completions API")

    response = await async_client.completions.create(
        model=model,
        prompt=config.prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        logprobs=config.logprobs,
        extra_body=extra_body if extra_body else None,
    )
    if return_full_response:
        return response
    return response.choices[0].text


async def send_chat_request(async_client,
                            model: str,
                            config: RequestConfig,
                            return_full_response: bool = False):
    """Send a single async chat completion request (new API).
    """
    extra_body: dict[str, Any] = {}
    if config.top_k is not None:
        extra_body["top_k"] = config.top_k
    if config.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = config.repetition_penalty
    if config.min_p != 0.0:
        extra_body["min_p"] = config.min_p
    if config.min_tokens != 0:
        extra_body["min_tokens"] = config.min_tokens
    if config.bad_words is not None:
        extra_body["bad_words"] = config.bad_words
    if config.logit_bias is not None:
        extra_body["logit_bias"] = config.logit_bias
    if config.allowed_token_ids is not None:
        extra_body["allowed_token_ids"] = config.allowed_token_ids

    response = await async_client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": config.prompt
        }],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        logprobs=config.logprobs is not None,
        top_logprobs=config.logprobs,
        extra_body=extra_body if extra_body else None,
    )
    if return_full_response:
        return response
    return response.choices[0].message.content


async def send_batch_concurrent(async_client,
                                model: str,
                                configs: list[RequestConfig],
                                use_chat: bool = False,
                                return_full_response: bool = False):
    """
    Send multiple requests concurrently with different per-request parameters.
    The vLLM server will batch these together internally.

    Args:
        use_chat: use chat completions API instead of legacy completions API.
        return_full_response: return full response objects vs just text.
    """
    send_fn = send_chat_request if use_chat else send_request
    tasks = [
        send_fn(async_client,
                model,
                cfg,
                return_full_response=return_full_response) for cfg in configs
    ]
    return await asyncio.gather(*tasks)


def run_concurrent_batch(tt_server,
                         tt_model_name,
                         configs: list[RequestConfig],
                         use_chat: bool = False,
                         return_full_response: bool = False):
    """
    Synchronous wrapper to run concurrent requests.
    Returns list of output texts (or full responses) in same order as configs.

    Args:
        use_chat: use chat completions API instead of legacy completions API.
        return_full_response: return full response objects vs just text.
    """

    async def _run():
        async_client = tt_server.get_async_client()
        try:
            return await send_batch_concurrent(
                async_client,
                tt_model_name,
                configs,
                use_chat=use_chat,
                return_full_response=return_full_response)
        finally:
            await async_client.close()

    return asyncio.run(_run())


def assert_varied(results, min_varied, explanation):
    unique_results = set(results)
    assert len(unique_results) >= min_varied, (
        f"{explanation}\n"
        f"Expected at least {min_varied} unique results.\n"
        f"Only {len(unique_results)}/{len(results)} were varied.\n"
        f"Results: {results}")


def assert_pairwise_varied(results1, results2, min_varied, explanation):
    different = [x != y for x, y in zip(results1, results2)]
    assert sum(different) >= min_varied, (f"{explanation}\n"
                                          f"Expected difference on re-run.\n"
                                          f"Results: {results1} + {results2}")


def assert_deterministic(results, explanation):
    unique_results = set(results)
    assert len(unique_results) == 1, (
        f"{explanation}\n"
        f"Expected reproducible outputs.\n"
        f"Got {len(unique_results)} unique results out of {len(results)}.\n"
        f"Results: {results}")
