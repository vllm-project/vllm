# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import replace

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

DP_SIZE = int(os.getenv("DP_SIZE", 2))


@pytest.mark.asyncio
async def test_run_eagle_dp(monkeypatch: pytest.MonkeyPatch):
    # This test checks that running a model with and without eagle
    # leads to identical tokens. This is only true in batch invariant mode
    # (because the target model verifies all draft tokens in one big forward pass)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")

    target_model = "meta-llama/Llama-3.1-8B-Instruct"
    draft_model = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

    engine_args = AsyncEngineArgs(
        model=target_model,
        tokenizer_mode="auto",
        enforce_eager=False,
        tensor_parallel_size=int(os.getenv("TP_SIZE", 1)),
        data_parallel_size=DP_SIZE,
        data_parallel_backend="mp",  # ray takes more time
        trust_remote_code=True,
        max_model_len=16384,
        attention_config={"backend": "FLASH_ATTN"},
    )

    eagle_engine_args = replace(
        engine_args,
        speculative_config={
            "model": draft_model,
            "method": "eagle",
            "num_speculative_tokens": 3,
        },
    )

    prompt = "This is a test of data parallel with eagle"
    num_expected_tokens = 100
    sampling_params = SamplingParams(
        min_tokens=num_expected_tokens,
        max_tokens=num_expected_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
        temperature=0,
    )

    async def generate_with_timeout(given_engine: AsyncLLM):
        async for out in given_engine.generate(
            request_id="test-eagle-dp", prompt=prompt, sampling_params=sampling_params
        ):
            token_ids = out.outputs[0].token_ids
            assert len(token_ids) == num_expected_tokens
            return token_ids

    async def engine_create_and_generate(engine_args: AsyncEngineArgs):
        async with AsyncExitStack() as after:
            engine = AsyncLLM.from_engine_args(engine_args)
            after.callback(engine.shutdown)

            token_ids = await asyncio.wait_for(
                generate_with_timeout(engine), timeout=30
            )

            assert not engine.output_processor.has_unfinished_requests()
        return token_ids

    token_ids_with_eagle = await engine_create_and_generate(eagle_engine_args)
    token_ids_no_eagle = await engine_create_and_generate(engine_args)

    # Test for correctness
    assert token_ids_with_eagle == token_ids_no_eagle
