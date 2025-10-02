# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
from contextlib import ExitStack

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

DP_SIZE = int(os.getenv("DP_SIZE", 2))


@pytest.fixture
def use_vllm_v1(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_V1", "1")


@pytest.mark.asyncio
async def test_run_eagle_dp(use_vllm_v1):
    target_model = "meta-llama/Llama-3.1-8B-Instruct"
    draft_model = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

    engine_args = AsyncEngineArgs(
        model=target_model,
        tokenizer_mode="auto",
        enforce_eager=True,
        tensor_parallel_size=int(os.getenv("TP_SIZE", 1)),
        data_parallel_size=DP_SIZE,
        data_parallel_backend="mp",  # ray takes more time
        trust_remote_code=True,
        speculative_config={
            "model": draft_model,
            "method": "eagle",
            "num_speculative_tokens": 3,
        })

    if not current_platform.supports_v1(engine_args.create_model_config()):
        pytest.skip(reason="Requires V1-supporting platform.",
                    allow_module_level=True)

    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        prompt = "This is a test of data parallel with eagle"
        num_expected_tokens = 100
        sampling_params = SamplingParams(
            min_tokens=num_expected_tokens,
            max_tokens=num_expected_tokens,
            ignore_eos=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
            temperature=0)
        async with asyncio.timeout(30):
            async for out in engine.generate(request_id="eagle-dp",
                                             prompt=prompt,
                                             sampling_params=sampling_params):
                num_tokens = len(out.outputs[0].token_ids)
                assert num_tokens == num_expected_tokens

        assert not engine.output_processor.has_unfinished_requests()
