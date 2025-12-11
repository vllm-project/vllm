# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import ExitStack

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.async_llm import AsyncLLM

TEXT_ENGINE_ARGS = AsyncEngineArgs(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enforce_eager=True,
)


@pytest.mark.asyncio
async def test_multi_prompt_generate():
    """Test generating with multiple prompts in a single call."""
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        prompts = ["Hello, my name is", "The capital of France is", "1+1="]
        request_ids = [f"req-{i}" for i in range(len(prompts))]
        sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

        outputs = {}
        async for out in engine.generate(
            request_id=request_ids, prompt=prompts, sampling_params=sampling_params
        ):
            if out.finished:
                outputs[out.request_id] = out

        assert len(outputs) == len(prompts)
        for rid in request_ids:
            assert rid in outputs
            assert outputs[rid].finished
            assert len(outputs[rid].outputs[0].token_ids) > 0


@pytest.mark.asyncio
async def test_multi_prompt_generate_mismatched_lengths():
    """Test that mismatched lengths raise ValueError."""
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(TEXT_ENGINE_ARGS)
        after.callback(engine.shutdown)

        prompts = ["A", "B"]
        request_ids = ["req-1"]  # Mismatch

        sampling_params = SamplingParams(max_tokens=5)

        with pytest.raises(ValueError, match="same length"):
            async for _ in engine.generate(
                request_id=request_ids, prompt=prompts, sampling_params=sampling_params
            ):
                pass
