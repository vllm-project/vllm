# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from tests.utils import multi_gpu_test
from vllm import SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

MODEL = "hmellor/tiny-random-LlamaForCausalLM"


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "parallel_kwargs",
    [
        {"tensor_parallel_size": 2, "enable_batch_sharded_sampling": True},
        {"pipeline_parallel_size": 2},
    ],
    ids=["tp2-sharded-sampling", "pp2"],
)
def test_watermarked_generation_with_model_parallelism(vllm_runner, parallel_kwargs):
    prompts = ["Parallel watermark smoke"] * 2

    with vllm_runner(
        MODEL,
        dtype="half",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=4,
        watermark_config={"algorithm": "gumbel", "key": 42},
        **parallel_kwargs,
    ) as runner:
        first = runner.llm.generate(
            prompts,
            SamplingParams(
                temperature=0.8,
                seed=1,
                max_tokens=8,
                ignore_eos=True,
            ),
        )
        second = runner.llm.generate(
            prompts,
            SamplingParams(
                temperature=0.8,
                seed=2,
                max_tokens=8,
                ignore_eos=True,
            ),
        )

    first_tokens = [output.outputs[0].token_ids for output in first]
    second_tokens = [output.outputs[0].token_ids for output in second]
    assert all(len(token_ids) == 8 for token_ids in first_tokens)
    assert first_tokens[0] == first_tokens[1]
    assert first_tokens == second_tokens


@multi_gpu_test(num_gpus=2)
def test_watermarked_generation_with_data_parallelism():
    async def run() -> None:
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=MODEL,
                dtype="half",
                data_parallel_size=2,
                enforce_eager=True,
                max_model_len=128,
                max_num_seqs=4,
                watermark_config={"algorithm": "gumbel", "key": 42},
            )
        )

        async def generate(index: int) -> tuple[int, ...]:
            final_output = None
            async for output in engine.generate(
                "Data-parallel watermark smoke",
                SamplingParams(
                    temperature=0.8,
                    seed=index,
                    max_tokens=8,
                    ignore_eos=True,
                ),
                request_id=f"watermark-dp-{index}",
                data_parallel_rank=index,
            ):
                final_output = output
            assert final_output is not None
            return tuple(final_output.outputs[0].token_ids)

        try:
            token_ids = await asyncio.gather(generate(0), generate(1))
            assert len(token_ids[0]) == 8
            assert token_ids[0] == token_ids[1]
        finally:
            engine.shutdown()

    try:
        asyncio.run(run())
    finally:
        cleanup_dist_env_and_memory()
