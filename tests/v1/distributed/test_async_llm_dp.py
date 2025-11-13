# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from contextlib import ExitStack
from dataclasses import dataclass

import pytest

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import DPAsyncMPClient
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats

DP_SIZE = int(os.getenv("DP_SIZE", 2))


async def generate(
    engine: AsyncLLM,
    request_id: str,
    prompt: PromptType,
    output_kind: RequestOutputKind,
    max_tokens: int,
    prompt_logprobs: int | None = None,
    data_parallel_rank: int | None = None,
) -> tuple[int, str]:
    # Ensure generate doesn't complete too fast for cancellation test.
    await asyncio.sleep(0.2)

    count = 0
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        output_kind=output_kind,
        temperature=0,
        prompt_logprobs=prompt_logprobs,
    )
    async for out in engine.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params,
        data_parallel_rank=data_parallel_rank,
    ):
        num_tokens = len(out.outputs[0].token_ids)
        if output_kind == RequestOutputKind.DELTA:
            count += num_tokens
        else:
            count = num_tokens

        await asyncio.sleep(0.0)

    return count, request_id


@pytest.mark.parametrize(
    "model",
    [
        "ibm-research/PowerMoE-3b",
        "hmellor/tiny-random-LlamaForCausalLM",
    ],
)
@pytest.mark.parametrize(
    "output_kind",
    [
        RequestOutputKind.DELTA,
        RequestOutputKind.FINAL_ONLY,
    ],
)
@pytest.mark.parametrize("data_parallel_backend", ["mp", "ray"])
@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.asyncio
async def test_load(
    model: str,
    output_kind: RequestOutputKind,
    data_parallel_backend: str,
    async_scheduling: bool,
):
    if async_scheduling and data_parallel_backend == "ray":
        # TODO(NickLucche) Re-enable when async scheduling is supported
        pytest.skip("Async scheduling is not supported with ray")
    stats_loggers = {}

    @dataclass
    class SimpleStatsLogger(StatLoggerBase):
        init_count: int = 0
        finished_req_count: int = 0

        def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
            stats_loggers[engine_index] = self

        def record(
            self,
            scheduler_stats: SchedulerStats | None,
            iteration_stats: IterationStats | None,
            mm_cache_stats: MultiModalCacheStats | None = None,
            engine_idx: int = 0,
        ):
            if iteration_stats:
                self.finished_req_count += len(iteration_stats.finished_requests)

        def log_engine_initialized(self):
            self.init_count += 1

    with ExitStack() as after:
        prompt = "This is a test of data parallel"

        engine_args = AsyncEngineArgs(
            model=model,
            enforce_eager=True,
            tensor_parallel_size=int(os.getenv("TP_SIZE", 1)),
            data_parallel_size=DP_SIZE,
            data_parallel_backend=data_parallel_backend,
            async_scheduling=async_scheduling,
        )
        engine = AsyncLLM.from_engine_args(
            engine_args, stat_loggers=[SimpleStatsLogger]
        )
        after.callback(engine.shutdown)

        NUM_REQUESTS = 100
        NUM_EXPECTED_TOKENS = 10

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks = []
        for request_id in request_ids:
            tasks.append(
                asyncio.create_task(
                    generate(
                        engine, request_id, prompt, output_kind, NUM_EXPECTED_TOKENS
                    )
                )
            )
            # Short sleep to ensure that requests are distributed.
            await asyncio.sleep(0.01)
        # Confirm that we got all the EXPECTED tokens from the requests.
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            num_generated_tokens, request_id = await task
            assert num_generated_tokens == NUM_EXPECTED_TOKENS, (
                f"{request_id} generated {num_generated_tokens} but "
                f"expected {NUM_EXPECTED_TOKENS}"
            )

        assert not engine.output_processor.has_unfinished_requests()

        # testing internals here which may break
        core_client: DPAsyncMPClient = engine.engine_core
        # the engines only synchronize stopping every N steps so
        # allow a small amount of time here.
        for _ in range(10):
            if not core_client.engines_running:
                break
            await asyncio.sleep(0.5)

        assert not core_client.engines_running
        assert not core_client.reqs_in_flight

        # Check that requests were distributed between the engines
        print(f"Stats loggers after test: {stats_loggers}")
        assert len(stats_loggers) == DP_SIZE
        assert stats_loggers[0].init_count == 1

        for sl in stats_loggers.values():
            slogger: SimpleStatsLogger = sl

            assert slogger.finished_req_count > NUM_REQUESTS // (DP_SIZE + 1), (
                f"requests are imbalanced: {stats_loggers}"
            )
