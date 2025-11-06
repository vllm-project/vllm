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
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import LoggingStatLogger, StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

DP_SIZE = 1

engine_args_user_spec = AsyncEngineArgs(
    model="distilbert/distilgpt2",
    tensor_parallel_size=int(os.getenv("TP_SIZE", 2)),
    gpu_memory_utilization=0.1,
    mfu_analysis_interval=0,
    mfu_analysis_mode="manual",
    mfu_analysis_active_parameters=1e12,
    disable_log_stats=False,
)

engine_args_auto_moe = AsyncEngineArgs(
    model="ibm-research/PowerMoE-3b",
    tensor_parallel_size=int(os.getenv("TP_SIZE", 2)),
    gpu_memory_utilization=0.1,
    mfu_analysis_interval=0,
    mfu_analysis_mode="fast",  # induce automatic derivation
    disable_log_stats=False,
)

engine_args_detailed = AsyncEngineArgs(
    model="distilbert/distilgpt2",
    tensor_parallel_size=int(os.getenv("TP_SIZE", 2)),
    gpu_memory_utilization=0.1,
    mfu_analysis_interval=0,
    mfu_analysis_mode="detailed",
    disable_log_stats=False,
)

# Check if platform supports v1, handling None case
supports_v1_fn = getattr(current_platform, "supports_v1", None)
if supports_v1_fn is not None and not supports_v1_fn(
    engine_args_user_spec.create_model_config()
):
    pytest.skip(reason="Requires V1-supporting platform.", allow_module_level=True)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine_args", [engine_args_user_spec, engine_args_detailed, engine_args_auto_moe]
)
async def test_detailed(engine_args):
    stats_loggers = {}

    @dataclass
    class SimpleStatsLogger(StatLoggerBase):
        init_count: int = 0
        finished_req_count: int = 0
        mfu_count: int = 0
        mfu_min_flops: float = 1e12

        def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
            stats_loggers[engine_index] = self

        def record(
            self,
            scheduler_stats: SchedulerStats | None,
            iteration_stats: IterationStats | None,
            mm_cache_stats=None,
            engine_idx: int = 0,
        ):
            if iteration_stats:
                if iteration_stats.mfu_info is not None:
                    self.mfu_count += 1
                    self.mfu_min_flops = min(
                        self.mfu_min_flops, iteration_stats.mfu_info.flops
                    )
                self.finished_req_count += len(iteration_stats.finished_requests)

        def log_engine_initialized(self):
            self.init_count += 1

    with ExitStack() as after:
        prompt = "This is a test of data parallel"

        engine_args.async_scheduling = True
        engine = AsyncLLM.from_engine_args(
            engine_args, stat_loggers=[SimpleStatsLogger, LoggingStatLogger]
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
                        engine,
                        request_id,
                        prompt,
                        RequestOutputKind.DELTA,
                        NUM_EXPECTED_TOKENS,
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
        core_client = engine.engine_core
        # the engines only synchronize stopping every N steps so
        # allow a small amount of time here.
        for _ in range(10):
            if not core_client.engines_running:
                break
            await asyncio.sleep(0.5)

        assert not core_client.engines_running

        # Check that requests were distributed between the engines
        assert len(stats_loggers) == DP_SIZE
        assert stats_loggers[0].init_count == 1

        for sl in stats_loggers.values():
            slogger: SimpleStatsLogger = sl

            assert slogger.finished_req_count > NUM_REQUESTS // (DP_SIZE + 1), (
                f"requests are imbalanced: {stats_loggers}"
            )
            # make sure we're collecting MFU
            assert slogger.mfu_count > 0
            assert slogger.mfu_min_flops > 0  # actually tracked something
