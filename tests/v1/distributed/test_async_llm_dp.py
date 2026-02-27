# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import pytest

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
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
    elif data_parallel_backend == "ray" and current_platform.is_rocm():
        pytest.skip(
            "Ray as the distributed executor backend is not supported with ROCm."
        )
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


# =============================================================================
# DP Pause/Resume Tests
# =============================================================================
# When expert_parallel=False: uses non-MoE model (DP replicas as separate engines).
# When expert_parallel=True: uses MoE model + EP (DPEngineCoreProc, sync pause path).

DP_PAUSE_MODEL = "hmellor/tiny-random-LlamaForCausalLM"
DP_PAUSE_MODEL_MOE = "ibm-research/PowerMoE-3b"
DP_PAUSE_PROMPT = "This is a test of data parallel pause"


def _get_dp_pause_engine_args(expert_parallel: bool) -> AsyncEngineArgs:
    """Engine args for DP pause tests: MoE+EP when expert_parallel else small Llama."""
    model = DP_PAUSE_MODEL_MOE if expert_parallel else DP_PAUSE_MODEL
    return AsyncEngineArgs(
        model=model,
        enforce_eager=True,
        tensor_parallel_size=int(os.getenv("TP_SIZE", 1)),
        data_parallel_size=DP_SIZE,
        data_parallel_backend="mp",
        enable_expert_parallel=expert_parallel,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("expert_parallel", [False, True])
async def test_dp_pause_resume_basic(expert_parallel: bool):
    """Pausing from the client (one call) pauses all DP ranks; resume clears it."""
    with ExitStack() as after:
        engine_args = _get_dp_pause_engine_args(expert_parallel)
        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        assert not await engine.is_paused()
        await engine.pause_generation(mode="abort")
        assert await engine.is_paused()
        await engine.resume_generation()
        assert not await engine.is_paused()

        # Engine still works after resume
        sampling_params = SamplingParams(max_tokens=5)
        async for out in engine.generate(
            request_id="after-resume",
            prompt=DP_PAUSE_PROMPT,
            sampling_params=sampling_params,
        ):
            pass
        assert out.finished


@pytest.mark.asyncio
@pytest.mark.parametrize("expert_parallel", [False, True])
async def test_dp_pause_abort(expert_parallel: bool):
    """Pause with abort from one client aborts in-flight requests on all DP ranks."""
    with ExitStack() as after:
        engine_args = _get_dp_pause_engine_args(expert_parallel)
        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        # Start several requests so they are distributed across ranks
        sampling_params = SamplingParams(max_tokens=500, ignore_eos=True)
        num_requests = 4
        outputs_by_id: dict[str, list[RequestOutput]] = {}

        async def gen(rid: str):
            out_list: list[RequestOutput] = []
            outputs_by_id[rid] = out_list
            async for out in engine.generate(
                request_id=rid,
                prompt=DP_PAUSE_PROMPT,
                sampling_params=sampling_params,
            ):
                out_list.append(out)
            return out_list[-1] if out_list else None

        tasks = [asyncio.create_task(gen(f"req-{i}")) for i in range(num_requests)]
        # Wait for some tokens on at least one request
        while not any(len(o) >= 2 for o in outputs_by_id.values()):
            await asyncio.sleep(0.02)

        await engine.pause_generation(mode="abort")

        finals = await asyncio.gather(*tasks)
        for i, final in enumerate(finals):
            assert final is not None, f"req-{i} had no output"
            assert final.finished
            assert final.outputs[0].finish_reason == "abort"

        assert await engine.is_paused()
        await engine.resume_generation()
        assert not await engine.is_paused()

        # New request completes after resume
        async for out in engine.generate(
            request_id="after-abort",
            prompt=DP_PAUSE_PROMPT,
            sampling_params=SamplingParams(max_tokens=5),
        ):
            pass
        assert out.finished
        assert not engine.output_processor.has_unfinished_requests()


@pytest.mark.asyncio
@pytest.mark.parametrize("expert_parallel", [False, True])
async def test_dp_pause_keep_then_resume(expert_parallel: bool):
    """Start generation, pause after a few tokens (keep mode), resume; verify gap."""

    pause_duration = 2.0
    min_tokens_before_pause = 3

    with ExitStack() as after:
        engine_args = _get_dp_pause_engine_args(expert_parallel)
        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        sampling_params = SamplingParams(max_tokens=15, ignore_eos=True)
        token_times: list[tuple[int, float]] = []
        pause_token_idx = 0

        async def generator_task():
            nonlocal pause_token_idx
            out = None
            async for output in engine.generate(
                request_id="keep-resume-req",
                prompt=DP_PAUSE_PROMPT,
                sampling_params=sampling_params,
            ):
                token_count = len(output.outputs[0].token_ids)
                token_times.append((token_count, time.monotonic()))
                out = output
            return out

        async def controller_task():
            nonlocal pause_token_idx
            while len(token_times) < min_tokens_before_pause:
                await asyncio.sleep(0.01)
            await engine.pause_generation(mode="keep")
            await asyncio.sleep(pause_duration)
            pause_token_idx = len(token_times)
            await engine.resume_generation()

        gen_task = asyncio.create_task(generator_task())
        ctrl_task = asyncio.create_task(controller_task())
        final_output, _ = await asyncio.gather(gen_task, ctrl_task)

        assert final_output is not None and final_output.finished
        assert await engine.is_paused() is False
        assert pause_token_idx >= min_tokens_before_pause
        if pause_token_idx > 0 and pause_token_idx < len(token_times):
            pause_gap = (
                token_times[pause_token_idx][1] - token_times[pause_token_idx - 1][1]
            )
            assert pause_gap >= pause_duration * 0.8, (
                f"Expected gap ~{pause_duration}s after pause, got {pause_gap:.3f}s"
            )


@pytest.mark.asyncio
async def test_dp_pause_keep_race_staggered_engines():
    """Race: send pause(keep) to engine 0, then add two requests,
    then pause(keep) to engine 1. Ensures no deadlock when pause
    requests are staggered and requests arrive in between."""
    if DP_SIZE != 2:
        pytest.skip("test_dp_pause_keep_race_staggered_engines requires DP_SIZE=2")

    with ExitStack() as after:
        engine_args = _get_dp_pause_engine_args(expert_parallel=True)
        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        client = engine.engine_core

        original_call_utility = client.call_utility_async
        mid_pause_tasks: list[asyncio.Task] = []

        async def staggered_pause_keep(method: str, *args) -> Any:
            if method != "pause_scheduler" or not args or args[0] != "keep":
                return await original_call_utility(method, *args)
            # Send pause(keep) to engine 0 first
            await client._call_utility_async(
                method, *args, engine=client.core_engines[0]
            )
            # In the middle: send two requests (race window)
            sp = SamplingParams(max_tokens=5, ignore_eos=True)

            async def consume_gen(req_id: str) -> None:
                async for _ in engine.generate(
                    request_id=req_id,
                    prompt=DP_PAUSE_PROMPT,
                    sampling_params=sp,
                ):
                    pass

            t1 = asyncio.create_task(consume_gen("race-1"))
            t2 = asyncio.create_task(consume_gen("race-2"))
            mid_pause_tasks.extend([t1, t2])
            await asyncio.sleep(3)
            # Then send pause(keep) to engine 1
            result = await client._call_utility_async(
                method, *args, engine=client.core_engines[1]
            )
            return result

        client.call_utility_async = staggered_pause_keep

        await engine.pause_generation(mode="keep")
        assert await engine.is_paused()
        await engine.resume_generation()
        assert not await engine.is_paused()
        # Let the two requests we sent mid-pause complete
        await asyncio.gather(*mid_pause_tasks)
