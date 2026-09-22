# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 ModelRunner + pipeline parallel + data parallel integration tests.

Covers the interaction between the V2 model runner's PP sampled-token
broadcast and the DP per-step all-reduce across a few concurrency
regimes. Requires 4 GPUs (DP=2, PP=2, TP=1) on CUDA.
"""

import asyncio
import contextlib
import os
from contextlib import ExitStack

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

PP_DP_MODEL = "ibm-research/PowerMoE-3b"  # smallest cached MoE that supports PP
PROMPT = "This is a test of data parallel and pipeline parallel together"


def _gpu_skip_reason() -> str | None:
    if not current_platform.is_cuda():
        return "requires CUDA"
    n = current_platform.device_count()
    if n < 4:
        return f"requires 4 GPUs, got {n}"
    return None


_GPU_SKIP = _gpu_skip_reason()

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "1",
        reason="VLLM_USE_V2_MODEL_RUNNER=1 required",
    ),
    pytest.mark.skipif(_GPU_SKIP is not None, reason=_GPU_SKIP or ""),
]


def _engine_args(async_scheduling: bool) -> AsyncEngineArgs:
    return AsyncEngineArgs(
        model=PP_DP_MODEL,
        pipeline_parallel_size=2,
        data_parallel_size=2,
        data_parallel_backend="mp",
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        async_scheduling=async_scheduling,
        enable_prefix_caching=False,
        enforce_eager=False,
        enable_expert_parallel=False,
    )


async def _generate(engine: AsyncLLM, prompt: str, max_tokens: int) -> int:
    """Run one streaming completion and return the number of tokens it yielded."""
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
        temperature=0.0,
    )
    request_id = f"req-{id(prompt):x}-{max_tokens}"
    total = 0
    async for out in engine.generate(
        request_id=request_id, prompt=prompt, sampling_params=sampling_params
    ):
        total += len(out.outputs[0].token_ids)
    return total


@pytest.mark.asyncio
@pytest.mark.parametrize("async_scheduling", [True, False])
async def test_pp_dp_v2_low_concurrency(async_scheduling: bool):
    """A single in-flight request at a time, repeated, to exercise the
    PP slot ring under empty batches between decodes."""
    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(_engine_args(async_scheduling))
        after.callback(engine.shutdown)

        for _ in range(4):
            n = await _generate(engine, PROMPT, max_tokens=16)
            assert n == 16


@pytest.mark.asyncio
@pytest.mark.parametrize("async_scheduling", [True, False])
async def test_pp_dp_v2_mid_concurrency(async_scheduling: bool):
    """64 concurrent requests, staggered, to exercise the steady-state
    DP all-reduce + PP slot-ring path."""
    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(_engine_args(async_scheduling))
        after.callback(engine.shutdown)

        async def _one(i: int) -> int:
            await asyncio.sleep(0.01 * i)  # stagger so DP load-balances
            return await _generate(engine, f"{PROMPT} {i}", max_tokens=64)

        results = await asyncio.gather(*[_one(i) for i in range(64)])
        assert all(n == 64 for n in results), results


@pytest.mark.asyncio
async def test_pp_dp_v2_abort_mid_decode():
    """Cancel half the in-flight requests mid-stream and confirm the
    engine survives the abort storm."""

    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(_engine_args(async_scheduling=True))
        after.callback(engine.shutdown)

        async def _maybe_cancel(i: int):
            sampling_params = SamplingParams(
                max_tokens=64,
                ignore_eos=True,
                output_kind=RequestOutputKind.DELTA,
                temperature=0.0,
            )
            request_id = f"abort-req-{i}"
            count = 0
            cancel_at = 4 if i % 2 == 0 else 64
            async for out in engine.generate(
                request_id=request_id,
                prompt=f"{PROMPT} {i}",
                sampling_params=sampling_params,
            ):
                count += len(out.outputs[0].token_ids)
                if count >= cancel_at:
                    break
            return count, i

        results = await asyncio.gather(*[_maybe_cancel(i) for i in range(32)])
        for count, i in results:
            if i % 2 == 0:
                assert count >= 4
            else:
                assert count == 64

        # Engine must still serve after the abort storm.
        final = await _generate(engine, "post-abort warmup", max_tokens=8)
        assert final == 8


@pytest.mark.asyncio
async def test_pp_dp_v2_pause_resume():
    """Pause an engine with a request in flight, then resume and confirm
    new requests still work."""

    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(_engine_args(async_scheduling=True))
        after.callback(engine.shutdown)

        # Start a long-running generation, let some decoding happen, then
        # pause (abort mode) and confirm the in-flight task terminates.
        inflight = asyncio.create_task(_generate(engine, PROMPT, max_tokens=128))
        await asyncio.sleep(0.5)

        assert not await engine.is_paused()
        await engine.pause_generation(mode="abort")
        assert await engine.is_paused()

        with contextlib.suppress(Exception):
            await inflight

        await engine.resume_generation()
        assert not await engine.is_paused()

        n = await _generate(engine, PROMPT, max_tokens=8)
        assert n == 8
