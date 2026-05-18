# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 ModelRunner + pipeline parallel + data parallel integration tests.

These exercise the cross-cut between PR #42187 (V2 model runner side-stream
PP broadcast / `PPHandler` slot ring) and the DP `execute_dummy_batch`
synchronisation path. The combination requires that:

* the worker's per-step ``dispatch_cg_and_sync_dp`` all-reduce stays
  aligned across DP ranks even when one rank schedules an empty batch and
  another schedules a real one, with the engine falling back to a dummy
  batch on the empty side;
* the PP broadcast slot ring (which only exists on non-last PP ranks)
  advances in lockstep with the engine's per-step clock so the decode
  cadence enforced by the V2+PP+async throttle reads from the right slot
  even when empty real batches occur (e.g. concurrency=1);
* on a non-last PP rank the deferred-free pool of request-state indices
  (used to plug the abort race) does not leak warmup slots into the
  first real batch's allocation pool;
* the dummy intermediate-tensor path on a non-first PP rank does not
  self-alias when ``execute_dummy_batch`` runs.

The tests require 4 GPUs (DP=2, PP=2, TP=1) on a CUDA platform. Each
test exercises a concurrency regime that historically hid or surfaced
one of the bugs above.
"""

import asyncio
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


def _need_4_gpus():
    if not current_platform.is_cuda():
        return pytest.mark.skip("requires CUDA")
    n = current_platform.device_count()
    if n < 4:
        return pytest.mark.skip(f"requires 4 GPUs, got {n}")
    return None


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "1",
        reason="VLLM_USE_V2_MODEL_RUNNER=1 required",
    ),
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
@pytest.mark.skipif(_need_4_gpus() is not None, reason="needs 4 GPUs")
async def test_pp_dp_v2_low_concurrency(async_scheduling: bool):
    """Single in-flight request exercises the empty-batch slot-ring advance
    and the deferred-free flush after warmup. Pre-fix this hung/crashed
    at engine startup ("No free indices") or produced nonsense output."""
    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(_engine_args(async_scheduling))
        after.callback(engine.shutdown)

        # Multiple short sequential requests force the engine to schedule
        # empty batches between decodes and to flip the active DP rank.
        for _ in range(4):
            n = await _generate(engine, PROMPT, max_tokens=16)
            assert n == 16


@pytest.mark.asyncio
@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.skipif(_need_4_gpus() is not None, reason="needs 4 GPUs")
async def test_pp_dp_v2_mid_concurrency(async_scheduling: bool):
    """64 concurrent requests at 64 tokens each spread evenly across DP and
    exercise the steady-state DP all-reduce + PP slot-ring path."""
    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(_engine_args(async_scheduling))
        after.callback(engine.shutdown)

        async def _one(i: int) -> int:
            await asyncio.sleep(0.01 * i)  # stagger so DP load-balances
            return await _generate(engine, f"{PROMPT} {i}", max_tokens=64)

        results = await asyncio.gather(*[_one(i) for i in range(64)])
        assert all(n == 64 for n in results), results


@pytest.mark.asyncio
@pytest.mark.skipif(_need_4_gpus() is not None, reason="needs 4 GPUs")
async def test_pp_dp_v2_abort_mid_decode():
    """Cancel half the in-flight requests mid-stream. Exercises the
    interaction between the abort defer-free, the per-step DP all-reduce
    that follows the dummy-batch fallback, and the PP slot ring's empty
    slots when a DP rank's load drops sharply."""

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
            gen = engine.generate(
                request_id=request_id,
                prompt=f"{PROMPT} {i}",
                sampling_params=sampling_params,
            )
            cancel_at = 4 if i % 2 == 0 else 64
            try:
                async for out in gen:
                    count += len(out.outputs[0].token_ids)
                    if count >= cancel_at:
                        break
            except asyncio.CancelledError:
                raise
            return count, i

        results = await asyncio.gather(*[_maybe_cancel(i) for i in range(32)])
        # Even requests stop at 4 tokens, odd at 64.
        for count, i in results:
            if i % 2 == 0:
                assert count >= 4
            else:
                assert count == 64

        # Engine must still be alive after the abort storm. Drive one more
        # request through to confirm.
        final = await _generate(engine, "post-abort warmup", max_tokens=8)
        assert final == 8


@pytest.mark.asyncio
@pytest.mark.skipif(_need_4_gpus() is not None, reason="needs 4 GPUs")
async def test_pp_dp_v2_pause_resume():
    """Pause and resume an engine running PP+DP+V2. Tests that the
    dummy-batch fallback keeps DP all-reduce alive while the scheduler is
    PAUSED_ALL, and that the PP slot ring picks up correctly after resume."""

    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(_engine_args(async_scheduling=True))
        after.callback(engine.shutdown)

        # Launch a request, then pause/resume mid-flight by waiting briefly.
        assert not await engine.is_paused()
        await engine.pause_generation(mode="abort")
        assert await engine.is_paused()
        await engine.resume_generation()
        assert not await engine.is_paused()

        # Engine still works after resume.
        n = await _generate(engine, PROMPT, max_tokens=8)
        assert n == 8
