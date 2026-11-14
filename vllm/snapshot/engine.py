# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math
from typing import Any, Literal

from vllm.snapshot.types import Oracle, oracles_match

_CANARY_PROMPT = "The capital of France is"


class SnapshotCanaryError(RuntimeError):
    """The initialized engine did not produce a valid snapshot oracle."""


class SnapshotSession:
    """One capture and restore cycle with admission owned by the caller.

    Preparation records a canary, releases and restores resources to rehearse
    recovery, then releases them again for capture. The caller must not expose
    the engine to request producers until restore succeeds, and owns
    termination when a phase fails: a failed session cannot be resumed. The
    engine must be an out-of-process ``EngineClient``; ``wait`` pause mode is
    not available to an in-process engine.
    """

    def __init__(self, engine: Any, *, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("snapshot phase timeout must be positive and finite")
        self.engine = engine
        self.timeout_s = timeout_s
        self.state: Literal[
            "new", "preparing", "prepared", "restoring", "ready", "failed"
        ] = "new"
        self.phase = "new"
        self._oracle: Oracle | None = None

    async def prepare(self) -> Oracle:
        if self.state == "prepared":
            assert self._oracle is not None
            return self._oracle
        if self.state != "new":
            raise RuntimeError(f"cannot prepare snapshot session in state {self.state}")
        self.state = "preparing"
        try:
            await asyncio.wait_for(self._prepare(), timeout=self.timeout_s)
        except BaseException:
            self.state = "failed"
            raise
        self.state = "prepared"
        self.phase = "prepared"
        assert self._oracle is not None
        return self._oracle

    async def restore(self) -> None:
        if self.state == "ready":
            return
        if self.state != "prepared":
            raise RuntimeError(f"cannot restore snapshot session in state {self.state}")
        self.state = "restoring"
        try:
            await asyncio.wait_for(self._restore("recovery"), timeout=self.timeout_s)
        except BaseException:
            self.state = "failed"
            raise
        self.state = "ready"
        self.phase = "ready"

    async def _prepare(self) -> None:
        self.phase = "reference canary"
        if self.engine.get_num_unfinished_requests():
            raise RuntimeError("snapshot preparation requires an idle engine")
        self._oracle = await run_engine_canary(self.engine)
        await self._release()
        await self._restore("rehearsal")
        await self._release()

    async def _release(self) -> None:
        self.phase = "pause generation"
        await self.engine.pause_generation(mode="wait")
        self.phase = "prepare engine resources"
        await self.engine.prepare_snapshot_resources()

    async def _restore(self, phase: str) -> None:
        self.phase = f"{phase}: restore engine resources"
        await self.engine.restore_snapshot_resources()
        await self._validate(phase)

    async def _validate(self, phase: str) -> None:
        self.phase = f"{phase}: synchronize device"
        await self.engine.collective_rpc("synchronize_device", timeout=self.timeout_s)
        self.phase = f"{phase}: resume private generation"
        await self.engine.resume_generation()
        self.phase = f"{phase}: canary"
        oracle = await run_engine_canary(self.engine)
        assert self._oracle is not None
        if not oracles_match(self._oracle, oracle):
            raise SnapshotCanaryError(f"snapshot {phase} changed canary output")


def oracle_from_request_output(request_output: Any) -> Oracle:
    try:
        candidate = request_output.outputs[0]
        token_ids = tuple(candidate.token_ids)
        (sampled_token_id,) = token_ids
        return Oracle(
            token_ids=token_ids,
            text=candidate.text,
            sampled_token_logprob=candidate.logprobs[0][sampled_token_id].logprob,
        )
    except ValueError as error:
        raise SnapshotCanaryError(
            "snapshot canary must produce exactly one finite token logprob"
        ) from error
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise SnapshotCanaryError(
            "snapshot canary did not return sampled token logprob"
        ) from error


async def run_engine_canary(engine: Any) -> Oracle:
    from vllm import SamplingParams

    final_output = None
    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=1,
        max_tokens=1,
        seed=0,
        logprobs=0,
    )
    async for output in engine.generate(
        _CANARY_PROMPT,
        sampling_params,
        request_id="vllm-snapshot-canary",
    ):
        final_output = output
    if final_output is None:
        raise SnapshotCanaryError("canary generation returned no output")
    return oracle_from_request_output(final_output)
