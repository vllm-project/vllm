# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive an in-process engine with a timed arrival schedule.

`simulate()` replays a list of requests, each with an arrival time, against an
`LLMEngine` running in the caller's process, and timestamps every request's
output tokens. With ``simulation_config.clock`` set to ``"virtual"`` or
``"fast_forward"`` the timestamps are simulated time and the run takes only as
long as the host work, which is what makes configuration sweeps cheap. With
the real clock (simulated or real execution) the same driver produces the
wall-clock baseline to compare against.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.worker.simulated.timing import (
    VirtualClock,
    building_driven_engine,
    get_clock,
)

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

logger = init_logger(__name__)


@dataclass
class TimedRequest:
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float = 0.0
    """Seconds after the start of the run."""


@dataclass
class RequestTiming:
    request_id: str
    arrival_time: float
    num_prompt_tokens: int
    num_output_tokens: int = 0
    token_times: list[float] = field(default_factory=list)
    """Time of every output chunk (one per step that produced tokens)."""
    finish_time: float | None = None

    @property
    def ttft(self) -> float | None:
        return self.token_times[0] - self.arrival_time if self.token_times else None

    @property
    def e2el(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        if self.finish_time is None or self.num_output_tokens < 2:
            return None
        return (self.finish_time - self.token_times[0]) / (self.num_output_tokens - 1)

    @property
    def itls(self) -> list[float]:
        return list(np.diff(self.token_times)) if len(self.token_times) > 1 else []


@dataclass
class SimulationResult:
    requests: list[RequestTiming]
    duration: float
    """From the start of the run to the last request's completion, in the
    run's clock (simulated seconds under the virtual clock)."""
    wall_time: float

    def summary(self) -> dict[str, float]:
        """Latency percentiles in milliseconds and throughputs, named as in
        `vllm bench serve`."""

        def stats(name: str, values: list[float]) -> dict[str, float]:
            if not values:
                return {}
            ms = np.asarray(values) * 1e3
            return {
                f"mean_{name}_ms": float(ms.mean()),
                f"median_{name}_ms": float(np.median(ms)),
                f"p99_{name}_ms": float(np.percentile(ms, 99)),
            }

        done = [r for r in self.requests if r.finish_time is not None]
        out = {
            "completed": float(len(done)),
            "duration_s": self.duration,
            "wall_time_s": self.wall_time,
            "request_throughput": len(done) / self.duration if self.duration else 0,
            "output_throughput": (
                sum(r.num_output_tokens for r in done) / self.duration
                if self.duration
                else 0
            ),
        }
        out |= stats("ttft", [r.ttft for r in done if r.ttft is not None])
        out |= stats("tpot", [r.tpot for r in done if r.tpot is not None])
        out |= stats("itl", [itl for r in done for itl in r.itls])
        out |= stats("e2el", [r.e2el for r in done if r.e2el is not None])
        return out


def make_engine(engine_args: EngineArgs) -> LLMEngine:
    """An `LLMEngine` whose engine core runs in this process, so that it and
    the driver share the simulation clock."""
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    vllm_config = engine_args.create_engine_config()
    with building_driven_engine():
        return LLMEngine(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
            multiprocess_mode=False,
        )


def run(
    engine: LLMEngine,
    requests: Sequence[TimedRequest],
    max_stalled_steps: int = 10_000,
) -> SimulationResult:
    """Feed `requests` to `engine` (from `make_engine()`) at their arrival
    times and timestamp the outputs. A request is admitted at the first
    engine step at or after its arrival, as the engine core of a server picks
    up requests between steps. Under the virtual clock, `max_stalled_steps`
    consecutive engine steps without output or clock progress abort the
    run."""
    from vllm.v1.engine.core_client import InprocClient

    sim = engine.vllm_config.simulation_config
    if not isinstance(engine.engine_core, InprocClient):
        raise ValueError("run() needs an engine built by make_engine().")
    if sim.clock != "real" and not sim.simulate_forward:
        raise ValueError(f"The {sim.clock} clock needs simulate_forward.")
    if any(r.sampling_params.n != 1 for r in requests):
        raise ValueError("Timed runs support n=1 only.")
    clock = get_clock(sim.clock)
    wall_start = time.perf_counter()
    t0 = clock.now()
    pending = deque(sorted(enumerate(requests), key=lambda x: x[1].arrival_time))
    timings: dict[str, RequestTiming] = {}
    stalled = 0
    while pending or engine.has_unfinished_requests():
        # Compare absolute times: after jumping to `t0 + arrival_time`,
        # `now - t0` can round to just below `arrival_time`.
        now = clock.now()
        while pending and t0 + pending[0][1].arrival_time <= now:
            i, req = pending.popleft()
            request_id = f"timed-{i}"
            params = req.sampling_params.clone()
            params.output_kind = RequestOutputKind.DELTA
            engine.add_request(
                request_id, {"prompt_token_ids": req.prompt_token_ids}, params
            )
            timings[request_id] = RequestTiming(
                request_id, req.arrival_time, len(req.prompt_token_ids)
            )
        if not engine.has_unfinished_requests():
            # Idle until the next arrival.
            clock.sleep_until(t0 + pending[0][1].arrival_time)
            continue

        before = clock.now()
        outputs = engine.step()
        t = clock.now() - t0
        for out in outputs:
            timing = timings[out.request_id]
            n = sum(len(c.token_ids) for c in out.outputs)
            if n:
                timing.num_output_tokens += n
                timing.token_times.append(t)
            if out.finished:
                timing.finish_time = t

        if outputs or clock.now() != before:
            stalled = 0
        else:
            stalled += 1
            if stalled > max_stalled_steps and isinstance(clock, VirtualClock):
                raise RuntimeError(
                    f"No progress in {stalled} engine steps at simulated time "
                    f"{t:.6f} s with {engine.get_num_unfinished_requests()} "
                    "unfinished requests."
                )

    finished = [r.finish_time for r in timings.values() if r.finish_time is not None]
    return SimulationResult(
        requests=[timings[f"timed-{i}"] for i in range(len(requests))],
        duration=max(finished, default=0.0),
        wall_time=time.perf_counter() - wall_start,
    )


def simulate(
    engine_args: EngineArgs, requests: Sequence[TimedRequest]
) -> SimulationResult:
    """Build an in-process engine from `engine_args` and run `requests`
    through it. Pass ``simulation_config={"simulate_forward": True,
    "clock": "fast_forward", ...}`` for a run in simulated time."""
    engine = make_engine(engine_args)
    try:
        return run(engine, requests)
    finally:
        engine.engine_core.shutdown()
