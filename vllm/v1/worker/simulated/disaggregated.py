# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simulate disaggregated prefill: prefill and decode instances in one process.

The driver plays the proxy: a request goes to a prefill instance with
``max_tokens=1``, then to a decode instance with the kv_transfer_params the
prefill instance returned. Each engine runs on its own virtual clock and the
engine furthest behind steps next; an event reaches a busy engine when its
step ends. Engines that only wait for KV transfers skip ahead to the next
time anything can happen.
"""

from __future__ import annotations

import dataclasses
import heapq
import math
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from vllm.config import KVTransferConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.worker.simulated.kv_nixl import simulated_nixl_worker
from vllm.v1.worker.simulated.offline import (
    MAX_STALLED_STEPS,
    REQUEST_ID_PREFIX,
    RequestTiming,
    SimulationResult,
    TimedRequest,
    add_token_request,
    engine_clock,
    make_engine,
)
from vllm.v1.worker.simulated.timing import VirtualClock

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.llm_engine import LLMEngine

# How the proxy picks an instance: round-robin, as vLLM's example proxies do,
# or by fewest unfinished requests.
Routing = Literal["round_robin", "least_unfinished"]

# The kv_transfer_params a proxy sends with a request to a prefill instance.
_PREFILL_KV_TRANSFER_PARAMS: dict[str, Any] = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


def make_instance(
    engine_args: EngineArgs,
    kv_role: Literal["kv_producer", "kv_consumer"],
    name: str,
    usage_context: UsageContext | None = None,
) -> LLMEngine:
    """A prefill (`kv_role="kv_producer"`) or decode (`"kv_consumer"`) engine on
    its own virtual clock. Extra NixlConnector settings are taken from
    `engine_args.kv_transfer_config`; a set `log_path` gets `.<name>` appended.
    `usage_context` is as for `make_engine()`."""
    base = engine_args.kv_transfer_config
    if isinstance(base, dict):
        base = KVTransferConfig(**base)
    sim = dataclasses.replace(engine_args.simulation_config, clock="virtual")
    if sim.log_path is not None:
        sim.log_path = f"{sim.log_path}.{name}"
    args = dataclasses.replace(
        engine_args,
        kv_transfer_config=KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role=kv_role,
            engine_id=f"{name}-{uuid.uuid4().hex[:8]}",
            kv_connector_extra_config=dict(base.kv_connector_extra_config)
            if base is not None
            else {},
        ),
        simulation_config=sim,
    )
    return make_engine(args, VirtualClock(), usage_context)


def next_transfer_completion(engine: LLMEngine) -> float:
    """When the engine's next KV transfer completes; inf if none is in flight."""
    assert engine.vllm_config.kv_transfer_config is not None
    engine_id = engine.vllm_config.kv_transfer_config.engine_id
    assert engine_id is not None
    t = simulated_nixl_worker(engine_id).next_completion()
    return math.inf if t is None else t


def _core(engine: LLMEngine) -> EngineCore:
    from vllm.v1.engine.core_client import InprocClient

    assert isinstance(engine.engine_core, InprocClient)
    return engine.engine_core.engine_core


def _scheduler(engine: LLMEngine) -> Scheduler:
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = _core(engine).scheduler
    assert isinstance(scheduler, Scheduler)
    return scheduler


def _has_work(engine: LLMEngine) -> bool:
    # Includes finished requests whose blocks a prefill instance still holds.
    return _scheduler(engine).has_requests()


def _only_waits(engine: LLMEngine) -> bool:
    """Whether, having scheduled nothing, no step can make progress until a KV
    transfer completes: no running requests and no received KV to resume. A
    waiting request then waits for its remote KV, or for blocks that only
    those requests free."""
    scheduler = _scheduler(engine)
    return not scheduler.running and not scheduler.finished_recving_kv_req_ids


def run_disaggregated(
    prefill: Sequence[LLMEngine],
    decode: Sequence[LLMEngine],
    requests: Sequence[TimedRequest],
    handoff_latency_s: float = 0.0,
    routing: Routing = "round_robin",
    max_stalled_steps: int = MAX_STALLED_STEPS,
    frontend_latency_s: float = 0.0,
) -> SimulationResult:
    """Run `requests` through engines from `make_instance()`, timestamping the
    decode instances' outputs. `handoff_latency_s` is the proxy's time from a
    finished prefill to the decode instance receiving the request;
    `frontend_latency_s` is as in `offline.run()`."""
    if any(r.sampling_params.n != 1 for r in requests):
        raise ValueError("Timed runs support n=1 only.")
    engines = [*prefill, *decode]
    clocks = {id(e): engine_clock(e) for e in engines}
    wall_start = time.perf_counter()
    t0 = max(c.now() for c in clocks.values())
    for c in clocks.values():
        c.sleep_until(t0)

    # (time, seq, request index, kv_transfer_params of a handed-over request).
    events: list[tuple[float, int, int, dict[str, Any] | None]] = [
        (t0 + frontend_latency_s + r.arrival_time, i, i, None)
        for i, r in enumerate(requests)
    ]
    heapq.heapify(events)
    seq = len(events)
    timings = {
        i: RequestTiming(
            f"{REQUEST_ID_PREFIX}{i}", r.arrival_time, len(r.prompt_token_ids)
        )
        for i, r in enumerate(requests)
    }
    is_prefill = {id(e) for e in prefill}
    # Engines that only wait for KV transfers: they made no progress in as
    # many steps as they keep in flight, so that every step's output (which
    # reports completed transfers) has reached their scheduler.
    waiting: set[int] = set()
    idle_steps: dict[int, int] = {}
    stalled = 0

    picks = {id(prefill): 0, id(decode): 0}

    def pick(pool: Sequence[LLMEngine]) -> LLMEngine:
        if routing == "least_unfinished":
            return min(pool, key=lambda e: e.get_num_unfinished_requests())
        picks[id(pool)] += 1
        return pool[(picks[id(pool)] - 1) % len(pool)]

    while events or any(_has_work(e) for e in engines):
        active = [e for e in engines if _has_work(e)]
        t_active = min((clocks[id(e)].now() for e in active), default=math.inf)
        if events and events[0][0] <= t_active:
            t, _, i, handover = heapq.heappop(events)
            req = requests[i]
            params = req.sampling_params.clone()
            params.output_kind = RequestOutputKind.DELTA
            params.extra_args = dict(params.extra_args or {})
            if handover is None:
                engine = pick(prefill)
                params.max_tokens = 1
                params.min_tokens = 0
                params.extra_args["kv_transfer_params"] = dict(
                    _PREFILL_KV_TRANSFER_PARAMS
                )
            else:
                engine = pick(decode)
                params.extra_args["kv_transfer_params"] = handover
            clocks[id(engine)].sleep_until(t)
            waiting.discard(id(engine))
            idle_steps.pop(id(engine), None)
            add_token_request(
                engine, f"{REQUEST_ID_PREFIX}{i}", req.prompt_token_ids, params
            )
            continue

        # Step the engine furthest behind; of several at the same time, each
        # until it only waits.
        tied = [e for e in active if clocks[id(e)].now() == t_active]
        fresh = [e for e in tied if id(e) not in waiting]
        if not fresh:
            # All of them only wait: move them to the earliest time anything
            # can happen (an event, an engine ahead, a transfer completing).
            t_next = min(
                [events[0][0] if events else math.inf]
                + [clocks[id(e)].now() for e in active if e not in tied]
                + [next_transfer_completion(e) for e in tied]
            )
            if t_next == math.inf:
                raise RuntimeError(
                    f"All engines wait at simulated time {t_active - t0:.6f} s "
                    "and nothing is pending."
                )
            for e in tied:
                clocks[id(e)].sleep_until(t_next)
            waiting.clear()
            idle_steps.clear()
            continue
        engine = fresh[0]
        clock = clocks[id(engine)]
        # The engine must not skip ahead (to its transfers' completion) past
        # the next event or another engine, which may hand it a request.
        clock.horizon = min(
            [events[0][0] if events else math.inf]
            + [clocks[id(e)].now() for e in active if e is not engine]
        )
        before = clock.now()
        outputs = engine.step()
        t = clock.now()
        for out in outputs:
            assert isinstance(out, RequestOutput)
            i = int(out.request_id.removeprefix(REQUEST_ID_PREFIX))
            if id(engine) in is_prefill:
                if out.finished:
                    # The proxy hands the request to a decode instance.
                    heapq.heappush(
                        events,
                        (t + handoff_latency_s, seq, i, out.kv_transfer_params or {}),
                    )
                    seq += 1
                continue
            timing = timings[i]
            n = sum(len(c.token_ids) for c in out.outputs)
            if n:
                timing.num_output_tokens += n
                timing.token_times.append(t - t0)
            if out.finished:
                timing.finish_time = t - t0

        if outputs or t != before:
            waiting.clear()
            idle_steps.clear()
            stalled = 0
        else:
            idle_steps[id(engine)] = idle_steps.get(id(engine), 0) + 1
            if idle_steps[id(engine)] >= _core(engine).batch_queue_size and _only_waits(
                engine
            ):
                waiting.add(id(engine))
            stalled += 1
            if stalled > max_stalled_steps:
                raise RuntimeError(
                    f"No progress in {stalled} engine steps at simulated time "
                    f"{t - t0:.6f} s."
                )

    finished = [r.finish_time for r in timings.values() if r.finish_time is not None]
    return SimulationResult(
        requests=[timings[i] for i in range(len(requests))],
        duration=max(finished, default=0.0),
        wall_time=time.perf_counter() - wall_start,
    )


def simulate_disaggregated(
    prefill_args: EngineArgs,
    decode_args: EngineArgs,
    requests: Sequence[TimedRequest],
    num_prefill: int = 1,
    num_decode: int = 1,
    handoff_latency_s: float = 0.0,
    routing: Routing = "round_robin",
    max_stalled_steps: int = MAX_STALLED_STEPS,
    frontend_latency_s: float = 0.0,
) -> SimulationResult:
    """Build `num_prefill` prefill and `num_decode` decode instances and run
    `requests` through them. Both argument sets need ``simulate_forward``;
    the instances use the virtual clock."""
    prefill = [
        make_instance(prefill_args, "kv_producer", f"prefill{i}")
        for i in range(num_prefill)
    ]
    decode = [
        make_instance(decode_args, "kv_consumer", f"decode{i}")
        for i in range(num_decode)
    ]
    try:
        return run_disaggregated(
            prefill,
            decode,
            requests,
            handoff_latency_s,
            routing,
            max_stalled_steps,
            frontend_latency_s,
        )
    finally:
        for engine in [*prefill, *decode]:
            engine.engine_core.shutdown()
