# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Timing model for simulated execution.

A `TimingProvider` turns a `StepDescriptor` (what the scheduler asked the
worker to do in one step) into a `StepCost` (how long the device would be
busy). Providers see plain Python data only, so they can be thin wrappers
over compiled engines, measured tables or analytical models without any torch
dependency. The interface is deliberately per step, not per operator: how a
step decomposes into kernels is the provider's business.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# --------------------------------------------------------------------------- #
# Step description and cost
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimRequestStep:
    """One request's share of a simulated step."""

    req_id: str
    num_scheduled_tokens: int
    """Tokens computed for this request in this step: a prefill chunk, or
    ``1 + num_spec_tokens`` for decode."""
    num_computed_tokens: int
    """KV length before this step (prompt tokens already computed + outputs)."""
    num_prompt_tokens: int
    num_cached_tokens: int
    """Prefix-cache hits credited at admission (0 for already-running reqs)."""
    num_spec_tokens: int
    is_prefill: bool


@dataclass(frozen=True)
class ParallelSummary:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1


def parallel_summary(vllm_config: VllmConfig) -> ParallelSummary:
    pc = vllm_config.parallel_config
    return ParallelSummary(
        tensor_parallel_size=pc.tensor_parallel_size,
        pipeline_parallel_size=pc.pipeline_parallel_size,
        data_parallel_size=pc.data_parallel_size,
        expert_parallel_size=(
            pc.tensor_parallel_size * pc.data_parallel_size
            if pc.enable_expert_parallel
            else 1
        ),
    )


@dataclass(frozen=True)
class StepDescriptor:
    """Everything a timing model may need to price one scheduler step."""

    step_index: int
    requests: tuple[SimRequestStep, ...]
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefill_reqs: int
    num_decode_reqs: int
    num_common_prefix_blocks: int
    num_kv_tokens: int
    """Sum over scheduled requests of KV length after this step."""
    parallel: ParallelSummary = field(default_factory=ParallelSummary)

    @property
    def num_reqs(self) -> int:
        return len(self.requests)

    @property
    def num_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_decode_only(self) -> bool:
        return self.num_prefill_reqs == 0

    def replay_key(self) -> str:
        """Batch composition key shared with recorded step logs."""
        pairs = sorted(
            (r.num_scheduled_tokens, r.num_computed_tokens) for r in self.requests
        )
        return json.dumps(pairs, separators=(",", ":"))

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["requests"] = [asdict(r) for r in self.requests]
        return d


def build_step_descriptor(
    scheduler_output: SchedulerOutput,
    prompt_lens: dict[str, int],
    step_index: int,
    parallel: ParallelSummary,
) -> StepDescriptor:
    """Describe one scheduled step from the `SchedulerOutput` alone.

    `prompt_lens` is the caller's running map of request id to prompt length;
    this function adds newly scheduled requests to it and drops the ones the
    scheduler reports finished. Keeping the builder independent of worker
    state lets a real worker and the simulated runner emit identical
    descriptors, which is what makes recorded steps replayable."""
    for req_id in scheduler_output.finished_req_ids:
        prompt_lens.pop(req_id, None)
    num_scheduled = scheduler_output.num_scheduled_tokens
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    computed: dict[str, int] = {}
    cached_at_admission: dict[str, int] = {}
    for req in scheduler_output.scheduled_new_reqs:
        prompt_lens[req.req_id] = req.prompt_len
        computed[req.req_id] = req.num_computed_tokens
        cached_at_admission[req.req_id] = req.num_computed_tokens
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, num_computed in zip(cached.req_ids, cached.num_computed_tokens):
        computed[req_id] = num_computed

    requests: list[SimRequestStep] = []
    num_prefill_tokens = num_decode_tokens = 0
    num_prefill_reqs = num_decode_reqs = 0
    num_kv_tokens = 0
    for req_id in sorted(num_scheduled):
        scheduled = num_scheduled[req_id]
        num_computed = computed.get(req_id, 0)
        prompt_len = prompt_lens.get(req_id, 0)
        is_prefill = num_computed < prompt_len
        requests.append(
            SimRequestStep(
                req_id=req_id,
                num_scheduled_tokens=scheduled,
                num_computed_tokens=num_computed,
                num_prompt_tokens=prompt_len,
                num_cached_tokens=cached_at_admission.get(req_id, 0),
                num_spec_tokens=len(spec_tokens.get(req_id, ())),
                is_prefill=is_prefill,
            )
        )
        if is_prefill:
            num_prefill_tokens += scheduled
            num_prefill_reqs += 1
        else:
            num_decode_tokens += scheduled
            num_decode_reqs += 1
        num_kv_tokens += num_computed + scheduled
    common = scheduler_output.num_common_prefix_blocks
    return StepDescriptor(
        step_index=step_index,
        requests=tuple(requests),
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        num_prefill_reqs=num_prefill_reqs,
        num_decode_reqs=num_decode_reqs,
        num_common_prefix_blocks=max(common) if common else 0,
        num_kv_tokens=num_kv_tokens,
        parallel=parallel,
    )


def empty_step_descriptor(step_index: int, parallel: ParallelSummary) -> StepDescriptor:
    """A rank's share of a lockstep step in which it has no requests (an idle
    data-parallel rank's dummy batch)."""
    return StepDescriptor(step_index, (), 0, 0, 0, 0, 0, 0, parallel)


def merge_dp_descriptors(descriptors: list[StepDescriptor]) -> StepDescriptor:
    """All MoE data-parallel ranks' shares of one lockstep step, as one step.
    Their forward passes meet in all-to-all collectives, so the step takes as
    long as the whole joint batch dictates on every rank."""
    own = descriptors[0]
    return StepDescriptor(
        step_index=own.step_index,
        requests=tuple(
            sorted((r for d in descriptors for r in d.requests), key=lambda r: r.req_id)
        ),
        num_prefill_tokens=sum(d.num_prefill_tokens for d in descriptors),
        num_decode_tokens=sum(d.num_decode_tokens for d in descriptors),
        num_prefill_reqs=sum(d.num_prefill_reqs for d in descriptors),
        num_decode_reqs=sum(d.num_decode_reqs for d in descriptors),
        num_common_prefix_blocks=max(d.num_common_prefix_blocks for d in descriptors),
        num_kv_tokens=sum(d.num_kv_tokens for d in descriptors),
        parallel=own.parallel,
    )


def gather_dp_step(descriptor: StepDescriptor) -> StepDescriptor:
    """The joint step of all MoE data-parallel ranks. Must be called exactly
    where the real runner synchronizes the DP ranks: every step that runs the
    model, and every dummy batch of an idle rank."""
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_dp_group

    group = get_dp_group()
    gathered: list[StepDescriptor | None] = [None] * group.world_size
    dist.all_gather_object(gathered, descriptor, group=group.cpu_group)
    others = [d for i, d in enumerate(gathered) if i != group.rank_in_group]
    return merge_dp_descriptors([descriptor, *others])  # type: ignore[list-item]


@dataclass(frozen=True)
class StepCost:
    """Predicted cost of one step.

    `device_s` is the time the device is busy with the step. `host_s` is the
    host time the target deployment spends inside the step's
    `execute_model`/`sample_tokens` calls (input preparation, kernel launches,
    sampling); the simulated runner holds the engine thread for that long, as
    the real calls would, while the device time runs on the device queue. The
    rest of the host loop (scheduling, output processing) is the real code and
    pays its own cost."""

    device_s: float
    host_s: float = 0.0
    num_accepted: tuple[int, ...] | None = None
    """Per-request accepted draft tokens (speculative decoding), or None."""
    breakdown: dict[str, float] | None = None
    """Optional named components (e.g. per kernel class), for logging."""


@runtime_checkable
class TimingProvider(Protocol):
    """Prices one simulated step. Implementations are constructed with the
    engine configuration plus the user's `timing_args` as keyword arguments."""

    def estimate(self, step: StepDescriptor) -> StepCost: ...


# --------------------------------------------------------------------------- #
# Built-in providers
# --------------------------------------------------------------------------- #


class ZeroTimingProvider:
    """Every step takes no device time (pure scheduler / KV-cache simulation)."""

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any) -> None:
        if kwargs:
            raise ValueError(f"zero timing provider takes no arguments, got {kwargs}")

    def estimate(self, step: StepDescriptor) -> StepCost:
        return StepCost(device_s=0.0)


class ConstantTimingProvider:
    """Linear model: ``fixed_us + prefill_us_per_token * P + decode_us_per_req * D
    + decode_us_per_kv_token * K`` where P is prefill tokens, D decode
    requests and K the KV tokens read by decode requests."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        fixed_us: float = 0.0,
        prefill_us_per_token: float = 0.0,
        decode_us_per_req: float = 0.0,
        decode_us_per_kv_token: float = 0.0,
        host_us: float = 0.0,
    ) -> None:
        self.fixed_us = fixed_us
        self.prefill_us_per_token = prefill_us_per_token
        self.decode_us_per_req = decode_us_per_req
        self.decode_us_per_kv_token = decode_us_per_kv_token
        self.host_us = host_us

    def estimate(self, step: StepDescriptor) -> StepCost:
        decode_kv_tokens = sum(
            r.num_computed_tokens + r.num_scheduled_tokens
            for r in step.requests
            if not r.is_prefill
        )
        prefill = self.prefill_us_per_token * step.num_prefill_tokens
        decode = (
            self.decode_us_per_req * step.num_decode_reqs
            + self.decode_us_per_kv_token * decode_kv_tokens
        )
        device_us = self.fixed_us + prefill + decode
        return StepCost(
            device_s=device_us * 1e-6,
            host_s=self.host_us * 1e-6,
            breakdown={"prefill_us": prefill, "decode_us": decode},
        )


def _step_features(step: StepDescriptor) -> tuple[float, float, float]:
    """Coarse batch features used for nearest-neighbour lookups."""
    past = sum(r.num_computed_tokens for r in step.requests)
    return (float(step.num_reqs), float(step.num_tokens), float(past))


class _KNNTable:
    """Inverse-distance weighted k-NN over z-scored (num_reqs, num_tokens,
    sum_past_kv) features, interpolating (device_s, host_s). Points are
    normalized once; a query is one vectorized distance pass, so lookups stay
    off the step's critical path."""

    def __init__(
        self,
        points: list[tuple[tuple[float, float, float], tuple[float, float]]],
        k: int,
    ):
        if not points:
            raise ValueError("timing table is empty")
        feats = np.asarray([p for p, _ in points], dtype=np.float64)
        self.values = np.asarray([v for _, v in points], dtype=np.float64)
        self.mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        self.points = (feats - self.mean) / self.std
        self.k = min(max(1, k), len(points))

    def query(self, f: tuple[float, float, float]) -> StepCost:
        q = (np.asarray(f, dtype=np.float64) - self.mean) / self.std
        dist = np.sqrt(((self.points - q) ** 2).sum(axis=1))
        idx = np.argpartition(dist, self.k - 1)[: self.k]
        d = dist[idx]
        if (d == 0.0).any():
            device_s, host_s = self.values[idx[d == 0.0]].mean(axis=0)
        else:
            w = 1.0 / d
            device_s, host_s = (w[:, None] * self.values[idx]).sum(axis=0) / w.sum()
        return StepCost(device_s=float(device_s), host_s=float(host_s))


def _load_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{") and "\n" not in stripped.strip():
        obj = json.loads(text)
        return obj.get("steps", [obj])
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _record_cost(rec: dict[str, Any]) -> tuple[float, float]:
    """(device_s, host_s) of a record from `--simulation-config log_path` or
    a hand-written table: {"cost": {"device_s", "host_s"}}, {"device_s"} or
    {"device_ms"}, with an optional "host_s"/"host_ms"."""
    cost = rec["cost"] if isinstance(rec.get("cost"), dict) else rec
    if "device_s" in cost:
        device_s = float(cost["device_s"])
    elif "device_ms" in cost:
        device_s = float(cost["device_ms"]) * 1e-3
    else:
        raise ValueError(f"record has no device time: {list(rec)}")
    host_s = float(cost.get("host_s", cost.get("host_ms", 0.0) * 1e-3))
    return device_s, host_s


def _record_features(rec: dict[str, Any]) -> tuple[float, float, float]:
    if "step" in rec and isinstance(rec["step"], dict):
        reqs = rec["step"]["requests"]
        return (
            float(len(reqs)),
            float(sum(r["num_scheduled_tokens"] for r in reqs)),
            float(sum(r["num_computed_tokens"] for r in reqs)),
        )
    return (
        float(rec["num_reqs"]),
        float(rec["num_tokens"]),
        float(rec.get("sum_past_kv", rec.get("num_kv_tokens", 0.0))),
    )


def acceptance_rates_from_log(path: str) -> list[float]:
    """Per-position conditional draft acceptance rates of a step log recorded
    with speculative decoding: rate[i] is the fraction of requests that had a
    draft at position i, and had all earlier drafts accepted, whose draft at
    position i was accepted too."""
    reached = np.zeros(0)
    accepted = np.zeros(0)
    for rec in _load_json_or_jsonl(path):
        num_accepted = rec.get("cost", {}).get("num_accepted")
        if not num_accepted:
            continue
        for req, n in zip(rec["step"]["requests"], num_accepted):
            drafts = req["num_spec_tokens"]
            if drafts > len(reached):
                reached = np.pad(reached, (0, drafts - len(reached)))
                accepted = np.pad(accepted, (0, drafts - len(accepted)))
            n = min(n, drafts)
            reached[: min(n + 1, drafts)] += 1
            accepted[:n] += 1
    if not reached.any():
        raise ValueError(f"{path} has no speculative decoding steps")
    return (accepted / np.maximum(reached, 1)).tolist()


class TableTimingProvider:
    """Interpolates a measured table of step costs.

    `path` is a JSON/JSONL file of records; each record carries a device time,
    optionally a host time, and either a full `step` descriptor (as written by
    `log_path`) or the features `num_reqs`, `num_tokens`, `sum_past_kv`."""

    def __init__(self, vllm_config: VllmConfig, *, path: str, k: int = 4) -> None:
        records = _load_json_or_jsonl(path)
        self.table = _KNNTable(
            [(_record_features(r), _record_cost(r)) for r in records], k
        )
        logger.info("Loaded %d timing records from %s", len(records), path)

    def estimate(self, step: StepDescriptor) -> StepCost:
        return self.table.query(_step_features(step))


class ReplayTimingProvider:
    """Exact replay of recorded step times keyed by batch composition.

    A recorded run (real or simulated) is matched step by step on the sorted
    ``[(num_scheduled_tokens, num_computed_tokens), ...]`` composition. On a
    miss, ``fallback="knn"`` interpolates like `TableTimingProvider` and
    ``fallback="zero"`` returns `miss_fallback_s`. The miss rate is reported
    in `metrics()`; a high miss rate means the simulated scheduler diverged
    from the recorded one, which is exactly what this provider is for."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        path: str,
        fallback: str = "knn",
        k: int = 4,
        miss_fallback_s: float = 0.0,
    ) -> None:
        records = _load_json_or_jsonl(path)
        self.exact: dict[str, tuple[float, float]] = {}
        points = []
        for rec in records:
            cost = _record_cost(rec)
            if "step" in rec:
                reqs = rec["step"]["requests"]
                key = json.dumps(
                    sorted(
                        (r["num_scheduled_tokens"], r["num_computed_tokens"])
                        for r in reqs
                    ),
                    separators=(",", ":"),
                )
                self.exact[key] = cost
            points.append((_record_features(rec), cost))
        if fallback not in ("knn", "zero"):
            raise ValueError(f"unknown replay fallback {fallback!r}")
        self.fallback = fallback
        self.miss_fallback_s = miss_fallback_s
        self.table = _KNNTable(points, k) if fallback == "knn" else None
        self.hits = 0
        self.misses = 0
        logger.info(
            "Loaded %d replay records (%d exact keys) from %s",
            len(records),
            len(self.exact),
            path,
        )

    def estimate(self, step: StepDescriptor) -> StepCost:
        cost = self.exact.get(step.replay_key())
        if cost is not None:
            self.hits += 1
            device_s, host_s = cost
            return StepCost(
                device_s=device_s, host_s=host_s, breakdown={"replay_exact": 1.0}
            )
        self.misses += 1
        if self.table is not None:
            return dataclasses.replace(
                self.table.query(_step_features(step)),
                breakdown={"replay_exact": 0.0},
            )
        return StepCost(device_s=self.miss_fallback_s, breakdown={"replay_exact": 0.0})

    def metrics(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "replay_exact_match_steps": self.hits,
            "replay_miss_steps": self.misses,
            "replay_miss_rate": (self.misses / total) if total else 0.0,
        }


BUILTIN_TIMING_PROVIDERS: dict[str, type] = {
    "zero": ZeroTimingProvider,
    "constant": ConstantTimingProvider,
    "table": TableTimingProvider,
    "replay": ReplayTimingProvider,
}


def get_timing_provider(
    name: str, args: dict[str, Any], vllm_config: VllmConfig
) -> TimingProvider:
    cls = BUILTIN_TIMING_PROVIDERS.get(name)
    if cls is None:
        if "." not in name:
            raise ValueError(
                f"Unknown timing provider {name!r}. Built-ins: "
                f"{sorted(BUILTIN_TIMING_PROVIDERS)}; otherwise pass a "
                "fully-qualified class name."
            )
        cls = resolve_obj_by_qualname(name)
    provider = cls(vllm_config, **args)
    if not isinstance(provider, TimingProvider):
        raise TypeError(f"{name} does not implement TimingProvider.estimate")
    return provider


# --------------------------------------------------------------------------- #
# Clock and async output
# --------------------------------------------------------------------------- #


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep_until(self, deadline: float) -> None: ...


class RealClock:
    """Wall clock: simulated steps take real time, so every existing metric,
    benchmark client and router sees realistic latencies."""

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


class VirtualClock:
    """Simulated time: waiting for a deadline moves the clock to it at once.

    With the in-process engine the worker and the driver share one process
    and therefore one `VIRTUAL_CLOCK`. The engine's own host loop
    (scheduling, output processing) takes no simulated time; a step's
    predicted `host_s` still holds the engine thread."""

    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def sleep_until(self, deadline: float) -> None:
        self._now = max(self._now, deadline)


class FastForwardClock:
    """The wall clock with waits skipped: host work takes its real time, and
    waiting for a deadline shifts the clock forward instead of sleeping. With
    the in-process engine nothing else runs while the engine core waits for a
    step, so this reproduces the `real` clock's timeline in a fraction of the
    time, host overheads included."""

    def __init__(self) -> None:
        self._offset = 0.0

    def now(self) -> float:
        return time.monotonic() + self._offset

    def sleep_until(self, deadline: float) -> None:
        self._offset += max(0.0, deadline - self.now())


# One instance per process, shared by the in-process engine's worker and the
# driver that timestamps requests.
VIRTUAL_CLOCK = VirtualClock()
FAST_FORWARD_CLOCK = FastForwardClock()
_building_driven_engine = False


@contextlib.contextmanager
def building_driven_engine() -> Iterator[None]:
    """Engines built in this context run in the process of a driver that
    reads the same clock (`vllm.v1.worker.simulated.offline`)."""
    global _building_driven_engine
    _building_driven_engine = True
    try:
        yield
    finally:
        _building_driven_engine = False


def is_building_driven_engine() -> bool:
    return _building_driven_engine


def get_clock(kind: str) -> Clock:
    if kind == "real":
        return RealClock()
    if kind == "virtual":
        return VIRTUAL_CLOCK
    if kind == "fast_forward":
        return FAST_FORWARD_CLOCK
    raise ValueError(f"unknown clock {kind!r}")


class SimulatedAsyncOutput(AsyncModelRunnerOutput):
    """Defers the step's completion to `get_output()`.

    Executors resolve `AsyncModelRunnerOutput` lazily (when the engine core
    needs the result), so blocking here instead of inside `execute_model`
    lets the scheduler prepare step N+1 while "step N" is still running,
    exactly as with a real device under async scheduling."""

    def __init__(self, output: ModelRunnerOutput, deadline: float, clock: Clock):
        self.output = output
        self.deadline = deadline
        self.clock = clock

    def get_output(self) -> ModelRunnerOutput:
        self.clock.sleep_until(self.deadline)
        return self.output


# --------------------------------------------------------------------------- #
# Step log
# --------------------------------------------------------------------------- #


def step_log_path(vllm_config: VllmConfig) -> str | None:
    """This rank's step log, if it writes one. Only the rank whose output the
    executor returns (last pipeline stage, first tensor-parallel rank) writes
    one. Under pipeline parallelism a recorded device time therefore runs from
    the step's arrival at the last stage, whose first kernels wait for the
    earlier stages, to its end: the step's time through the whole pipeline,
    which is what `device_s` means to the simulated runner (while the pipeline
    is not saturated). MoE data-parallel ranks step in lockstep and log the
    joint step, so only DP rank 0 writes; independent (dense) DP rank i > 0
    writes `<log_path>.dp<i>`."""
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    path = vllm_config.simulation_config.log_path
    if path is None or not get_pp_group().is_last_rank:
        return None
    if get_tp_group().rank_in_group != 0:
        return None
    parallel_config = vllm_config.parallel_config
    dp_index = parallel_config.data_parallel_index
    if parallel_config.data_parallel_size > 1:
        return path if parallel_config.data_parallel_rank == 0 else None
    if dp_index > 0:
        root, ext = os.path.splitext(path)
        path = f"{root}.dp{dp_index}{ext}"
    return path


class StepLogWriter:
    """Appends one JSON line per simulated step. The same records feed the
    `table` and `replay` providers."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a")

    def write(
        self,
        step: StepDescriptor,
        cost: StepCost,
        t_start: float,
        t_deadline: float,
        finished_req_ids: list[str],
    ) -> None:
        rec = {
            "step": step.to_json_dict(),
            "cost": asdict(cost),
            "t_start": t_start,
            "t_deadline": t_deadline,
            "finished_req_ids": finished_req_ids,
        }
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
