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

import json
import time
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


@dataclass(frozen=True)
class StepCost:
    """Predicted cost of one step.

    `device_s` is the time the device is busy. `host_s` is an optional extra
    host-side term for the *target* machine; the host where the simulation
    runs already pays its own scheduler cost in real time, so providers should
    leave it at zero unless they model a different host."""

    device_s: float
    host_s: float = 0.0
    num_accepted: tuple[int, ...] | None = None
    """Per-request accepted draft tokens (speculative decoding), or None."""
    breakdown: dict[str, float] | None = None
    """Optional named components (e.g. per kernel class), for logging."""

    @property
    def total_s(self) -> float:
        return self.device_s + self.host_s


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
    sum_past_kv) features. Points are normalized once; a query is one
    vectorized distance pass, so lookups stay off the step's critical path."""

    def __init__(self, points: list[tuple[tuple[float, float, float], float]], k: int):
        if not points:
            raise ValueError("timing table is empty")
        feats = np.asarray([p for p, _ in points], dtype=np.float64)
        self.values = np.asarray([v for _, v in points], dtype=np.float64)
        self.mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        self.points = (feats - self.mean) / self.std
        self.k = min(max(1, k), len(points))

    def query(self, f: tuple[float, float, float]) -> float:
        q = (np.asarray(f, dtype=np.float64) - self.mean) / self.std
        dist = np.sqrt(((self.points - q) ** 2).sum(axis=1))
        idx = np.argpartition(dist, self.k - 1)[: self.k]
        d = dist[idx]
        if (d == 0.0).any():
            return float(self.values[idx[d == 0.0]].mean())
        w = 1.0 / d
        return float((w * self.values[idx]).sum() / w.sum())


def _load_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{") and "\n" not in stripped.strip():
        obj = json.loads(text)
        return obj.get("steps", [obj])
    return [json.loads(line) for line in text.splitlines() if line.strip()]


_CRITICAL_PATHS = ("max", "device")


def _record_device_s(rec: dict[str, Any], critical_path: str = "device") -> float:
    """Step time of a record from `--simulation-config log_path` or a
    hand-written table: {"cost": {"device_s", "host_s"}}, {"device_s"} or
    {"device_ms"}.

    With ``critical_path="max"`` a recorded step costs
    ``max(device_s, host_s)``: a real step whose host work (input preparation,
    kernel launches, sampling) outlasts its device work is host-bound, e.g.
    large prefills under piecewise CUDA graphs. ``"device"`` uses the device
    time alone."""
    if critical_path not in _CRITICAL_PATHS:
        raise ValueError(f"critical_path must be one of {_CRITICAL_PATHS}")
    cost = rec["cost"] if isinstance(rec.get("cost"), dict) else rec
    if "device_s" in cost:
        device_s = float(cost["device_s"])
    elif "device_ms" in cost:
        device_s = float(cost["device_ms"]) * 1e-3
    else:
        raise ValueError(f"record has no device time: {list(rec)}")
    if critical_path == "max":
        return max(device_s, float(cost.get("host_s", 0.0)))
    return device_s


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


class TableTimingProvider:
    """Interpolates a measured table of step costs.

    `path` is a JSON/JSONL file of records; each record carries a device time
    and either a full `step` descriptor (as written by `log_path`) or the
    features `num_reqs`, `num_tokens`, `sum_past_kv`. `critical_path` is as in
    `_record_device_s`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        path: str,
        k: int = 4,
        critical_path: str = "max",
    ) -> None:
        records = _load_json_or_jsonl(path)
        self.table = _KNNTable(
            [
                (_record_features(r), _record_device_s(r, critical_path))
                for r in records
            ],
            k,
        )
        logger.info("Loaded %d timing records from %s", len(records), path)

    def estimate(self, step: StepDescriptor) -> StepCost:
        return StepCost(device_s=self.table.query(_step_features(step)))


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
        critical_path: str = "max",
    ) -> None:
        records = _load_json_or_jsonl(path)
        self.exact: dict[str, float] = {}
        points = []
        for rec in records:
            device_s = _record_device_s(rec, critical_path)
            if "step" in rec:
                reqs = rec["step"]["requests"]
                key = json.dumps(
                    sorted(
                        (r["num_scheduled_tokens"], r["num_computed_tokens"])
                        for r in reqs
                    ),
                    separators=(",", ":"),
                )
                self.exact[key] = device_s
            points.append((_record_features(rec), device_s))
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
        device_s = self.exact.get(step.replay_key())
        if device_s is not None:
            self.hits += 1
            return StepCost(device_s=device_s, breakdown={"replay_exact": 1.0})
        self.misses += 1
        if self.table is not None:
            return StepCost(
                device_s=self.table.query(_step_features(step)),
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
