# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step timing for simulated execution: a `TimingProvider` prices a
`StepDescriptor` as a `StepCost`."""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Recorded steps a k-NN lookup interpolates.
_DEFAULT_KNN_K = 4
# Relative error below which a smaller host-time fit wins a tie.
_FIT_TIE_TOLERANCE = 1e-9


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


def parallel_summary(vllm_config: VllmConfig) -> ParallelSummary:
    pc = vllm_config.parallel_config
    return ParallelSummary(
        tensor_parallel_size=pc.tensor_parallel_size,
        pipeline_parallel_size=pc.pipeline_parallel_size,
        data_parallel_size=pc.data_parallel_size,
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

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> StepDescriptor:
        return cls(
            **d
            | {
                "requests": tuple(SimRequestStep(**r) for r in d["requests"]),
                "parallel": ParallelSummary(**d.get("parallel", {})),
            }
        )


def accepted_in_step_order(
    step: StepDescriptor, accepted: Mapping[str, int]
) -> tuple[int, ...] | None:
    """Accepted drafts of the step's requests, in its order; None for a joint
    MoE-DP step, whose other ranks' requests are not in `accepted`."""
    if not all(r.req_id in accepted for r in step.requests):
        return None
    return tuple(accepted[r.req_id] for r in step.requests)


def build_step_descriptor(
    scheduler_output: SchedulerOutput,
    prompt_lens: dict[str, int],
    step_index: int,
    parallel: ParallelSummary,
) -> StepDescriptor:
    """Describe one scheduled step from the `SchedulerOutput`, as the simulated
    runner and a recording worker both do. `prompt_lens` maps request
    ids to prompt lengths across steps."""
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
    """Predicted cost of one step: device time, and the host time the step's
    `execute_model`/`sample_tokens` calls take."""

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


class ZeroTimingProvider:
    """Every step takes no device time (pure scheduler / KV-cache simulation)."""

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any) -> None:
        if kwargs:
            raise ValueError(f"zero timing provider takes no arguments, got {kwargs}")

    def estimate(self, step: StepDescriptor) -> StepCost:
        return StepCost(device_s=0.0)


@dataclass(frozen=True)
class HostTimeModel:
    """A step's host time: ``fixed_us + per_req_us * R + per_prefill_token_us *
    P`` for R scheduled requests and P prefill tokens."""

    fixed_us: float = 0.0
    per_req_us: float = 0.0
    per_prefill_token_us: float = 0.0

    @classmethod
    def from_log(cls, path: str) -> HostTimeModel:
        """Non-negative least-squares fit to a recorded step log; terms the log
        cannot determine stay at zero."""
        rows, host_us = [], []
        for rec in load_step_log(path):
            if "step" not in rec:
                raise ValueError(f"{path}: host time needs a recorded step log")
            step = StepDescriptor.from_json_dict(rec["step"])
            rows.append((1.0, step.num_reqs, step.num_prefill_tokens))
            host_us.append(_record_cost(rec)[1] * 1e6)
        if not rows:
            raise ValueError(f"{path} has no recorded steps")
        a, y = np.asarray(rows, dtype=np.float64), np.asarray(host_us)
        if not a[:, 2].any():
            logger.warning("%s has no prefill steps: no per-token host time", path)
        # NNLS over the determined subsets of terms; on a tie the smaller
        # subset, preferring the fixed term, wins.
        best, best_err = np.zeros(3), np.inf
        subsets = sorted(
            (c for k in (1, 2, 3) for c in itertools.combinations(range(3), k)),
            key=lambda c: (len(c), c[0] != 0),
        )
        for cols in subsets:
            if np.linalg.matrix_rank(a[:, cols]) < len(cols):
                continue
            fit, *_ = np.linalg.lstsq(a[:, cols], y, rcond=None)
            err = float(np.sum((a[:, cols] @ fit - y) ** 2))
            if (fit >= 0).all() and err < best_err * (1 - _FIT_TIE_TOLERANCE):
                best, best_err = np.zeros(3), err
                best[list(cols)] = fit
        model = cls(*(float(c) for c in best))
        logger.info("Host time fitted to %s: %s", path, model)
        return model

    def host_s(self, step: StepDescriptor) -> float:
        return 1e-6 * (
            self.fixed_us
            + self.per_req_us * step.num_reqs
            + self.per_prefill_token_us * step.num_prefill_tokens
        )


class LinearTimingProvider:
    """Linear model: ``fixed_us + prefill_us_per_token * P + decode_us_per_req * D
    + decode_us_per_kv_token * K`` where P is prefill tokens, D decode
    requests and K the KV tokens read by decode requests. Host time is
    `host_us` per step, or fitted to the recorded log `host_fit_log`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        fixed_us: float = 0.0,
        prefill_us_per_token: float = 0.0,
        decode_us_per_req: float = 0.0,
        decode_us_per_kv_token: float = 0.0,
        host_us: float = 0.0,
        host_fit_log: str | None = None,
    ) -> None:
        self.fixed_us = fixed_us
        self.prefill_us_per_token = prefill_us_per_token
        self.decode_us_per_req = decode_us_per_req
        self.decode_us_per_kv_token = decode_us_per_kv_token
        if host_fit_log is not None and host_us:
            raise ValueError("Pass host_us or host_fit_log, not both.")
        self.host = (
            HostTimeModel.from_log(host_fit_log)
            if host_fit_log is not None
            else HostTimeModel(fixed_us=host_us)
        )

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
            host_s=self.host.host_s(step),
            breakdown={"prefill_us": prefill, "decode_us": decode},
        )


def _step_features(step: StepDescriptor) -> tuple[float, float, float]:
    """Coarse batch features used for nearest-neighbour lookups."""
    past = sum(r.num_computed_tokens for r in step.requests)
    return (float(step.num_reqs), float(step.num_tokens), float(past))


# Lookups of this process's k-NN tables, and those outside the recorded range.
_knn_lookups = [0, 0]


def knn_lookup_counts() -> tuple[int, int]:
    """(lookups, lookups outside the recorded range) of this process's k-NN
    timing tables so far."""
    return _knn_lookups[0], _knn_lookups[1]


class _KNNTable:
    """Inverse-distance weighted k-NN over z-scored (num_reqs, num_tokens,
    sum_past_kv) features of recorded steps. A lookup outside the
    recorded range, which k-NN underestimates, is counted and warned
    about."""

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
        # One contiguous array per feature: summing three columns is several
        # times faster than reducing over a length-3 axis, with the same result.
        self.columns = np.ascontiguousarray(((feats - self.mean) / self.std).T)
        self.k = min(max(1, k), len(points))
        self.lo = feats.min(axis=0).tolist()
        self.hi = feats.max(axis=0).tolist()
        self.warned = False

    def query(self, f: tuple[float, float, float]) -> StepCost:
        _knn_lookups[0] += 1
        if any(not lo <= v <= hi for v, lo, hi in zip(f, self.lo, self.hi)):
            if not self.warned:
                self.warned = True
                logger.warning(
                    "Step (num_reqs, num_tokens, sum_past_kv) = %s is outside "
                    "the recorded range %s to %s: its time is that of the "
                    "nearest recorded steps. Record under a load that covers "
                    "the simulated one.",
                    f,
                    self.lo,
                    self.hi,
                )
            _knn_lookups[1] += 1
        q = (np.asarray(f, dtype=np.float64) - self.mean) / self.std
        x, y, z = self.columns
        dist = np.sqrt((x - q[0]) ** 2 + (y - q[1]) ** 2 + (z - q[2]) ** 2)
        idx = np.argpartition(dist, self.k - 1)[: self.k]
        d = dist[idx]
        if (d == 0.0).any():
            device_s, host_s = self.values[idx[d == 0.0]].mean(axis=0)
        else:
            w = 1.0 / d
            device_s, host_s = (w[:, None] * self.values[idx]).sum(axis=0) / w.sum()
        return StepCost(device_s=float(device_s), host_s=float(host_s))


def load_step_log(
    path: str, kind: Literal["step", "kv_transfers", "kv_post"] = "step"
) -> list[dict[str, Any]]:
    """The step records of a JSONL step log, or its records of completed KV
    transfers (`kv_transfers`) or NIXL read postings (`kv_post`)."""
    lines = Path(path).read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if kind == "step":
        return [r for r in records if "kv_transfers" not in r and "kv_post" not in r]
    return [r for r in records if kind in r]


def _record_cost(rec: dict[str, Any]) -> tuple[float, float]:
    """(device_s, host_s) of a step record, or of a hand-written one."""
    cost = rec.get("cost", rec)
    return float(cost["device_s"]), float(cost.get("host_s") or 0.0)


def _record_features(rec: dict[str, Any]) -> tuple[float, float, float]:
    """`_step_features` of a step record, or of a hand-written one:
    {"num_reqs", "num_tokens", "sum_past_kv", "device_s"[, "host_s"]}."""
    if "step" in rec:
        return _step_features(StepDescriptor.from_json_dict(rec["step"]))
    return (
        float(rec["num_reqs"]),
        float(rec["num_tokens"]),
        float(rec["sum_past_kv"]),
    )


def acceptance_rates_from_log(path: str) -> list[float]:
    """Per-position draft acceptance rates of a step log recorded with
    speculative decoding, each conditional on the earlier drafts being
    accepted."""
    reached = np.zeros(0)
    accepted = np.zeros(0)
    for rec in load_step_log(path):
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
    """Interpolates recorded steps (`path`, a step log or a hand-written JSONL
    table)."""

    def __init__(
        self, vllm_config: VllmConfig, *, path: str, k: int = _DEFAULT_KNN_K
    ) -> None:
        records = load_step_log(path)
        self.table = _KNNTable(
            [(_record_features(r), _record_cost(r)) for r in records], k
        )
        logger.info("Loaded %d timing records from %s", len(records), path)

    def estimate(self, step: StepDescriptor) -> StepCost:
        return self.table.query(_step_features(step))


class ReplayTimingProvider:
    """Replays the mean recorded time of a step's exact batch composition; on a
    miss, interpolates (``fallback="knn"``) or returns
    `miss_fallback_s`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        path: str,
        fallback: str = "knn",
        k: int = _DEFAULT_KNN_K,
        miss_fallback_s: float = 0.0,
    ) -> None:
        records = load_step_log(path)
        # Steps of one composition, recorded more than once, vary run to run:
        # replay their mean.
        same_key: dict[str, list[tuple[float, float]]] = {}
        points = []
        for rec in records:
            cost = _record_cost(rec)
            if "step" in rec:
                key = StepDescriptor.from_json_dict(rec["step"]).replay_key()
                same_key.setdefault(key, []).append(cost)
            points.append((_record_features(rec), cost))
        self.exact: dict[str, tuple[float, float]] = {
            key: (
                math.fsum(d for d, _ in costs) / len(costs),
                math.fsum(h for _, h in costs) / len(costs),
            )
            for key, costs in same_key.items()
        }
        if fallback not in ("knn", "zero"):
            raise ValueError(f"unknown replay fallback {fallback!r}")
        self.fallback = fallback
        self.miss_fallback_s = miss_fallback_s
        self.table = _KNNTable(points, k) if fallback == "knn" else None
        logger.info(
            "Loaded %d replay records (%d exact keys) from %s",
            len(records),
            len(self.exact),
            path,
        )

    def estimate(self, step: StepDescriptor) -> StepCost:
        cost = self.exact.get(step.replay_key())
        if cost is not None:
            device_s, host_s = cost
            return StepCost(
                device_s=device_s, host_s=host_s, breakdown={"replay_exact": 1.0}
            )
        if self.table is not None:
            return dataclasses.replace(
                self.table.query(_step_features(step)),
                breakdown={"replay_exact": 0.0},
            )
        return StepCost(device_s=self.miss_fallback_s, breakdown={"replay_exact": 0.0})


BUILTIN_TIMING_PROVIDERS: dict[str, type] = {
    "zero": ZeroTimingProvider,
    "linear": LinearTimingProvider,
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


class Clock(Protocol):
    horizon: float
    """When the driver of an in-process engine next has work for it (a
    request arrival); an engine that is only waiting may skip ahead to it."""

    def now(self) -> float: ...

    def sleep_until(self, deadline: float) -> None: ...


class RealClock:
    """Wall clock: simulated steps take real time, so every existing metric,
    benchmark client and router sees realistic latencies."""

    horizon = math.inf

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


class VirtualClock:
    """Simulated time: waiting for a deadline moves the clock to it."""

    def __init__(self) -> None:
        self._now = 0.0
        self.horizon = math.inf

    def now(self) -> float:
        return self._now

    def sleep_until(self, deadline: float) -> None:
        self._now = max(self._now, deadline)


class FastForwardClock:
    """The wall clock with waits skipped: waiting for a deadline shifts the
    clock forward instead of sleeping."""

    def __init__(self) -> None:
        self._offset = 0.0
        self.horizon = math.inf

    def now(self) -> float:
        return time.monotonic() + self._offset

    def sleep_until(self, deadline: float) -> None:
        self._offset += max(0.0, deadline - self.now())


_CLOCKS: dict[str, type[Clock]] = {
    "real": RealClock,
    "virtual": VirtualClock,
    "fast_forward": FastForwardClock,
}
# The clock of the engine being built in the process of its driver.
_driven_engine_clock: Clock | None = None


@contextlib.contextmanager
def building_driven_engine(clock: Clock) -> Iterator[None]:
    """Engines built in this context share `clock` with their driver."""
    global _driven_engine_clock
    _driven_engine_clock = clock
    try:
        yield
    finally:
        _driven_engine_clock = None


def driven_engine_clock() -> Clock | None:
    """The clock of the driven engine being built, if any."""
    return _driven_engine_clock


def new_clock(kind: str) -> Clock:
    """A new clock of `SimulationConfig.clock` kind `kind`."""
    return _CLOCKS[kind]()


class SimulatedAsyncOutput(AsyncModelRunnerOutput):
    """Defers a step's completion to `get_output()`, so the engine core
    schedules the next step meanwhile, as under async scheduling."""

    def __init__(self, output: ModelRunnerOutput, deadline: float, clock: Clock):
        self.output = output
        self.deadline = deadline
        self.clock = clock
        self.resolved = False

    def get_output(self) -> ModelRunnerOutput:
        self.clock.sleep_until(self.deadline)
        self.resolved = True
        return self.output


def is_output_rank() -> bool:
    """Whether this rank returns the step output: the last pipeline stage's
    first tensor-parallel rank."""
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    return get_pp_group().is_last_rank and get_tp_group().rank_in_group == 0


def step_log_path(vllm_config: VllmConfig) -> str | None:
    """This rank's step log, if it writes one: the output rank's, of DP rank 0
    under MoE DP (which logs joint steps), else `<log_path>.dp<i>` for an
    independent DP rank i > 0. Under pipeline parallelism a recorded step is
    timed on the last stage, so its device time spans the whole pipeline."""
    path = vllm_config.simulation_config.log_path
    if path is None or not is_output_rank():
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
    """Appends one JSON line per simulated step, plus one per step in which
    KV connector transfers completed. The step records feed the `table` and
    `replay` providers, the transfer records `kv_transfer_timing`."""

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

    def write_transfers(self, transfers: dict[str, dict[str, float]], t: float) -> None:
        """Append the KV transfers that completed in a step, per direction:
        their count, bytes and summed duration."""
        rec = {"kv_transfers": transfers, "t": t}
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()

    def write_kv_post(self, reads: int, runs: int, seconds: float, t: float) -> None:
        """Append the NIXL reads a decode rank posted in a step: their count,
        runs of consecutive blocks (see `PostCost`) and the time posting took,
        which the step's own record leaves out."""
        rec = {"kv_post": {"reads": reads, "runs": runs, "seconds": seconds}, "t": t}
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
