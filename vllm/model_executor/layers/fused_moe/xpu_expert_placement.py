# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts placed per expert between VRAM and pinned host memory on XPU.

``--cpu-offload-gb`` offloads whole parameters in module order, so for a MoE model
the first layers end up with every expert in host memory and the last ones with
none. With ``--cpu-offload-expert-rankings`` on XPU the budget is spent per expert
instead: each layer keeps its most-routed experts in VRAM (hot) and the rest in
pinned host memory (cold), the same number of cold experts in every layer.

Every expert stays routable. The hot and the cold set each run the stock grouped
GEMM, each with an ``expert_map`` that maps only its own experts (-1 elsewhere),
and the two outputs are summed, so results equal the all-resident layer up to the
order of that sum. The maps are device tensors, so a later placement change can
rewrite them in place without recapturing graphs.

With ``--cpu-offload-expert-state`` the server learns its own rankings: each
placed layer counts its routes per expert on the device (inside captured graphs
too), the worker folds the counts into a long-term score per expert between
steps, and the order is written to the state file, in the rankings format,
every few minutes and at shutdown. The next boot places by the state file ahead
of the rankings file.
"""

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config.offload import OffloadConfig, UVAOffloadConfig

logger = init_logger(__name__)

# Route counting and the learned state.
_LARGE_STEP_TOKENS = 512  # steps this large count as prefill in the counters
_PREFILL_WEIGHT = 0.1  # prefill touches nearly every expert; decode tells more
_FOLD_S = 5.0  # counters -> scores
_STATE_SAVE_S = 300.0
_STATE_DECAY = 0.999  # per active fold
_STATE_PRIOR = 0.05  # weight of the boot order in the saved order
_STATE_MIN_FOLDS = 12  # a boot with less traffic does not overwrite the state

# Decoder layers only: other MoE layers (e.g. MTP) keep their experts in VRAM, so
# the budget below is exact.
_LAYER_RE = re.compile(r"(?:^|\.)(?:model|language_model)\.layers\.(\d+)\.")
_EXPERT_PARAMS = (
    "w13_qweight",
    "w13_scales",
    "w13_qzeros",
    "w13_bias",
    "w2_qweight",
    "w2_scales",
    "w2_qzeros",
    "w2_bias",
)


def _uva_enabled() -> bool:
    """The decision `UVAOffloader` takes: host views are available and not
    disabled with VLLM_WEIGHT_OFFLOADING_DISABLE_UVA."""
    import vllm.envs as envs
    from vllm.utils.platform_utils import is_uva_available

    return is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA


def places_experts(offload_config: "OffloadConfig") -> bool:
    """True when the UVA offload settings select per-expert placement.

    Raises ValueError when they select it while UVA is disabled or unavailable:
    cold experts are only ever read through host views, there is no copy
    fallback as in `UVAOffloader`.
    """
    uva = offload_config.uva
    selected = (
        (
            uva.cpu_offload_expert_rankings is not None
            or uva.cpu_offload_expert_state is not None
        )
        and current_platform.is_xpu()
        and offload_config.offload_backend in ("auto", "uva")
        and offload_config.prefetch.offload_group_size == 0
        and uva.cpu_offload_gb > 0
    )
    if selected and not _uva_enabled():
        raise ValueError(
            "--cpu-offload-expert-rankings / --cpu-offload-expert-state read cold "
            "experts from pinned host memory through UVA, which is disabled "
            "(VLLM_WEIGHT_OFFLOADING_DISABLE_UVA) or unavailable on this "
            "platform. Drop them to offload parameters in module order."
        )
    return selected


def _offload_config() -> "OffloadConfig | None":
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None or not places_experts(config.offload_config):
        return None
    return config.offload_config


def expert_bytes(
    quant_method: Any,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
) -> int:
    """One routed expert's packed weights and scales, as
    ``AutoGPTQMoEMethod.create_weights`` lays them out (int32-packed qweight,
    ``params_dtype`` scales per group)."""
    quant_config = quant_method.quant_config
    w13_out = quant_method.moe.w13_num_shards * intermediate_size_per_partition
    pack = quant_config.pack_factor
    groups13 = groups2 = 1
    if quant_config.group_size != -1:
        groups13 = hidden_size // quant_config.group_size
        groups2 = intermediate_size_per_partition // quant_config.group_size
    qweight = (
        hidden_size // pack * w13_out
        + intermediate_size_per_partition // pack * hidden_size
    ) * torch.int32.itemsize
    scales = (groups13 * w13_out + groups2 * hidden_size) * params_dtype.itemsize
    return qweight + scales


def budget_cold_count(
    gb: float, num_layers: int, expert_bytes: int, num_local: int
) -> int:
    """Cold experts per layer per rank that ``gb`` per rank pays for.

    At least one expert stays hot in each layer.
    """
    cold = int(gb * 2**30 / max(num_layers * expert_bytes, 1))
    return max(0, min(cold, num_local - 1))


def _parse_rankings(raw: Any, path: str) -> dict[int, list[int]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of layer index -> expert ids")
    return {int(k): [int(e) for e in v] for k, v in raw.items()}


@cache
def load_rankings(path: str) -> dict[int, list[int]]:
    """JSON ``{"<layer index>": [global expert ids, hottest first], ...}``."""
    with open(path) as handle:
        return _parse_rankings(json.load(handle), path)


@cache
def load_state(path: str) -> dict[int, list[int]]:
    """The learned state, in the rankings format; empty when it is missing or
    unreadable. It only moves where experts live, so a bad file falls back to
    the rankings instead of failing the boot."""
    try:
        with open(path) as handle:
            return _parse_rankings(json.load(handle), path)
    except FileNotFoundError:
        logger.info("Per-expert offload placement: no learned state at %s yet", path)
    except (OSError, ValueError, TypeError, AttributeError) as err:
        logger.warning(
            "Per-expert offload placement: ignoring the learned state %s: %s",
            path,
            err,
        )
    return {}


def boot_order(uva: "UVAOffloadConfig", layer_idx: int) -> tuple[list[int], str]:
    """(global expert ids hottest first, its source) for one layer at boot: the
    learned state, else the rankings file, else none (id order)."""
    if uva.cpu_offload_expert_state is not None:
        learned = load_state(uva.cpu_offload_expert_state).get(layer_idx)
        if learned:
            return learned, "state"
    if uva.cpu_offload_expert_rankings is not None:
        ranked = load_rankings(uva.cpu_offload_expert_rankings).get(layer_idx)
        if ranked:
            return ranked, "rankings"
    return [], "id"


def hot_experts(
    ranked: list[int], local_globals: list[int], hot_count: int
) -> list[int]:
    """The ``hot_count`` local experts that rank highest; unranked ones by id."""
    local = set(local_globals)
    order = [g for g in dict.fromkeys(ranked) if g in local]
    seen = set(order)
    order += [g for g in local_globals if g not in seen]
    return order[:hot_count]


class _PinnedArena:
    """Pinned host memory carved from 1 GiB chunks (first fit, 4 KiB aligned).

    The pinned allocator rounds each block up to a power of two, which can waste
    up to half of a layer's cold experts. Chunks of exactly 2^30 B are not
    rounded, so ``--cpu-offload-gb`` is close to the host memory it takes.

    The arena lives as long as the process: releasing a model does not free its
    chunks, and a second model loaded in the same process adds chunks to it.
    """

    CHUNK = 1 << 30
    ALIGN = 4096

    def __init__(self) -> None:
        self.chunks: list[torch.Tensor] = []
        self.used: list[int] = []
        self.total = 0

    def empty(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        for i, used in enumerate(self.used):
            if used + nbytes <= self.chunks[i].numel():
                break
        else:
            size = (
                self.CHUNK if nbytes <= self.CHUNK else 1 << (nbytes - 1).bit_length()
            )
            self.chunks.append(torch.empty(size, dtype=torch.uint8, pin_memory=True))
            self.used.append(0)
            self.total += size
            i, used = len(self.chunks) - 1, 0
        self.used[i] = used + -(-nbytes // self.ALIGN) * self.ALIGN
        return self.chunks[i][used : used + nbytes].view(dtype).view(shape)

    @property
    def gib(self) -> float:
        return self.total / 2**30


_ARENA = _PinnedArena()
# Placements that count routes, by decoder layer index.
_PLACEMENTS: dict[int, "ExpertPlacement"] = {}
# Set by the first runner step after the warm-up: the memory profile and the
# warm-up don't count.
_LIVE = [False]
# True inside the worker's warm-up (``warmup()``): its steps are not traffic.
_WARMUP = [False]
_ONES: dict[str, torch.Tensor] = {}
_ONES_N = 1 << 16


def _ones(device: torch.device, n: int = _ONES_N) -> torch.Tensor:
    """``n`` int64 ones for index_add_: a slice of one buffer per device, so
    counting adds no fill kernel. Created at finalize, outside any capture."""
    buf = _ONES.get(str(device))
    if buf is None:
        buf = _ONES[str(device)] = torch.ones(_ONES_N, dtype=torch.int64, device=device)
    if n > buf.numel():
        return torch.ones(n, dtype=torch.int64, device=device)
    return buf[:n]


# Stream and event helpers, one place for the device API (and for tests).
def _capturing() -> bool:
    return current_platform.is_xpu() and bool(torch.xpu.is_current_stream_capturing())


def _current_stream() -> Any:
    return torch.xpu.current_stream()


def _record_event(stream: Any = None) -> Any:
    event = torch.xpu.Event()
    event.record(stream)
    return event


def _pinned_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, pin_memory=True)


def _synchronize() -> None:
    torch.xpu.synchronize()


def _now() -> float:
    return time.monotonic()


def _is_writer() -> bool:
    """One writer per pipeline stage: TP rank 0 of DP rank 0. Under TP every
    rank routes the same tokens, so its counts are the stage's."""
    from vllm.distributed import parallel_state as ps

    if not ps.model_parallel_is_initialized():
        return True
    return (
        ps.get_tensor_model_parallel_rank() == 0
        and ps.get_dp_group().rank_in_group == 0
    )


def _write_state(path: str, layers: dict[int, list[int]]) -> None:
    """Replace this writer's layers in the state file, atomically. Pipeline
    stages own disjoint layers, so the read-merge-write is under a file lock."""
    with open(f"{path}.lock", "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # released when the file closes
        merged: dict[int, list[int]] = {}
        try:
            with open(path) as handle:
                merged = _parse_rankings(json.load(handle), path)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        merged.update(layers)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as handle:
            json.dump({str(k): v for k, v in sorted(merged.items())}, handle)
        os.replace(tmp, path)


class _Recorder:
    """Route counters -> a long-term score per expert -> the state file.

    Every ``_FOLD_S`` the counters of all counting layers are copied into a
    pinned snapshot on the forward stream; a later step folds the snapshot
    once its event is complete, so the forward never waits on the host. A fold
    turns each layer's new routes into a distribution (decode routes, plus
    prefill routes at ``_PREFILL_WEIGHT``) and adds it to a decayed score.
    """

    def __init__(self) -> None:
        self.placements: list[ExpertPlacement] = []
        self.ready = False
        self.event: Any = None
        self.snap: torch.Tensor | None = None
        self.last: np.ndarray | None = None  # [layers, 2, num_global]
        self.score: np.ndarray | None = None  # [layers, num_global]
        self.prior: np.ndarray | None = None  # boot order, 1 for the hottest
        self.folds = 0
        self.t_fold = 0.0
        self.t_save = 0.0
        self.path: str | None = None

    def _setup(self, now: float) -> bool:
        placements = [_PLACEMENTS[k] for k in sorted(_PLACEMENTS)]
        if not placements:
            return False
        num_global = placements[0].num_global
        self.placements = placements
        self.path = placements[0].state_path
        self.snap = _pinned_empty((len(placements), 2, num_global + 1), torch.int64)
        self.last = np.zeros((len(placements), 2, num_global), dtype=np.int64)
        self.score = np.zeros((len(placements), num_global))
        self.prior = np.zeros((len(placements), num_global))
        for k, p in enumerate(placements):
            order = [g for g in dict.fromkeys(p.boot_order) if 0 <= g < num_global]
            seen = set(order)
            order += [g for g in range(num_global) if g not in seen]
            self.prior[k, order] = 1 - np.arange(len(order)) / len(order)
        self.t_fold = self.t_save = now
        self.ready = True
        logger.info(
            "Per-expert offload placement: counting routes in %d layers; learned "
            "state %s",
            len(placements),
            self.path or "off",
        )
        return True

    def _snapshot(self) -> None:
        assert self.snap is not None
        for k, p in enumerate(self.placements):
            assert p.route_counts is not None
            self.snap[k].copy_(p.route_counts, non_blocking=True)

    def poll(self, now: float) -> np.ndarray | None:
        """Start a snapshot when one is due; the new routes per layer
        ``[layers, 2, num_global]`` when one is complete, else None."""
        if not self.ready and not self._setup(now):
            return None
        if self.event is None:
            if now - self.t_fold >= _FOLD_S:
                self.t_fold = now
                self._snapshot()
                self.event = _record_event(_current_stream())
            return None
        if not self.event.query():
            return None  # copy still in flight: the forward goes on
        self.event = None
        return self._delta()

    def _delta(self) -> np.ndarray:
        assert self.snap is not None and self.last is not None
        counts = self.snap.numpy()[:, :, :-1].copy()
        delta, self.last = counts - self.last, counts
        return delta

    def fold(self, delta: np.ndarray) -> None:
        assert self.score is not None
        active = False
        for k in range(delta.shape[0]):
            pre, dec = delta[k, 0].astype(np.float64), delta[k, 1].astype(np.float64)
            mass = np.zeros_like(dec)
            if dec.sum() > 0:
                mass += dec / dec.sum()
            if pre.sum() > 0:
                mass += _PREFILL_WEIGHT * pre / pre.sum()
            if mass.sum() <= 0:
                continue  # idle time doesn't decay the score
            active = True
            self.score[k] = _STATE_DECAY * self.score[k] + mass / mass.sum()
        if active:
            self.folds += 1

    def order(self, k: int) -> list[int]:
        """Layer ``k``'s experts, hottest first: the learned score, with the
        boot order as a small prior that also orders unrouted experts."""
        assert self.score is not None and self.prior is not None
        top = float(self.score[k].max()) or 1.0
        key = self.score[k] / top + _STATE_PRIOR * self.prior[k]
        return [int(g) for g in np.argsort(-key, kind="stable")]

    def maybe_save(self, now: float, force: bool = False) -> bool:
        if (
            not self.path
            or self.folds < _STATE_MIN_FOLDS
            or (not force and now - self.t_save < _STATE_SAVE_S)
            or not _is_writer()
        ):
            return False
        self.t_save = now
        layers = {p.layer_idx: self.order(k) for k, p in enumerate(self.placements)}
        try:
            _write_state(self.path, layers)
        except OSError as err:
            logger.warning(
                "Per-expert offload placement: cannot write %s: %s", self.path, err
            )
            return False
        logger.debug(
            "Per-expert offload placement: learned state %s, %d layers, %d folds",
            self.path,
            len(layers),
            self.folds,
        )
        return True

    def flush(self) -> None:
        """At shutdown: fold what the counters hold now and save."""
        if not self.ready:
            return
        _synchronize()
        self.event = None
        self._snapshot()
        _synchronize()
        self.fold(self._delta())
        self.maybe_save(_now(), force=True)


_RECORDER = _Recorder()


@contextmanager
def warmup() -> Iterator[None]:
    """Around the worker's compile and warm-up, whose dummy batches reach
    ``execute_model``: their steps neither start counting nor set up the
    recorder."""
    was_warmup = _WARMUP[0]
    _WARMUP[0] = True
    try:
        yield
    finally:
        _WARMUP[0] = was_warmup


def _start_counting() -> None:
    """At the first serving step: zero the counters, which captured graphs
    replayed in the warm-up may have added dummy routes to, and count from
    here. The zeroing is queued on the stream ahead of this step's forward."""
    for p in _PLACEMENTS.values():
        if p.route_counts is not None:
            p.route_counts.zero_()
    _LIVE[0] = True


@torch.inference_mode()
def step() -> None:
    """Once per runner step, on the host, before the forward is issued. A no-op
    unless a placed layer counts routes, and inside ``warmup()``.

    Under inference mode like the runner, whichever caller reaches it (the
    kernel warm-up calls the worker's ``execute_model`` inside the mode, serving
    steps come before the runner enters it): the buffers a step sets up are
    inference tensors, and later steps update them in place."""
    if not _PLACEMENTS or _WARMUP[0]:
        return
    if not _LIVE[0]:
        _start_counting()
    if _capturing():
        return
    now = _now()
    delta = _RECORDER.poll(now)
    if delta is not None:
        _RECORDER.fold(delta)
        _RECORDER.maybe_save(now)


@torch.inference_mode()
def shutdown() -> None:
    """Save the learned state and forget the placements of this process."""
    global _RECORDER
    try:
        _RECORDER.flush()
    finally:
        _PLACEMENTS.clear()
        _LIVE[0] = False
        _RECORDER = _Recorder()


def _repack_int4_pinned(qweight: torch.Tensor, device: torch.device) -> torch.Tensor:
    """GPTQ int32 ``[E, in/8, out]`` (CPU) -> signed-int4 bytes ``[E, out, in/2]``.

    The layout transform of the method's post-load step plus ``implement_zp``,
    run on ``device`` a few experts at a time into pinned host memory, so device
    temporaries stay small.
    """
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    e, k8, n = qweight.shape
    out = _ARENA.empty((e, n, k8 * 4), torch.uint8)
    step = 16
    for i in range(0, e, step):
        chunk = qweight[i : i + step].to(device)
        packed = chunk.transpose(1, 2).contiguous().view(torch.uint8)
        out[i : i + chunk.shape[0]].copy_(implement_zp(packed))
    return out


def _pin(t: torch.Tensor) -> torch.Tensor:
    pinned = _ARENA.empty(tuple(t.shape), t.dtype)
    pinned.copy_(t)
    return pinned


class ExpertPlacement:
    """Which local experts of one layer live where, plus the cold storage."""

    def __init__(
        self,
        layer_name: str,
        local_ids: list[int],
        global_ids: list[int],
        num_global: int,
        hot_local: list[int],
    ):
        self.layer_name = layer_name
        self.num_global = num_global
        hot_set = set(hot_local)
        self.hot_local = sorted(hot_set)
        self.cold_local = [i for i in local_ids if i not in hot_set]
        # local expert id -> (is cold, row in its set)
        self.slot: dict[int, tuple[bool, int]] = {}
        for s, i in enumerate(self.hot_local):
            self.slot[i] = (False, s)
        for s, i in enumerate(self.cold_local):
            self.slot[i] = (True, s)
        self.local_to_global = dict(zip(local_ids, global_ids))
        self.twins: dict[str, torch.nn.Parameter] = {}
        self._param_names: dict[int, str] = {}
        self.cold: dict[str, torch.Tensor] = {}  # pinned, kernel layout
        self.cold_views: dict[str, torch.Tensor] = {}
        self.hot_map: torch.Tensor | None = None
        self.cold_map: torch.Tensor | None = None
        self._impls: tuple[Any, Any] | None = None
        match = _LAYER_RE.search(layer_name)
        self.layer_idx = int(match.group(1)) if match else -1
        # Route counting (set by maybe_place): the boot order, hottest first,
        # and the state file it updates.
        self.boot_order: list[int] = []
        self.state_path: str | None = None
        self.count = False
        # [2, num_global + 1] on the device: row 0 large steps, row 1 small
        # ones; the last bucket takes negative (unrouted) ids.
        self.route_counts: torch.Tensor | None = None

    @property
    def num_hot(self) -> int:
        return len(self.hot_local)

    @property
    def num_cold(self) -> int:
        return len(self.cold_local)

    def attach_twins(self, layer: torch.nn.Module) -> None:
        """A pageable CPU twin ``[num_cold, ...]`` for every per-expert param."""
        for name, param in layer.named_parameters(recurse=False):
            if (
                name not in _EXPERT_PARAMS
                or param.ndim == 0
                or param.shape[0] != self.num_hot
            ):
                continue
            twin = torch.nn.Parameter(
                # explicit CPU: the model is built under a default-device context
                torch.zeros(
                    (self.num_cold, *param.shape[1:]), dtype=param.dtype, device="cpu"
                ),
                requires_grad=False,
            )
            for key, value in vars(param).items():
                if not key.startswith("_"):
                    setattr(twin, key, value)
            self.twins[name] = twin
            self._param_names[id(param)] = name

    def route(
        self, param: torch.nn.Parameter, local_id: int
    ) -> tuple[torch.nn.Parameter, int]:
        """(param, row) the loader should write expert ``local_id`` into."""
        is_cold, row = self.slot[local_id]
        if not is_cold:
            return param, row
        name = self._param_names.get(id(param))
        if name is None:
            raise RuntimeError(
                f"{self.layer_name}: no cold twin for a param of shape "
                f"{tuple(param.shape)}"
            )
        return self.twins[name], row

    def finalize(self, layer: torch.nn.Module, moe_kernel: Any) -> None:
        """Repack the cold twins into pinned kernel-layout tensors; wire the kernel."""
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        for bias in ("w13_bias", "w2_bias"):
            if getattr(layer, bias, None) is not None:
                raise NotImplementedError(
                    f"{self.layer_name}: per-expert placement does not support {bias}"
                )
        experts = getattr(moe_kernel, "fused_experts", None)
        if experts is None or not hasattr(experts, "placement"):
            raise NotImplementedError(
                f"per-expert placement needs XPUExperts, got {type(experts).__name__}"
            )
        twins, self.twins, self._param_names = self.twins, {}, {}
        device = layer.w13_weight.device
        for key in ("w13", "w2"):
            self.cold[key] = _repack_int4_pinned(
                twins.pop(f"{key}_qweight").data, device
            )
            scales = twins.pop(f"{key}_scales").data.transpose(1, 2).contiguous()
            self.cold[f"{key}_scales"] = _pin(
                scales.to(getattr(layer, f"{key}_scales").dtype)
            )
        del twins  # qzeros: symmetric int4 has none on XPU
        for key, tensor in self.cold.items():
            self.cold_views[key] = get_accelerator_view_from_cpu_tensor(tensor)

        hot_map = torch.full((self.num_global,), -1, dtype=torch.int32)
        cold_map = torch.full((self.num_global,), -1, dtype=torch.int32)
        for local_id, (is_cold, row) in self.slot.items():
            (cold_map if is_cold else hot_map)[self.local_to_global[local_id]] = row
        self.hot_map = hot_map.to(device)
        self.cold_map = cold_map.to(device)
        experts.placement = self
        if self.count:
            self.route_counts = torch.zeros(
                2, self.num_global + 1, dtype=torch.int64, device=device
            )
            _ones(device)
            _PLACEMENTS[self.layer_idx] = self

    def count_routes(self, topk_ids: torch.Tensor) -> None:
        """Add this step's routes to the counters: two device kernels, no
        host sync, recorded into captured graphs like any other kernel."""
        counts = self.route_counts
        if counts is None:
            return
        if not _LIVE[0] and not _capturing():
            return  # the memory profile and warm-up are not traffic
        row = counts[1 if topk_ids.shape[0] < _LARGE_STEP_TOKENS else 0]
        ids = topk_ids.reshape(-1).remainder(row.shape[0])  # -1 -> last bucket
        row.index_add_(0, ids, _ones(row.device, ids.numel()))

    def impls(
        self,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        topk: int,
        activation: str,
        gemm1_clamp_limit: float | None,
    ) -> tuple[Any, Any]:
        """(hot, cold) ``XpuFusedMoe``, built on first use like the stock path."""
        if self._impls is None:
            from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

            # ep_size > 1 only makes XpuFusedMoe index through the expert_map
            # given here.
            common: dict[str, Any] = {
                "n_experts_per_token": topk,
                "activation": activation,
                "ep_rank": 0,
                "ep_size": 2,
                "gemm1_clamp_limit": gemm1_clamp_limit,
            }
            hot = XpuFusedMoe(
                w13=w1,
                w13_scales=w1_scale,
                w13_bias=None,
                w2=w2,
                w2_scales=w2_scale,
                w2_bias=None,
                num_experts=self.num_hot,
                expert_map=self.hot_map,
                **common,
            )
            cw13, cw2 = self.cold_views["w13"], self.cold_views["w2"]
            cw13.xpu_fused_moe = True  # already signed int4, in kernel layout
            cold = XpuFusedMoe(
                w13=cw13,
                w13_scales=self.cold_views["w13_scales"],
                w13_bias=None,
                w2=cw2,
                w2_scales=self.cold_views["w2_scales"],
                w2_bias=None,
                num_experts=self.num_cold,
                expert_map=self.cold_map,
                **common,
            )
            # Sizes come from the uint8 view; the kernel reads packed int4 as int8.
            cold.w13, cold.w2 = cw13.view(torch.int8), cw2.view(torch.int8)
            self._impls = (hot, cold)
        return self._impls


def remap_unrouted(
    topk_ids: torch.Tensor, num_global: int, ep_rank: int
) -> torch.Tensor:
    """The stock remap of ``topk_ids == -1`` to an expert this rank doesn't own.

    That expert is -1 in both maps, so both kernel calls skip the row.
    """
    sentinel = num_global - 1 if ep_rank == 0 else 0
    return torch.where(topk_ids < 0, topk_ids.new_full((), sentinel), topk_ids)


def maybe_place(
    layer_name: str,
    moe_config: Any,
    expert_map_manager: Any,
    quant_method: Any,
    moe_quant_params: dict[str, Any],
) -> ExpertPlacement | None:
    """A placement for this layer, or None: the setting is off or doesn't apply."""
    config = _offload_config()
    if config is None:
        return None
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod

    if (
        not isinstance(quant_method, AutoGPTQMoEMethod)
        or quant_method.quant_config.weight_bits != 4
    ):
        logger.warning_once(
            "Per-expert offload placement supports int4 GPTQ experts only; %s "
            "experts stay in VRAM",
            type(quant_method).__name__,
        )
        return None
    match = _LAYER_RE.search(layer_name)
    if match is None:
        logger.info_once(
            "Per-expert offload placement: %s is not a decoder layer; its experts "
            "stay in VRAM",
            layer_name,
        )
        return None
    layer_idx = int(match.group(1))
    num_global = moe_config.num_experts
    local_of = {g: expert_map_manager.map_global_to_local(g) for g in range(num_global)}
    pairs = sorted((loc, g) for g, loc in local_of.items() if loc >= 0)
    local_ids = [loc for loc, _ in pairs]
    global_ids = [g for _, g in pairs]
    from vllm.config import get_current_vllm_config

    num_layers = get_current_vllm_config().model_config.hf_text_config.num_hidden_layers
    bytes_each = expert_bytes(
        quant_method,
        moe_quant_params["hidden_size"],
        moe_quant_params["intermediate_size_per_partition"],
        moe_quant_params["params_dtype"],
    )
    cold_count = budget_cold_count(
        config.uva.cpu_offload_gb, num_layers, bytes_each, len(local_ids)
    )
    if cold_count == 0:
        return None
    logger.info_once(
        "Per-expert offload placement: --cpu-offload-gb %s: %d of %d experts per "
        "layer per rank in host memory (%d B each, %d layers)",
        config.uva.cpu_offload_gb,
        cold_count,
        len(local_ids),
        bytes_each,
        num_layers,
    )
    ranked, source = boot_order(config.uva, layer_idx)
    logger.info_once(
        "Per-expert offload placement: the first placed layer's hot set comes "
        "from %s (learned state, then rankings, then expert id)",
        {
            "state": config.uva.cpu_offload_expert_state,
            "rankings": config.uva.cpu_offload_expert_rankings,
            "id": "expert ids",
        }[source],
    )
    hot = hot_experts(ranked, global_ids, len(local_ids) - cold_count)
    placement = ExpertPlacement(
        layer_name, local_ids, global_ids, num_global, [local_of[g] for g in hot]
    )
    placement.boot_order = ranked
    placement.state_path = config.uva.cpu_offload_expert_state
    placement.count = placement.state_path is not None
    return placement
