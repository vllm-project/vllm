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

With ``--cpu-offload-expert-swap`` the hot set follows the traffic while serving:
the same counters drive an asynchronous exchange of hot and cold experts through
spare cold rows, with the device maps rewritten in place between steps (see
``_Swapper``).
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
# The swap (--cpu-offload-expert-swap): checked at every fold.
_SWAP_SPARE = 16  # spare cold rows per layer: the most exchanges per fold
_SWAP_DECAY = 0.85  # short-term score, per fold
_SWAP_MARGIN = 1.1  # a cold expert must beat the hot one it replaces by this
_SWAP_PRIOR = 0.01  # weight of the boot order in the swap score
_SWAP_STEP_BYTES = 256 << 20  # the most a runner step copies, per copy phase

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
            or uva.cpu_offload_expert_swap
        )
        and current_platform.is_xpu()
        and offload_config.offload_backend in ("auto", "uva")
        and offload_config.prefetch.offload_group_size == 0
        and uva.cpu_offload_gb > 0
    )
    if selected and not _uva_enabled():
        raise ValueError(
            "--cpu-offload-expert-rankings / --cpu-offload-expert-state / "
            "--cpu-offload-expert-swap read cold experts from pinned host "
            "memory through UVA, which is disabled "
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


def _side_stream(device: torch.device) -> Any:
    return torch.xpu.Stream(device)


def _stream_ctx(stream: Any) -> Any:
    return torch.xpu.stream(stream)


def _side_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """One copy on the swap's side stream (the caller holds its context)."""
    dst.copy_(src, non_blocking=True)


class _Swapper:
    """The hot set follows the traffic, without a host wait on the forward path.

    Each fold of the route counters also feeds a short-term score per expert
    (decay ``_SWAP_DECAY``). From it each layer pairs its coldest hot experts
    with its hottest cold ones, at most ``_SWAP_SPARE`` pairs, and exchanges a
    pair when the cold one scores ``_SWAP_MARGIN`` times the hot one. The pairs
    of all layers are cut into chunks of at most ``_SWAP_STEP_BYTES``, and one
    chunk runs as three phases, at most one per runner step:

    copy1: side stream, after the previous chunk's flip2: hot ``a`` in row ``i``
        -> spare cold row ``s``.
    flip1: forward stream, once copy1's event is complete: the maps send ``a`` to
        cold row ``s``. Hot row ``i`` is referenced by no map from here on,
        though forwards issued before the flip may still read it.
    copy2: side stream, after flip1: cold ``b`` in row ``j`` -> hot row ``i``.
    flip2: forward stream, once copy2 is complete: the maps send ``b`` to hot row
        ``i``; cold row ``j`` becomes spare after every forward issued so far.

    Invariant: a row is written only while no map references it and after
    every forward that referenced it; a map points at a row only after the
    write into it is complete. So every forward reads each routed expert from
    complete bytes, every expert stays routable, and each copy phase moves at
    most ``_SWAP_STEP_BYTES``. The maps and weights change in place, so captured
    graphs read them through the same pointers and nothing is recaptured; maps
    change only between steps (``step`` runs before a forward is issued and
    never while a graph is being captured).

    Staging (``--cpu-offload-stage-min-tokens``) copies whole cold blocks on the
    forward stream inside a step. It may copy a spare row while the side stream
    writes it, but no map of that step references the row, so the half-written
    bytes are never read. That is the ordering rule between the two: the maps
    move on the forward stream at step boundaries, the side stream writes only
    unreferenced rows.
    """

    def __init__(self) -> None:
        self.ready = False
        self.disabled = False
        self.placements: list[ExpertPlacement] = []
        self.ema: np.ndarray | None = None  # [layers, num_global]
        self.prior: np.ndarray | None = None
        self.queue: list[list[tuple[int, int, int]]] = []  # chunks of (k, a, b)
        # (layer k, a, b, hot row i, cold row j, spare row s) of the chunk
        self.pending: list[tuple[int, int, int, int, int, int]] = []
        self.phase = "idle"
        self.event: Any = None
        self.ev_spare: Any = None
        self.stream: Any = None
        self.total = 0
        self.step_bytes = 0  # issued by the last step, for the budget check

    def _setup(self) -> bool:
        placements = [_PLACEMENTS[k] for k in sorted(_PLACEMENTS)]
        if not placements or not all(p.swap for p in placements):
            self.disabled = True
            return False
        if any(p._impls is None for p in placements):
            return False  # the kernels are built by the first forward
        for p in placements:
            hot, _ = p._impls  # type: ignore[misc]
            p.hot_tensors = {
                "w13": hot.w13.view(torch.uint8),
                "w2": hot.w2.view(torch.uint8),
                "w13_scales": hot.gemm1_wei_scales,
                "w2_scales": hot.gemm2_wei_scales,
            }
            for key, cold in p.cold.items():
                h = p.hot_tensors[key]
                if h.shape[1:] != cold.shape[1:] or h.element_size() != (
                    cold.element_size()
                ):
                    logger.warning(
                        "Per-expert offload swap off: %s %s hot %s %s vs cold %s %s",
                        p.layer_name,
                        key,
                        tuple(h.shape),
                        h.dtype,
                        tuple(cold.shape),
                        cold.dtype,
                    )
                    self.disabled = True
                    return False
        self.placements = placements
        assert _RECORDER.prior is not None
        self.prior = _RECORDER.prior
        self.ema = np.zeros_like(self.prior)
        self.stream = _side_stream(placements[0].hot_tensors["w13"].device)
        self.ready = True
        return True

    def _row_bytes(self, p: "ExpertPlacement") -> int:
        return sum(t[0].numel() * t.element_size() for t in p.cold.values())

    def _fold(self, delta: np.ndarray) -> None:
        assert self.ema is not None
        for k in range(delta.shape[0]):
            pre, dec = delta[k, 0].astype(np.float64), delta[k, 1].astype(np.float64)
            mass = np.zeros_like(dec)
            if dec.sum() > 0:
                mass += dec / dec.sum()
            if pre.sum() > 0:
                mass += _PREFILL_WEIGHT * pre / pre.sum()
            if mass.sum() > 0:  # idle time doesn't decay the score
                self.ema[k] = _SWAP_DECAY * self.ema[k] + mass

    def _plan(self) -> list[list[tuple[int, int, int]]]:
        """The exchanges worth making, best first, in chunks within the budget."""
        assert self.ema is not None and self.prior is not None
        top = float(self.ema.max()) or 1.0
        pairs: list[tuple[float, int, int, int]] = []
        for k, p in enumerate(self.placements):
            assert p.hot_owner is not None and p.cold_owner is not None
            score = self.ema[k] / top + _SWAP_PRIOR * self.prior[k]
            cold_g = p.cold_owner[p.cold_owner >= 0]
            cold_best = cold_g[np.argsort(-score[cold_g], kind="stable")]
            hot_worst = p.hot_owner[np.argsort(score[p.hot_owner], kind="stable")]
            for a, b in list(zip(hot_worst, cold_best))[:_SWAP_SPARE]:
                if score[b] > _SWAP_MARGIN * score[a]:
                    pairs.append((float(score[b] - score[a]), k, int(a), int(b)))
        pairs.sort(key=lambda x: -x[0])
        chunks: list[list[tuple[int, int, int]]] = []
        size = 0
        for _, k, a, b in pairs:
            nbytes = self._row_bytes(self.placements[k])
            if not chunks or size + nbytes > _SWAP_STEP_BYTES:
                chunks.append([])
                size = 0
            chunks[-1].append((k, a, b))
            size += nbytes
        return chunks

    def _push_maps(self, p: "ExpertPlacement") -> None:
        """Host mirrors -> device maps, in place, on the forward stream.

        The pinned stage is rewritten only at the next flip, which comes after
        an event that follows this copy on the forward stream (copy2 waits for
        flip1, the next copy1 for flip2), so the copy has read it by then.
        """
        assert p.map_stage is not None and p.hot_map is not None
        assert p.hot_map_np is not None and p.cold_map_np is not None
        assert p.cold_map is not None
        p.map_stage[0].copy_(torch.from_numpy(p.hot_map_np))
        p.map_stage[1].copy_(torch.from_numpy(p.cold_map_np))
        p.hot_map.copy_(p.map_stage[0], non_blocking=True)
        p.cold_map.copy_(p.map_stage[1], non_blocking=True)

    def _copy1(self, chunk: list[tuple[int, int, int]]) -> bool:
        if self.ev_spare is not None:
            self.stream.wait_event(self.ev_spare)
        with _stream_ctx(self.stream):
            for k, a, b in chunk:
                p = self.placements[k]
                assert p.hot_map_np is not None and p.cold_map_np is not None
                i, j = int(p.hot_map_np[a]), int(p.cold_map_np[b])
                if i < 0 or j < 0 or not p.spare_free:
                    continue  # moved by an earlier chunk, or no spare row left
                s = p.spare_free.pop(0)
                for key, cold in p.cold.items():
                    h = p.hot_tensors[key]
                    _side_copy(cold[s].view(h.dtype), h[i])
                self.step_bytes += self._row_bytes(p)
                self.pending.append((k, a, b, i, j, s))
            self.event = _record_event(self.stream)
        return bool(self.pending)

    def _flip1_copy2(self, fwd: Any) -> None:
        fwd.wait_event(self.event)  # complete already: orders the flip after it
        touched = set()
        for k, a, _b, _i, _j, s in self.pending:
            p = self.placements[k]
            assert p.hot_map_np is not None and p.cold_map_np is not None
            p.hot_map_np[a], p.cold_map_np[a] = -1, s
            touched.add(k)
        for k in touched:
            self._push_maps(self.placements[k])
        self.stream.wait_event(_record_event(fwd))
        with _stream_ctx(self.stream):
            for k, _a, _b, i, j, _s in self.pending:
                p = self.placements[k]
                for key, cold in p.cold.items():
                    h = p.hot_tensors[key]
                    _side_copy(h[i], cold[j].view(h.dtype))
                self.step_bytes += self._row_bytes(p)
            self.event = _record_event(self.stream)

    def _flip2(self, fwd: Any) -> int:
        fwd.wait_event(self.event)
        touched = set()
        for k, a, b, i, j, s in self.pending:
            p = self.placements[k]
            assert p.hot_map_np is not None and p.cold_map_np is not None
            assert p.hot_owner is not None and p.cold_owner is not None
            p.hot_map_np[b], p.cold_map_np[b] = i, -1
            p.hot_owner[i], p.cold_owner[s], p.cold_owner[j] = b, a, -1
            p.spare_free.append(j)
            touched.add(k)
        for k in touched:
            self._push_maps(self.placements[k])
        self.ev_spare = _record_event(fwd)
        n = len(self.pending)
        self.pending = []
        return n

    def step(self, delta: np.ndarray | None) -> None:
        """One runner step, on the host, before the forward is issued."""
        self.step_bytes = 0
        if self.disabled or (not self.ready and not self._setup()):
            return
        if delta is not None:
            self._fold(delta)
        fwd = _current_stream()
        if self.phase == "idle":
            if not self.queue and delta is not None:
                self.queue = self._plan()
            while self.queue:
                if self._copy1(self.queue.pop(0)):
                    self.phase = "copy1"
                    break
        elif not self.event.query():
            return  # copies in flight: the forward goes on with the current maps
        elif self.phase == "copy1":
            self._flip1_copy2(fwd)
            self.phase = "copy2"
        else:
            n = self._flip2(fwd)
            self.phase = "idle"
            if not self.total:
                logger.info("Per-expert offload placement: first experts swapped")
            self.total += n
            logger.debug(
                "Per-expert offload swap: %d experts, %d since start", n, self.total
            )


_SWAPPER = _Swapper()


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
    if _RECORDER.ready:
        _SWAPPER.step(delta)


@torch.inference_mode()
def shutdown() -> None:
    """Save the learned state and forget the placements of this process."""
    global _RECORDER, _SWAPPER
    try:
        _RECORDER.flush()
    finally:
        _PLACEMENTS.clear()
        _LIVE[0] = False
        _RECORDER = _Recorder()
        _SWAPPER = _Swapper()


def _repack_int4_pinned(
    qweight: torch.Tensor, device: torch.device, extra: int = 0
) -> torch.Tensor:
    """GPTQ int32 ``[E, in/8, out]`` (CPU) -> signed-int4 bytes ``[E, out, in/2]``,
    plus ``extra`` zeroed rows.

    The layout transform of the method's post-load step plus ``implement_zp``,
    run on ``device`` a few experts at a time into pinned host memory, so device
    temporaries stay small.
    """
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    e, k8, n = qweight.shape
    out = _ARENA.empty((e + extra, n, k8 * 4), torch.uint8)
    step = 16
    for i in range(0, e, step):
        chunk = qweight[i : i + step].to(device)
        packed = chunk.transpose(1, 2).contiguous().view(torch.uint8)
        out[i : i + chunk.shape[0]].copy_(implement_zp(packed))
    if extra:
        out[e:].zero_()
    return out


def _pin(t: torch.Tensor, extra: int = 0) -> torch.Tensor:
    """A pinned copy of ``t`` with ``extra`` zeroed rows appended."""
    pinned = _ARENA.empty((t.shape[0] + extra, *t.shape[1:]), t.dtype)
    pinned[: t.shape[0]].copy_(t)
    if extra:
        pinned[t.shape[0] :].zero_()
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
        # The swap (set by maybe_place): spare cold rows, host mirrors of the
        # maps and who owns which row, written only by the swapper.
        self.swap = False
        self.spare = 0
        self.hot_map_np: np.ndarray | None = None
        self.cold_map_np: np.ndarray | None = None
        self.hot_owner: np.ndarray | None = None  # hot row -> global id
        self.cold_owner: np.ndarray | None = None  # cold row -> global id or -1
        self.spare_free: list[int] = []
        self.map_stage: torch.Tensor | None = None
        self.hot_tensors: dict[str, torch.Tensor] = {}

    @property
    def cold_rows(self) -> int:
        """Rows of the cold tensors: the cold experts plus the spare rows."""
        return self.num_cold + self.spare

    def set_maps(
        self, hot_map: torch.Tensor, cold_map: torch.Tensor, device: torch.device
    ) -> None:
        """Device maps from CPU ones, with the host mirrors the swap keeps."""
        self.hot_map = hot_map.to(device)
        self.cold_map = cold_map.to(device)
        if not self.swap:
            return
        self.hot_map_np = hot_map.numpy().copy()
        self.cold_map_np = cold_map.numpy().copy()
        self.hot_owner = np.full(self.num_hot, -1, dtype=np.int64)
        self.cold_owner = np.full(self.cold_rows, -1, dtype=np.int64)
        for g in np.nonzero(self.hot_map_np >= 0)[0]:
            self.hot_owner[self.hot_map_np[g]] = g
        for g in np.nonzero(self.cold_map_np >= 0)[0]:
            self.cold_owner[self.cold_map_np[g]] = g
        self.spare_free = [r for r in range(self.cold_rows) if self.cold_owner[r] < 0]
        self.map_stage = _pinned_empty((2, self.num_global), hot_map.dtype)

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
        self.spare = _SWAP_SPARE if self.swap else 0
        for key in ("w13", "w2"):
            self.cold[key] = _repack_int4_pinned(
                twins.pop(f"{key}_qweight").data, device, self.spare
            )
            scales = twins.pop(f"{key}_scales").data.transpose(1, 2).contiguous()
            self.cold[f"{key}_scales"] = _pin(
                scales.to(getattr(layer, f"{key}_scales").dtype), self.spare
            )
        del twins  # qzeros: symmetric int4 has none on XPU
        for key, tensor in self.cold.items():
            self.cold_views[key] = get_accelerator_view_from_cpu_tensor(tensor)

        hot_map = torch.full((self.num_global,), -1, dtype=torch.int32)
        cold_map = torch.full((self.num_global,), -1, dtype=torch.int32)
        for local_id, (is_cold, row) in self.slot.items():
            (cold_map if is_cold else hot_map)[self.local_to_global[local_id]] = row
        self.set_maps(hot_map, cold_map, device)
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
                num_experts=self.cold_rows,
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
    placement.swap = config.uva.cpu_offload_expert_swap
    placement.count = placement.state_path is not None or placement.swap
    if placement.swap:
        logger.info_once(
            "Per-expert offload placement: swap on, up to %d experts per layer "
            "every %.0f s through %d spare cold rows per layer (%.2f GiB pinned "
            "per rank on top of --cpu-offload-gb), at most %d MiB copied per step",
            _SWAP_SPARE,
            _FOLD_S,
            _SWAP_SPARE,
            _SWAP_SPARE * bytes_each * num_layers / 2**30,
            _SWAP_STEP_BYTES >> 20,
        )
    return placement
