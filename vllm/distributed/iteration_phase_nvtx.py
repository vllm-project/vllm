# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread-local iteration phase + NVTX markers for comm and worker steps."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import vllm.envs as envs

if TYPE_CHECKING:
    import torch

    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.utils import IterationDetails


_local = threading.local()

ITERATION_PHASES = frozenset({"prefill", "decode", "mixed", "idle", "unknown"})


@dataclass(frozen=True)
class IterationNVTXContext:
    """Scheduler-derived labels for one engine step (worker iteration)."""

    iter_id: int
    phase: str
    rank: int
    pp: int
    tp: int
    ctx_tokens: int
    gen_tokens: int
    ctx_reqs: int
    gen_reqs: int

    @property
    def reqs(self) -> int:
        return self.ctx_reqs + self.gen_reqs


def get_iteration_nvtx_context() -> IterationNVTXContext | None:
    """Active iteration context (thread-local, then bound worker fallback)."""
    ctx = getattr(_local, "iteration_ctx", None)
    if ctx is not None:
        return ctx
    worker = getattr(_local, "bound_worker", None)
    if worker is not None:
        return getattr(worker, "_nvtx_iteration_ctx", None)
    return None


def get_iteration_phase() -> str | None:
    ctx = get_iteration_nvtx_context()
    return ctx.phase if ctx is not None else getattr(_local, "phase", None)


def set_iteration_phase(phase: str | None) -> None:
    _local.phase = phase


def _parallel_coords() -> tuple[int, int, int]:
    try:
        import torch.distributed as dist

        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        rank = dist.get_rank() if dist.is_initialized() else 0
        pp = get_pp_group().rank_in_group
        tp = get_tp_group().rank_in_group
        return rank, pp, tp
    except Exception:
        return 0, 0, 0


def build_iteration_nvtx_context(
    scheduler_output: SchedulerOutput,
    details: IterationDetails,
    *,
    phase: str | None = None,
) -> IterationNVTXContext:
    from vllm.v1.utils import iteration_phase_nvtx_label

    if phase is None:
        phase = iteration_phase_nvtx_label(details)
    rank, pp, tp = _parallel_coords()
    return IterationNVTXContext(
        iter_id=int(scheduler_output.engine_step_id),
        phase=phase,
        rank=rank,
        pp=pp,
        tp=tp,
        ctx_tokens=details.num_ctx_tokens,
        gen_tokens=details.num_generation_tokens,
        ctx_reqs=details.num_ctx_requests,
        gen_reqs=details.num_generation_requests,
    )


def format_iter_nvtx_label(ctx: IterationNVTXContext) -> str:
    """Outer worker-step NVTX range (visible in Nsight)."""
    return (
        f"iter|id={ctx.iter_id}|phase={ctx.phase}|rank={ctx.rank}|pp={ctx.pp}|"
        f"tp={ctx.tp}|ctx={ctx.ctx_tokens}|gen={ctx.gen_tokens}|reqs={ctx.reqs}"
    )[:240]


def format_comm_nvtx_label(
    op: str,
    *,
    phase: str | None = None,
    iter_id: int | None = None,
    rank: int | None = None,
    pp: int | None = None,
    tp: int | None = None,
    shape: str | None = None,
    nbytes: int | None = None,
    peer: int | None = None,
    key: str | None = None,
) -> str:
    """
    Inner comm NVTX label.

    New format:
      ``comm|iter=12346|op=pp_send|phase=decode|rank=0|pp=0|tp=0|peer=4|...``

    Legacy colon format is still parsed by plotting_tools for older traces.
    """
    ctx = get_iteration_nvtx_context()
    if ctx is not None:
        phase = phase or ctx.phase
        iter_id = ctx.iter_id if iter_id is None else iter_id
        rank = ctx.rank if rank is None else rank
        pp = ctx.pp if pp is None else pp
        tp = ctx.tp if tp is None else tp
    phase = phase or get_iteration_phase() or "unknown"
    iter_id = 0 if iter_id is None else iter_id
    rank = 0 if rank is None else rank
    pp = 0 if pp is None else pp
    tp = 0 if tp is None else tp
    parts = [
        f"comm|iter={iter_id}",
        f"op={op}",
        f"phase={phase}",
        f"rank={rank}",
        f"pp={pp}",
        f"tp={tp}",
    ]
    if peer is not None:
        parts.append(f"peer={peer}")
    if key:
        parts.append(f"key={key}")
    parts.append(f"shape={shape or '-'}")
    parts.append(f"bytes={nbytes if nbytes is not None else '-'}")
    return "|".join(parts)[:240]


def parse_iter_nvtx_label(name: str) -> dict[str, Any] | None:
    """Parse ``iter|id=…|phase=…|…`` outer range labels."""
    if not name.startswith("iter|"):
        return None
    fields: dict[str, str] = {}
    for part in name.split("|")[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        fields[key.strip()] = val.strip()
    phase = fields.get("phase", "").lower()
    if phase not in ITERATION_PHASES - {"unknown"}:
        return None
    try:
        iter_id = int(fields.get("id", "0"))
    except ValueError:
        return None
    return {
        "scope": "iteration",
        "iter_id": iter_id,
        "phase": phase,
        "rank": int(fields.get("rank", "0")),
        "pp": int(fields.get("pp", "0")),
        "tp": int(fields.get("tp", "0")),
        "ctx_tokens": int(fields.get("ctx", "0")),
        "gen_tokens": int(fields.get("gen", "0")),
        "reqs": int(fields.get("reqs", "0")),
        "nvtx_name": name,
    }


def parse_comm_nvtx_label_pipe(name: str) -> dict[str, Any] | None:
    """Parse ``comm|iter=…|op=…|phase=…|…`` labels."""
    if not name.startswith("comm|"):
        return None
    fields: dict[str, str] = {}
    for part in name.split("|")[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        fields[key.strip()] = val.strip()
    op = fields.get("op", "")
    phase = fields.get("phase", "unknown").lower()
    if not op or phase not in ITERATION_PHASES:
        return None
    peer: int | None = None
    if fields.get("peer", "").isdigit():
        peer = int(fields["peer"])
    nbytes: int | None = None
    if fields.get("bytes", "").isdigit():
        nbytes = int(fields["bytes"])
    try:
        iter_id = int(fields.get("iter", "0"))
    except ValueError:
        iter_id = 0
    return {
        "op": op,
        "phase": phase,
        "iter_id": iter_id,
        "tensor_key": fields.get("key"),
        "shape": fields.get("shape", "-"),
        "logical_tensor_bytes": nbytes,
        "peer": peer,
        "rank": int(fields.get("rank", "0")),
        "pp": int(fields.get("pp", "0")),
        "tp": int(fields.get("tp", "0")),
        "nvtx_name": name,
    }


def tensor_comm_metadata(
    tensor: torch.Tensor | None = None,
    *,
    shape: Sequence[int] | None = None,
    numel: int | None = None,
    element_size: int | None = None,
) -> tuple[str, int]:
    """Return (shape_label, nbytes) for NVTX comm markers."""
    if tensor is not None:
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        element_size = int(tensor.element_size())
    elif shape is not None:
        shape = tuple(shape)
        if numel is None:
            numel = 1
            for dim in shape:
                numel *= int(dim)
        if element_size is None:
            element_size = 1
    else:
        return "-", 0
    shape_label = "x".join(str(int(d)) for d in shape) if shape else "scalar"
    nbytes = int(numel) * int(element_size)
    return shape_label, nbytes


@contextlib.contextmanager
def _nvtx_range(label: str) -> Iterator[None]:
    if not envs.VLLM_ITERATION_NVTX or not label:
        yield
        return
    try:
        try:
            from nvtx import nvtx_range

            with nvtx_range(label):
                yield
        except ImportError:
            import torch.cuda.nvtx as cuda_nvtx

            cuda_nvtx.range_push(label)
            try:
                yield
            finally:
                cuda_nvtx.range_pop()
    except Exception:
        # Never let tracing kill the engine.
        yield


@contextlib.contextmanager
def iteration_nvtx_context(
    scheduler_output: SchedulerOutput,
    *,
    worker: Any | None = None,
) -> Iterator[IterationNVTXContext | None]:
    """
    Set scheduler-derived iteration context and push outer NVTX range.

    Covers PP recv, forward, TP collectives, and PP send when the caller
    wraps the full ``Worker.execute_model`` body. Also stores context on
    ``worker`` so Ray PP comm on other threads can inherit labels.
    """
    from vllm.v1.utils import compute_iteration_details

    if scheduler_output.total_num_scheduled_tokens <= 0:
        yield None
        return

    if not envs.VLLM_ITERATION_NVTX:
        yield None
        return

    details = compute_iteration_details(scheduler_output)
    ctx = build_iteration_nvtx_context(scheduler_output, details)
    prev_ctx = getattr(_local, "iteration_ctx", None)
    prev_phase = getattr(_local, "phase", None)
    prev_worker = getattr(_local, "bound_worker", None)
    prev_worker_ctx = (
        getattr(worker, "_nvtx_iteration_ctx", None) if worker is not None else None
    )
    _local.iteration_ctx = ctx
    _local.phase = ctx.phase
    if worker is not None:
        worker._nvtx_iteration_ctx = ctx
        _local.bound_worker = worker
    label = format_iter_nvtx_label(ctx)
    try:
        with _nvtx_range(label):
            yield ctx
    finally:
        _local.iteration_ctx = prev_ctx
        _local.phase = prev_phase
        _local.bound_worker = prev_worker
        if worker is not None:
            if prev_worker_ctx is None:
                delattr(worker, "_nvtx_iteration_ctx")
            else:
                worker._nvtx_iteration_ctx = prev_worker_ctx


@contextlib.contextmanager
def iteration_phase_scope(phase: str) -> Iterator[None]:
    """Legacy: phase-only scope. Prefer ``iteration_nvtx_context``."""
    if not envs.VLLM_ITERATION_NVTX:
        yield
        return
    prev = get_iteration_phase()
    set_iteration_phase(phase)
    try:
        with _nvtx_range(phase):
            yield
    finally:
        set_iteration_phase(prev)


@contextlib.contextmanager
def comm_nvtx_mark(
    op: str,
    *,
    tensor: torch.Tensor | None = None,
    shape: Sequence[int] | None = None,
    nbytes: int | None = None,
    peer: int | None = None,
    key: str | None = None,
    phase: str | None = None,
    iter_id: int | None = None,
) -> Iterator[None]:
    """NVTX range for a comm op; label includes iter_id and phase from context."""
    if not envs.VLLM_ITERATION_NVTX:
        yield
        return
    shape_label, tensor_bytes = tensor_comm_metadata(
        tensor, shape=shape, numel=None, element_size=None
    )
    if nbytes is None:
        nbytes = tensor_bytes
    label = format_comm_nvtx_label(
        op,
        shape=shape_label,
        nbytes=nbytes,
        peer=peer,
        key=key,
        phase=phase,
        iter_id=iter_id,
    )
    with _nvtx_range(label):
        yield
