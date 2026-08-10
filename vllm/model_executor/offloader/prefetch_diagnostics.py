# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional logging/instrumentation helpers for prefetch weight offloading.

These helpers are kept out of :mod:`prefetch` so the core control flow stays
close to the upstream prefetch implementation.  They are activated only when
the corresponding env vars are set:

* ``VLLM_PREFETCH_LOG_TRANSFER_STATS`` enables :class:`PrefetchTransferStats`
  bookkeeping during forward passes.
* ``VLLM_PREFETCH_LOG_SCHEDULE`` triggers the human schedule table and the
  post-init machine-readable manifest.
* ``VLLM_PREFETCH_LOG_OFFLOADED_PARAMS`` triggers
  :func:`log_prefetch_offload_plan`.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.prefetch_helpers import (
    maybe_retarget_offload_unit,
)
from vllm.model_executor.offloader.prefetch_tail_copy import (
    PREFETCH_H2D_CHUNK_BYTES,
    TAIL_PREFETCH_H2D_CHUNK_BYTES,
)
from vllm.model_executor.offloader.runtime import PrefetchRuntimeController
from vllm.model_executor.offloader.selectors import select_module_parameters

logger = init_logger(__name__)


def should_log_transfer_stats() -> bool:
    return envs.VLLM_PREFETCH_LOG_TRANSFER_STATS


def should_log_prefetch_schedule() -> bool:
    return envs.VLLM_PREFETCH_LOG_SCHEDULE


def should_collect_prefetch_debug_metadata() -> bool:
    return envs.VLLM_PREFETCH_LOG_SCHEDULE or envs.VLLM_NVTX_SCOPES_FOR_PROFILING


@dataclass
class PrefetchTransferStats:
    """Lightweight HtoD instrumentation for prefetch offloading."""

    h2d_bytes: int = 0
    copy_count: int = 0
    copy_time_s: float = 0.0
    wait_time_s: float = 0.0
    _pending_copy_events: list[tuple[Any, Any]] = field(default_factory=list)
    _pending_wait_events: list[tuple[Any, Any]] = field(default_factory=list)

    @property
    def effective_copy_bandwidth_bytes_per_s(self) -> float:
        if self.copy_time_s <= 0:
            return 0.0
        return self.h2d_bytes / self.copy_time_s

    def record_copy(
        self,
        num_bytes: int,
        start_event: Any | None = None,
        end_event: Any | None = None,
    ) -> None:
        self.h2d_bytes += num_bytes
        self.copy_count += 1
        if start_event is not None and end_event is not None:
            self._pending_copy_events.append((start_event, end_event))

    def record_wait(self, start_event: Any, end_event: Any) -> None:
        self._pending_wait_events.append((start_event, end_event))

    def flush_copy_timings(self, *, skip_query: bool = False) -> None:
        if skip_query:
            return
        remaining: list[tuple[Any, Any]] = []
        for start_event, end_event in self._pending_copy_events:
            if hasattr(end_event, "query") and not end_event.query():
                remaining.append((start_event, end_event))
                continue
            self.copy_time_s += start_event.elapsed_time(end_event) / 1000
        self._pending_copy_events = remaining

    def flush_wait_timings(self, *, skip_query: bool = False) -> None:
        if skip_query:
            return
        remaining: list[tuple[Any, Any]] = []
        for start_event, end_event in self._pending_wait_events:
            if hasattr(end_event, "query") and not end_event.query():
                remaining.append((start_event, end_event))
                continue
            self.wait_time_s += start_event.elapsed_time(end_event) / 1000
        self._pending_wait_events = remaining

    def reset(self) -> None:
        self.h2d_bytes = 0
        self.copy_count = 0
        self.copy_time_s = 0.0
        self.wait_time_s = 0.0
        self._pending_copy_events.clear()
        self._pending_wait_events.clear()

    def snapshot(self) -> dict[str, float | int]:
        return {
            "h2d_bytes": self.h2d_bytes,
            "copy_count": self.copy_count,
            "copy_time_s": self.copy_time_s,
            "wait_time_s": self.wait_time_s,
            "effective_copy_bandwidth_bytes_per_s": (
                self.effective_copy_bandwidth_bytes_per_s
            ),
        }

    def forward_snapshot(self) -> dict[str, float | int]:
        h2d_gb = self.h2d_bytes / 1e9
        bw_gb_s = 0.0 if self.copy_time_s <= 0 else h2d_gb / self.copy_time_s
        return {
            "h2d_gb": round(h2d_gb, 2),
            "h2d_copy_ops": self.copy_count,
            "gpu_copy_time_s": round(self.copy_time_s, 6),
            "gpu_wait_time_s": round(self.wait_time_s, 6),
            "gpu_copy_bandwidth_gb_s": round(bw_gb_s, 2),
        }


@dataclass(frozen=True)
class PrefetchScheduleRow:
    """One row in the static prefetch schedule table."""

    layer_idx: int
    unit_idx: int | None
    slot_idx: int | None
    initial: bool
    load_after_layer_idx: int | None
    lead_layers: int | None
    steady_state_load_after_layer_idx: int | None = None
    steady_state_lead_layers: int | None = None


@dataclass(frozen=True)
class PrefetchCopySegment:
    """One base CPU-to-GPU copy submitted for an offload unit."""

    kind: str
    num_bytes: int


def build_prefetch_copy_segments(
    module_offloader: Any,
) -> tuple[PrefetchCopySegment, ...]:
    """Describe the copies submitted by ``run_onload_to_static``."""
    segments: list[PrefetchCopySegment] = []
    if module_offloader.uses_slab_buffers and module_offloader._use_slab_copy:
        segments.extend(
            PrefetchCopySegment("slab_chunk", chunk.data.numel())
            for chunk in module_offloader._cpu_slab_chunks
        )
    elif module_offloader.uses_slab_buffers:
        for name in module_offloader._slab_param_names:
            storage = module_offloader._param_offloaders[name]._cpu_storage
            assert storage is not None
            segments.append(
                PrefetchCopySegment(
                    "slab_parameter", storage.numel() * storage.element_size()
                )
            )

    for group_info in module_offloader._storage_group_infos:
        source = group_info.cpu_source
        segments.append(
            PrefetchCopySegment("storage_group", source.numel() * source.element_size())
        )
    for name in module_offloader._direct_param_names:
        storage = module_offloader._param_offloaders[name]._cpu_storage
        assert storage is not None
        segments.append(
            PrefetchCopySegment(
                "direct_parameter", storage.numel() * storage.element_size()
            )
        )
    return tuple(segments)


def _layout_id(kind: str, signature: Any) -> str:
    digest = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    return f"{kind}-{digest}"


def _slab_layout_metadata(module_offloader: Any) -> tuple[str, int] | None:
    layout = module_offloader._slab_layout
    if layout is None:
        return None
    signature = tuple(
        (
            spec.name,
            spec.shape,
            spec.stride,
            str(spec.dtype),
            spec.offset_bytes,
            spec.storage_bytes,
        )
        for spec in layout.specs
    )
    return _layout_id("slab", signature), layout.total_bytes


def _storage_group_layout_metadata(
    module_offloader: Any,
) -> tuple[str, int] | None:
    group_infos = module_offloader._storage_group_infos
    if not group_infos:
        return None
    signature = tuple(group_info.key for group_info in group_infos)
    bytes_per_slot = sum(
        group_info.cpu_source.numel() * group_info.cpu_source.element_size()
        for group_info in group_infos
    )
    return _layout_id("storage_group", signature), bytes_per_slot


def _candidate_positions(
    modules: tuple[Any, ...],
    *,
    selectors: set[Any],
    include_names: set[str],
) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for module_index, module in enumerate(modules):
        param_names = select_module_parameters(
            module,
            selectors=selectors,
            include_names=include_names,
        )
        if param_names:
            target_module, target_names = maybe_retarget_offload_unit(
                module,
                param_names,
                selectors=selectors,
                include_names=include_names,
            )
            named_parameters = dict(target_module.named_parameters())
            selected = [
                named_parameters[name]
                for name in target_names
                if name in named_parameters
            ]
        else:
            target_names = ()
            selected = []
        positions.append(
            {
                "module_index": module_index,
                "offloadable": bool(selected),
                "selected_parameter_count": len(selected),
                "logical_parameter_bytes": sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in selected
                ),
            }
        )
    return positions


def build_prefetch_manifest(
    plan_units: tuple[Any, ...],
    runtime: PrefetchRuntimeController,
    module_offloaders: list[Any],
    all_modules: tuple[Any, ...],
    *,
    group_size: int,
    num_in_group: int,
    prefetch_step: int,
    selectors: set[Any],
    include_names: set[str],
    comm_aware: bool,
    total_offloaded_bytes: int,
    runtime_buffer_bytes: int,
) -> dict[str, Any]:
    """Build a compact, rank-local manifest for the offline optimizer."""
    initial_units = {unit.unit_idx for unit in runtime.initial_prefetches()}
    loaded_after: dict[int, int] = {}
    for source in runtime.units:
        target = runtime.prefetch_after(source.unit_idx)
        if target is not None:
            loaded_after[target.unit_idx] = source.unit_idx

    pooled_layouts: dict[str, dict[str, Any]] = {}
    units: list[dict[str, Any]] = []
    for runtime_unit, plan_unit, module_offloader in zip(
        runtime.units, plan_units, module_offloaders
    ):
        copy_segments = module_offloader._copy_segments
        pooled_layout_ids: list[str] = []
        for kind, metadata in (
            ("slab", _slab_layout_metadata(module_offloader)),
            ("storage_group", _storage_group_layout_metadata(module_offloader)),
        ):
            if metadata is None:
                continue
            layout_id, bytes_per_slot = metadata
            pooled_layout_ids.append(layout_id)
            layout = pooled_layouts.setdefault(
                layout_id,
                {
                    "layout_id": layout_id,
                    "kind": kind,
                    "bytes_per_slot": bytes_per_slot,
                    "unit_indices": [],
                },
            )
            layout["unit_indices"].append(runtime_unit.unit_idx)

        target = runtime.prefetch_after(runtime_unit.unit_idx)
        source_unit_idx = loaded_after.get(runtime_unit.unit_idx)
        source_module_index = None
        if source_unit_idx is not None:
            source_module_index = plan_units[source_unit_idx].module_index
        units.append(
            {
                "unit_idx": runtime_unit.unit_idx,
                "module_index": plan_unit.module_index,
                "slot_idx": runtime_unit.slot_idx,
                "initial": runtime_unit.unit_idx in initial_units,
                "prefetch_after_unit_idx": (
                    None if target is None else target.unit_idx
                ),
                "loaded_after_unit_idx": source_unit_idx,
                "loaded_after_module_index": source_module_index,
                "logical_parameter_bytes": module_offloader.offloaded_bytes,
                "h2d_bytes_per_prefetch": sum(
                    segment.num_bytes for segment in copy_segments
                ),
                "copy_segments": [
                    {"kind": segment.kind, "bytes": segment.num_bytes}
                    for segment in copy_segments
                ],
                "pooled_layout_ids": pooled_layout_ids,
                "direct_runtime_buffer_bytes": module_offloader.direct_buffer_bytes,
            }
        )

    rank = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    return {
        "schema_version": 1,
        "rank": rank,
        "module_count": len(all_modules),
        "config": {
            "group_size": group_size,
            "num_in_group": num_in_group,
            "prefetch_step": prefetch_step,
        },
        "tail_copy_chunk_bytes": TAIL_PREFETCH_H2D_CHUNK_BYTES,
        "comm_aware": comm_aware,
        "regular_copy_chunk_bytes": (PREFETCH_H2D_CHUNK_BYTES if comm_aware else None),
        "total_offloaded_bytes": total_offloaded_bytes,
        "runtime_buffer_bytes": runtime_buffer_bytes,
        "positions": _candidate_positions(
            all_modules,
            selectors=selectors,
            include_names=include_names,
        ),
        "pooled_buffer_layouts": sorted(
            pooled_layouts.values(), key=lambda item: item["layout_id"]
        ),
        "units": units,
    }


def log_prefetch_manifest(*args: Any, **kwargs: Any) -> None:
    """Log one machine-readable manifest when schedule diagnostics are enabled."""
    if not should_log_prefetch_schedule():
        return
    manifest = build_prefetch_manifest(*args, **kwargs)
    logger.info(
        "[PrefetchOffloader] manifest_json=%s",
        json.dumps(manifest, separators=(",", ":"), sort_keys=True),
    )


def build_prefetch_schedule_rows(
    plan_units: list[Any],
    runtime: PrefetchRuntimeController,
    module_count: int | None = None,
) -> list[PrefetchScheduleRow]:
    """Build static layer/unit/slot mapping with copy lead metadata."""
    initial_unit_idxs = {
        runtime_unit.unit_idx for runtime_unit in runtime.initial_prefetches()
    }
    load_source_by_target: dict[int, int] = {}
    for source_unit in runtime.units:
        target_unit = runtime.prefetch_after(source_unit.unit_idx)
        if target_unit is not None:
            load_source_by_target[target_unit.unit_idx] = source_unit.unit_idx

    layer_cycle_count = module_count
    if layer_cycle_count is None and plan_units:
        layer_cycle_count = max(unit.module_index for unit in plan_units) + 1

    offloaded_rows: list[PrefetchScheduleRow] = []
    for runtime_unit, plan_unit in zip(runtime.units, plan_units):
        unit_idx = runtime_unit.unit_idx
        initial = unit_idx in initial_unit_idxs
        load_after_layer_idx = None
        lead_layers = None
        steady_load_after_layer_idx = None
        steady_lead_layers = None

        steady_load_after_unit_idx = load_source_by_target.get(unit_idx)
        if steady_load_after_unit_idx is not None:
            steady_load_after_layer_idx = plan_units[
                steady_load_after_unit_idx
            ].module_index
            assert layer_cycle_count is not None
            steady_lead_layers = (
                plan_unit.module_index - steady_load_after_layer_idx
            ) % layer_cycle_count

        if not initial and steady_load_after_unit_idx is not None:
            load_after_layer_idx = steady_load_after_layer_idx
            lead_layers = steady_lead_layers

        offloaded_rows.append(
            PrefetchScheduleRow(
                layer_idx=plan_unit.module_index,
                unit_idx=unit_idx,
                slot_idx=runtime_unit.slot_idx,
                initial=initial,
                load_after_layer_idx=load_after_layer_idx,
                lead_layers=lead_layers,
                steady_state_load_after_layer_idx=steady_load_after_layer_idx,
                steady_state_lead_layers=steady_lead_layers,
            )
        )

    if module_count is None:
        return offloaded_rows

    rows_by_layer = {row.layer_idx: row for row in offloaded_rows}
    rows: list[PrefetchScheduleRow] = []
    for layer_idx in range(module_count):
        row = rows_by_layer.get(layer_idx)
        if row is not None:
            rows.append(row)
        else:
            rows.append(
                PrefetchScheduleRow(
                    layer_idx=layer_idx,
                    unit_idx=None,
                    slot_idx=None,
                    initial=False,
                    load_after_layer_idx=None,
                    lead_layers=None,
                    steady_state_load_after_layer_idx=None,
                    steady_state_lead_layers=None,
                )
            )
    return rows


def _format_prefetch_schedule_table(rows: list[PrefetchScheduleRow]) -> str:
    headers = (
        "layer_idx",
        "unit_idx",
        "slot_idx",
        "initial",
        "load_after_layer_idx",
        "lead_layers",
        "weights_loaded_when",
        "steady_state_loaded_when",
    )

    def value(row: PrefetchScheduleRow, name: str) -> str:
        raw = getattr(row, name)
        if raw is None:
            return "initial" if row.initial else "none"
        return str(raw)

    def load_desc(row: PrefetchScheduleRow) -> str:
        if row.unit_idx is None:
            return "not offloaded"
        if row.initial:
            return f"initial -> load layer {row.layer_idx} into slot {row.slot_idx}"
        if row.load_after_layer_idx is None:
            return f"none -> load layer {row.layer_idx} into slot {row.slot_idx}"
        return (
            f"after layer {row.load_after_layer_idx} -> "
            f"load layer {row.layer_idx} into slot {row.slot_idx}"
        )

    def steady_desc(row: PrefetchScheduleRow) -> str:
        if row.unit_idx is None:
            return "not offloaded"
        if row.steady_state_load_after_layer_idx is None:
            return "resident -> no steady-state copy"
        return (
            f"after layer {row.steady_state_load_after_layer_idx} -> "
            f"load layer {row.layer_idx} into slot {row.slot_idx}"
        )

    lines = [
        f"{headers[0]:<10} {headers[1]:<9} {headers[2]:<9} "
        f"{headers[3]:<7} {headers[4]:<21} {headers[5]:<11} "
        f"{headers[6]:<44} {headers[7]}"
    ]
    for row in rows:
        lines.append(
            f"{value(row, 'layer_idx'):<10} "
            f"{value(row, 'unit_idx'):<9} "
            f"{value(row, 'slot_idx'):<9} "
            f"{value(row, 'initial'):<7} "
            f"{value(row, 'load_after_layer_idx'):<21} "
            f"{value(row, 'lead_layers'):<11} "
            f"{load_desc(row):<44} "
            f"{steady_desc(row)}"
        )
    return "\n".join(lines)


def log_prefetch_schedule(
    plan_units: list[Any],
    runtime: PrefetchRuntimeController,
    module_count: int | None = None,
) -> None:
    """Log the static prefetch schedule table when enabled."""
    if not should_log_prefetch_schedule():
        return
    rows = build_prefetch_schedule_rows(plan_units, runtime, module_count=module_count)
    logger.info(
        "[PrefetchOffloader] prefetch schedule:\n%s",
        _format_prefetch_schedule_table(rows),
    )


def log_prefetch_offload_plan(units: list[Any]) -> None:
    """Log detailed parameter metadata for the selected offload units."""
    if not envs.VLLM_PREFETCH_LOG_OFFLOADED_PARAMS:
        return

    for runtime_idx, unit in enumerate(units):
        named_parameters = dict(unit.module.named_parameters())
        descriptions: list[str] = []
        total_bytes = 0
        for name in unit.param_names:
            param = named_parameters[name]
            num_bytes = param.numel() * param.element_size()
            total_bytes += num_bytes
            descriptions.append(
                f"{name}: shape={tuple(param.shape)}, dtype={param.dtype}, "
                f"bytes={num_bytes}"
            )
        logger.info(
            "[PrefetchOffloader] Offload unit %d "
            "(module_index=%d, module=%s) selects %d parameter(s), %.4f GB: %s",
            runtime_idx,
            unit.module_index,
            unit.module.__class__.__name__,
            len(unit.param_names),
            total_bytes / 1e9,
            "; ".join(descriptions),
        )
