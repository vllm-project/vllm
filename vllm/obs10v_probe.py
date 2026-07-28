# SPDX-License-Identifier: Apache-2.0
"""OBS-10V strict L2 lifecycle and request-state probe helpers."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import itertools
import json
import os
import threading
import time
from typing import Any, Iterator

from vllm.obs_probe import obs_emit, obs_request_id_hash


ABI_VERSION = "obs10w.strict_l2.v3"
STRICT_L2_ENABLED = os.environ.get("VLLM_OBS10V_STRICT_L2_ENABLE", "0") == "1"
STATE_SNAPSHOT_ENABLED = (
    os.environ.get("VLLM_OBS10V_STATE_SNAPSHOT_ENABLE", "0") == "1"
)
LIFECYCLE_DETAIL_ENABLED = (
    os.environ.get("VLLM_OBS10V_LIFECYCLE_DETAIL_ENABLE", "0") == "1"
)

_local = threading.local()
_lock = threading.Lock()
_transition_counter = itertools.count()
_owners_by_block: dict[int, set[int]] = {}
_refcount_by_block: dict[int, int] = {}
_last_freed_owner_by_block: dict[int, int] = {}
_allocation_generation_by_block: dict[int, int] = {}
_free_ledger_by_block: dict[int, dict[str, Any]] = {}
_preemption_ledger_by_request: dict[int, dict[str, Any]] = {}
_last_allocation_transition_by_request: dict[int, str] = {}


def _stable_u64(value: Any) -> int:
    digest = hashlib.blake2b(
        str(value).encode("utf-8", errors="replace"), digest_size=8
    )
    return int.from_bytes(digest.digest(), "big", signed=False)


def _trace_session_id_hash() -> int:
    explicit = os.environ.get("VLLM_OBS10V_TRACE_SESSION_ID_HASH")
    if explicit:
        try:
            return int(explicit)
        except ValueError:
            pass
    session = os.environ.get("VLLM_OBS10V_TRACE_SESSION_ID", "obs10v")
    return _stable_u64(session)


def _owner_set_hash(owners: set[int]) -> int:
    if not owners:
        return 0
    if len(owners) == 1:
        return next(iter(owners))
    return _stable_u64(",".join(str(value) for value in sorted(owners)))


def _json_list(values: list[int] | tuple[int, ...] | None) -> str:
    return json.dumps(list(values or []), separators=(",", ":"))


def _snapshot_hash(values: list[int] | tuple[int, ...] | None) -> str:
    return hashlib.blake2b(
        _json_list(values).encode("utf-8"), digest_size=8
    ).hexdigest()


def current_scheduler_context() -> dict[str, Any]:
    return dict(getattr(_local, "scheduler_context", {}) or {})


@contextmanager
def obs10v_scheduler_scope(
    scheduler_step_id: int,
    scheduler_output_id: int,
    batch_id: str,
) -> Iterator[None]:
    previous = getattr(_local, "scheduler_context", None)
    _local.scheduler_context = {
        "scheduler_step_id": scheduler_step_id,
        "scheduler_output_id": scheduler_output_id,
        "batch_id": batch_id,
    }
    try:
        yield
    finally:
        _local.scheduler_context = previous


def _strict_fields(
    request_id: str | None,
    worker_req_index: int | None = None,
    scheduler_step_id: int | None = None,
    scheduler_output_id: int | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    context = current_scheduler_context()
    if scheduler_step_id is None:
        scheduler_step_id = context.get("scheduler_step_id")
    if scheduler_output_id is None:
        scheduler_output_id = context.get("scheduler_output_id")
    if batch_id is None:
        batch_id = context.get("batch_id")
    in_scope = scheduler_step_id is not None and scheduler_output_id is not None
    return {
        "abi_version": ABI_VERSION,
        "trace_session_id_hash": _trace_session_id_hash(),
        "scheduler_step_id": scheduler_step_id,
        "scheduler_output_id": scheduler_output_id,
        "batch_id": batch_id,
        "request_id": request_id,
        "request_id_hash": obs_request_id_hash(request_id),
        "worker_req_index": worker_req_index,
        "step_scope": "in_scheduler_step" if in_scope else "outside_scheduler_step",
    }


def emit_transition(
    *,
    request_id: str | None,
    physical_block_id: int,
    logical_block_index: int | None,
    block_action: str,
    block_action_reason: str,
    event_stage: str,
    owner_before_hash: int = 0,
    owner_after_hash: int = 0,
    refcount_before: int | None = None,
    refcount_after: int | None = None,
    is_cached_before: bool = False,
    is_cached_after: bool = False,
    is_shared_before: bool = False,
    is_shared_after: bool = False,
    worker_req_index: int | None = None,
    scheduler_step_id: int | None = None,
    scheduler_output_id: int | None = None,
    batch_id: str | None = None,
    owner_count_before: int | None = None,
    owner_count_after: int | None = None,
    allocation_generation: int | None = None,
    previous_free_transition_id: str | None = None,
    previous_free_ts_ns: int | None = None,
    previous_free_scheduler_step_id: int | None = None,
    previous_free_scheduler_output_id: int | None = None,
    lifecycle_episode_id: str | None = None,
    preemption_event_id: str | None = None,
    rescheduled_allocation_event_id: str | None = None,
    recompute_episode_id: str | None = None,
) -> str | None:
    if not STRICT_L2_ENABLED:
        return None
    transition_id = (
        f"obs10v-{os.getpid()}-{threading.get_ident()}-"
        f"{next(_transition_counter)}"
    )
    fields = _strict_fields(
        request_id,
        worker_req_index,
        scheduler_step_id,
        scheduler_output_id,
        batch_id,
    )
    fields.update(
        {
            "physical_block_id": int(physical_block_id),
            "logical_block_index": logical_block_index,
            "transition_id": transition_id,
            "block_action": block_action,
            "block_action_reason": block_action_reason,
            "event_stage": event_stage,
            "state_before_flags": json.dumps(
                {
                    "cached": bool(is_cached_before),
                    "shared": bool(is_shared_before),
                    "allocated": bool(refcount_before),
                },
                separators=(",", ":"),
            ),
            "state_after_flags": json.dumps(
                {
                    "cached": bool(is_cached_after),
                    "shared": bool(is_shared_after),
                    "allocated": bool(refcount_after),
                },
                separators=(",", ":"),
            ),
            "owner_before_hash": owner_before_hash,
            "owner_after_hash": owner_after_hash,
            "owner_set_hash_before": owner_before_hash,
            "owner_set_hash_after": owner_after_hash,
            "owner_count_before": owner_count_before,
            "owner_count_after": owner_count_after,
            "refcount_before": refcount_before,
            "refcount_after": refcount_after,
            "is_cached_before": bool(is_cached_before),
            "is_cached_after": bool(is_cached_after),
            "is_shared_before": bool(is_shared_before),
            "is_shared_after": bool(is_shared_after),
            "source_event_id": transition_id,
            "allocation_generation": allocation_generation,
            "previous_free_transition_id": previous_free_transition_id,
            "previous_free_ts_ns": previous_free_ts_ns,
            "previous_free_scheduler_step_id": previous_free_scheduler_step_id,
            "previous_free_scheduler_output_id": previous_free_scheduler_output_id,
            "lifecycle_episode_id": lifecycle_episode_id,
            "preemption_event_id": preemption_event_id,
            "rescheduled_allocation_event_id": rescheduled_allocation_event_id,
            "recompute_episode_id": recompute_episode_id,
        }
    )
    obs_emit("kv", "kv_block_transition_v1", **fields)
    return transition_id


def record_allocation(
    request_id: str,
    physical_block_id: int,
    logical_block_index: int,
    refcount_after: int,
    is_cached_after: bool,
) -> str:
    if not STRICT_L2_ENABLED:
        return "alloc"
    request_hash = obs_request_id_hash(request_id)
    with _lock:
        owners_before = set(_owners_by_block.get(physical_block_id, set()))
        ref_before = _refcount_by_block.get(physical_block_id, 0)
        free_ledger = _free_ledger_by_block.pop(physical_block_id, None)
        freed_owner = (
            int(free_ledger["last_owner_request_id_hash"])
            if free_ledger
            else _last_freed_owner_by_block.get(physical_block_id)
        )
        action = (
            "reuse"
            if not owners_before
            and free_ledger is not None
            and freed_owner not in (None, 0, request_hash)
            else "alloc"
        )
        previous_generation = (
            int(free_ledger["allocation_generation"])
            if free_ledger
            else _allocation_generation_by_block.get(physical_block_id, -1)
        )
        generation = previous_generation + 1
        _allocation_generation_by_block[physical_block_id] = generation
        owners_after = set(owners_before)
        owners_after.add(request_hash)
        _owners_by_block[physical_block_id] = owners_after
        _refcount_by_block[physical_block_id] = refcount_after
    episode_id = (
        f"reuse-{physical_block_id}-g{generation}"
        if action == "reuse"
        else None
    )
    common = {
        "request_id": request_id,
        "physical_block_id": physical_block_id,
        "logical_block_index": logical_block_index,
        "block_action_reason": "free_then_reallocate" if action == "reuse" else "allocate_slots",
        "event_stage": "scheduler_after_allocate",
        "owner_before_hash": (
            int(freed_owner or 0)
            if action == "reuse"
            else _owner_set_hash(owners_before)
        ),
        "owner_after_hash": _owner_set_hash(owners_after),
        "owner_count_before": len(owners_before),
        "owner_count_after": len(owners_after),
        "refcount_before": ref_before,
        "refcount_after": refcount_after,
        "is_cached_before": False,
        "is_cached_after": is_cached_after,
        "is_shared_before": len(owners_before) > 1,
        "is_shared_after": len(owners_after) > 1,
        "allocation_generation": generation,
        "previous_free_transition_id": (
            free_ledger.get("free_transition_id") if free_ledger else None
        ),
        "previous_free_ts_ns": (
            free_ledger.get("free_ts_ns") if free_ledger else None
        ),
        "previous_free_scheduler_step_id": (
            free_ledger.get("free_scheduler_step_id") if free_ledger else None
        ),
        "previous_free_scheduler_output_id": (
            free_ledger.get("free_scheduler_output_id") if free_ledger else None
        ),
        "lifecycle_episode_id": episode_id,
    }
    allocation_transition_id = emit_transition(block_action=action, **common)
    if allocation_transition_id:
        with _lock:
            _last_allocation_transition_by_request[request_hash] = (
                allocation_transition_id
            )
    emit_transition(block_action="owner_change", **common)
    emit_transition(block_action="refcount_change", **common)
    return action


def record_preemption(request_id: str) -> str | None:
    if not STRICT_L2_ENABLED:
        return None
    request_hash = obs_request_id_hash(request_id)
    context = current_scheduler_context()
    event_id = (
        f"obs10w-preempt-{os.getpid()}-{threading.get_ident()}-"
        f"{next(_transition_counter)}"
    )
    episode_id = f"recompute-{request_hash}-{event_id}"
    with _lock:
        _preemption_ledger_by_request[request_hash] = {
            "preemption_event_id": event_id,
            "recompute_episode_id": episode_id,
            "scheduler_step_id": context.get("scheduler_step_id"),
            "scheduler_output_id": context.get("scheduler_output_id"),
        }
        _last_allocation_transition_by_request.pop(request_hash, None)
    obs_emit(
        "sched",
        "request_preempted",
        request_id=request_id,
        preemption_event_id=event_id,
        recompute_episode_id=episode_id,
        scheduler_step_id=context.get("scheduler_step_id"),
        scheduler_output_id=context.get("scheduler_output_id"),
        batch_id=context.get("batch_id"),
    )
    return event_id


def record_recompute(
    request_id: str,
    physical_block_id: int,
    logical_block_index: int,
    refcount: int,
    is_cached: bool,
) -> str | None:
    if not STRICT_L2_ENABLED:
        return None
    request_hash = obs_request_id_hash(request_id)
    with _lock:
        preemption = _preemption_ledger_by_request.pop(request_hash, None)
        allocation_id = _last_allocation_transition_by_request.get(request_hash)
        generation = _allocation_generation_by_block.get(physical_block_id)
    if not preemption or not allocation_id:
        return None
    episode_id = str(preemption["recompute_episode_id"])
    return emit_transition(
        request_id=request_id,
        physical_block_id=physical_block_id,
        logical_block_index=logical_block_index,
        block_action="recompute",
        block_action_reason="rescheduled_after_preemption",
        event_stage="preemption_end",
        owner_before_hash=request_hash,
        owner_after_hash=request_hash,
        owner_count_before=1,
        owner_count_after=1,
        refcount_before=refcount,
        refcount_after=refcount,
        is_cached_before=is_cached,
        is_cached_after=is_cached,
        is_shared_before=False,
        is_shared_after=False,
        allocation_generation=generation,
        lifecycle_episode_id=episode_id,
        preemption_event_id=str(preemption["preemption_event_id"]),
        rescheduled_allocation_event_id=allocation_id,
        recompute_episode_id=episode_id,
    )


def record_prefix_attach(
    request_id: str,
    physical_block_id: int,
    logical_block_index: int,
    refcount_after: int,
    is_cached: bool,
) -> None:
    if not STRICT_L2_ENABLED:
        return
    request_hash = obs_request_id_hash(request_id)
    with _lock:
        owners_before = set(_owners_by_block.get(physical_block_id, set()))
        ref_before = _refcount_by_block.get(
            physical_block_id, max(0, refcount_after - 1)
        )
        owners_after = set(owners_before)
        owners_after.add(request_hash)
        _owners_by_block[physical_block_id] = owners_after
        _refcount_by_block[physical_block_id] = refcount_after
    common = {
        "request_id": request_id,
        "physical_block_id": physical_block_id,
        "logical_block_index": logical_block_index,
        "block_action_reason": "prefix_cache_hit",
        "event_stage": "prefix_cache_lookup",
        "owner_before_hash": _owner_set_hash(owners_before),
        "owner_after_hash": _owner_set_hash(owners_after),
        "owner_count_before": len(owners_before),
        "owner_count_after": len(owners_after),
        "refcount_before": ref_before,
        "refcount_after": refcount_after,
        "is_cached_before": is_cached,
        "is_cached_after": is_cached,
        "is_shared_before": len(owners_before) > 1,
        "is_shared_after": len(owners_after) > 1,
    }
    emit_transition(block_action="prefix_cache_hit", **common)
    emit_transition(block_action="prefix_block_attach", **common)
    if (
        len(owners_before) == 1
        and len(owners_after) >= 2
    ) or (ref_before >= 1 and refcount_after >= 2):
        emit_transition(
            block_action="concurrent_shared_ownership_begin",
            **common,
        )
    emit_transition(block_action="refcount_change", **common)


def record_free(
    request_id: str,
    physical_block_id: int,
    logical_block_index: int,
    refcount_before: int,
    refcount_after: int,
    is_cached_before: bool,
    is_cached_after: bool,
) -> None:
    if not STRICT_L2_ENABLED:
        return
    request_hash = obs_request_id_hash(request_id)
    with _lock:
        owners_before = set(_owners_by_block.get(physical_block_id, {request_hash}))
        owners_after = set(owners_before)
        owners_after.discard(request_hash)
        _owners_by_block[physical_block_id] = owners_after
        _refcount_by_block[physical_block_id] = refcount_after
        generation = _allocation_generation_by_block.get(physical_block_id, 0)
    stage = (
        "scheduler_after_free"
        if current_scheduler_context()
        else "request_finish_cleanup"
    )
    common = {
        "request_id": request_id,
        "physical_block_id": physical_block_id,
        "logical_block_index": logical_block_index,
        "block_action_reason": "free_request",
        "event_stage": stage,
        "owner_before_hash": _owner_set_hash(owners_before),
        "owner_after_hash": _owner_set_hash(owners_after),
        "owner_count_before": len(owners_before),
        "owner_count_after": len(owners_after),
        "refcount_before": refcount_before,
        "refcount_after": refcount_after,
        "is_cached_before": is_cached_before,
        "is_cached_after": is_cached_after,
        "is_shared_before": len(owners_before) > 1,
        "is_shared_after": len(owners_after) > 1,
        "allocation_generation": generation,
    }
    free_transition_id = emit_transition(block_action="free", **common)
    if (
        len(owners_before) >= 2
        and len(owners_after) < 2
    ) or (refcount_before >= 2 and refcount_after < 2):
        emit_transition(
            block_action="concurrent_shared_ownership_end",
            **common,
        )
    emit_transition(block_action="owner_change", **common)
    emit_transition(block_action="refcount_change", **common)
    if refcount_after == 0 and free_transition_id:
        context = current_scheduler_context()
        free_ts_ns = time.time_ns()
        with _lock:
            _last_freed_owner_by_block[physical_block_id] = request_hash
            _free_ledger_by_block[physical_block_id] = {
                "physical_block_id": physical_block_id,
                "allocation_generation": generation,
                "last_owner_request_id_hash": request_hash,
                "free_transition_id": free_transition_id,
                "free_ts_ns": free_ts_ns,
                "free_scheduler_step_id": context.get("scheduler_step_id"),
                "free_scheduler_output_id": context.get("scheduler_output_id"),
            }


def emit_request_state_snapshot(
    *,
    request_id: str,
    snapshot_stage: str,
    block_table_physical_ids: list[int] | tuple[int, ...] | None,
    slot_mapping_physical_ids: list[int] | tuple[int, ...] | None,
    num_logical_blocks: int,
    token_start_index: int | None = None,
    token_end_index: int | None = None,
    worker_req_index: int | None = None,
    scheduler_step_id: int | None = None,
    scheduler_output_id: int | None = None,
    batch_id: str | None = None,
    free_pool_physical_ids: list[int] | tuple[int, ...] | None = None,
    ownership_physical_ids: list[int] | tuple[int, ...] | None = None,
) -> None:
    if not STATE_SNAPSHOT_ENABLED:
        return
    block_ids = list(block_table_physical_ids or [])
    slot_ids = list(slot_mapping_physical_ids or [])
    free_pool_ids = list(free_pool_physical_ids or [])
    ownership_ids = list(ownership_physical_ids or [])
    fields = _strict_fields(
        request_id,
        worker_req_index,
        scheduler_step_id,
        scheduler_output_id,
        batch_id,
    )
    fields.update(
        {
            "snapshot_stage": snapshot_stage,
            "block_table_hash": _snapshot_hash(block_ids),
            "block_table_physical_ids": _json_list(block_ids),
            "slot_mapping_hash": _snapshot_hash(slot_ids),
            "slot_mapping_physical_ids": _json_list(slot_ids),
            "free_pool_physical_ids": _json_list(free_pool_ids),
            "free_pool_hash": _snapshot_hash(free_pool_ids),
            "ownership_physical_ids": _json_list(ownership_ids),
            "ownership_hash": _snapshot_hash(ownership_ids),
            "num_logical_blocks": num_logical_blocks,
            "num_physical_blocks": len(
                set(block_ids) | set(slot_ids) | set(free_pool_ids) | set(ownership_ids)
            ),
            "token_start_index": token_start_index,
            "token_end_index": token_end_index,
            "detail_enabled": LIFECYCLE_DETAIL_ENABLED,
        }
    )
    obs_emit("kv", "kv_request_state_snapshot_v1", **fields)
