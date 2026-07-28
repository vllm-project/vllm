# SPDX-License-Identifier: Apache-2.0
"""OBS request-centric canonical probe adapter.

This module keeps the OBS-02/02B canonical event names close to runtime hooks.
It emits through the existing v2 semantic trace path, preserving source fields
instead of synthesizing collector-side identifiers.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from vllm.v2_trace import v2_instant

OBS_P0_CANONICAL_PROBES = (
    "request_arrival",
    "request_engine_admitted",
    "request_terminal",
    "first_token",
    "output_token",
    "scheduler_step_enter",
    "scheduler_step_end",
    "request_selected",
    "batch_member",
    "scheduler_output_snapshot",
    "scheduler_output_member",
    "scheduler_queue_snapshot",
    "scheduler_queue_member_sample",
    "request_preempted",
    "request_batch_index_mapping",
    "worker_request_index_mapping",
    "request_kv_state_snapshot",
    "block_table_entry",
    "slot_mapping_entry",
    "worker_slot_mapping_entry",
    "worker_slot_mapping_request_summary",
    "kv_block_detail",
    "kv_block_alloc_detail",
    "kv_block_free_detail",
    "kv_block_refcount_change",
    "block_owner_change",
    "prefix_cache_lookup",
    "prefix_cache_hit_detail",
    "prefix_cache_miss_detail",
)

OBS_P1_CANONICAL_PROBES = (
    "kv_block_reuse_detail",
    "kv_block_share_detail",
    "kv_block_evict_detail",
    "prefix_cache_insert_detail",
    "request_deferred",
    "kv_block_transition_v1",
    "kv_request_state_snapshot_v1",
)

_PROVIDER_NAME = "vllm_obs"
_USDT_ENABLED = os.environ.get("VLLM_OBS_ENABLE_USDT", "0") == "1"
_FORCE_USDT_FIRE = os.environ.get("VLLM_OBS_FORCE_USDT_FIRE", "0") == "1"
_provider: Any | None = None
_probes: dict[str, Any] = {}
_usdt_init_attempted = False

_EVENT_TYPE = {
    "request_arrival": 101,
    "request_engine_admitted": 102,
    "first_token": 103,
    "output_token": 104,
    "request_terminal": 106,
    "scheduler_step_enter": 201,
    "scheduler_step_end": 202,
    "request_selected": 203,
    "scheduler_output_snapshot": 204,
    "batch_member": 205,
    "request_batch_index_mapping": 206,
    "worker_request_index_mapping": 210,
    "scheduler_queue_snapshot": 207,
    "request_preempted": 208,
    "request_deferred": 209,
    "scheduler_output_member": 211,
    "scheduler_queue_member_sample": 212,
    "request_kv_state_snapshot": 301,
    "block_table_entry": 302,
    "slot_mapping_entry": 303,
    "worker_slot_mapping_entry": 309,
    "worker_slot_mapping_request_summary": 310,
    "kv_block_detail": 304,
    "prefix_cache_lookup": 305,
    "prefix_cache_hit_detail": 306,
    "prefix_cache_miss_detail": 307,
    "prefix_cache_insert_detail": 308,
    "kv_block_alloc_detail": 401,
    "kv_block_free_detail": 402,
    "kv_block_refcount_change": 403,
    "block_owner_change": 404,
    "kv_block_reuse_detail": 405,
    "kv_block_share_detail": 406,
    "kv_block_evict_detail": 407,
    "kv_block_transition_v1": 501,
    "kv_request_state_snapshot_v1": 502,
}


def obs_event_type(event_name: str) -> int:
    return _EVENT_TYPE.get(event_name, 0)


def obs_request_id_hash(request_id: Any | None) -> int:
    if request_id is None:
        return 0
    digest = hashlib.blake2b(str(request_id).encode("utf-8", errors="replace"),
                             digest_size=8)
    return int.from_bytes(digest.digest(), "big", signed=False)


def _stapsdt_char_ptr_arg(stapsdt: Any) -> Any:
    arg_types = stapsdt.ArgTypes
    return getattr(arg_types, "char_ptr", getattr(arg_types, "uint64"))


def _init_usdt_provider() -> None:
    global _provider, _usdt_init_attempted
    if _usdt_init_attempted or not _USDT_ENABLED:
        return
    _usdt_init_attempted = True
    try:
        import stapsdt

        provider = stapsdt.Provider(_PROVIDER_NAME)
        char_ptr_arg = _stapsdt_char_ptr_arg(stapsdt)
        for probe_name in OBS_P0_CANONICAL_PROBES + OBS_P1_CANONICAL_PROBES:
            _probes[probe_name] = provider.add_probe(
                probe_name,
                char_ptr_arg,
                char_ptr_arg,
                char_ptr_arg,
                char_ptr_arg,
                char_ptr_arg,
                char_ptr_arg,
            )
        provider.load()
        _provider = provider
    except Exception:
        _probes.clear()
        _provider = None


def _payload_layout_id(event_name: str) -> str:
    return f"obs03_{event_name}_v1"


def _to_usdt_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return "<unprintable>"


def _field_json(fields: dict[str, Any]) -> str:
    try:
        return json.dumps(fields, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        return "{}"


def _pack_pipe(*values: Any) -> str:
    return "|".join(_to_usdt_text(value).replace("|", "/") for value in values)


def _field(fields: dict[str, Any], key: str) -> Any:
    value = fields.get(key)
    return "" if value is None else value


def _usdt_args(event_name: str, layer: str, fields: dict[str, Any]) -> tuple[str, ...]:
    event_type = obs_event_type(event_name)
    layout = _payload_layout_id(event_name)
    request_id = fields.get("request_id") or ""
    case_name = fields.get("case_name") or ""

    if event_name == "request_terminal":
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(fields.get("terminal_type") or ""),
            _to_usdt_text(fields.get("terminal_source") or ""),
            _to_usdt_text(fields.get("terminal_confidence") or ""),
            _pack_pipe(
                fields.get("raw_finish_reason") or fields.get("finish_reason") or "",
                fields.get("raw_terminal_code") or "",
                case_name,
                request_id,
            ),
        )

    if event_name in {"first_token", "output_token", "request_arrival", "request_engine_admitted"}:
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(layer),
            _to_usdt_text(case_name),
            _to_usdt_text(request_id),
            _pack_pipe(
                fields.get("output_token_count") or fields.get("token_count") or "",
                fields.get("client_request_id") or "",
                fields.get("event_category") or "",
            ),
        )

    if event_name == "slot_mapping_entry":
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(fields.get("slot_mapping_source") or ""),
            _to_usdt_text(fields.get("scheduler_output_id") or ""),
            _to_usdt_text(request_id),
            _pack_pipe(
                fields.get("request_order_in_batch") or "",
                fields.get("token_or_slot_index") or "",
                fields.get("slot_index") or "",
                fields.get("logical_block_id") or "",
                fields.get("physical_block_id") or "",
                case_name,
            ),
        )

    if event_name == "worker_request_index_mapping":
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(_field(fields, "request_id_hash")),
            _to_usdt_text(_field(fields, "scheduler_output_id")),
            _to_usdt_text(_field(fields, "worker_req_index")),
            _pack_pipe(
                _field(fields, "request_order_in_batch"),
                _field(fields, "pre_reorder_index"),
                _field(fields, "post_reorder_index"),
                int(bool(fields.get("batch_reordered"))),
                _field(fields, "token_start_index"),
                _field(fields, "token_end_index"),
                case_name,
                request_id,
            ),
        )

    if event_name == "worker_slot_mapping_entry":
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(fields.get("request_id_hash_if_available")
                          or _field(fields, "request_id_hash")),
            _to_usdt_text(_field(fields, "scheduler_output_id")),
            _to_usdt_text(_field(fields, "worker_req_index")),
            _pack_pipe(
                _field(fields, "position"),
                _field(fields, "token_index"),
                _field(fields, "slot_index"),
                _field(fields, "logical_block_index"),
                _field(fields, "physical_block_id"),
                _field(fields, "kernel_block_id"),
                _field(fields, "kv_cache_group_id"),
                case_name,
                request_id,
                int(bool(fields.get("slot_mapping_truncated"))),
                _field(fields, "slot_mapping_sample_policy"),
                _field(fields, "sample_index_within_request"),
                _field(fields, "sample_limit_per_request"),
                _field(fields, "total_entries_for_request"),
                _field(fields, "token_start_index"),
                _field(fields, "token_end_index"),
                _field(fields, "num_tokens_for_request"),
                _field(fields, "sample_reason"),
            ),
        )

    if event_name == "worker_slot_mapping_request_summary":
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(fields.get("request_id_hash_if_available")
                          or _field(fields, "request_id_hash")),
            _to_usdt_text(_field(fields, "scheduler_output_id")),
            _to_usdt_text(_field(fields, "worker_req_index")),
            _pack_pipe(
                _field(fields, "scheduler_step_id"),
                _field(fields, "batch_id"),
                _field(fields, "worker_batch_id"),
                _field(fields, "request_order_in_batch"),
                _field(fields, "request_phase"),
                _field(fields, "token_start_index"),
                _field(fields, "token_end_index"),
                _field(fields, "num_tokens_for_request"),
                _field(fields, "total_slot_mapping_entries"),
                _field(fields, "sampled_slot_mapping_entries"),
                int(bool(fields.get("slot_mapping_truncated"))),
                _field(fields, "slot_mapping_sample_policy"),
                _field(fields, "slot_mapping_min_per_request_rows"),
                _field(fields, "slot_mapping_sample_limit_per_request"),
                _field(fields, "slot_mapping_total_detail_limit"),
                _field(fields, "unique_slot_indices_count"),
                _field(fields, "min_token_index"),
                _field(fields, "max_token_index"),
                _field(fields, "min_position"),
                _field(fields, "max_position"),
                _field(fields, "slot_mapping_hash"),
                _field(fields, "kv_cache_group_id"),
                case_name,
            ),
        )

    if event_name == "scheduler_output_snapshot":
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(_field(fields, "scheduler_step_id")),
            _to_usdt_text(_field(fields, "scheduler_output_id")),
            _to_usdt_text(_field(fields, "batch_id")),
            _pack_pipe(
                _field(fields, "request_count"),
                _field(fields, "batch_size"),
                _field(fields, "selected_count"),
                _field(fields, "num_prefill_tokens"),
                _field(fields, "num_decode_tokens"),
                _field(fields, "token_budget"),
                _field(fields, "phase"),
                _field(fields, "preempted_count"),
                _field(fields, "max_num_batched_tokens"),
                _field(fields, "max_num_seqs"),
                _field(fields, "token_budget_total"),
                _field(fields, "token_budget_used"),
                _field(fields, "token_budget_remaining"),
            ),
        )

    if event_name == "scheduler_queue_snapshot":
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(_field(fields, "scheduler_step_id")),
            _to_usdt_text(_field(fields, "scheduler_output_id")),
            _to_usdt_text(_field(fields, "batch_id")),
            _pack_pipe(
                _field(fields, "running_queue_size"),
                _field(fields, "waiting_queue_size"),
                _field(fields, "preempted_count"),
                _field(fields, "skipped_waiting_count"),
                _field(fields, "selected_count"),
                _field(fields, "max_num_batched_tokens"),
                _field(fields, "max_num_seqs"),
                _field(fields, "token_budget_total"),
                _field(fields, "token_budget_remaining"),
            ),
        )

    if event_name == "scheduler_output_member":
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(_field(fields, "request_id_hash")),
            _to_usdt_text(_field(fields, "scheduler_output_id")),
            _to_usdt_text(request_id),
            _pack_pipe(
                _field(fields, "scheduler_step_id"),
                _field(fields, "batch_id"),
                _field(fields, "request_order_in_batch"),
                _field(fields, "request_phase"),
                _field(fields, "scheduled_token_count_for_request"),
                _field(fields, "batch_size"),
                case_name,
            ),
        )

    if event_name == "scheduler_queue_member_sample":
        return (
            _to_usdt_text(event_type),
            layout,
            _to_usdt_text(_field(fields, "request_id_hash")),
            _to_usdt_text(_field(fields, "scheduler_output_id")),
            _to_usdt_text(request_id),
            _pack_pipe(
                _field(fields, "scheduler_step_id"),
                _field(fields, "queue_snapshot_phase"),
                _field(fields, "queue_name"),
                _field(fields, "queue_position"),
                _field(fields, "sample_index"),
                _field(fields, "sample_limit"),
                int(bool(fields.get("queue_snapshot_truncated"))),
                _field(fields, "request_phase"),
                case_name,
            ),
        )

    return (
        _to_usdt_text(event_type),
        layout,
        _to_usdt_text(layer),
        _to_usdt_text(fields.get("event_category") or ""),
        _to_usdt_text(request_id),
        _field_json(fields),
    )


def _fire_usdt(event_name: str, layer: str, fields: dict[str, Any]) -> None:
    if not _USDT_ENABLED:
        return
    _init_usdt_provider()
    probe = _probes.get(event_name)
    if probe is None:
        return
    args = _usdt_args(event_name, layer, fields)
    try:
        if _FORCE_USDT_FIRE:
            import ctypes
            import stapsdt

            raw_probe = getattr(probe, "_probe", None)
            if raw_probe:
                stapsdt.probeFire(
                    raw_probe,
                    *[
                        ctypes.c_char_p(arg.encode("utf-8", errors="replace"))
                        for arg in args
                    ],
                )
        else:
            probe.fire(*args)
    except Exception:
        pass


def obs_emit(layer: str, event_name: str, **fields: Any) -> None:
    if event_name not in OBS_P0_CANONICAL_PROBES and event_name not in OBS_P1_CANONICAL_PROBES:
        return
    request_id = fields.get("request_id")
    if request_id is not None and "request_id_hash" not in fields:
        fields["request_id_hash"] = obs_request_id_hash(request_id)
    case_name = os.environ.get("VLLM_OBS03_CASE_NAME")
    if case_name and "case_name" not in fields:
        fields["case_name"] = case_name
    fields.setdefault("event_name", event_name)
    fields.setdefault("schema_version", "obs03.runtime_probe.v1")
    fields.setdefault("event_type", obs_event_type(event_name))
    fields.setdefault("payload_layout_id", _payload_layout_id(event_name))
    fields.setdefault("collector_provider", _PROVIDER_NAME)
    _fire_usdt(event_name, layer, fields)
    v2_instant(layer, event_name, **fields)


_init_usdt_provider()
