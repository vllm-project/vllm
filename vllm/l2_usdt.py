# SPDX-License-Identifier: Apache-2.0
"""L2 KV-block-state native USDT provider wrapper, v0.4 core/detail ABI."""

from __future__ import annotations

import atexit
from collections import Counter
import ctypes
import hashlib
import json
import os
import threading
from typing import Any

PROVIDER = "vllm_l2"
BACKEND = "elf_shared_library"
ABI_SCHEMA_VERSION = "l2_p2_usdt_abi_v0.4_core_detail_bpftrace6"
ABI_SCHEMA_VERSION_NUMERIC = 4
DEFAULT_NATIVE_SO = "/home/guohao/L2_KV_Block_State/native_usdt/build/libvllm_l2_usdt.so"
NATIVE_SO_PATH = os.environ.get("VLLM_L2_USDT_SO", DEFAULT_NATIVE_SO)

REQUIRED_PROBES = (
    "kv_prefix_lookup_core",
    "kv_alloc_core",
    "kv_free_core",
    "block_pool_delta_core",
    "block_table_core",
    "slot_mapping_core",
    "kv_block_detail",
    "block_table_meta",
    "block_table_entry",
    "slot_mapping_meta",
    "slot_mapping_entry",
)

EVENT_TO_CORE_PROBE = {
    "l2.kv.prefix_lookup": "kv_prefix_lookup_core",
    "l2.kv.alloc_summary": "kv_alloc_core",
    "l2.kv.free_summary": "kv_free_core",
    "l2.block_pool.delta": "block_pool_delta_core",
    "l2.bridge.block_table_summary": "block_table_core",
    "l2.bridge.slot_mapping_summary": "slot_mapping_core",
}

PROBE_TO_FUNCTION = {
    "kv_prefix_lookup_core": "vllm_l2_emit_kv_prefix_lookup_core",
    "kv_alloc_core": "vllm_l2_emit_kv_alloc_core",
    "kv_free_core": "vllm_l2_emit_kv_free_core",
    "block_pool_delta_core": "vllm_l2_emit_block_pool_delta_core",
    "block_table_core": "vllm_l2_emit_block_table_core",
    "slot_mapping_core": "vllm_l2_emit_slot_mapping_core",
    "kv_block_detail": "vllm_l2_emit_kv_block_detail",
    "block_table_meta": "vllm_l2_emit_block_table_meta",
    "block_table_entry": "vllm_l2_emit_block_table_entry",
    "slot_mapping_meta": "vllm_l2_emit_slot_mapping_meta",
    "slot_mapping_entry": "vllm_l2_emit_slot_mapping_entry",
}

ABI_FIELD_ORDER: dict[str, tuple[str, ...]] = {
    "kv_prefix_lookup_core": (
        "schema_version", "trace_session_id_hash", "scheduler_step_id",
        "scheduler_output_id_or_zero", "kv_event_id", "prefix_summary_packed",
    ),
    "kv_alloc_core": (
        "schema_version", "trace_session_id_hash", "scheduler_step_id",
        "scheduler_output_id", "kv_event_id", "alloc_summary_packed",
    ),
    "kv_free_core": (
        "schema_version", "trace_session_id_hash", "scheduler_step_id",
        "scheduler_output_id_or_zero", "kv_event_id", "free_summary_packed",
    ),
    "block_pool_delta_core": (
        "schema_version", "trace_session_id_hash", "scheduler_step_id",
        "scheduler_output_id_or_zero", "kv_event_id", "block_pool_delta_packed",
    ),
    "block_table_core": (
        "schema_version", "trace_session_id_hash", "scheduler_step_id",
        "scheduler_output_id", "execution_step_id", "block_table_event_id",
    ),
    "slot_mapping_core": (
        "schema_version", "trace_session_id_hash", "scheduler_step_id",
        "scheduler_output_id", "execution_step_id", "slot_mapping_event_id",
    ),
    "kv_block_detail": (
        "schema_version", "trace_session_id_hash", "kv_event_id",
        "raw_request_id_hash", "detail_kind_index_packed", "physical_block_id_hash_or_sample",
    ),
    "block_table_meta": (
        "schema_version", "trace_session_id_hash", "block_table_event_id",
        "batch_id_hash", "block_table_hash", "block_table_meta_packed",
    ),
    "block_table_entry": (
        "schema_version", "trace_session_id_hash", "block_table_event_id",
        "request_id_hash", "logical_block_index", "physical_block_id_hash_or_sample",
    ),
    "slot_mapping_meta": (
        "schema_version", "trace_session_id_hash", "slot_mapping_event_id",
        "batch_id_hash", "slot_mapping_hash", "slot_mapping_meta_packed",
    ),
    "slot_mapping_entry": (
        "schema_version", "trace_session_id_hash", "slot_mapping_event_id",
        "request_id_hash", "token_or_slot_index", "slot_mapping_value_hash_or_sample",
    ),
}

FORBIDDEN_USDT_FIELDS = {
    "prompt_text", "output_text", "raw_token_text", "full_token_ids_list",
    "full_prompt_token_ids", "full_generated_token_ids", "sampling_output_text",
    "user_metadata", "raw_http_payload", "full_block_table_array",
    "full_slot_mapping_array", "full_block_id_list_default",
    "full_computed_block_id_list_default", "full_new_block_id_list_default",
    "full_freed_block_id_list_default", "large_physical_block_id_list_default",
}

LOOKUP_STATUS_ENUM = {"unknown": 0, "disabled_or_skipped": 1, "miss": 2, "hit": 3}
ALLOCATION_KIND_ENUM = {"unknown": 0, "fresh": 1, "prefix_hit": 2, "external_kv": 3}
FREE_REASON_ENUM = {"unknown": 0, "request_finish_or_preempt": 1}
BLOCK_POOL_ACTION_ENUM = {"unknown": 0, "allocate": 1, "free": 2}
DETAIL_KIND = {"computed": 1, "new": 2, "freed": 3}

_ENABLED = os.environ.get("VLLM_L2_ENABLE", "1").lower() not in {"0", "false", "no"}
_ATTEMPT_PATH = os.environ.get("VLLM_L2_USDT_ATTEMPT_PATH")
_TIMELINE_PATH = os.environ.get("VLLM_L2_USDT_TIMELINE_PATH")
_lib: ctypes.CDLL | None = None
_native_so_load_error: str | None = None
_native_functions: dict[str, Any] = {}
_attempt_count: Counter[str] = Counter({name: 0 for name in REQUIRED_PROBES})
_error_count: Counter[str] = Counter({name: 0 for name in REQUIRED_PROBES})
_abi_arg_count = {name: len(fields) for name, fields in ABI_FIELD_ORDER.items()}
_scheduler_output_missing = 0
_scheduler_output_is_surrogate = 0
_scheduler_output_equals_step = 0
_forbidden_field_count = 0
_entry_owner_unresolved_count = 0
_lock = threading.Lock()


def _u64(value: Any) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value & 0xFFFFFFFFFFFFFFFF
    if isinstance(value, (list, tuple, dict)):
        return _stable_u64_hash(value)
    text = str(value)
    try:
        return int(text, 16) & 0xFFFFFFFFFFFFFFFF
    except ValueError:
        digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8)
        return int.from_bytes(digest.digest(), "big", signed=False)


def _i64(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _stable_u64_hash(value: Any) -> int:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.blake2b(data.encode("utf-8", errors="replace"), digest_size=8)
    return int.from_bytes(digest.digest(), "big", signed=False)


def _pack_u16(*values: Any) -> int:
    packed = 0
    for index, value in enumerate(values[:4]):
        packed |= (_i64(value) & 0xFFFF) << (index * 16)
    return packed & 0xFFFFFFFFFFFFFFFF


def _enum(mapping: dict[str, int], value: Any) -> int:
    return mapping.get(str(value), 0)


def _list_from_debug(event: dict[str, Any], key: str) -> list[int]:
    debug = event.get("debug_payload")
    if not isinstance(debug, dict):
        return []
    values = debug.get(key) or []
    out: list[int] = []
    for item in values:
        try:
            out.append(int(item))
        except Exception:
            pass
    return out


def _request_owner_ranges(event: dict[str, Any]) -> list[dict[str, int]]:
    debug = event.get("debug_payload")
    if not isinstance(debug, dict):
        return []
    ranges = debug.get("request_owner_ranges") or []
    out: list[dict[str, int]] = []
    for item in ranges:
        if not isinstance(item, dict):
            continue
        request_id_hash = _u64(
            item.get("canonical_request_id_hash")
            or item.get("raw_request_id_hash")
            or item.get("request_id_hash")
        )
        if not request_id_hash:
            continue
        out.append(
            {
                "row_index": _i64(item.get("row_index")),
                "request_id_hash": request_id_hash,
                "token_range_start": _i64(item.get("token_range_start")),
                "token_range_end": _i64(item.get("token_range_end")),
                "num_tokens_for_request_in_step": _i64(
                    item.get("num_tokens_for_request_in_step")
                ),
            }
        )
    return out


def _slot_owner_hash(token_index: int, ranges: list[dict[str, int]]) -> int:
    for item in ranges:
        if item["token_range_start"] <= token_index < item["token_range_end"]:
            return item["request_id_hash"]
    return 0


def _block_table_event_id(event: dict[str, Any]) -> int:
    return _u64(event.get("block_table_event_id") or event.get("kv_event_id") or _stable_u64_hash({"block_table": event}))


def _slot_mapping_event_id(event: dict[str, Any]) -> int:
    return _u64(event.get("slot_mapping_event_id") or event.get("kv_event_id") or _stable_u64_hash({"slot_mapping": event}))


def _base(event: dict[str, Any]) -> dict[str, int]:
    scheduler_step_value = (
        event.get("scheduler_step_index")
        if event.get("scheduler_step_index") is not None
        else event.get("scheduler_step_id")
    )
    return {
        "schema_version": ABI_SCHEMA_VERSION_NUMERIC,
        "trace_session_id_hash": _u64(event.get("trace_session_id_hash")),
        "scheduler_step_id": _i64(scheduler_step_value),
        "scheduler_output_id": _u64(event.get("scheduler_output_id")),
        "scheduler_output_id_or_zero": _u64(event.get("scheduler_output_id")),
        "kv_event_id": _u64(event.get("kv_event_id")),
        "raw_request_id_hash": _u64(event.get("raw_request_id_hash")),
        "batch_id_hash": _u64(event.get("batch_id")),
        "execution_step_id": _i64(event.get("execution_step_id")),
        "block_table_event_id": _block_table_event_id(event),
        "slot_mapping_event_id": _slot_mapping_event_id(event),
        "request_id_hash": _u64(event.get("raw_request_id_hash") or event.get("request_id_hash")),
    }


def _payloads_for_event(event: dict[str, Any]) -> list[tuple[str, dict[str, int]]]:
    global _entry_owner_unresolved_count
    etype = str(event.get("event_type"))
    common = _base(event)
    rows: list[tuple[str, dict[str, int]]] = []
    if etype == "l2.kv.prefix_lookup":
        rows.append(("kv_prefix_lookup_core", {**common, "prefix_summary_packed": _pack_u16(
            event.get("computed_block_count"), event.get("prefix_hit_tokens"),
            _enum(LOOKUP_STATUS_ENUM, event.get("lookup_status")), 0)}))
        for index, block_id in enumerate(_list_from_debug(event, "computed_block_id_sample")):
            rows.append(("kv_block_detail", {**common, "detail_kind_index_packed": _pack_u16(DETAIL_KIND["computed"], index), "physical_block_id_hash_or_sample": _u64(block_id)}))
    elif etype == "l2.kv.alloc_summary":
        rows.append(("kv_alloc_core", {**common, "alloc_summary_packed": _pack_u16(
            event.get("num_new_blocks"), event.get("num_computed_blocks"),
            event.get("num_scheduled_tokens"), _enum(ALLOCATION_KIND_ENUM, event.get("allocation_kind")))}))
        for kind, key in (("new", "new_block_id_sample"), ("computed", "computed_block_id_sample")):
            for index, block_id in enumerate(_list_from_debug(event, key)):
                rows.append(("kv_block_detail", {**common, "detail_kind_index_packed": _pack_u16(DETAIL_KIND[kind], index), "physical_block_id_hash_or_sample": _u64(block_id)}))
    elif etype == "l2.kv.free_summary":
        rows.append(("kv_free_core", {**common, "free_summary_packed": _pack_u16(
            event.get("num_freed_blocks"), _enum(FREE_REASON_ENUM, event.get("free_reason")),
            event.get("num_total_blocks_before"), event.get("num_total_blocks_after"))}))
        for index, block_id in enumerate(_list_from_debug(event, "freed_block_id_sample")):
            rows.append(("kv_block_detail", {**common, "detail_kind_index_packed": _pack_u16(DETAIL_KIND["freed"], index), "physical_block_id_hash_or_sample": _u64(block_id)}))
    elif etype == "l2.block_pool.delta":
        rows.append(("block_pool_delta_core", {**common, "block_pool_delta_packed": _pack_u16(
            _enum(BLOCK_POOL_ACTION_ENUM, event.get("block_pool_action")),
            event.get("num_blocks_delta"), event.get("free_pool_size_before"),
            event.get("free_pool_size_after"))}))
    elif etype == "l2.bridge.block_table_summary":
        table_hash = _u64(event.get("block_table_hash"))
        event_id = common["block_table_event_id"]
        owner_ranges = _request_owner_ranges(event)
        rows.append(("block_table_core", {**common, "block_table_event_id": event_id}))
        rows.append(("block_table_meta", {**common, "block_table_event_id": event_id, "block_table_hash": table_hash, "block_table_meta_packed": _pack_u16(event.get("block_table_len"), event.get("request_count"), event.get("total_block_entries"), 0)}))
        for owner in owner_ranges:
            rows.append(("block_table_entry", {**common, "block_table_event_id": event_id, "request_id_hash": owner["request_id_hash"], "logical_block_index": owner["row_index"], "physical_block_id_hash_or_sample": table_hash or _stable_u64_hash({"block_table": event, "owner": owner})}))
        if not owner_ranges:
            with _lock:
                _entry_owner_unresolved_count += 1
            rows.append(("block_table_entry", {**common, "block_table_event_id": event_id, "logical_block_index": 0, "physical_block_id_hash_or_sample": table_hash or _stable_u64_hash(event)}))
    elif etype == "l2.bridge.slot_mapping_summary":
        slot_hash = _u64(event.get("slot_mapping_hash"))
        event_id = common["slot_mapping_event_id"]
        owner_ranges = _request_owner_ranges(event)
        rows.append(("slot_mapping_core", {**common, "slot_mapping_event_id": event_id}))
        rows.append(("slot_mapping_meta", {**common, "slot_mapping_event_id": event_id, "slot_mapping_hash": slot_hash, "slot_mapping_meta_packed": _pack_u16(event.get("slot_mapping_len"), event.get("total_num_scheduled_tokens"), 0, 0)}))
        emitted_slot_owner = False
        for owner in owner_ranges:
            if owner["token_range_end"] <= owner["token_range_start"]:
                continue
            token_index = owner["token_range_start"]
            request_id_hash = _slot_owner_hash(token_index, owner_ranges)
            rows.append(("slot_mapping_entry", {**common, "slot_mapping_event_id": event_id, "request_id_hash": request_id_hash, "token_or_slot_index": token_index, "slot_mapping_value_hash_or_sample": slot_hash or _stable_u64_hash({"slot_mapping": event, "owner": owner})}))
            emitted_slot_owner = True
        if not emitted_slot_owner:
            with _lock:
                _entry_owner_unresolved_count += 1
            rows.append(("slot_mapping_entry", {**common, "slot_mapping_event_id": event_id, "token_or_slot_index": 0, "slot_mapping_value_hash_or_sample": slot_hash or _stable_u64_hash(event)}))
    return rows


def _init_native_provider() -> None:
    global _lib, _native_so_load_error
    if not _ENABLED:
        return
    try:
        lib = ctypes.CDLL(NATIVE_SO_PATH)
        argtypes = [ctypes.c_uint64] * 6
        for probe_name, function_name in PROBE_TO_FUNCTION.items():
            func = getattr(lib, function_name)
            func.argtypes = argtypes
            func.restype = None
            _native_functions[probe_name] = func
        _lib = lib
        _native_so_load_error = None
    except Exception as exc:
        _lib = None
        _native_functions.clear()
        _native_so_load_error = f"{exc.__class__.__name__}: {exc}"


def _write_timeline(probe_name: str, event: dict[str, Any], payload: dict[str, int]) -> None:
    if not _TIMELINE_PATH:
        return
    os.makedirs(os.path.dirname(_TIMELINE_PATH), exist_ok=True)
    row = {
        "backend": BACKEND,
        "probe_name": probe_name,
        "event_type": event.get("event_type"),
        "kv_event_id": payload.get("kv_event_id"),
        "scheduler_step_id": payload.get("scheduler_step_id"),
        "scheduler_output_id": payload.get("scheduler_output_id", payload.get("scheduler_output_id_or_zero")),
        "scheduler_output_id_source": event.get("scheduler_output_id_source"),
        "context_generation_point": event.get("context_generation_point"),
        "context_resolution": event.get("context_resolution"),
        "join_required": event.get("join_required"),
        "l2_local_derived_scheduler_output_id_used": event.get(
            "l2_local_derived_scheduler_output_id_used", False
        ),
        "abi_arg_count": len(ABI_FIELD_ORDER[probe_name]),
    }
    with open(_TIMELINE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def emit_probe(probe_name: str, payload: dict[str, int]) -> None:
    func = _native_functions.get(probe_name)
    if func is None:
        with _lock:
            _error_count[probe_name] += 1
        raise RuntimeError(f"native USDT emit function unavailable for {probe_name}: {_native_so_load_error}")
    args = [payload[field] & 0xFFFFFFFFFFFFFFFF for field in ABI_FIELD_ORDER[probe_name]]
    func(*args)


def fire_l2_probe(event: dict[str, Any]) -> None:
    global _scheduler_output_missing, _scheduler_output_is_surrogate
    global _scheduler_output_equals_step, _forbidden_field_count
    if not _ENABLED:
        return
    payloads = _payloads_for_event(event)
    if not payloads:
        return
    forbidden = FORBIDDEN_USDT_FIELDS.intersection(event.keys())
    first_error: BaseException | None = None
    for probe_name, payload in payloads:
        with _lock:
            _attempt_count[probe_name] += 1
            _forbidden_field_count += len(forbidden)
            if probe_name in {"kv_alloc_core", "block_table_core", "slot_mapping_core"} and not payload.get("scheduler_output_id"):
                _scheduler_output_missing += 1
            if probe_name in {"kv_prefix_lookup_core", "kv_free_core", "block_pool_delta_core"} and not payload.get("scheduler_output_id_or_zero"):
                _scheduler_output_missing += 1
            if event.get("scheduler_output_id_is_surrogate") is True:
                _scheduler_output_is_surrogate += 1
            if event.get("scheduler_output_id") is not None and str(event.get("scheduler_output_id")) == str(event.get("scheduler_step_id")):
                _scheduler_output_equals_step += 1
        try:
            emit_probe(probe_name, payload)
            _write_timeline(probe_name, event, payload)
        except BaseException as exc:
            with _lock:
                _error_count[probe_name] += 1
            if first_error is None:
                first_error = exc
    write_emit_summary()
    if first_error is not None:
        raise RuntimeError(f"native USDT emit failed: {first_error}") from first_error


def get_abi_manifest() -> dict[str, Any]:
    return {
        "provider": PROVIDER,
        "backend": BACKEND,
        "native_so_path": NATIVE_SO_PATH,
        "abi_schema_version": ABI_SCHEMA_VERSION,
        "abi_schema_version_numeric": ABI_SCHEMA_VERSION_NUMERIC,
        "schema_version": ABI_SCHEMA_VERSION_NUMERIC,
        "required_probes": list(REQUIRED_PROBES),
        "abi_field_order_by_probe": {name: list(fields) for name, fields in ABI_FIELD_ORDER.items()},
        "abi_arg_count_by_probe": dict(_abi_arg_count),
        "physical_abi_max_arg_count": 6,
        "provider_type": "ELF-backed native USDT provider",
        "forbidden_fields": sorted(FORBIDDEN_USDT_FIELDS),
        "numeric_only": True,
    }


def get_emit_summary() -> dict[str, Any]:
    with _lock:
        return {
            "provider": PROVIDER,
            "backend": BACKEND,
            "native_so_path": NATIVE_SO_PATH,
            "abi_schema_version": ABI_SCHEMA_VERSION,
            "schema_version": ABI_SCHEMA_VERSION_NUMERIC,
            "emit_attempt_count_by_probe": {name: int(_attempt_count.get(name, 0)) for name in REQUIRED_PROBES},
            "emit_error_count_by_probe": {name: int(_error_count.get(name, 0)) for name in REQUIRED_PROBES},
            "native_so_load_error": _native_so_load_error,
            "forbidden_field_in_usdt_payload_count": int(_forbidden_field_count),
            "scheduler_output_id_missing_in_usdt_payload_count": int(_scheduler_output_missing),
            "scheduler_output_id_is_surrogate_count": int(_scheduler_output_is_surrogate),
            "scheduler_output_id_equals_scheduler_step_id_count": int(_scheduler_output_equals_step),
            "entry_owner_unresolved_count": int(_entry_owner_unresolved_count),
        }


def write_emit_summary(path: str | None = None) -> None:
    target = path or _ATTEMPT_PATH
    if not target:
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(get_emit_summary(), f, indent=2, sort_keys=True)
        f.write("\n")


def native_so_load_success() -> bool:
    return _lib is not None and _native_so_load_error is None


_init_native_provider()
atexit.register(write_emit_summary)
