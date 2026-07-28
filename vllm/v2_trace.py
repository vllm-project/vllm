from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import itertools
import json
import os
import threading
import time
from typing import Any

_ENABLED = os.environ.get("VLLM_V2_SEMANTIC_TRACE", "0") == "1"
_ENABLE_NVTX = os.environ.get("VLLM_V2_NVTX", "0") == "1"
_ENABLE_RECORD_FUNCTION = os.environ.get("VLLM_V2_RECORD_FUNCTION", "0") == "1"
_ENABLE_TRACE_FIELDS = os.environ.get("VLLM_V2_TRACE_FIELDS", "0") == "1"
_TRACE_JSONL = os.environ.get("VLLM_V2_TRACE_JSONL")
_OBS10V_MODE = os.environ.get("VLLM_OBS10V_STRICT_L2_ENABLE", "0") == "1"
_OBS10V_EVENT_NAMES = frozenset(
    {
        "request_arrival",
        "request_engine_admitted",
        "first_token",
        "output_token",
        "request_terminal",
        "request_finish",
        "scheduler_step",
        "scheduler_step_enter",
        "scheduler_step_end",
        "scheduler_output_snapshot",
        "scheduler_output_member",
        "scheduler_queue_snapshot",
        "scheduler_queue_member_sample",
        "request_selected",
        "batch_member",
        "request_batch_index_mapping",
        "worker_request_index_mapping",
        "worker_slot_mapping_entry",
        "worker_slot_mapping_request_summary",
        "kv_block_transition_v1",
        "kv_request_state_snapshot_v1",
        "kv_block_alloc",
        "kv_block_free",
        "kv_block_alloc_detail",
        "kv_block_free_detail",
        "prefix_cache_lookup",
        "prefix_cache_hit_detail",
        "prefix_cache_miss_detail",
        "prefix_cache_insert_detail",
        "request_preempted",
    }
)

_TORCH = None
if _ENABLED:
    try:
        import torch as _TORCH
    except Exception:
        _TORCH = None

_LAYERS = frozenset(("req", "sched", "kv", "exec", "semop", "rtop", "cuda", "hw"))
_ID_KEYS = (
    "request_id",
    "sequence_id",
    "scheduler_step_id",
    "batch_id",
    "block_table_id",
    "kv_block_id",
    "logical_block_id",
    "physical_block_id",
    "execution_step_id",
    "semantic_op_id",
    "runtime_op_id",
    "cuda_kernel_id",
    "hardware_counter_id",
)
_ATTR_KEYS = (
    "phase",
    "num_requests",
    "num_tokens",
    "num_prefill_tokens",
    "num_decode_tokens",
    "layer_id",
    "op_type",
    "stream_id",
    "correlation_id",
    "source_file",
    "source_func",
)

_event_counter = itertools.count()
_jsonl_file = None
_jsonl_disabled = False
_jsonl_lock = threading.Lock()
_local_state = threading.local()


def _is_torch_compiling() -> bool:
    if _TORCH is None:
        return False
    try:
        return bool(_TORCH.compiler.is_compiling()) or bool(
            _TORCH._dynamo.is_compiling()
        )
    except Exception:
        return False


def _short_value(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 48 else text[:45] + "..."


def _make_label(layer: str, name: str, fields: dict[str, Any]) -> str:
    label = f"v2.{layer}::{name}"
    if not _ENABLE_TRACE_FIELDS or not fields:
        return label
    parts = [label]
    for key in (
        "request_id",
        "scheduler_step_id",
        "batch_id",
        "execution_step_id",
        "semantic_op_id",
        "layer_id",
        "phase",
    ):
        value = fields.get(key)
        if value is not None:
            parts.append(f"{key}={_short_value(value)}")
    return "|".join(parts)


def _get_jsonl_file():
    global _jsonl_file, _jsonl_disabled
    if _jsonl_disabled or not _TRACE_JSONL:
        return None
    if _jsonl_file is None:
        try:
            os.makedirs(os.path.dirname(_TRACE_JSONL), exist_ok=True)
            _jsonl_file = open(_TRACE_JSONL, "a", encoding="utf-8", buffering=1)
        except Exception:
            _jsonl_disabled = True
            return None
    return _jsonl_file


def _split_fields(fields: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = {key: fields.get(key) for key in _ID_KEYS}
    attrs = {key: fields.get(key) for key in _ATTR_KEYS}
    for key, value in fields.items():
        if key not in ids and key not in attrs:
            attrs[key] = value
    return ids, attrs


def _write_event(
    event_type: str,
    layer: str,
    name: str,
    event_id: str,
    fields: dict[str, Any],
    value: Any | None = None,
) -> None:
    if not _ENABLED:
        return
    if _OBS10V_MODE and name not in _OBS10V_EVENT_NAMES:
        return
    sink = _get_jsonl_file()
    if sink is None:
        return
    ids, attrs = _split_fields(fields)
    payload = {
        "schema_version": "v2.1",
        "event_id": event_id,
        "event_type": event_type,
        "layer": layer,
        "name": name,
        "ts_ns": time.time_ns(),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "ids": ids,
        "attrs": attrs,
    }
    if value is not None:
        payload["value"] = value
    try:
        with _jsonl_lock:
            sink.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        global _jsonl_disabled
        _jsonl_disabled = True


def _next_event_id(layer: str, name: str) -> str:
    return f"{os.getpid()}-{threading.get_ident()}-{layer}-{name}-{next(_event_counter)}"


def current_ids() -> dict[str, Any]:
    ids = getattr(_local_state, "ids", None)
    if ids is None:
        return {}
    return dict(ids)


def _merged_fields(fields: dict[str, Any]) -> dict[str, Any]:
    merged = current_ids()
    merged.update({key: value for key, value in fields.items() if value is not None})
    return merged


@contextmanager
def v2_range(layer: str, name: str, **fields: Any):
    if not _ENABLED or _is_torch_compiling():
        yield
        return
    if layer not in _LAYERS:
        layer = "semop"

    merged = _merged_fields(fields)
    label = _make_label(layer, name, merged)
    event_id = _next_event_id(layer, name)
    rf_ctx = None
    nvtx_pushed = False
    previous_ids = getattr(_local_state, "ids", {})
    scoped_ids = dict(previous_ids)
    for key in _ID_KEYS:
        value = merged.get(key)
        if value is not None:
            scoped_ids[key] = value

    try:
        _local_state.ids = scoped_ids
        _write_event("begin", layer, name, event_id, merged)
        if _ENABLE_NVTX:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(label)
                    nvtx_pushed = True
            except Exception:
                pass
        if _ENABLE_RECORD_FUNCTION:
            try:
                from torch.profiler import record_function

                rf_ctx = record_function(label)
                rf_ctx.__enter__()
            except Exception:
                rf_ctx = None
        yield
    finally:
        if rf_ctx is not None:
            try:
                rf_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if nvtx_pushed:
            try:
                import torch

                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
        _write_event("end", layer, name, event_id, merged)
        _local_state.ids = previous_ids


def v2_instant(layer: str, name: str, **fields: Any) -> None:
    if not _ENABLED or _is_torch_compiling():
        return
    if layer not in _LAYERS:
        layer = "semop"
    merged = _merged_fields(fields)
    _write_event("instant", layer, name, _next_event_id(layer, name), merged)


def v2_counter(layer: str, name: str, value: Any, **fields: Any) -> None:
    if not _ENABLED or _is_torch_compiling():
        return
    if layer not in _LAYERS:
        layer = "hw"
    merged = _merged_fields(fields)
    _write_event("counter", layer, name, _next_event_id(layer, name), merged, value)


def v2_trace_func(layer: str, name: str, **fields: Any):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with v2_range(layer, name, **fields):
                return func(*args, **kwargs)

        return wrapper

    return decorator
