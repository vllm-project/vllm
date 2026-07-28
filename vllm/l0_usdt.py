# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import hashlib
import threading
import time
import uuid
from typing import Any

_PROVIDER_NAME = "vllm_l0"
_SCHEMA_VERSION = "vllm_l0_v1"
_DEFAULT_LEVEL = "L0_request_lifecycle"
_PROBE_NAMES = (
    "request_arrival",
    "request_id_mapping",
    "request_id_assigned",
    "request_engine_admitted",
    "first_token",
    "output_token",
    "request_terminal",
    "request_first_output",
    "request_output",
    "request_finish",
    "request_abort",
    "request_reject",
    "request_error",
)

_TRACE_PATH = os.environ.get("VLLM_L0_TRACE_PATH")
_INTERNAL_TRACE_PATH = os.environ.get("VLLM_L0_INTERNAL_TRACE_PATH")
_L0_ENABLED = os.environ.get("VLLM_L0_ENABLE", "1").lower() not in {
    "0",
    "false",
    "no",
}
_USDT_ENV = os.environ.get("VLLM_L0_ENABLE_USDT", os.environ.get("VLLM_L0_USDT_ENABLE", "0"))
_SIDECAR_ENV = os.environ.get("VLLM_L0_ENABLE_SIDECAR")
_USDT_ENABLED = _L0_ENABLED and _USDT_ENV == "1"
_FORCE_USDT_FIRE = os.environ.get("VLLM_L0_FORCE_USDT_FIRE", "0") == "1"
_SIDECAR_ENABLED = _L0_ENABLED and (
    _SIDECAR_ENV is None or _SIDECAR_ENV.lower() not in {"0", "false", "no"}
)
_INTERNAL_TRACE_ENABLED = (
    _L0_ENABLED and os.environ.get("VLLM_L0_ENABLE_INTERNAL_TRACE", "0") == "1"
)

_provider: Any | None = None
_probes: dict[str, Any] = {}
_jsonl_files: dict[str, Any] = {}
_jsonl_disabled_paths: set[str] = set()
_internal_trace_path_warning_emitted = False
_jsonl_lock = threading.Lock()
_event_seq_lock = threading.Lock()
_event_seq = 0
_TRACE_SESSION_ID = os.environ.get("VLLM_L0_TRACE_SESSION_ID") or uuid.uuid4().hex
_EXPERIMENT_ID = os.environ.get("VLLM_L0_EXPERIMENT_ID")
_TRACE_SCOPE = os.environ.get("VLLM_L0_TRACE_SCOPE")
_TRACE_ENDPOINT = os.environ.get("VLLM_L0_TRACE_ENDPOINT")
_TRACE_STREAM = os.environ.get("VLLM_L0_TRACE_STREAM")
_DISABLE_INTERNAL_TRACE = os.environ.get("VLLM_L0_DISABLE_INTERNAL_TRACE") == "1"


def _stapsdt_char_ptr_arg(stapsdt: Any) -> Any:
    arg_types = stapsdt.ArgTypes
    return getattr(arg_types, "char_ptr", getattr(arg_types, "uint64"))


def _init_usdt_provider() -> None:
    global _provider
    if not _USDT_ENABLED:
        return

    try:
        import stapsdt

        provider = stapsdt.Provider(_PROVIDER_NAME)
        char_ptr_arg = _stapsdt_char_ptr_arg(stapsdt)
        for probe_name in _PROBE_NAMES:
            _probes[probe_name] = provider.add_probe(
                probe_name, char_ptr_arg, char_ptr_arg
            )
        provider.load()
        _provider = provider
    except Exception:
        _probes.clear()
        _provider = None


def _to_bytes(value: Any | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    try:
        return str(value).encode("utf-8", errors="replace")
    except Exception:
        return b"<unprintable>"


def _to_text(value: Any | None) -> str | None:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return "<unprintable>"


def _extra_to_json(extra: dict[str, Any]) -> str:
    try:
        return json.dumps(
            extra, ensure_ascii=False, default=str, separators=(",", ":")
        )
    except Exception:
        return "{}"


def _next_event_seq() -> int:
    global _event_seq
    with _event_seq_lock:
        _event_seq += 1
        return _event_seq


def _canonical_extra_json(extra: dict[str, Any]) -> str:
    try:
        return json.dumps(
            extra,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception:
        return "{}"


def _extra_hash(extra: dict[str, Any]) -> str:
    return hashlib.sha1(_canonical_extra_json(extra).encode("utf-8")).hexdigest()[:12]


def _stable_u64_hash(value: Any) -> int:
    data = str(value).encode("utf-8", errors="replace")
    digest = hashlib.blake2b(data, digest_size=8)
    return int.from_bytes(digest.digest(), "big", signed=False)


def _is_internal_trace(signal_scope: str | None, extra: dict[str, Any]) -> bool:
    return (
        signal_scope == "engine_internal"
        or extra.get("internal_trace") is True
        or extra.get("path") == "offline_or_engine"
    )


def _get_l0_trace_path(signal_scope: str | None) -> str | None:
    if signal_scope == "engine_internal":
        return os.environ.get("VLLM_L0_INTERNAL_TRACE_PATH") or _INTERNAL_TRACE_PATH
    return os.environ.get("VLLM_L0_TRACE_PATH") or _TRACE_PATH


def _warn_internal_trace_path_missing() -> None:
    global _internal_trace_path_warning_emitted
    if _internal_trace_path_warning_emitted:
        return
    _internal_trace_path_warning_emitted = True
    try:
        import sys

        print(
            "VLLM_L0_ENABLE_INTERNAL_TRACE=1 but "
            "VLLM_L0_INTERNAL_TRACE_PATH is not set; engine_internal sidecar "
            "events will not be written to VLLM_L0_TRACE_PATH.",
            file=sys.stderr,
        )
    except Exception:
        pass


def _force_fire_usdt_probe(probe: Any, request_id: str, extra_json: str) -> bool:
    try:
        import ctypes
        import stapsdt

        raw_probe = getattr(probe, "_probe", None)
        if not raw_probe:
            return False
        stapsdt.probeFire(
            raw_probe,
            ctypes.c_char_p(request_id.encode("utf-8", errors="replace")),
            ctypes.c_char_p(extra_json.encode("utf-8", errors="replace")),
        )
        return True
    except Exception:
        return False


def _get_jsonl_file(signal_scope: str | None) -> Any | None:
    trace_path = _get_l0_trace_path(signal_scope)
    if not _SIDECAR_ENABLED or not trace_path or trace_path in _jsonl_disabled_paths:
        return None
    if trace_path not in _jsonl_files:
        try:
            trace_dir = os.path.dirname(trace_path)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            _jsonl_files[trace_path] = open(trace_path, "a", encoding="utf-8", buffering=1)
        except Exception:
            _jsonl_disabled_paths.add(trace_path)
            return None
    return _jsonl_files[trace_path]


def _write_jsonl(
    event: str,
    request_id: str | None,
    extra_json: str,
    wall_ts_ns: int,
    monotonic_ts_ns: int,
    event_seq: int,
    top_level: dict[str, Any],
) -> None:
    signal_scope = top_level.get("signal_scope") or _TRACE_SCOPE
    if signal_scope == "engine_internal" and not _get_l0_trace_path(signal_scope):
        _warn_internal_trace_path_missing()
        return

    sink = _get_jsonl_file(signal_scope)
    if sink is None:
        return

    try:
        extra = json.loads(extra_json)
    except Exception:
        extra = {}

    payload = {
        "schema_version": top_level.get("schema_version") or _SCHEMA_VERSION,
        "legacy_schema_version": "v4_l0_usdt_validation.v1",
        "level": top_level.get("level") or _DEFAULT_LEVEL,
        "signal_scope": signal_scope,
        "provider": _PROVIDER_NAME,
        "source": "sidecar",
        "payload_mode": top_level.get("payload_mode") or "full",
        "experiment_id": top_level.get("experiment_id") or _EXPERIMENT_ID,
        "trace_session_id": top_level.get("trace_session_id") or _TRACE_SESSION_ID,
        "event_seq": event_seq,
        "extra_hash": top_level.get("extra_hash") or _extra_hash(extra),
        "event": event,
        "ts_ns": wall_ts_ns,
        "monotonic_ns": monotonic_ts_ns,
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "request_id": request_id,
        "extra": extra,
    }
    try:
        with _jsonl_lock:
            sink.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        trace_path = _get_l0_trace_path(signal_scope)
        if trace_path:
            _jsonl_disabled_paths.add(trace_path)


def fire_l0_probe(event: str, request_id: Any | None = None, **extra: Any) -> None:
    if not _L0_ENABLED:
        return
    if event not in _PROBE_NAMES:
        return
    if not _TRACE_PATH and event not in _probes:
        return

    try:
        requested_signal_scope = extra.get("signal_scope") or _TRACE_SCOPE
        is_internal_trace = _is_internal_trace(requested_signal_scope, extra)
        if is_internal_trace:
            if not _INTERNAL_TRACE_ENABLED:
                return
            extra["signal_scope"] = "engine_internal"
            extra["internal_trace"] = True
            extra.setdefault("path", "offline_or_engine")

        top_level = {
            key: extra.pop(key)
            for key in (
                "schema_version",
                "level",
                "signal_scope",
                "payload_mode",
                "experiment_id",
                "trace_session_id",
                "extra_hash",
            )
            if key in extra
        }
        if is_internal_trace:
            top_level["signal_scope"] = "engine_internal"
        if _TRACE_ENDPOINT and "endpoint" not in extra:
            extra["endpoint"] = _TRACE_ENDPOINT
        if _TRACE_STREAM is not None and "stream" not in extra:
            extra["stream"] = _TRACE_STREAM.lower() in {"1", "true", "yes"}
        obs03_case_name = os.environ.get("VLLM_OBS03_CASE_NAME")
        if obs03_case_name and "case_name" not in extra:
            extra["case_name"] = obs03_case_name
        request_id_text = _to_text(request_id)
        usdt_request_id = request_id_text or ""
        extra_json = _extra_to_json(extra)
        wall_ts_ns = time.time_ns()
        monotonic_ts_ns = time.monotonic_ns()
        event_seq = _next_event_seq()
    except Exception:
        return

    probe = _probes.get(event)
    if probe is not None:
        try:
            if _FORCE_USDT_FIRE:
                _force_fire_usdt_probe(probe, usdt_request_id, extra_json)
            else:
                probe.fire(usdt_request_id, extra_json)
        except Exception:
            pass

    _write_jsonl(
        event,
        request_id_text,
        extra_json,
        wall_ts_ns,
        monotonic_ts_ns,
        event_seq,
        top_level,
    )
    if event in {
        "request_arrival",
        "request_engine_admitted",
        "first_token",
        "output_token",
        "request_terminal",
    }:
        try:
            from vllm.obs_probe import obs_emit

            obs_emit(
                "req",
                event,
                request_id=request_id_text,
                event_category="lifecycle",
                ts_ns=wall_ts_ns,
                monotonic_ns=monotonic_ts_ns,
                **extra,
            )
        except Exception:
            pass


def fire_request_arrival(request_id: Any | None = None, **extra: Any) -> None:
    fire_l0_probe("request_arrival", request_id, **extra)


def fire_request_id_mapping(
    external_request_id: Any | None,
    internal_request_id: Any | None,
    **extra: Any,
) -> None:
    mapping_extra = {
        "path": "online",
        "external_request_id": _to_text(external_request_id),
        "internal_request_id": _to_text(internal_request_id),
    }
    mapping_extra.update(extra)
    fire_l0_probe(
        "request_id_mapping",
        external_request_id,
        **mapping_extra,
    )


def fire_request_id_assigned(request_id: Any | None = None, **extra: Any) -> None:
    fire_l0_probe("request_id_assigned", request_id, **extra)


def fire_request_engine_admitted(
    request_id: Any | None = None, **extra: Any
) -> None:
    fire_l0_probe("request_engine_admitted", request_id, **extra)


def fire_request_first_output(request_id: Any | None = None, **extra: Any) -> None:
    fire_l0_probe("request_first_output", request_id, **extra)


def fire_first_token(request_id: Any | None = None, **extra: Any) -> None:
    extra.setdefault("request_id_hash", l0_request_id_hash(request_id))
    fire_l0_probe("first_token", request_id, **extra)


def fire_request_output(request_id: Any | None = None, **extra: Any) -> None:
    fire_l0_probe("request_output", request_id, **extra)


def fire_output_token(request_id: Any | None = None, **extra: Any) -> None:
    extra.setdefault("request_id_hash", l0_request_id_hash(request_id))
    fire_l0_probe("output_token", request_id, **extra)


def fire_request_finish(request_id: Any | None = None, **extra: Any) -> None:
    fire_l0_probe("request_finish", request_id, **extra)


def fire_request_terminal(request_id: Any | None = None, **extra: Any) -> None:
    terminal_type = extra.get("terminal_type")
    if terminal_type is None:
        raw_finish_reason = extra.get("raw_finish_reason") or extra.get("finish_reason")
        terminal_type = "finished" if raw_finish_reason not in (None, "", "abort") else "unknown"
    extra.setdefault("request_id_hash", l0_request_id_hash(request_id))
    extra.setdefault("terminal_type", terminal_type)
    extra.setdefault("terminal_source", "vllm_l0_runtime")
    extra.setdefault("terminal_confidence", "high")
    extra.setdefault("raw_finish_reason", extra.get("finish_reason"))
    extra.setdefault("raw_terminal_code", extra.get("terminal_type"))
    fire_l0_probe("request_terminal", request_id, **extra)


def fire_request_abort(request_id: Any | None = None, **extra: Any) -> None:
    fire_l0_probe("request_abort", request_id, **extra)


def fire_request_reject(request_id: Any | None = None, **extra: Any) -> None:
    fire_l0_probe("request_reject", request_id, **extra)


def fire_request_error(request_id: Any | None = None, **extra: Any) -> None:
    fire_l0_probe("request_error", request_id, **extra)


def l0_output_token_count(output: Any) -> int | None:
    outputs = getattr(output, "outputs", None)
    if not outputs:
        return None

    total = 0
    seen_token_ids = False
    for item in outputs:
        token_ids = getattr(item, "token_ids", None)
        if token_ids is None:
            continue
        try:
            total += len(token_ids)
            seen_token_ids = True
        except Exception:
            continue
    return total if seen_token_ids else None


def l0_request_id_hash(request_id: Any | None) -> int:
    return _stable_u64_hash(request_id or "")


def l0_finish_reason(output: Any) -> Any | None:
    outputs = getattr(output, "outputs", None)
    if not outputs:
        return None

    reasons = []
    for item in outputs:
        reason = getattr(item, "finish_reason", None)
        if reason is not None:
            reasons.append(reason)
    if not reasons:
        return None
    if len(reasons) == 1:
        return reasons[0]
    return reasons


try:
    import vllm.obs_probe  # noqa: F401
except Exception:
    pass

_init_usdt_provider()
