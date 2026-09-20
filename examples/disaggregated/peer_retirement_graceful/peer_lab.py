# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit, graceful-only D-side retirement for an isolated vLLM 0.26 lab.

Run via worker collective RPC, never a background HTTP thread. The controller
must first stop old-P admissions and drain accepted requests. A successful RPC
does NOT certify native close or reclaimed GPU memory.
"""

import os
import re
import threading

PEER_MAPS = (
    "_remote_agents",
    "dst_xfer_side_handles",
    "kv_caches_base_addr",
    "dst_num_blocks",
    "tp_mappings",
    "_engine_clock_offset",
    "_engine_last_active",
)


def _state(worker):
    if not hasattr(worker, "_peer_lab_state"):
        worker._peer_lab_state = {"thread": threading.get_ident(), "ops": {}}
    state = worker._peer_lab_state
    if state["thread"] != threading.get_ident():
        raise RuntimeError("Retirement must run on the connector worker thread")
    return state


def is_fenced(worker, engine_id):
    state = getattr(worker, "_peer_lab_state", None)
    return state is not None and engine_id in state["ops"]


def busy_reason(worker):
    # The first lab milestone requires all D-side remote reads to quiesce.
    # Already-local decode generation may continue between worker RPCs.
    for name in (
        "_handshake_futures",
        "_recving_metadata",
        "_recving_transfers",
        "_reqs_to_send",
        "_reqs_to_process",
    ):
        if getattr(worker, name):
            return name
    for name in ("_ready_requests", "_failed_recv_reqs"):
        if not getattr(worker, name).empty():
            return name
    return None


def _result(worker, target, operation, state, reason=""):
    return {
        "decode_engine_id": worker.engine_id,
        "tp_rank": worker.tp_rank,
        "target_engine_id": target,
        "operation_id": operation,
        "state": state,
        "reason": reason,
        "native_release_verified": False,
        "gpu_memory_measured": False,
    }


def snapshot(worker):
    with worker._handshake_lock:
        return {
            "decode_engine_id": worker.engine_id,
            "tp_rank": worker.tp_rank,
            "peers": sorted(worker._remote_agents),
            "busy_reason": busy_reason(worker),
            "retired": sorted(getattr(worker, "_peer_lab_state", {}).get("ops", {})),
        }


def retire(worker, target, operation, expected_decode, phase):
    if os.environ.get("PEER_LAB_RETIRE_ENABLED") != "1":
        raise RuntimeError("Retirement is disabled in this control image")
    for value in (target, operation, expected_decode):
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z0-9_.:-]{1,128}", value
        ):
            raise ValueError("Invalid exact engine generation or operation ID")
    if expected_decode != worker.engine_id or target == worker.engine_id:
        raise ValueError("Engine generation mismatch")
    if phase not in {"prepare", "commit"}:
        raise ValueError("Unknown phase")
    if worker.kv_transfer_config.kv_role != "kv_consumer":
        raise RuntimeError("Only unidirectional pull Decode is supported")
    if worker._bidirectional_kv_xfer_enabled:
        raise RuntimeError("Bidirectional KV is outside this experiment")
    state = _state(worker)
    with worker._handshake_lock:
        previous = state["ops"].get(target)
        if previous is not None and previous["operation_id"] != operation:
            return _result(worker, target, operation, "conflict", "different_operation")
        if previous is not None and previous["state"] in {"cleanup_returned", "failed"}:
            return dict(previous)
        busy = busy_reason(worker)
        if busy:
            return _result(worker, target, operation, "busy", busy)
        if phase == "prepare":
            if len(state["ops"]) >= 1024 and previous is None:
                return _result(worker, target, operation, "refused", "tombstone_limit")
            if target not in worker._remote_agents and any(
                target in getattr(worker, key) for key in PEER_MAPS
            ):
                return _result(
                    worker, target, operation, "refused", "partial_peer_state"
                )
            result = _result(worker, target, operation, "prepared")
            state["ops"][target] = result
            return dict(result)
        if previous is None:
            return _result(worker, target, operation, "refused", "prepare_required")
        try:
            if target in worker._remote_agents:
                worker._cleanup_remote_engine(target, log_eviction=False)
            elif any(target in getattr(worker, key) for key in PEER_MAPS):
                raise RuntimeError("Peer state changed after prepare")
        except Exception as exc:
            # Cleanup may have freed only some handles. Do not retry or let TTL
            # re-enter this generation's cleanup and double-release native state.
            worker._engine_last_active.pop(target, None)
            result = _result(worker, target, operation, "failed", type(exc).__name__)
        else:
            result = _result(
                worker,
                target,
                operation,
                "cleanup_returned",
                "native_close_and_vram_need_external_verification",
            )
        state["ops"][target] = result
        return dict(result)


def _connector():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
        NixlPullConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group

    worker = get_kv_transfer_group().connector_worker
    if type(worker) is not NixlPullConnectorWorker:
        raise RuntimeError("Only the pinned NixlPullConnectorWorker is supported")
    return worker


class WorkerExtension:
    def peer_lab_snapshot(self):
        return snapshot(_connector())

    def peer_lab_retire(self, target, operation, expected_decode, phase):
        return retire(_connector(), target, operation, expected_decode, phase)

    def peer_lab_libraries(self):
        from pathlib import Path

        paths = set()
        for row in Path("/proc/self/maps").read_text().splitlines():
            path = row.split()[-1]
            if path.startswith("/") and any(
                x in path for x in ("libuc", "libplugin_UCX", "libnixl")
            ):
                paths.add(path)
        return {
            "pid": os.getpid(),
            "tp_rank": _connector().tp_rank,
            "libraries": sorted(paths),
        }
