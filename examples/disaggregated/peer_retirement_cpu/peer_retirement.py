# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only design prototype, not a vLLM plugin or a production endpoint.

The caller must stop admitting the exact old engine generation and serialize
calls on the connector's worker thread. This module does not detect dead peers,
cancel native transfers, or establish that CUDA memory was actually released.
"""

from dataclasses import dataclass
from threading import get_ident

PEER_MAPS = (
    "_remote_agents",
    "dst_xfer_side_handles",
    "kv_caches_base_addr",
    "dst_num_blocks",
    "dst_region_num_blocks",
    "dst_region_group_ids",
    "dst_uses_region_group_mapping",
    "dst_region_mem_types",
    "tp_mappings",
    "_engine_clock_offset",
    "_engine_last_active",
)


@dataclass(frozen=True)
class DropResult:
    engine_id: str
    state: str
    reason: str
    native_release_verified: bool = False


class PeerRetirement:
    """Experiment with a drained, unidirectional pull consumer's cleanup.

    A successful result acknowledges Python bookkeeping only. Retirement is
    irreversible in this prototype: a replacement must use a new engine ID.
    Entries are intentionally not expired; generation/tombstone storage is an
    integration concern, not implemented here.
    """

    def __init__(self, worker):
        self.worker = worker
        self._thread_id = get_ident()
        self._retiring = set()
        self._terminal = {}

    def _check_thread(self):
        if get_ident() != self._thread_id:
            raise RuntimeError("Call on the serialized connector worker thread")

    def check_admission(self, engine_id):
        """Model the admission gate; real scheduler wiring is NOT installed."""
        self._check_thread()
        if engine_id in self._retiring:
            raise RuntimeError("Old engine generation has been retired")

    def _busy_reason(self, engine_id):
        worker = self.worker
        # NIXL is not assumed thread-safe, even for a different peer.
        if worker._handshake_futures:
            return "handshake_callbacks_pending"
        if not worker._ready_requests.empty():
            return "ready_callbacks_pending"
        if not worker._failed_recv_reqs.empty():
            return "failed_callbacks_pending"
        if worker._reqs_to_send or worker._reqs_to_process:
            return "send_side_not_drained"

        for meta in worker._recving_metadata.values():
            if meta.remote is None:
                return "unattributed_receive_metadata"
            if meta.remote.engine_id == engine_id:
                return "target_receive_pending"
        # Do not treat a handle without metadata as an idle peer.
        for req_id in worker._recving_transfers:
            if req_id not in worker._recving_metadata:
                return "unattributed_transfer"
        for req_id in worker._pending_recv_notifs:
            meta = worker._recving_metadata.get(req_id)
            if meta is None or meta.remote is None:
                return "unattributed_notification"
            if meta.remote.engine_id == engine_id:
                return "target_notification_pending"
        return None

    def drop_peer(self, engine_id):
        """Retire an exact generation; return busy instead of forcing cleanup.

        The caller retries busy after normal completion processing. A cleanup
        exception is terminal here: native calls may have partially succeeded,
        so blindly retrying or reporting an absent peer would be unsafe.
        """
        self._check_thread()
        if not isinstance(engine_id, str) or not engine_id.strip():
            raise ValueError("A nonempty engine generation ID is required")
        if engine_id in self._terminal:
            return self._terminal[engine_id]
        self._retiring.add(engine_id)
        worker = self.worker

        # Local calls are serialized; the lock excludes handshake callbacks.
        with worker._handshake_lock:
            reason = self._busy_reason(engine_id)
            if reason is not None:
                return DropResult(engine_id, "busy", reason)
            if engine_id not in worker._remote_agents:
                partial = any(engine_id in getattr(worker, name) for name in PEER_MAPS)
                result = DropResult(
                    engine_id,
                    "cleanup_failed" if partial else "absent",
                    "inconsistent_peer_state" if partial else "no_python_peer_state",
                )
            else:
                try:
                    worker._cleanup_remote_engine(engine_id, log_eviction=False)
                except Exception as exc:
                    result = DropResult(engine_id, "cleanup_failed", type(exc).__name__)
                else:
                    result = DropResult(
                        engine_id, "cleanup_returned", "native_close_not_verified"
                    )
            self._terminal[engine_id] = result
            return result


def all_workers_acknowledged(results, expected_worker_ids):
    """Require every expected D/rank; never interpret this as a VRAM check."""
    return (
        bool(expected_worker_ids)
        and set(results) == set(expected_worker_ids)
        and all(r.state in {"cleanup_returned", "absent"} for r in results.values())
        and len({r.engine_id for r in results.values()}) == 1
    )
