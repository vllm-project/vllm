# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PSecondaryTierManager: Secondary tier for P2P KV cache sharing.

Owns transports, sessions, and the SecondaryTierManager interface.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.p2p.control import ControlTransport, ZmqTransport
from vllm.v1.kv_offload.tiering.p2p.data import DataTransport, NixlTransport
from vllm.v1.kv_offload.tiering.p2p.session import (
    P2PClientSession,
    P2PServerSession,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class P2PSecondaryTierManager(SecondaryTierManager):
    """Secondary tier for P2P KV cache sharing."""

    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str = "p2p",
        host: str = "0.0.0.0",
        port: int = 7777,
        **kwargs,
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        port = int(port)
        self._local_id = f"{host}:{port}"

        config_fields = self._build_config_fields(offloading_spec)
        self._data: DataTransport = NixlTransport(
            self._local_id, primary_kv_view, config_fields=config_fields
        )
        self._control: ControlTransport = ZmqTransport(self._local_id, host, port)

        self._server_sessions: dict[str, P2PServerSession] = {}
        self._client_sessions: dict[str, P2PClientSession] = {}

        self._finished_jobs: list[JobResult] = []
        self._failed_req_ids: set[str] = set()

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        kv_params = req_context.kv_transfer_params
        if (
            not kv_params
            or not kv_params.get("remote_host")
            or not kv_params.get("remote_port")
            or not kv_params.get("kv_request_id")
        ):
            return False

        kv_request_id = kv_params["kv_request_id"]
        return kv_request_id not in self._failed_req_ids

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """Ensures client session exists for the remote peer."""
        kv_params = req_context.kv_transfer_params
        if kv_params:
            peer_id = self._remote_id_from_params(kv_params)
            if peer_id and peer_id not in self._client_sessions:
                conn = self._control.connect(peer_id)
                self._client_sessions[peer_id] = P2PClientSession(
                    peer_id,
                    conn,
                    local_id=self._local_id,
                    transport=self._data,
                )
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        """Cancels pending loads and prunes state."""
        kv_params = req_context.kv_transfer_params
        if not kv_params:
            return
        kv_request_id = kv_params.get("kv_request_id")
        if not kv_request_id:
            return
        self._failed_req_ids.discard(kv_request_id)
        peer_id = self._remote_id_from_params(kv_params)
        if peer_id:
            session = self._client_sessions.get(peer_id)
            if session is not None:
                session.cancel_request(kv_request_id)

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids)

        kv_params = job_metadata.req_context.kv_transfer_params
        if not kv_params or not kv_params.get("do_remote_decode"):
            self._finished_jobs.append(JobResult(job_id=job_id, success=True))
            return

        kv_request_id = kv_params.get("kv_request_id")
        peer_id = self._remote_id_from_params(kv_params)
        if not kv_request_id or not peer_id:
            logger.warning(
                "P2P %s: submit_store missing kv_request_id or peer", self._local_id
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            return

        session = self._server_sessions.get(peer_id)
        if session is None:
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            return
        session.add_stored_blocks(kv_request_id, keys, block_ids, job_id)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids

        kv_params = job_metadata.req_context.kv_transfer_params
        if (
            not kv_params
            or not kv_params.get("remote_host")
            or not kv_params.get("remote_port")
            or not kv_params.get("kv_request_id")
        ):
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            return

        kv_request_id = kv_params["kv_request_id"]
        peer_id = self._remote_id_from_params(kv_params)
        assert peer_id is not None  # guaranteed by kv_params checks above

        if not keys:
            self._finished_jobs.append(JobResult(job_id=job_id, success=True))
            return

        session = self._client_sessions.get(peer_id)
        if session is None:
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            self._failed_req_ids.add(kv_request_id)
            return
        session.request_blocks(job_id, kv_request_id, keys, block_ids)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        self._accept_new_peers(self._control.poll())

        for srv in self._server_sessions.values():
            for sr in srv.poll():
                self._finished_jobs.append(
                    JobResult(job_id=sr.job_id, success=sr.success)
                )

        for cli in self._client_sessions.values():
            for lr in cli.poll():
                self._finished_jobs.append(
                    JobResult(job_id=lr.job_id, success=lr.success)
                )
                if not lr.success:
                    self._failed_req_ids.add(lr.kv_request_id)

        self._reap_dead_servers()
        self._reap_dead_clients()

        result = self._finished_jobs
        self._finished_jobs = []
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config_fields(offloading_spec: OffloadingSpec) -> dict | None:
        """Extract config fields for fingerprint from the offloading spec."""
        try:
            model_config = offloading_spec.vllm_config.model_config
            parallel_config = offloading_spec.vllm_config.parallel_config
            gpu_block_sizes = list(offloading_spec.gpu_block_size)
            fields: dict = {
                "model": model_config.model,
                "dtype": str(model_config.dtype),
                "hash_block_size": offloading_spec.hash_block_size,
                "block_size_factor": offloading_spec.block_size_factor,
                "gpu_block_size": gpu_block_sizes,
            }
            # TP size only matters with multiple kv_cache_groups
            if len(gpu_block_sizes) > 1:
                fields["tp_size"] = parallel_config.tensor_parallel_size
            return fields
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _remote_id_from_params(kv_params: dict) -> str | None:
        host = kv_params.get("remote_host")
        port = kv_params.get("remote_port")
        if host and port:
            return f"{host}:{port}"
        return None

    def _accept_new_peers(self, new_connections: list) -> None:
        for conn in new_connections:
            try:
                session = P2PServerSession(
                    conn,
                    local_id=self._local_id,
                    transport=self._data,
                    local_block_len=self._data.block_len,
                )
                self._server_sessions[session.peer_id] = session
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                logger.error("P2P %s: rejecting peer: %s", self._local_id, exc)
                conn.close()

    def _reap_dead_servers(self) -> None:
        dead = [pid for pid, s in self._server_sessions.items() if not s.alive]
        for pid in dead:
            session = self._server_sessions.pop(pid)
            for job_id in session.close():
                self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            self._data.remove_remote_peer(pid)
            logger.warning("P2P %s: server peer %s down", self._local_id, pid)

    def _reap_dead_clients(self) -> None:
        dead = [pid for pid, s in self._client_sessions.items() if not s.alive]
        for pid in dead:
            session = self._client_sessions.pop(pid)
            for job_id, kv_request_id in session.close():
                self._finished_jobs.append(JobResult(job_id=job_id, success=False))
                self._failed_req_ids.add(kv_request_id)
            logger.warning("P2P %s: client peer %s down", self._local_id, pid)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @override
    def shutdown(self) -> None:
        for srv in self._server_sessions.values():
            srv.close()
        for cli in self._client_sessions.values():
            cli.close()
        self._server_sessions.clear()
        self._client_sessions.clear()
        self._control.close()
        self._data.close()
