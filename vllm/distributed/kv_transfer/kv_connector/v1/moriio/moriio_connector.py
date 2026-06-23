# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import logging
import math
import os
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Collection
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import msgpack
import msgspec
import numpy as np
import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    EngineId,
    HandshakeError,
    MoRIIOAgentMetadata,
    MoRIIOConfig,
    MoRIIOConnectorMetadata,
    MoRIIOConstants,
    MoRIIOMode,
    MoRIIOTransferAck,
    ReqId,
    ReqMeta,
    TransferId,
    WriteTask,
    get_moriio_mode,
    get_peer_zmq_from_request_id,
    get_port_offset,
    get_role,
    parse_moriio_zmq_address,
    resolve_host_ip,
    set_role,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
    MoRIIOWriter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout import (
    LayerTransferGeometry,
    build_layer_to_spec,
    compute_block_transfer_offsets,
    get_layer_transfer_geometry,
    is_mla_cache_layer,
    iter_layer_registration_regions,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    get_world_group,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import (
    make_zmq_path,
    make_zmq_socket,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


try:
    from mori.io import (
        BackendType,
        IOEngine,
        IOEngineConfig,
    )

    logger.info("MoRIIO is available")
    MoRIIO_enabled = True
except ImportError:
    logger.error("MoRIIO is not available")
    MoRIIO_enabled = False


def is_moriio_available() -> bool:
    return MoRIIO_enabled


def get_moriio_remote_tp_rank(
    local_tp_rank: int, local_tp_size: int, remote_tp_size: int
) -> int:
    if local_tp_size <= 0 or remote_tp_size <= 0:
        raise ValueError("TP sizes must be positive")
    if local_tp_rank < 0 or local_tp_rank >= local_tp_size:
        raise ValueError(
            f"local_tp_rank {local_tp_rank} must be in [0, {local_tp_size})"
        )
    if remote_tp_size == local_tp_size:
        return local_tp_rank
    if remote_tp_size > local_tp_size:
        if remote_tp_size % local_tp_size != 0:
            raise ValueError(
                f"remote tp_size {remote_tp_size} must be a multiple of local "
                f"tp_size {local_tp_size} for heterogeneous-TP P/D"
            )
        return local_tp_rank * (remote_tp_size // local_tp_size)
    if local_tp_size % remote_tp_size != 0:
        raise ValueError(
            f"local tp_size {local_tp_size} must be a multiple of remote "
            f"tp_size {remote_tp_size} for heterogeneous-TP P/D"
        )
    return local_tp_rank // (local_tp_size // remote_tp_size)


def validate_moriio_heterogeneous_tp_kv_heads(
    local_tp_size: int,
    remote_tp_size: int,
    total_num_kv_heads: int,
    is_mla: bool,
) -> None:
    if is_mla or local_tp_size == remote_tp_size:
        return
    if local_tp_size <= 0 or remote_tp_size <= 0 or total_num_kv_heads <= 0:
        raise ValueError("TP sizes and total_num_kv_heads must be positive")
    if min(local_tp_size, remote_tp_size) >= total_num_kv_heads:
        return
    raise NotImplementedError(
        "MoRIIO heterogeneous TP requires replicated KV heads on both "
        f"prefill and decode. Got total_num_kv_heads={total_num_kv_heads}, "
        f"local_tp_size={local_tp_size}, remote_tp_size={remote_tp_size}."
    )


def get_moriio_expected_ack_count(producer_tp_size: int, consumer_tp_size: int) -> int:
    if producer_tp_size <= 0 or consumer_tp_size <= 0:
        raise ValueError("TP sizes must be positive")
    if consumer_tp_size <= producer_tp_size:
        return 1
    if consumer_tp_size % producer_tp_size != 0:
        raise ValueError(
            f"consumer tp_size {consumer_tp_size} must be a multiple of "
            f"producer tp_size {producer_tp_size} for heterogeneous-TP P/D"
        )
    return consumer_tp_size // producer_tp_size


def resolve_moriio_transfer_ack(
    ack: MoRIIOTransferAck | TransferId,
    producer_tp_size: int,
    live_transfer_ids: Collection[TransferId],
    notification_counts: dict[TransferId, int],
    completed_transfer_ids: set[TransferId],
) -> TransferId | None:
    if isinstance(ack, str):
        ack = MoRIIOTransferAck(ack)
    transfer_id = ack.transfer_id
    if transfer_id not in live_transfer_ids:
        return None
    if transfer_id in completed_transfer_ids:
        return None

    expected_acks = get_moriio_expected_ack_count(
        producer_tp_size, ack.consumer_tp_size
    )
    count = notification_counts.get(transfer_id, 0) + 1
    if count < expected_acks:
        notification_counts[transfer_id] = count
        return None

    notification_counts.pop(transfer_id, None)
    completed_transfer_ids.add(transfer_id)
    return transfer_id


class MoRIIOConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None, (
            "kv_transfer_config must be set for MoRIIOConnector"
        )

        self.kv_transfer_config = vllm_config.kv_transfer_config
        self._set_port_defaults(vllm_config)

        self.engine_id = (
            str(resolve_host_ip(self.kv_transfer_config.kv_connector_extra_config))
            + ":"
            + str(self.kv_transfer_config.kv_connector_extra_config["handshake_port"])
        )
        self.mode = get_moriio_mode(self.kv_transfer_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MoRIIOConnectorScheduler | None = (
                MoRIIOConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MoRIIOConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MoRIIOConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
        logger.info(
            "Initialized MoRIIO Connector,engine_id:%s,role: %s",
            self.engine_id,
            role.value,
        )

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def _set_port_defaults(self, vllm_config: VllmConfig):
        assert vllm_config.kv_transfer_config is not None, (
            "kv_transfer_config must be set for MoRIIOConnector"
        )
        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = kv_transfer_config.kv_connector_extra_config

        if "handshake_port" not in extra_config or not extra_config["handshake_port"]:
            extra_config["handshake_port"] = MoRIIOConstants.DEFAULT_HANDSHAKE_PORT

        if "notify_port" not in extra_config or not extra_config["notify_port"]:
            extra_config["notify_port"] = MoRIIOConstants.DEFAULT_NOTIFY_PORT

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens, self.connector_worker
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        if self.mode == MoRIIOMode.WRITE and get_role() == ROLE.CONSUMER:
            self.connector_worker.moriio_wrapper.async_wait_reqid()

        assert isinstance(self._connector_metadata, MoRIIOConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        # Only producer/prefill saves KV Cache
        if get_role() == ROLE.CONSUMER:
            return
        assert self.connector_worker is not None, (
            "save_kv_layer called on scheduler role"
        )

        assert isinstance(self._connector_metadata, MoRIIOConnectorMetadata), (
            "Connector metadata not initialized yet"
        )
        self.connector_worker.save_kv_layer(
            self._connector_metadata, layer_name, kv_layer, attn_metadata, **kwargs
        )

        return None

    def wait_for_save(self):
        if self.mode != MoRIIOMode.WRITE or get_role() != ROLE.PRODUCER:
            return
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MoRIIOConnectorMetadata), (
            "Connector metadata not initialized yet"
        )
        self.connector_worker.wait_for_save(self._connector_metadata)

    def shutdown(self):
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()

    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        try:
            return self._connector_metadata is not None
        except AttributeError:
            return False


class MoRIIOConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config

        assert vllm_config.kv_transfer_config is not None, (
            "kv_transfer_config must be set for MoRIIOConnector"
        )
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.mode = get_moriio_mode(self.kv_transfer_config)
        self.host_ip = resolve_host_ip(
            self.kv_transfer_config.kv_connector_extra_config
        )
        self.handshake_port = self.kv_transfer_config.kv_connector_extra_config[
            "handshake_port"
        ]
        logger.info("Initializing MoRIIO Scheduler engine_id = %s", engine_id)

        self.side_notify_port = self.kv_transfer_config.kv_connector_extra_config[
            "notify_port"
        ]
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        # Local DP rank (0..dp_local-1) for port arithmetic.
        self.dp_rank = (
            self.vllm_config.parallel_config.data_parallel_rank
            % self.vllm_config.parallel_config.data_parallel_size_local
        )
        # Only first-pod ranks originate notify to avoid duplicates.
        self._is_kv_master = (
            self.vllm_config.parallel_config.data_parallel_rank
            < self.vllm_config.parallel_config.data_parallel_size_local
        )
        # Global DP rank for pinned request ownership check.
        self._global_dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        self.is_producer = self.kv_transfer_config.kv_role == "kv_producer"
        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}
        # Snapshot of kv_transfer_params for chunked prefill recovery.
        self._req_kv_params: dict[ReqId, dict] = {}

        # For chunked prefill, we perform layer-wise access within the final chunk.
        # TODO: Perform transfer at end chunk.
        self._reqs_need_pending_save: dict[ReqId, tuple[Request, list[int]]] = {}

        if self.is_producer:
            set_role(ROLE.PRODUCER)
        else:
            set_role(ROLE.CONSUMER)
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        # Deadlines for requests whose block freeing was deferred.
        # Survives across scheduler steps. If the worker never reports
        # finished_sending before the deadline, we inject them into
        # connector_output.finished_sending so the scheduler frees the blocks to avoid
        # hanging indefinitely waiting for a free notification that never comes.
        # Value: (deadline, transfer_id) for unmapping after mutation.
        self._deferred_send_deadlines: dict[ReqId, tuple[float, TransferId | None]] = {}
        self._defer_timeout = float(
            self.kv_transfer_config.kv_connector_extra_config.get(
                "defer_timeout", MoRIIOConstants.DEFAULT_DEFER_TIMEOUT
            )
        )
        # Buffer for early ACKs that arrive before request_finished.
        self._pending_sent_acks: dict[ReqId, float] = {}
        self.paths: dict[str, zmq.Socket] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

    def map_request_id(self, request_id: ReqId, transfer_id: TransferId):
        self.transfer_id_to_request_id[transfer_id] = request_id
        self.request_id_to_transfer_id[request_id] = transfer_id

    def unmap_request_id(
        self, request_id: ReqId, transfer_id: TransferId | None = None
    ):
        """Unmap request_id/transfer_id. Uses transfer_id for lookup if
        exact request_id match fails (handles input_processor mutation)."""
        if request_id in self.request_id_to_transfer_id:
            tid = self.request_id_to_transfer_id[request_id]
            del self.request_id_to_transfer_id[request_id]
            if tid in self.transfer_id_to_request_id:
                del self.transfer_id_to_request_id[tid]
            return

        # Fallback: use transfer_id to find original request_id.
        if transfer_id is not None and transfer_id in self.transfer_id_to_request_id:
            original_rid = self.transfer_id_to_request_id[transfer_id]
            if original_rid != request_id:
                logger.debug(
                    "MoRI-IO unmap via transfer_id: %r -> %r", request_id, original_rid
                )
            if original_rid in self.request_id_to_transfer_id:
                del self.request_id_to_transfer_id[original_rid]
            del self.transfer_id_to_request_id[transfer_id]
            return

        logger.warning(
            "MoRI-IO unmap MISS: rid=%r transfer_id=%r table_size=%d",
            request_id,
            transfer_id,
            len(self.request_id_to_transfer_id),
        )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """
        if self.is_producer:
            return 0, False

        token_ids = request.prompt_token_ids or []
        if self.mode == MoRIIOMode.WRITE:
            # MoriiO in write mode, no remote prefill

            return len(token_ids) - num_computed_tokens, True

        return len(token_ids) - 1 - num_computed_tokens, False

    def send_notify_block(
        self,
        req_id: ReqId,
        transfer_id: TransferId,
        block_notify_list: list[int],
        host=None,
        port=None,
    ):
        path = make_zmq_path("tcp", host, port)
        if path not in self.paths:
            ctx = zmq.Context.instance()
            sock = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.DEALER, bind=False
            )
            self.paths[path] = sock

        data = {
            "req_id": req_id,
            "transfer_id": transfer_id,
            "block_notify_list": block_notify_list or [],
            # GLOBAL decode dp rank: the producer derives the per-pod local
            # notify offset (% dp_local), the owning pod index (// dp_local),
            # and the write-target session from it. Sending the local rank
            # made child-pod consumers look like master rank 0 and hang in
            # WAITING_FOR_REMOTE_KVS. Single-pod is unaffected (local==global).
            "decode_rank": self._global_dp_rank,
            "type": "remote_blocks",
        }
        serialized_data = msgpack.dumps(data)
        self.paths[path].send(serialized_data)

    def _send_transfer_release(self, transfer_id: TransferId, host: str, port: int):
        path = make_zmq_path("tcp", host, port)
        if path not in self.paths:
            ctx = zmq.Context.instance()
            sock = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.DEALER, bind=False
            )
            self.paths[path] = sock

        # Align with the upstream READ-mode release (see _pop_done_transfers):
        # advertise the consumer (decode) TP size so the prefill side counts
        # the right number of ACKs via get_moriio_expected_ack_count(). For the
        # homogeneous-TP configs we run (1P1D/2P2D, TP=1) this resolves to 1
        # ACK exactly as before; it only matters under heterogeneous-TP fan-in.
        self.paths[path].send(
            msgpack.dumps(
                {
                    "type": "release",
                    "transfer_id": transfer_id,
                    "consumer_tp_size": self.tp_size,
                }
            )
        )

    def _release_write_prefill_blocks(self, request_id: ReqId, params: dict[str, Any]):
        transfer_id = params.get("transfer_id")
        if transfer_id is None:
            logger.warning(
                "Cannot release WRITE prefill blocks for request %s: "
                "missing transfer_id",
                request_id,
            )
            return

        remote_dp_rank = params.get("remote_dp_rank", 0)
        remote_host = params.get("remote_host")
        remote_notify_port = params.get("remote_notify_port")
        if remote_host is None or remote_notify_port is None:
            try:
                peer_zmq = get_peer_zmq_from_request_id(request_id, is_producer=False)
                if peer_zmq is None:
                    raise ValueError("no peer zmq address for request")
                remote_host, _, remote_notify_port = parse_moriio_zmq_address(peer_zmq)
            except ValueError:
                logger.warning(
                    "Cannot release WRITE prefill blocks for request %s: "
                    "missing remote notify address",
                    request_id,
                )
                return

        remote_notify_port = int(remote_notify_port)
        for tp_index in range(self.tp_size):
            target_port = remote_notify_port + get_port_offset(remote_dp_rank, tp_index)
            self._send_transfer_release(transfer_id, remote_host, target_port)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
        connector_worker: "MoRIIOConnectorWorker | None" = None,
    ):
        """Scheduler-side post-allocation hook (decode leg in WRITE mode).

        In WRITE mode this fires the decode->prefill "blocks ready" notify
        that lets the producer RDMA-Write its KV into the freshly allocated
        decode blocks.

        DP-rank pinning contract (override-primary / hash-failsafe)
        ----------------------------------------------------------
        Both legs of a disagg pair must agree on a single prefill DP rank,
        otherwise the notify lands on a rank that never handshook and the
        request hangs until ``VLLM_MORIIO_DEFERRED_TIMEOUT_S``.

        * Primary (production): the llm-d routing sidecar owns rank
          selection. It computes ``H = pickDPRank(uuid, dp_size)``
          (blake2s-256 of the canonical request UUID, see
          ``dp_rank.go``) and stamps both legs with ``remote_dp_rank=H``
          plus the sentinel ``remote_dp_rank_override=True``. When the
          sentinel is present we honour ``H`` verbatim and our own hash
          below never runs -- so the Go blake2s-256 vs Python blake2s-8
          divergence is irrelevant (only the sidecar's value reaches the
          wire).

        * Failsafe (sidecar-less debug, or a sidecar regression that drops
          the sentinel): when ``remote_dp_rank_override`` is absent and
          ``dp_size > 1`` we self-derive the rank from
          ``blake2s(base_rid) % dp_size`` so the two legs still converge
          on their own. ``base_rid`` strips MoRI-IO's per-transfer
          ``-<8 hex>`` suffix so it matches the rid the dispatcher hashed.

        The owning rank is also the only one that originates the notify
        (``remote_dp_rank_override`` -> global-rank match; otherwise the
        ``_is_kv_master`` anti-duplicate gate). See the inline comments
        below for the exactly-once-across-pods reasoning.
        """
        params = request.kv_transfer_params
        if not params:
            return
        # LLM-D sidecar compat: the routing-sidecar (--kv-connector=nixlv2)
        # emits NIXL-shaped kv_transfer_params that do not include MoRI-IO's
        # transfer_id. Synthesize one deterministically from request_id so the
        # producer and consumer (both downstream of the same sidecar fan-out)
        # observe the same transfer_id without requiring a wire-protocol
        # change in the sidecar.
        transfer_id = params.get("transfer_id") or f"sidecar-{request.request_id}"
        params.setdefault("transfer_id", transfer_id)
        request_id = request.request_id
        self.map_request_id(request_id, transfer_id)
        if params.get("do_remote_decode"):
            local_block_ids = blocks.get_block_ids()[0]
            self._reqs_need_save[request.request_id] = (request, local_block_ids)
            # Snapshot params now so chunked-prefill build_connector_meta
            # can recover them on the final chunk even if the live
            # request.kv_transfer_params has been mutated/cleared.
            self._req_kv_params[request.request_id] = dict(params)

        if params is not None and params.get("do_remote_prefill"):
            if self.mode == MoRIIOMode.READ:
                if remote_block_ids := params.get("remote_block_ids"):
                    # remote_engine_id is returned by the prefill's request_finished.
                    # host/ports come from the request_id (parsed in add_new_req).
                    if "remote_engine_id" in params:
                        # If remote_blocks and num_external_tokens = 0, we have
                        # a full prefix cache hit on the D worker. We need to call
                        # send_notify in _read_blocks to free the memory on the P.

                        # Get unhashed blocks to pull from remote.
                        local_block_ids = blocks.get_block_ids()[0]
                        assert len(local_block_ids) <= len(remote_block_ids)
                        if len(local_block_ids) == len(remote_block_ids):
                            pass
                        else:
                            local_block_ids = remote_block_ids[-len(local_block_ids) :]

                        self._reqs_need_recv[request.request_id] = (
                            request,
                            local_block_ids,
                        )
                        # Snapshot params for chunked prefill consumption
                        # in build_connector_meta (see comment above).
                        self._req_kv_params[request.request_id] = dict(params)
                    else:
                        logger.warning(
                            "Got invalid KVTransferParams: %s. This "
                            "request will not utilize KVTransfer",
                            params,
                        )

            else:
                # WRITE mode, decode side: notify P that blocks are ready
                assert request.kv_transfer_params is not None, (
                    "kv_transfer_params should not be None"
                )

                remote_dp_rank = request.kv_transfer_params.get("remote_dp_rank", 0)

                # Hash-route to prefill DP rank (vLLM native router fallback).
                _dp_size = int(request.kv_transfer_params.get("remote_dp_size", 1) or 1)
                try:
                    _dp_local = int(
                        request.kv_transfer_params.get("remote_dp_size_local", 0) or 0
                    )
                    if _dp_local > 0:
                        _dp_size = min(_dp_size, _dp_local)
                except (TypeError, ValueError):
                    _dp_local = 0
                # Use transfer_id for hashing (stable, not mutated).
                if (
                    _dp_size > 1
                    and "remote_dp_rank_override" not in request.kv_transfer_params
                ):
                    _hash_key = request.kv_transfer_params.get(
                        "transfer_id", request.request_id
                    )
                    _digest = hashlib.blake2s(
                        str(_hash_key).encode("utf-8"), digest_size=8
                    ).digest()
                    remote_dp_rank = int.from_bytes(_digest, "big") % _dp_size

                # Only the owning rank originates notify (exactly-once).
                # Priority: is_request_leader > remote_dp_rank_override > _is_kv_master
                _leader_flag = request.kv_transfer_params.get("is_request_leader")
                if _leader_flag is not None:
                    _should_notify = bool(_leader_flag)
                elif request.kv_transfer_params.get("remote_dp_rank_override"):
                    _should_notify = self._global_dp_rank == remote_dp_rank
                else:
                    _should_notify = self._is_kv_master
                if _should_notify:
                    peer_zmq = get_peer_zmq_from_request_id(
                        request.request_id, is_producer=False
                    )
                    if peer_zmq is not None:
                        remote_host, _, remote_notify_port = parse_moriio_zmq_address(
                            peer_zmq
                        )
                    else:
                        # Sidecar fallback: use explicit params fields.
                        params = request.kv_transfer_params or {}
                        remote_host = params.get("remote_host") or ""
                        try:
                            remote_notify_port = int(
                                params.get("remote_notify_port") or 0
                            )
                        except (TypeError, ValueError):
                            remote_notify_port = 0
                        if not remote_host or not remote_notify_port:
                            raise ValueError(
                                f"request {request.request_id!r}: "
                                f"request_id has no embedded peer "
                                f"zmq_address and kv_transfer_params is "
                                f"missing remote_host / remote_notify_port "
                                f"(got remote_host={remote_host!r}, "
                                f"remote_notify_port={remote_notify_port!r})"
                            )

                    # Wide-EP multi-pod: a pod binds notify sockets only for
                    # its LOCAL ranks, so the port offset must use the per-pod
                    # local rank (% dp_local), not the global rank. Single-pod
                    # is bit-identical (modulus is a no-op).
                    _remote_dp_rank_for_port = remote_dp_rank
                    if _dp_local > 0:
                        _remote_dp_rank_for_port = remote_dp_rank % _dp_local
                    # The target rank may live on a child pod at a different IP,
                    # so resolve the per-pod host (pod_idx = global // dp_local).
                    # Otherwise a notify for child ranks lands on the master
                    # pod and the request hangs in WAITING_FOR_REMOTE_KVS.
                    _notify_host = remote_host
                    _kvp = request.kv_transfer_params or {}
                    _remote_hosts = _kvp.get("remote_hosts") or []
                    if _dp_local > 0 and _remote_hosts:
                        _pod_idx = remote_dp_rank // _dp_local
                        if 0 <= _pod_idx < len(_remote_hosts):
                            _notify_host = _remote_hosts[_pod_idx]
                    for tp_index in range(self.tp_size):
                        target_port = remote_notify_port + get_port_offset(
                            _remote_dp_rank_for_port, tp_index
                        )

                        self.send_notify_block(
                            req_id=request.request_id,
                            transfer_id=request.kv_transfer_params["transfer_id"],
                            block_notify_list=blocks.get_block_ids()[0],
                            host=_notify_host,
                            port=target_port,
                        )

            # Only trigger 1 KV transfer per request.

            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MoRIIOConnectorMetadata()
        meta.transfer_id_to_request_id = self.transfer_id_to_request_id

        if self.mode == MoRIIOMode.WRITE and get_role() == ROLE.PRODUCER:
            # This is the logic for checking against chunked prefill.
            # When the last chunk is identified,
            # It places the request metadata into the saving queue.

            for i, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
                new_block_ids = scheduler_output.scheduled_cached_reqs.new_block_ids[i]

                if new_block_ids is not None:
                    block_ids = new_block_ids[0]
                    # TODO : hybrid attn, etc
                    # A non-disagg request (no kv_transfer_params, e.g. smoke
                    # test) is never registered in _reqs_need_pending_save;
                    # indexing it unconditionally would KeyError and crash the
                    # EngineCore. Skip it silently.
                    if req_id not in self._reqs_need_pending_save:
                        continue
                    req, existing_blocks = self._reqs_need_pending_save[req_id]
                    updated_blocks = list(existing_blocks) + (block_ids)
                    self._reqs_need_pending_save[req_id] = (req, updated_blocks)
                    if (
                        len(self._reqs_need_pending_save[req_id][1]) * self.block_size
                        >= req.num_prompt_tokens
                    ):
                        # Final chunk: live kv_transfer_params may be cleared,
                        # so prefer the snapshot from update_state_after_alloc.
                        kv_params = self._req_kv_params.pop(
                            req_id, req.kv_transfer_params or {}
                        )
                        meta.add_new_req(
                            request_id=req_id,
                            local_block_ids=self._reqs_need_pending_save[req_id][1],
                            kv_transfer_params=kv_params,
                            write_mode=True,
                        )
                        del self._reqs_need_pending_save[req_id]

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            kv_params = self._req_kv_params.get(req_id, req.kv_transfer_params or {})
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=kv_params,
            )

        for req_id, (req, block_ids) in self._reqs_need_save.items():
            kv_params = self._req_kv_params.get(req_id, req.kv_transfer_params or {})
            if req.num_prompt_tokens > len(block_ids) * self.block_size:
                # not last chunk prefill
                self._reqs_need_pending_save[req_id] = (req, block_ids)
                continue
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=kv_params,
                write_mode=True,
            )
        # Clear the list once workers start the transfers

        meta.reqs_to_send = self._reqs_need_send

        # Reclaim snapshot cache entries that completed this step. Keep
        # entries that are still pending (chunked prefill not yet at the
        # final chunk) — those will be popped above on the final chunk.
        for req_id in self._reqs_need_recv:
            self._req_kv_params.pop(req_id, None)
        for req_id in self._reqs_need_save:
            if req_id not in self._reqs_need_pending_save:
                self._req_kv_params.pop(req_id, None)

        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        self._reqs_need_send = {}

        return meta

    def shutdown(self):
        for path, sock in self.paths.items():
            try:
                sock.close(linger=0)
                logger.debug("Closed ZMQ socket for path: %s", path)
            except Exception as e:
                logger.warning("Error closing ZMQ socket for path %s: %s", path, e)
        self.paths.clear()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        request_id = request.request_id
        params = request.kv_transfer_params
        # Consumer: can unmap transfer_id<->request_id immediately since done_recving
        #   has fired at this point (i.e. KV has been transferred)
        # Producer: must keep the mapping until we get notification that blocks can
        #   be freed, which may be several scheduler steps later.
        if not self.is_producer:
            transfer_id = params.get("transfer_id") if params else None
            self.unmap_request_id(request_id, transfer_id=transfer_id)
        logger.debug(
            "MoriioConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s",
            request.status,
            params,
        )
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # READ mode adds empty block_ids to _reqs_need_recv so the worker
            # side notifies the prefill instance. WRITE mode should notify the
            # producer directly: there is no decode allocation for the producer
            # to write into, and a plain request_id may not contain router-
            # embedded MoRIIO ZMQ addresses.
            if self.mode == MoRIIOMode.WRITE:
                self._release_write_prefill_blocks(request.request_id, params)
            else:
                self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (
            not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        # computed_block_ids = block_ids if all_full else block_ids[:-1]
        computed_block_ids = block_ids
        # If prompt < block_size, no xfer so free blocks immediately.
        delay_free_blocks = len(computed_block_ids) > 0

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            self._reqs_need_send[request.request_id] = (
                time.perf_counter()
                + MoRIIOConstants.VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT
            )
            self._deferred_send_deadlines[request.request_id] = (
                time.monotonic() + self._defer_timeout,
                params.get("transfer_id") if params else None,
            )

        # Return KV transfer params forwarded verbatim to the decode instance by
        # the router.
        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.host_ip,
            remote_handshake_port=self.handshake_port,
            remote_notify_port=self.side_notify_port,
            remote_dp_size=self.vllm_config.parallel_config.data_parallel_size,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            transfer_id=params["transfer_id"],
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Reconcile worker finished_sending ACKs with the producer lifecycle.

        Called every scheduler step (when there is KV connector output),
        BEFORE the scheduler's finished_sending free loop. We rewrite
        ``connector_output.finished_sending`` in place so it contains
        exactly the producer requests that are safe for the scheduler to
        free this step, letting the scheduler keep the plain upstream
        ``assert req_id in self.requests`` + ``_free_blocks`` path:

        * An ACK whose request is in ``_deferred_send_deadlines``
          (request_finished ran with delay_free_blocks=True, so the
          scheduler is holding its blocks) is surfaced -> freed now.
        * An ACK that arrives BEFORE its request finished is parked in
          ``_pending_sent_acks`` and released on a later step once the
          request enters ``_deferred_send_deadlines``.
        * A deferred send whose ACK never arrives is reaped after
          ``_defer_timeout`` and surfaced, so leaked blocks are freed.
        * A parked ACK that never matches a deferral before its own
          deadline is a stale duplicate (e.g. a real ACK landing after the
          send was already reaped) and is dropped.

        Consumers never populate finished_sending (they report
        finished_recving), and they unmap in request_finished, so this is a
        no-op for them.
        """
        if not self.is_producer:
            return

        incoming = set(connector_output.finished_sending or ())
        now = time.monotonic()

        # Surface ACKs whose request is already finished (blocks held for
        # delayed free); park the rest until their request finishes.
        safe: set[ReqId] = set()
        for req_id in incoming:
            if req_id in self._deferred_send_deadlines:
                safe.add(req_id)
            else:
                self._pending_sent_acks.setdefault(req_id, now + self._defer_timeout)

        # Release previously parked ACKs whose request has since finished.
        for req_id in self._pending_sent_acks:
            if req_id in self._deferred_send_deadlines:
                safe.add(req_id)

        # Reap deferred sends whose ACK never arrived (avoid leaking blocks).
        expired = [
            req_id
            for req_id, (deadline, _) in self._deferred_send_deadlines.items()
            if now >= deadline
        ]
        if expired:
            safe.update(expired)
            logger.warning(
                "Reaped %d deferred sends with no finished_sending "
                "notification after %.0fs. This indicates lost async KV "
                "completion notifications from the KV connector.",
                len(expired),
                self._defer_timeout,
            )

        # Finalize the requests we are surfacing: drop their deferral/park
        # bookkeeping and unmap their transfer ids.
        for req_id in safe:
            deferred_info = self._deferred_send_deadlines.pop(req_id, None)
            transfer_id = deferred_info[1] if deferred_info else None
            self._pending_sent_acks.pop(req_id, None)
            self.unmap_request_id(req_id, transfer_id=transfer_id)

        # Drop stale parked ACKs that never matched a deferral in time.
        stale = [
            req_id
            for req_id, deadline in self._pending_sent_acks.items()
            if now >= deadline
        ]
        for req_id in stale:
            self._pending_sent_acks.pop(req_id, None)

        connector_output.finished_sending = safe or None


class MoRIIOConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        if not is_moriio_available():
            raise RuntimeError(
                "MoRIIO is not available. Please ensure the 'mori' package "
                "is installed and properly configured."
            )

        assert vllm_config.kv_transfer_config is not None
        self.moriio_config = MoRIIOConfig.from_vllm_config(vllm_config)
        self.mode = (
            MoRIIOMode.READ if self.moriio_config.read_mode else MoRIIOMode.WRITE
        )

        logger.info("Initializing MoRIIO worker %s", engine_id)

        logging.getLogger("aiter").disabled = True

        # Config.
        self.vllm_config = vllm_config
        assert vllm_config.kv_transfer_config is not None, (
            "kv_transfer_config must be set for MoRIIOConnector"
        )
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.is_producer = self.kv_transfer_config.is_kv_producer
        self.layer_to_spec = build_layer_to_spec(kv_cache_config)

        if self.is_producer:
            set_role(ROLE.PRODUCER)
        else:
            set_role(ROLE.CONSUMER)
        # mori engine
        self._rank = get_world_group().rank
        self._local_rank = get_world_group().local_rank
        self.tp_rank = self.moriio_config.tp_rank
        self.dp_rank = self.moriio_config.dp_rank

        self.local_ip = self.moriio_config.local_ip
        self.local_kv_port = self.moriio_config.local_kv_port
        self.proxy_ip = self.moriio_config.proxy_ip
        self.local_ping_port = self.moriio_config.local_ping_port
        self.proxy_ping_port = self.moriio_config.proxy_ping_port
        self.http_port = self.moriio_config.http_port
        self.handshake_port = self.moriio_config.handshake_port
        self.notify_port = self.moriio_config.notify_port

        self.zmq_context = zmq.Context()
        self.metadata_address = (
            f"{self.moriio_config.local_ip}:{self.moriio_config.local_ping_port}"
        )
        self.request_address = (
            f"{self.moriio_config.local_ip}:{self.moriio_config.http_port}"
        )

        self.moriio_engine = None
        self._handle_request_thread = None
        self._ping_thread = None
        self._writer = MoRIIOWriter(self)
        # Completions that arrived before transfer_id_to_request_id was populated.
        # Retried each step until the mapping is established.
        self._unmatched_write_completions: set[str] = set()
        # Producer-side READ-mode ACK fan-in. When decode TP is larger than
        # prefill TP, multiple decode ranks can read from one prefill rank and
        # notify the same transfer_id. Blocks are reusable only after all ACKs.
        self._consumer_notification_counts: dict[TransferId, int] = {}
        self._completed_consumer_notifications: set[TransferId] = set()

        role = "producer" if self.is_producer else "consumer"
        engine_suffix = (
            f"{self.moriio_config.local_ip}:{self.moriio_config.handshake_port}:"
            f"tp{self.tp_rank}:dp{self.dp_rank}"
        )
        self.moriio_engine = IOEngine(
            f"{role}:{engine_suffix}",
            IOEngineConfig(
                self.moriio_config.local_ip, self.moriio_config.local_kv_port
            ),
        )
        logger.debug(
            "build MORI IOEngine %s (ip=%s port=%s)",
            f"{role}:{engine_suffix}",
            self.moriio_config.local_ip,
            self.moriio_config.local_kv_port,
        )

        if self._rank == 0 and self.moriio_config.proxy_ip:
            self._ping_thread = threading.Thread(
                target=self._ping, args=(self.zmq_context,), daemon=True
            )
            self._ping_thread.start()

        logger.info(
            "Initializing MoRIIO Engine, engine = %s, role = %s",
            self.moriio_engine,
            "producer" if self.is_producer else "consumer",
        )

        # Agent.
        self.moriio_wrapper = MoRIIOWrapper(
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
            transfer_timeout=self.moriio_config.transfer_timeout,
        )
        self.moriio_wrapper.set_moriio_engine(self.moriio_engine)
        backend = (
            BackendType.XGMI
            if self.moriio_config.backend == "xgmi"
            else BackendType.RDMA
        )
        self.moriio_wrapper.set_backend_type(
            backend,
            qp_per_transfer=self.moriio_config.qp_per_transfer,
            post_batch_size=self.moriio_config.post_batch_size,
            num_workers=self.moriio_config.num_workers,
        )
        self.moriio_wrapper.notify_port = self.moriio_config.notify_port
        self.local_kv_cache_metadata: list[bytes] = []
        self.local_kv_cache_size: list[int] = []
        self.layer_name_to_local_kv_cache_metadata: dict[str, list[bytes]] = {}

        self.remote_kv_cache_metadata: list[bytes] = []
        self.remote_kv_cache_size: list[int] = []
        self.layer_name_to_remote_kv_cache_metadata: dict[str, dict[str, list[Any]]] = (
            dict()
        )
        self.remote_moriio_metadata: dict[EngineId, MoRIIOAgentMetadata] = {}
        self.slot_size_bytes = 0

        self.load_ready_flag: dict[str, bool] = {}
        self.write_ready_flags: dict[str, bool] = {}
        self.kv_cache_shape = None
        self.block_shape = None
        self.kv_element_size = 0
        self.kv_cache_shapes: dict[str, torch.Size] = {}
        self.block_lens: dict[str, int] = {}

        # Map of engine_id -> {agent_name0, agent_name1..}.
        self._remote_agents: dict[EngineId, set[str]] = {}

        self.side_channel_port: int = (
            self.moriio_config.handshake_port
            + get_port_offset(self.dp_rank, self.tp_rank)
        )
        self.engine_id: EngineId = engine_id

        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        # KV Caches and moriio tracking data.
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

        # Number of MoRIIO regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = 0

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        # In progress transfers.
        self._recving_transfers: defaultdict[ReqId, list] = defaultdict(list)
        # Values are (remote_host, remote_notify_port, transfer_id).
        self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str, str]] = {}
        # Monotonic-clock start times for each in-flight recv transfer.
        # Used by _pop_done_transfers to abort transfers whose RDMA
        # completion is lost, instead of hanging forever.
        self._recving_transfers_start: dict[str, float] = {}

        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}

        # Background thread for handling new handshake requests.
        self._moriio_handshake_listener_t: threading.Thread | None = None
        # Background thread for initializing new MoRIIO handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # MoRIIO is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-moriio-handshake-initiator",
        )
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[set[str]]] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        self.block_window_per_layer: list[int | None] = []
        self.use_mla = self.model_config.use_mla
        self.built_session = False
        self.built_write_session: defaultdict[str, list] = defaultdict(list)
        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            use_mla=self.use_mla,
        )
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}

        # TODO: consider the integration of flashinfer or other backends.
        self.backend_name = backend.get_name()
        logger.debug("Detected attention backend %s", self.backend_name)

    def schedule_write_blocks(
        self,
        request_id: ReqId,
        transfer_id: TransferId,
        dst_engine_id: str,
        local_block_ids: list[int],
        remote_block_ids: list[int] | None,
        layer_name: str,
        kv_layer: torch.Tensor,
        remote_notify_port: int,
        remote_ip: str,
    ) -> None:
        """Schedule a block write operation.

        Args:
            request_id: Unique identifier for the request
            transfer_id: Unique identifier for the transfer
            dst_engine_id: Destination engine ID
            local_block_ids: Local block IDs to transfer
            remote_block_ids: Hint for remote block IDs
            layer_name: Name of the layer
            kv_layer: KV cache tensor
            remote_notify_port: Port for completion notification
            remote_ip: IP address of remote node
        """

        # synchronization to prevent dirty reads between
        # transfer and attention operations
        # we can consider removing this synchronization after ibgda is enabled.
        # when mori-io supports ibgda functionality

        stream = torch.cuda.current_stream()
        event = torch.Event()
        event.record(stream)

        task = WriteTask(
            request_id=request_id,
            transfer_id=transfer_id,
            dst_engine_id=dst_engine_id,
            local_block_ids=local_block_ids,
            remote_block_ids_hint=remote_block_ids,
            layer_name=layer_name,
            event=event,
            remote_notify_port=remote_notify_port,
            remote_ip=remote_ip,
        )
        self._writer.schedule_write(task)

    def _get_built_session(self, remote_engine_id):
        if remote_engine_id not in self.built_write_session:
            cur_remote_engine_sessions = []
            for ln, local_meta in self.layer_name_to_local_kv_cache_metadata.items():
                unpacked_local_memory_meta = (
                    self.moriio_wrapper.get_unpack_memory_metadata(local_meta[0])
                )
                unpacked_remote_memory_meta = (
                    self.moriio_wrapper.get_unpack_memory_metadata(
                        self.layer_name_to_remote_kv_cache_metadata[remote_engine_id][
                            ln
                        ][0]
                    )
                )
                cur_remote_engine_sessions.append(
                    self.moriio_wrapper.build_session(
                        unpacked_local_memory_meta, unpacked_remote_memory_meta
                    )
                )
            self.built_write_session[remote_engine_id] = cur_remote_engine_sessions
        return self.built_write_session[remote_engine_id], self.remote_moriio_metadata[
            remote_engine_id
        ]

    def _ping(self, zmq_context):
        # Use host:port format for http_address (compatible with official router)
        http_address = f"{self.request_address}"
        # Include host so the router embeds it in the request_id; the connector
        # on the other side parses host/ports from there.
        zmq_address = (
            f"host:{self.local_ip},"
            f"handshake:{self.handshake_port},"
            f"notify:{self.notify_port}"
        )
        role = "P" if self.is_producer else "D"

        retry_count = 0
        index = 1
        with zmq_context.socket(zmq.DEALER) as sock:
            sock.connect(f"tcp://{self.proxy_ip}:{self.proxy_ping_port}")

            while True:
                try:
                    data = {
                        "type": role,  # "P" or "D"
                        "http_address": http_address,
                        "zmq_address": zmq_address,
                        # dp_size/tp_size are not used by the official vLLM router
                        # (routing operates at the http_address level); they are
                        # consumed only by the toy proxy server.
                        "dp_size": self.moriio_config.dp_size,
                        "tp_size": self.moriio_config.tp_size,
                        # transfer_mode is included so the router can distinguish
                        # READ (prefill-then-decode, sequential) from WRITE (concurrent)
                        # scheduling.
                        "transfer_mode": self.mode.name,
                    }

                    sock.send(msgpack.dumps(data))
                    # logger.debug(f"Successfully sent ping message #{index}")
                    retry_count = 0

                except ConnectionRefusedError:
                    logger.info(
                        "Connection refused: %s:%s -> %s:%s",
                        self.local_ip,
                        self.local_ping_port,
                        self.proxy_ip,
                        self.proxy_ping_port,
                    )
                    retry_count += 1

                except OSError as e:
                    logger.info("OS error when sending ping: %s", e)
                    retry_count += 1

                except Exception as e:
                    logger.info("Unexpected error when sending ping: %s", e)
                    retry_count += 1
                    if retry_count >= MoRIIOConstants.MAX_PING_RETRIES:
                        logger.error(
                            "Max retries (%s) exceeded. Stopping ping loop.",
                            MoRIIOConstants.MAX_PING_RETRIES,
                        )
                        raise RuntimeError(
                            f"Ping failed after {retry_count} retries"
                        ) from e

                finally:
                    time.sleep(MoRIIOConstants.PING_INTERVAL)
                    index += 1

    def shutdown(self):
        if hasattr(self, "moriio_wrapper") and self.moriio_wrapper:
            self.moriio_wrapper.shutdown()

        if hasattr(self, "_handshake_initiation_executor"):
            self._handshake_initiation_executor.shutdown(wait=False)

        if (
            hasattr(self, "_moriio_handshake_listener_t")
            and self._moriio_handshake_listener_t
        ):
            self._moriio_handshake_listener_t.join(timeout=0)

        if hasattr(self, "zmq_context") and self.zmq_context:
            self.zmq_context.destroy(linger=0)
            self.zmq_context = None

    def __del__(self):
        self.shutdown()

    @staticmethod
    def _moriio_handshake_listener(
        metadata: MoRIIOAgentMetadata,
        ready_event: threading.Event,
        base_port: int,
        tp_rank: int,
        dp_rank: int,
        layer_name_to_local_kv_cache_metadata: dict,
    ):
        """Background thread for getting new MoRIIO handshakes."""

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug(
            "Size of encoded MoRIIOAgentMetadata: %s bytes", str(size_in_bytes)
        )

        # Listen for new requests for metadata.
        host = "*"

        path = make_zmq_path("tcp", host, base_port)
        logger.debug("mori handshake starting listening on path: %s", path)

        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, msg = sock.recv_multipart()
                if (
                    msg != MoRIIOConstants.GET_META_MSG
                    and msg != MoRIIOConstants.POP_DONE_RECV
                ):
                    logger.error("Connection listener got unexpected message")
                    raise HandshakeError("handshake failed, unexpected msg type")
                elif msg == MoRIIOConstants.GET_META_MSG:
                    sock.send_multipart(
                        (identity, b"", encoded_data)
                    )  # send local mori io engine meta data
                    logger.debug("MoRIIO handshake listener sent metadata")
                    # now we send tensor meta data for each block
                    buf = msgpack.dumps(layer_name_to_local_kv_cache_metadata)
                    sock.send_multipart((identity, b"", buf))
                elif msg == MoRIIOConstants.POP_DONE_RECV:
                    _, req_id = sock.recv_multipart()
                    logger.debug(
                        "MoRIIO handshake listener received done recv for req",
                        req_id.decode(),
                    )

    def _moriio_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dp_rank: int = 0,
    ) -> set[str]:
        """Do a MoRIIO handshake with a remote instance."""

        start_time = time.perf_counter()

        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.

        port_offset = get_port_offset(
            remote_dp_rank, self._remote_tp_rank(remote_tp_size)
        )
        path = make_zmq_path("tcp", host, port + port_offset)
        logger.debug("handshake Querying metadata on path: %s", path)

        # Send query for the request.
        with zmq_ctx(zmq.DEALER, path) as sock:
            logger.debug("prepare send msg INSTAZNCE: %s", path)
            sock.send(MoRIIOConstants.GET_META_MSG)
            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                raise HandshakeError(f"Unexpected frame! {received_frame = }")

            metadata_bytes = received_frame[1]
            decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.info(
                "MoRIIO handshake: get metadata took: %s",
                got_metadata_time - start_time,
            )

            self.moriio_wrapper.remote_engine_ip = host
            remote_agent_name = self.moriio_wrapper.register_remote_engine(
                metadata.agent_metadata
            )

            logger.debug(
                "MoRIIO handshake: registered"
                "remote agent %s for engine ID %s, path = %s",
                remote_agent_name,
                expected_engine_id,
                path,
            )

            if len(self.local_kv_cache_metadata) > 0:
                logger.warning(
                    "len(self.local_kv_cache_metadata) = %s,"
                    "maybe you didnt clear this buffer correctly",
                    len(self.local_kv_cache_metadata),
                )
                self.local_kv_cache_metadata = []
            if len(self.remote_kv_cache_metadata) > 0:
                logger.warning(
                    "len(self.remote_kv_cache_metadata) = %s,"
                    "maybe you didnt clear this buffer correctly",
                    len(self.remote_kv_cache_metadata),
                )
                self.remote_kv_cache_metadata = []

            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                raise HandshakeError(f"unexpected frame! {received_frame = }")
            buf = received_frame[1]
            self.layer_name_to_remote_kv_cache_metadata[expected_engine_id] = (
                msgpack.loads(buf)
            )
            self.remote_moriio_metadata[expected_engine_id] = metadata
            setup_agent_time = time.perf_counter()
            logger.debug(
                "MoRIIO handshake: add agent took: %s",
                setup_agent_time - got_metadata_time,
            )

        return {remote_agent_name}

    def _remote_tp_rank(self, remote_tp_size: int) -> int:
        return get_moriio_remote_tp_rank(self.tp_rank, self.world_size, remote_tp_size)

    def _background_moriio_handshake(
        self, req_id: ReqId, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do MoRIIO handshake in background and add to _ready_requests when done.
        fut = None
        if remote_engine_id is not None:
            fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            host = meta.remote_host
            port = int(meta.remote_handshake_port)
            tp_size = int(meta.tp_size)
            remote_dp_size = int(meta.remote_dp_size)
            # Wide-EP multi-pod: remote DP ranks span pods at different IPs
            # (ranks per pod = dp_local), so resolve the host per cur_dp_rank
            # below instead of using a single host for all ranks.
            pod_hosts = list(meta.multi_pod_hosts) if meta.multi_pod_hosts else [host]
            remote_dp_size_local = int(meta.remote_dp_size_local) or remote_dp_size

        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            logger.info("MoRIIO handshake done for request %s", req_id)
            self._ready_requests.put(entry)
            self.load_ready_flag[remote_engine_id] = True
            self.write_ready_flags[remote_engine_id] = True

        fut_list = []

        # In dp(prefill)<->dp(decode) communication, we require an all-to-all handshake.

        for cur_dp_rank in range(remote_dp_size):
            dp_engine_id = self.get_engine_name_with_dp(remote_engine_id, cur_dp_rank)
            _pod_idx = (
                cur_dp_rank // remote_dp_size_local if remote_dp_size_local > 0 else 0
            )
            if _pod_idx >= len(pod_hosts):
                _pod_idx = 0
            _per_rank_host = pod_hosts[_pod_idx]
            # The handshake port offset must use the per-pod local rank
            # (% remote_dp_size_local), since each pod binds sockets only for
            # its local ranks; dp_engine_id keeps the global rank for
            # uniqueness. Single-pod is bit-identical (modulus is a no-op).
            _per_rank_local_dp = (
                cur_dp_rank % remote_dp_size_local
                if remote_dp_size_local > 0
                else cur_dp_rank
            )
            future = self._handshake_initiation_executor.submit(
                self._moriio_handshake,
                _per_rank_host,
                port,
                tp_size,
                dp_engine_id,
                _per_rank_local_dp,
            )
            fut_list.append(future)

            def done_callback(f: Future[set[str]], eid=dp_engine_id):
                with self._handshake_lock:
                    self._handshake_futures.pop(eid, None)
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            future.add_done_callback(done_callback)
            self._handshake_futures[dp_engine_id] = future

        # fut = fut_list
        def wait_all_dp():
            for future in fut_list:
                future.result()
            return True

        all_done_future = self._handshake_initiation_executor.submit(wait_all_dp)
        all_done_future.add_done_callback(request_ready)

    def _is_mla_cache_layer(self, layer_name: str) -> bool:
        return is_mla_cache_layer(self.layer_to_spec, layer_name)

    def _get_layer_transfer_geometry(
        self, layer_name: str, remote_num_blocks: int | None = None
    ) -> LayerTransferGeometry:
        return get_layer_transfer_geometry(
            layer_name,
            self.kv_caches[layer_name],
            self.layer_to_spec,
            remote_num_blocks,
        )

    def _iter_layer_registration_regions(
        self, layer_name: str
    ) -> list[tuple[torch.Tensor, int]]:
        return iter_layer_registration_regions(
            layer_name,
            self.kv_caches[layer_name],
            self.layer_to_spec,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in moriio."""

        self.kv_caches = kv_caches  # layer name to kv cache
        self.kv_cache_shapes = {
            layer_name: kv_cache.shape for layer_name, kv_cache in kv_caches.items()
        }

        first_layer_name, first_kv_cache = next(
            (
                (layer_name, kv_cache)
                for layer_name, kv_cache in kv_caches.items()
                if (
                    not self._is_mla_cache_layer(layer_name)
                    and len(kv_cache.shape) == 5
                    and (kv_cache.shape[0] == 2 or kv_cache.shape[1] == 2)
                )
            ),
            next(iter(kv_caches.items())),
        )
        kv_elem_size = first_kv_cache.element_size()

        use_mla = self._is_mla_cache_layer(first_layer_name)
        first_geometry = self._get_layer_transfer_geometry(first_layer_name)

        if use_mla:
            # MLA case.
            block_rank = 2  # [block_size, latent_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
        else:
            # [2, num_blocks, ...] or [num_blocks, 2, ...]
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
        self.num_blocks = first_geometry.num_blocks
        self.slot_size_bytes = first_geometry.slot_size_bytes
        if first_geometry.block_size != self.block_size:
            # DeepSeek-V3 / MLA backends (e.g. FlashMLA) override the
            # configured block_size at attention-layer creation time, so
            # the KV cache tensor is laid out with a different (usually
            # larger) block_size than vllm_config.cache_config.block_size.
            # Trust the actual tensor shape and reconcile.
            logger.info(
                "KV cache block_size=%d differs from config block_size=%d; "
                "using actual tensor shape (attention backend override).",
                first_geometry.block_size,
                self.block_size,
            )
            self.block_size = first_geometry.block_size
        # TODO(tms): self.block_len needs to be per-layer for sliding window,
        # hybrid attn, etc
        # block size in bytes
        self.block_len = first_geometry.block_len
        self.kv_cache_shape = first_kv_cache.shape
        self.block_shape = block_shape
        self.kv_element_size = kv_elem_size

        self.dst_num_blocks[self.engine_id] = self.num_blocks
        kv_caches_base_addr = []
        caches_data = []

        for layer_name in kv_caches:
            geometry = self._get_layer_transfer_geometry(layer_name)
            if geometry.block_size != self.block_size:
                raise ValueError(
                    "MoRIIO KV cache block size mismatch for layer "
                    f"{layer_name}: {geometry.block_size} != {self.block_size}"
                )
            self.block_lens[layer_name] = geometry.block_len
            for cache, region_len in self._iter_layer_registration_regions(layer_name):
                base_addr = cache.data_ptr()
                caches_data.append((base_addr, region_len, cache.device.index, ""))
                kv_caches_base_addr.append(base_addr)

        for layer_name, kv_cache in kv_caches.items():
            if layer_name not in self.layer_name_to_local_kv_cache_metadata:
                self.layer_name_to_local_kv_cache_metadata[layer_name] = []

            moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(kv_cache)
            self.layer_name_to_local_kv_cache_metadata[layer_name].append(
                moriio_mem_metadata
            )

            self.local_kv_cache_size.append(
                kv_cache.nelement() * kv_cache.element_size()
            )

        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
        self.num_regions = len(caches_data)
        self.num_layers = len(self.kv_caches.keys())

        # Optimization for models with local attention (Llama 4)
        if self.vllm_config.model_config.hf_config.model_type == "llama4":
            from transformers import Llama4TextConfig

            assert isinstance(
                self.vllm_config.model_config.hf_text_config, Llama4TextConfig
            )
            llama4_config = self.vllm_config.model_config.hf_text_config
            no_rope_layers = llama4_config.no_rope_layers
            chunk_size = llama4_config.attention_chunk_size
            chunk_block_size = math.ceil(chunk_size / self.block_size)
            for layer_idx in range(self.num_layers):
                # no_rope_layers[layer_idx] == 0 means NoPE (global)
                # Any other value means RoPE (local chunked)
                is_local_attention = no_rope_layers[layer_idx] != 0
                block_window = chunk_block_size if is_local_attention else None
                self.block_window_per_layer.append(block_window)
            logger.debug(
                "Llama 4 block window per layer mapping: %s",
                self.block_window_per_layer,
            )
            assert len(self.block_window_per_layer) == self.num_layers

        metadata = MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name=self.backend_name,
        )
        ready_event = threading.Event()
        self._moriio_handshake_listener_t = threading.Thread(
            target=self._moriio_handshake_listener,
            args=(
                metadata,
                ready_event,
                self.side_channel_port,
                self.tp_rank,
                self.dp_rank,
                self.layer_name_to_local_kv_cache_metadata,
            ),
            daemon=True,
            name="moriio_handshake_listener",
        )
        self._moriio_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.
        self.moriio_wrapper.async_wait_reqid()

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """

        done_sending, done_recving = set(), set()

        if self.is_producer:
            # pop_finished_req_ids returns release ACKs sent by decode. Keep
            # duplicate ACKs because heterogeneous TP can fan multiple decode
            # ranks into one prefill rank for the same transfer_id.
            finished_acks = self.moriio_wrapper.pop_finished_req_ids()
            resolved_transfer_ids: set[TransferId] = set()
            for ack in finished_acks:
                transfer_id = ack if isinstance(ack, str) else ack.transfer_id
                if transfer_id not in self.transfer_id_to_request_id:
                    logger.warning(
                        "Could not find %s in transfer_id_to_request_id "
                        "lookup table. This could lead to a possible hang.",
                        transfer_id,
                    )
                    continue
                resolved_transfer_id = resolve_moriio_transfer_ack(
                    ack,
                    producer_tp_size=self.world_size,
                    live_transfer_ids=self.transfer_id_to_request_id.keys(),
                    notification_counts=self._consumer_notification_counts,
                    completed_transfer_ids=(self._completed_consumer_notifications),
                )
                if resolved_transfer_id is not None:
                    resolved_transfer_ids.add(resolved_transfer_id)
            done_sending = {
                self.transfer_id_to_request_id[xfer_id]
                for xfer_id in resolved_transfer_ids
            }
        else:
            if self.mode == MoRIIOMode.WRITE:
                fresh = self.moriio_wrapper.pop_finished_write_req_ids()
                # Accumulate with any completions that arrived before their
                # transfer_id was registered in transfer_id_to_request_id.
                self._unmatched_write_completions |= fresh
                done_recving = self._unmatched_write_completions
            else:
                # READ mode: the scheduler treats KV loads as synchronous
                # (load_kv_async=False), so requests go directly to RUNNING
                # instead of WAITING_FOR_REMOTE_KVS. We still call
                # _pop_done_transfers() to send the notify to the prefill
                # side and clean up internal state, but we must NOT report
                # these as done_recving because the scheduler doesn't
                # expect a finished_recving signal for RUNNING requests.
                self._pop_done_transfers()

        done_recving = {
            self.transfer_id_to_request_id[id]
            for id in filter(
                lambda id: id in self.transfer_id_to_request_id, done_recving
            )
        }
        if self.mode == MoRIIOMode.WRITE and not self.is_producer:
            # Remove the ones we successfully matched; leave unmatched for retry.
            matched_xfer_ids = {
                id
                for id in self._unmatched_write_completions
                if id in self.transfer_id_to_request_id
            }
            self._unmatched_write_completions -= matched_xfer_ids

        return done_sending, done_recving

    def _pop_done_transfers(self) -> set[str]:
        done_req_ids: set[str] = set()
        _xfer_timeout = int(os.environ.get("VLLM_MORIIO_TRANSFER_TIMEOUT_S", "120"))
        with self.moriio_wrapper.lock:
            to_remove = []
            for req_id, status_list in self._recving_transfers.items():
                last = status_list[-1]
                if last.Succeeded():
                    host, port, xfer_id = self._recving_transfers_callback_addr[req_id]
                    done_req_ids.add(xfer_id)
                    self.moriio_wrapper.send_notify(
                        xfer_id,
                        host,
                        port,
                        message_type="release",
                        message_fields={"consumer_tp_size": self.world_size},
                    )
                    to_remove.append(req_id)
                elif last.Failed():
                    logger.error(
                        "RDMA transfer failed for request %s: %s (code=%s). "
                        "Notifying prefill to free blocks; request will be "
                        "aborted by timeout.",
                        req_id,
                        last.Message(),
                        last.Code(),
                    )
                    host, port, xfer_id = self._recving_transfers_callback_addr[req_id]
                    try:
                        self.moriio_wrapper.send_notify(
                            xfer_id,
                            host,
                            port,
                            message_type="release",
                            message_fields={"consumer_tp_size": self.world_size},
                        )
                    except Exception:
                        logger.exception(
                            "Failed to send error notification for request %s",
                            req_id,
                        )
                    to_remove.append(req_id)
                    # Do NOT add to done_req_ids: decode KV cache is incomplete.
                    # The request will expire via the normal request timeout.
                elif req_id in self._recving_transfers_start:
                    # Abort still-in-flight transfers that exceed the
                    # configured deadline. Otherwise a lost RDMA
                    # completion would leave the decode worker hung
                    # indefinitely on this request.
                    _age = time.monotonic() - self._recving_transfers_start[req_id]
                    if _age > _xfer_timeout:
                        logger.error(
                            "RDMA read TIMED OUT for req %s after %.1fs "
                            "(VLLM_MORIIO_TRANSFER_TIMEOUT_S=%d)",
                            req_id,
                            _age,
                            _xfer_timeout,
                        )
                        to_remove.append(req_id)
            for req_id in to_remove:
                del self._recving_transfers[req_id]
                del self._recving_transfers_callback_addr[req_id]
                self._recving_transfers_start.pop(req_id, None)

            return done_req_ids

    def save_kv_layer(
        self,
        metadata: MoRIIOConnectorMetadata,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata | None",
        **kwargs,
    ):
        if not self.is_producer:
            return
        if self.mode == MoRIIOMode.READ:
            return
        remote_engine_id = None

        for req_id, meta in metadata.reqs_to_save.items():
            # we only need to check if dp0 in rank
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )

            meta.remote_engine_id = remote_engine_id

            dp0_remote_engine_id = self.get_engine_name_with_dp(remote_engine_id, 0)
            if dp0_remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta
                        )

                        continue
            self._write_blocks_for_req(req_id, meta, layer_name, kv_layer)

        if remote_engine_id is None:
            return
        _deadline = time.monotonic() + self.moriio_config.transfer_timeout
        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.write_ready_flags
            ):
                if time.monotonic() > _deadline:
                    logger.warning(
                        "Timed out waiting for write_ready_flags[%s]; "
                        "adjust with kv_connector_extra_config.transfer_timeout",
                        remote_engine_id,
                    )
                    break
                time.sleep(0.001)
                continue
            elif not self._ready_requests.empty() and (
                remote_engine_id in self.write_ready_flags
            ):
                self._write_blocks_for_req(
                    *self._ready_requests.get_nowait(), layer_name, kv_layer
                )
                break
            else:
                break

    def get_engine_name_with_dp(self, engine_name, dp_rank):
        return f"{engine_name}_dp{dp_rank}"

    def start_load_kv(self, metadata: MoRIIOConnectorMetadata):
        """
        Start loading by triggering non-blocking moriio_xfer.
        We check for these trnxs to complete in each step().
        """
        self.transfer_id_to_request_id = metadata.transfer_id_to_request_id
        if self.is_producer:
            live_transfer_ids = set(self.transfer_id_to_request_id)
            self._consumer_notification_counts = {
                transfer_id: count
                for transfer_id, count in self._consumer_notification_counts.items()
                if transfer_id in live_transfer_ids
            }
            self._completed_consumer_notifications.intersection_update(
                live_transfer_ids
            )
            self.moriio_wrapper.async_wait_reqid()
            return
        if self.mode == MoRIIOMode.WRITE:
            return

        wait_handshake_readd_req = False
        remote_engine_id = None

        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )
            meta.remote_engine_id = remote_engine_id
            dp0_remote_engine_id = self.get_engine_name_with_dp(remote_engine_id, 0)
            if dp0_remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta
                        )
                        wait_handshake_readd_req = True

                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)
        # Start transfers for requests whose handshakes have now finished.

        if remote_engine_id is None and not wait_handshake_readd_req:
            return
        _deadline = time.monotonic() + self.moriio_config.transfer_timeout
        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.load_ready_flag
                and wait_handshake_readd_req
            ):
                if time.monotonic() > _deadline:
                    logger.warning(
                        "Timed out waiting for load_ready_flag[%s]; "
                        "adjust with kv_connector_extra_config.transfer_timeout",
                        remote_engine_id,
                    )
                    break
                time.sleep(0.001)
                continue
            elif (
                not self._ready_requests.empty()
                and remote_engine_id in self.load_ready_flag
            ):
                self._read_blocks_for_req(*self._ready_requests.get_nowait())
                break
            else:
                break

        self._reqs_to_send.update(metadata.reqs_to_send)

    def wait_for_save(self, metadata: MoRIIOConnectorMetadata):
        if self.mode == MoRIIOMode.WRITE and self.is_producer:
            for layer_name, kv_layer in self.kv_caches.items():
                self.save_kv_layer(metadata, layer_name, kv_layer, None)
            self._writer.seal_pending_transfers()

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        logger.debug(
            "Remote agent %s available, calling _read_blocks for req %s",
            meta.remote_engine_id,
            req_id,
        )
        self._read_blocks(
            request_id=req_id,
            transfer_id=meta.transfer_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            remote_host=meta.remote_host,
            remote_notify_port=meta.remote_notify_port,
            remote_tp_size=meta.tp_size,
        )

    def _write_blocks_for_req(self, req_id: ReqId, meta: ReqMeta, layer_name, kv_layer):
        # Stash multi_pod_hosts + local DP size on the worker so
        # MoRIIOEngine._finalize_if_complete (which sees only the WriteTask,
        # not ReqMeta) can pick the per-rank pod IP for the completion notify.
        # Last-writer-wins is safe: all requests share the same topology.
        if meta.multi_pod_hosts:
            self.multi_pod_hosts = list(meta.multi_pod_hosts)
        else:
            self.multi_pod_hosts = [meta.remote_host]
        if meta.remote_dp_size_local:
            self.remote_dp_size_local = int(meta.remote_dp_size_local)
        else:
            self.remote_dp_size_local = int(meta.remote_dp_size)
        self.schedule_write_blocks(
            request_id=req_id,
            transfer_id=meta.transfer_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            layer_name=layer_name,
            kv_layer=kv_layer,
            remote_notify_port=meta.remote_notify_port,
            remote_ip=meta.remote_host,
        )

    def merge_contiguous_blocks(
        self,
        offsets_local: list[int],
        offsets_remote: list[int],
        sizes: list[int],
        assume_sorted: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        n = len(offsets_local)
        if n == 0:
            return [], [], []
        if not (n == len(offsets_remote) == len(sizes)):
            raise ValueError("Input list lengths mismatch")
        local_arr = np.fromiter(offsets_local, dtype=np.int64, count=n)
        remote_arr = np.fromiter(offsets_remote, dtype=np.int64, count=n)
        sizes_arr = np.fromiter(sizes, dtype=np.int64, count=n)

        if assume_sorted:
            local_sorted = local_arr
            remote_sorted = remote_arr
            sizes_sorted = sizes_arr
        else:
            if np.all(local_arr[:-1] <= local_arr[1:]):
                local_sorted = local_arr
                remote_sorted = remote_arr
                sizes_sorted = sizes_arr
            else:
                sort_idx = np.argsort(local_arr, kind="stable")
                local_sorted = local_arr[sort_idx]
                remote_sorted = remote_arr[sort_idx]
                sizes_sorted = sizes_arr[sort_idx]

        if n == 1:
            return (
                [int(local_sorted[0])],
                [int(remote_sorted[0])],
                [int(sizes_sorted[0])],
            )

        diff_local = local_sorted[1:] - local_sorted[:-1]
        diff_remote = remote_sorted[1:] - remote_sorted[:-1]
        prev_size = sizes_sorted[:-1]

        contiguous = (diff_local == prev_size) & (diff_remote == prev_size)

        if not contiguous.any():
            return local_sorted.tolist(), remote_sorted.tolist(), sizes_sorted.tolist()

        if contiguous.all():
            total_size = int(sizes_sorted.sum())
            return [int(local_sorted[0])], [int(remote_sorted[0])], [total_size]

        break_positions = np.flatnonzero(~contiguous) + 1
        segment_starts = np.concatenate(([0], break_positions))
        segment_ends = np.concatenate((break_positions, [n]))

        seg_count = len(segment_starts)
        merged_local = [0] * seg_count
        merged_remote = [0] * seg_count
        merged_sizes = [0] * seg_count

        for si in range(seg_count):
            s = segment_starts[si]
            e = segment_ends[si]
            merged_local[si] = int(local_sorted[s])
            merged_remote[si] = int(remote_sorted[s])

            merged_sizes[si] = int(
                local_sorted[e - 1] + sizes_sorted[e - 1] - local_sorted[s]
            )

        return merged_local, merged_remote, merged_sizes

    def _compute_block_transfer_offsets(
        self,
        layer_name: str,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_moriio_meta: MoRIIOAgentMetadata,
        remote_tp_size: int | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Compute transfer offsets for block data.

        Args:
            layer_name: Name of the layer to transfer
            local_block_ids: IDs of local blocks
            remote_block_ids: IDs of remote blocks
            remote_moriio_meta: Metadata of the remote MoRIIO agent
        Returns:
            Tuple of (local_offsets, remote_offsets, transfer_sizes)
        """
        validate_moriio_heterogeneous_tp_kv_heads(
            local_tp_size=self.world_size,
            remote_tp_size=(
                remote_tp_size if remote_tp_size is not None else self.world_size
            ),
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            is_mla=self._is_mla_cache_layer(layer_name),
        )
        return compute_block_transfer_offsets(
            layer_name=layer_name,
            kv_cache=self.kv_caches[layer_name],
            layer_to_spec=self.layer_to_spec,
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
            remote_num_blocks=remote_moriio_meta.num_blocks,
            merge_fn=lambda local, remote, sizes: self.merge_contiguous_blocks(
                local, remote, sizes, assume_sorted=False
            ),
        )

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_engine_id: str,
        request_id: str,
        transfer_id: str,
        remote_host: str,
        remote_notify_port: int,
        remote_tp_size: int,
    ) -> None:
        if self.mode == MoRIIOMode.WRITE:
            return

        dp0_engine_id = self.get_engine_name_with_dp(dst_engine_id, 0)
        sessions, remote_moriio_meta = self._get_built_session(dp0_engine_id)

        for layer_name in self.layer_name_to_local_kv_cache_metadata:
            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(
                layer_name
            )
            offs = self._compute_block_transfer_offsets(
                layer_name,
                local_block_ids,
                remote_block_ids,
                remote_moriio_meta,
                remote_tp_size=remote_tp_size,
            )
            # TODO : apply multi-session batch-read when moriio support it
            transfer_status = self.moriio_wrapper.read_remote_data(
                offs[2], offs[0], offs[1], sessions[sess_idx]
            )
            with self.moriio_wrapper.lock:
                self._recving_transfers[request_id].append(transfer_status)
                self._recving_transfers_start.setdefault(request_id, time.monotonic())
                self._recving_transfers_callback_addr[request_id] = (
                    remote_host,
                    str(remote_notify_port + self._remote_tp_rank(remote_tp_size)),
                    transfer_id,
                )
