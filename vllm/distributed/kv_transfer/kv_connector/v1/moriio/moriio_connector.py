# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import math
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Collection
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import msgpack
import msgspec
import numpy as np
import regex as re
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
    LayerTransferPlan,
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
    _as_bool,
    _strip_vllm_request_suffix,
    get_moriio_mode,
    get_moriio_node_hosts,
    get_moriio_trusted_remote_hosts,
    get_peer_zmq_from_request_id,
    get_port_offset,
    get_role,
    parse_moriio_zmq_address,
    resolve_host_ip,
    set_role,
    validate_moriio_trusted_host,
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

_TRANSFER_ID_RE = re.compile(
    r"(tx-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
_READ_COMPLETION_ID_RE = re.compile(r"^(?P<transfer_id>.+):tp(?P<tp_rank>\d+)$")
_MAX_PENDING_UNMAPPED_DONE_TIDS = 4096


def _make_read_completion_id(transfer_id: TransferId, tp_rank: int) -> str:
    return f"{transfer_id}:tp{int(tp_rank)}"


CompletionId = str | MoRIIOTransferAck


def _completion_id_text(completion_id: CompletionId) -> str:
    if isinstance(completion_id, MoRIIOTransferAck):
        return completion_id.transfer_id
    return completion_id


def _read_completion_transfer_id(completion_id: CompletionId) -> TransferId | None:
    completion_id = _completion_id_text(completion_id)
    match = _READ_COMPLETION_ID_RE.fullmatch(completion_id)
    if match is None:
        return None
    return match.group("transfer_id")


def _read_completion_key(completion_id: CompletionId) -> str:
    completion_id = _completion_id_text(completion_id)
    match = _READ_COMPLETION_ID_RE.fullmatch(completion_id)
    if match is None:
        return completion_id
    return f"tp{match.group('tp_rank')}"


def _read_completion_quorum(
    remote_decode_tp_size: int,
    producer_tp_size: int,
    producer_tp_rank: int,
) -> int:
    remote_decode_tp_size = max(1, int(remote_decode_tp_size))
    producer_tp_size = max(1, int(producer_tp_size))
    producer_tp_rank = int(producer_tp_rank)
    if producer_tp_rank >= remote_decode_tp_size:
        return 0
    return ((remote_decode_tp_size - 1 - producer_tp_rank) // producer_tp_size) + 1


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


def _pick_remote_rank_host(
    default_host: str,
    remote_hosts: list[str] | None,
    tp_size: int,
    tp_rank: int,
    remote_dp_size: int = 1,
    remote_dp_rank: int = 0,
) -> str:
    if remote_hosts and len(remote_hosts) > 1:
        n_hosts = len(remote_hosts)
        if int(remote_dp_size) >= n_hosts:
            dp_per_node = max(1, int(remote_dp_size) // n_hosts)
            node_idx = int(remote_dp_rank) // dp_per_node
            if 0 <= node_idx < n_hosts:
                return remote_hosts[node_idx]
        ranks_per_node = max(1, int(tp_size) // n_hosts)
        node_idx = int(tp_rank) // ranks_per_node
        if 0 <= node_idx < n_hosts:
            return remote_hosts[node_idx]
    return default_host


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

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        if self.mode == MoRIIOMode.WRITE and get_role() == ROLE.CONSUMER:
            self.connector_worker.moriio_wrapper.async_wait_reqid()

        assert isinstance(self._connector_metadata, MoRIIOConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.mode != MoRIIOMode.READ or get_role() == ROLE.PRODUCER:
            return
        assert self.connector_worker is not None, (
            "wait_for_layer_load called on scheduler role"
        )
        self.connector_worker.wait_for_layer_load(layer_name)

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

    def get_finished_count(self) -> int | None:
        # Aggregation still waits for this producer TP group; each worker emits
        # only after its per-transfer decode-rank quorum is satisfied.
        return self._vllm_config.parallel_config.tensor_parallel_size

    def has_pending_deferred_sends(self) -> bool:
        if self.connector_scheduler is None:
            return False
        return self.connector_scheduler.has_pending_deferred_sends()

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """MoRIIO needs PIECEWISE around layerwise transfer hooks."""
        if _as_bool(extra_config.get("allow_full_cudagraph", False)):
            raise ValueError(
                "MoRIIO cannot honor allow_full_cudagraph=True: layerwise "
                "KV-transfer hooks must run outside full CUDA graph replay."
            )
        return True


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
        # Multi-node TP: node_hosts holds the ordered host IPs in this
        # engine's TP group (rank 0 first). Surfaced to the peer side via
        # request_finished's kv_transfer_params so workers can dial the rank
        # owner. Single-node TP falls back to [host_ip].
        self.node_hosts = get_moriio_node_hosts(self.kv_transfer_config, self.host_ip)
        self.trusted_remote_hosts = get_moriio_trusted_remote_hosts(
            self.kv_transfer_config
        )
        self.handshake_port = self.kv_transfer_config.kv_connector_extra_config[
            "handshake_port"
        ]
        logger.info(
            "Initializing MoRIIO Scheduler engine_id = %s node_hosts = %s "
            "trusted_remote_hosts = %s",
            engine_id,
            self.node_hosts,
            self.trusted_remote_hosts,
        )

        self.side_notify_port = self.kv_transfer_config.kv_connector_extra_config[
            "notify_port"
        ]
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        self.is_producer = self.kv_transfer_config.kv_role == "kv_producer"
        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}

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
        self._deferred_send_deadlines: dict[ReqId, float] = {}
        self._defer_timeout = float(
            self.kv_transfer_config.kv_connector_extra_config.get(
                "defer_timeout", MoRIIOConstants.DEFAULT_DEFER_TIMEOUT
            )
        )
        self.paths: dict[str, zmq.Socket] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}
        self._transfer_ids_to_forget: set[TransferId] = set()
        # Scheduler-to-worker ID mappings that have not been sent yet. The
        # worker merges them into a persistent map, so metadata only needs the
        # local delta instead of cloning the full map every scheduler tick.
        self._pending_transfer_id_to_request_id: dict[TransferId, ReqId] = {}
        self.transfer_id_to_remote_tp_size: dict[TransferId, int] = {}
        self._pending_transfer_id_to_remote_tp_size: dict[TransferId, int] = {}

    def map_request_id(
        self,
        request_id: ReqId,
        transfer_id: TransferId,
        remote_tp_size: int | None = None,
    ):
        previous_request_id = self.transfer_id_to_request_id.get(transfer_id)
        self.transfer_id_to_request_id[transfer_id] = request_id
        self.request_id_to_transfer_id[request_id] = transfer_id
        if previous_request_id != request_id:
            self._pending_transfer_id_to_request_id[transfer_id] = request_id
        if remote_tp_size is not None:
            if not hasattr(self, "transfer_id_to_remote_tp_size"):
                self.transfer_id_to_remote_tp_size = {}
            if not hasattr(self, "_pending_transfer_id_to_remote_tp_size"):
                self._pending_transfer_id_to_remote_tp_size = {}
            remote_tp_size = max(1, int(remote_tp_size))
            self.transfer_id_to_remote_tp_size[transfer_id] = remote_tp_size
            self._pending_transfer_id_to_remote_tp_size[transfer_id] = remote_tp_size

    def unmap_request_id(self, request_id: ReqId, warn_missing: bool = True):
        if request_id in self.request_id_to_transfer_id:
            transfer_id = self.request_id_to_transfer_id[request_id]
            del self.request_id_to_transfer_id[request_id]
            with suppress(AttributeError):
                self._pending_transfer_id_to_request_id.pop(transfer_id, None)
                self._pending_transfer_id_to_remote_tp_size.pop(transfer_id, None)
            with suppress(AttributeError):
                self.transfer_id_to_remote_tp_size.pop(transfer_id, None)
            if transfer_id in self.transfer_id_to_request_id:
                del self.transfer_id_to_request_id[transfer_id]
            else:
                logger.warning(
                    "transfer id not in transfer_id_to_request_id lookup"
                    "table. there is likely a bug!"
                )
        elif warn_missing:
            logger.warning(
                "Could not find %s  in transfer_id_to_request_id"
                "lookup table.  This could lead to a possible hang.",
                request_id,
            )

    def _get_transfer_block_count(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def _trim_block_ids_to_token_span(
        self,
        block_ids: list[int],
        num_tokens: int,
    ) -> list[int]:
        """Limit block ids to the token span MoRIIO actually transfers."""
        transfer_blocks = self._get_transfer_block_count(num_tokens)
        if transfer_blocks <= 0:
            return []
        if len(block_ids) <= transfer_blocks:
            return block_ids
        return block_ids[:transfer_blocks]

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
            "decode_rank": self.dp_rank,
            "type": "remote_blocks",
        }
        serialized_data = msgpack.dumps(data, use_bin_type=True)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "MoRIIO remote_blocks send: path=%s transfer_id=%s req_id=%s blocks=%d",
                path,
                transfer_id,
                req_id,
                len(block_notify_list or []),
            )
        self.paths[path].send(serialized_data)

    def _send_transfer_release(self, transfer_id: TransferId, host: str, port: int):
        path = make_zmq_path("tcp", host, port)
        if path not in self.paths:
            ctx = zmq.Context.instance()
            sock = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.DEALER, bind=False
            )
            self.paths[path] = sock

        self.paths[path].send(
            msgpack.dumps({"type": "release", "transfer_id": transfer_id})
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
        params = request.kv_transfer_params
        if not params:
            return
        transfer_id = params["transfer_id"]
        request_id = request.request_id
        remote_tp_size = int(
            params.get("remote_tp_size", params.get("tp_size", self.tp_size))
            or self.tp_size
        )
        self.map_request_id(request_id, transfer_id, remote_tp_size=remote_tp_size)
        logger.info(
            "MoRIIO update_state_after_alloc: request_id=%s transfer_id=%s "
            "mode=%s do_remote_decode=%s do_remote_prefill=%s "
            "num_external_tokens=%d",
            request_id,
            transfer_id,
            self.mode.name,
            params.get("do_remote_decode"),
            params.get("do_remote_prefill"),
            num_external_tokens,
        )

        if params.get("do_remote_decode"):
            local_block_ids = self._trim_block_ids_to_token_span(
                blocks.get_block_ids()[0], request.num_prompt_tokens
            )
            self._reqs_need_save[request.request_id] = (request, local_block_ids)

        if params is not None and params.get("do_remote_prefill"):
            if self.mode == MoRIIOMode.READ:
                if remote_block_ids := params.get("remote_block_ids"):
                    # remote_engine_id is returned by the prefill's request_finished.
                    # host/ports come from the request_id (parsed in add_new_req).
                    if "remote_engine_id" in params:
                        # If remote_blocks and num_external_tokens = 0, we have
                        # a full prefix cache hit on the D worker. We need to call
                        # send_notify in _read_blocks to free the memory on the P.

                        # Get local blocks to pull remote KV into. The two block
                        # lists must describe the same token span.
                        local_block_ids = blocks.get_block_ids()[0]
                        assert len(local_block_ids) == len(remote_block_ids), (
                            "MoRIIO READ block id count mismatch: "
                            f"local={len(local_block_ids)} "
                            f"remote={len(remote_block_ids)}"
                        )

                        self._reqs_need_recv[request.request_id] = (
                            request,
                            local_block_ids,
                        )
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
                remote_host = request.kv_transfer_params.get("remote_host")
                remote_notify_port = request.kv_transfer_params.get(
                    "remote_notify_port"
                )
                used_remote_zmq_fallback = False
                if remote_host is None or remote_notify_port is None:
                    try:
                        peer_zmq = get_peer_zmq_from_request_id(
                            request.request_id, is_producer=False
                        )
                    except ValueError:
                        peer_zmq = request.kv_transfer_params.get("remote_zmq_address")
                        if not peer_zmq:
                            raise ValueError(
                                "MoRIIO WRITE remote_blocks notification requires "
                                "kv_transfer_params.remote_zmq_address when the "
                                "request_id does not embed a prefill zmq address"
                            ) from None
                        used_remote_zmq_fallback = True
                    remote_host, _, remote_notify_port = parse_moriio_zmq_address(
                        peer_zmq
                    )
                    if used_remote_zmq_fallback:
                        validate_moriio_trusted_host(
                            remote_host,
                            self.trusted_remote_hosts,
                            "kv_transfer_params.remote_zmq_address",
                        )
                remote_notify_port = int(remote_notify_port)

                remote_hosts = request.kv_transfer_params.get("remote_hosts")
                if isinstance(remote_hosts, str):
                    remote_hosts = [remote_hosts] if remote_hosts else None

                remote_tp_size = int(
                    request.kv_transfer_params.get(
                        "remote_tp_size",
                        request.kv_transfer_params.get("tp_size", self.tp_size),
                    )
                    or self.tp_size
                )
                block_notify_list = blocks.get_block_ids()[0]
                if num_external_tokens > 0:
                    block_notify_list = self._trim_block_ids_to_token_span(
                        block_notify_list, num_external_tokens
                    )

                logger.info(
                    "MoRIIO WRITE decode remote_blocks branch: request_id=%s "
                    "transfer_id=%s mode=%s remote_host=%s remote_hosts=%s "
                    "remote_notify_port=%s remote_dp_rank=%s remote_dp_size=%s "
                    "remote_tp_size=%d num_external_tokens=%d blocks=%d",
                    request.request_id,
                    request.kv_transfer_params["transfer_id"],
                    self.mode.name,
                    remote_host,
                    remote_hosts,
                    remote_notify_port,
                    remote_dp_rank,
                    request.kv_transfer_params.get("remote_dp_size", 1),
                    remote_tp_size,
                    num_external_tokens,
                    len(block_notify_list or []),
                )

                for tp_index in range(remote_tp_size):
                    target_port = remote_notify_port + get_port_offset(
                        remote_dp_rank, tp_index, remote_tp_size
                    )
                    target_host = _pick_remote_rank_host(
                        remote_host,
                        remote_hosts,
                        remote_tp_size,
                        tp_index,
                        int(request.kv_transfer_params.get("remote_dp_size", 1)),
                        int(remote_dp_rank),
                    )

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "MoRIIO WRITE decode remote_blocks target: "
                            "transfer_id=%s tp_index=%d target_host=%s "
                            "target_port=%d remote_tp_size=%d",
                            request.kv_transfer_params["transfer_id"],
                            tp_index,
                            target_host,
                            target_port,
                            remote_tp_size,
                        )

                    if used_remote_zmq_fallback:
                        validate_moriio_trusted_host(
                            target_host,
                            self.trusted_remote_hosts,
                            "kv_transfer_params.remote_hosts",
                        )
                    self.send_notify_block(
                        req_id=request.request_id,
                        transfer_id=request.kv_transfer_params["transfer_id"],
                        block_notify_list=block_notify_list,
                        host=target_host,
                        port=target_port,
                    )

            # Only trigger 1 KV transfer per request.

            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MoRIIOConnectorMetadata()
        try:
            pending_transfer_ids = self._pending_transfer_id_to_request_id
        except AttributeError:
            meta.transfer_id_to_request_id = {
                transfer_id: req_id
                for transfer_id, req_id in self.transfer_id_to_request_id.items()
            }
            meta.transfer_id_to_remote_tp_size = dict(
                getattr(self, "transfer_id_to_remote_tp_size", {})
            )
        else:
            if pending_transfer_ids:
                meta.transfer_id_to_request_id = pending_transfer_ids
                self._pending_transfer_id_to_request_id = {}
            pending_remote_tp_sizes = getattr(
                self, "_pending_transfer_id_to_remote_tp_size", {}
            )
            if pending_remote_tp_sizes:
                meta.transfer_id_to_remote_tp_size = pending_remote_tp_sizes
                self._pending_transfer_id_to_remote_tp_size = {}
        if self._transfer_ids_to_forget:
            meta.freed_transfer_ids = set(self._transfer_ids_to_forget)
            self._transfer_ids_to_forget.clear()

        if self.mode == MoRIIOMode.WRITE and get_role() == ROLE.PRODUCER:
            # This is the logic for checking against chunked prefill.
            # When the last chunk is identified,
            # It places the request metadata into the saving queue.

            for i, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
                new_block_ids = scheduler_output.scheduled_cached_reqs.new_block_ids[i]

                if new_block_ids is not None:
                    block_ids = new_block_ids[0]
                    # TODO : hybrid attn, etc
                    req, existing_blocks = self._reqs_need_pending_save[req_id]
                    updated_blocks = list(existing_blocks) + (block_ids)
                    self._reqs_need_pending_save[req_id] = (req, updated_blocks)
                    if (
                        len(self._reqs_need_pending_save[req_id][1]) * self.block_size
                        >= req.num_prompt_tokens
                    ):
                        local_block_ids = self._trim_block_ids_to_token_span(
                            self._reqs_need_pending_save[req_id][1],
                            req.num_prompt_tokens,
                        )
                        meta.add_new_req(
                            request_id=req_id,
                            local_block_ids=local_block_ids,
                            kv_transfer_params=req.kv_transfer_params or {},
                            write_mode=True,
                            trusted_remote_hosts=self.trusted_remote_hosts,
                        )
                        del self._reqs_need_pending_save[req_id]

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                trusted_remote_hosts=self.trusted_remote_hosts,
            )

        for req_id, (req, block_ids) in self._reqs_need_save.items():
            assert req.kv_transfer_params is not None
            local_block_ids = self._trim_block_ids_to_token_span(
                block_ids, req.num_prompt_tokens
            )
            if req.num_prompt_tokens > len(local_block_ids) * self.block_size:
                # not last chunk prefill
                self._reqs_need_pending_save[req_id] = (req, local_block_ids)
                continue
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=local_block_ids,
                kv_transfer_params=req.kv_transfer_params,
                write_mode=True,
                trusted_remote_hosts=self.trusted_remote_hosts,
            )
        # Clear the list once workers start the transfers

        meta.reqs_to_send = self._reqs_need_send

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

        params = request.kv_transfer_params
        request_id = request.request_id
        # Consumer mappings can be removed once KV transfer has completed.
        # Producer mappings stay until the block-free notification arrives.
        if not self.is_producer:
            mapping_was_never_created = bool(params and params.get("do_remote_prefill"))
            self.unmap_request_id(
                request_id,
                warn_missing=not mapping_was_never_created,
            )
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
                time.monotonic() + self._defer_timeout
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
            # Multi-node TP: list of all prefill-instance host IPs in this
            # engine's TP group (rank 0 first). Decode workers use this to
            # pick the correct producer host per their tp_rank.
            remote_hosts=self.node_hosts,
            # Wall-clock TS captured at end of P-side prefill (request_finished
            # runs after FINISHED_LENGTH_CAPPED). Consumed by D's OutputProcessor
            # to derive stage-3 KV transfer time. Requires NTP sync across nodes.
            prefill_complete_ts=time.time(),
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Free KV blocks from sends that never received a completion signal.

        Called every scheduler step. When a send is deferred (request_finished
        returns True), blocks remain allocated until the worker reports
        finished_sending. If that notification is lost (e.g. ibv_post_send
        failure), blocks leak permanently. This method injects timed-out
        entries into connector_output.finished_sending so the scheduler
        frees them via the normal path.
        """
        # Producer: unmap transfer_id<->request_id for sends that are now (async)
        #   reported as completed. This unmapping have to be deferred until now
        #   so get_finished can use it in any scheduler step.
        # Consumer: unmapping already done in request_finished
        if self.is_producer and connector_output.finished_sending:
            for req_id in connector_output.finished_sending:
                transfer_id = self.request_id_to_transfer_id.get(req_id)
                if transfer_id is not None:
                    self._transfer_ids_to_forget.add(transfer_id)
                self.unmap_request_id(req_id)

        if not self._deferred_send_deadlines:
            return

        # Remove entries the worker already reported as finished_sending, these will be
        # freed anyways.
        for req_id in connector_output.finished_sending or ():
            self._deferred_send_deadlines.pop(req_id, None)

        if not self._deferred_send_deadlines:
            return

        now = time.monotonic()
        expired_reqs = [
            req_id
            for req_id, deadline in self._deferred_send_deadlines.items()
            if now >= deadline
        ]
        if not expired_reqs:
            return

        if connector_output.finished_sending is None:
            connector_output.finished_sending = set()
        # Register the expired requests as finished so the scheduler frees their blocks.
        for req_id in expired_reqs:
            transfer_id = self.request_id_to_transfer_id.get(req_id)
            if transfer_id is not None:
                self._transfer_ids_to_forget.add(transfer_id)
            connector_output.finished_sending.add(req_id)
            self._reqs_need_send.pop(req_id, None)
            del self._deferred_send_deadlines[req_id]
            if self.is_producer:
                self.unmap_request_id(req_id)
        logger.warning(
            "Reaped %d deferred sends with no finished_sending notification "
            "after %.0fs. This indicates lost async KV completion "
            "notifications from the KV connector.",
            len(expired_reqs),
            self._defer_timeout,
        )

    def has_pending_deferred_sends(self) -> bool:
        return bool(self._transfer_ids_to_forget or self._deferred_send_deadlines)


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

        assert vllm_config.kv_transfer_config is not None, (
            "kv_transfer_config must be set for MoRIIOConnector"
        )
        self.vllm_config = vllm_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.moriio_config = MoRIIOConfig.from_vllm_config(vllm_config)
        self.mode = (
            MoRIIOMode.READ if self.moriio_config.read_mode else MoRIIOMode.WRITE
        )

        logger.info("Initializing MoRIIO worker %s", engine_id)

        logging.getLogger("aiter").disabled = True

        # Config.
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
        self.tp_size = self.moriio_config.tp_size

        self.local_ip = self.moriio_config.local_ip
        self.local_kv_port = self.moriio_config.local_kv_port
        self.proxy_ip = self.moriio_config.proxy_ip
        self.local_ping_port = self.moriio_config.local_ping_port
        self.proxy_ping_port = self.moriio_config.proxy_ping_port
        self.http_port = self.moriio_config.http_port
        self.handshake_port = self.moriio_config.handshake_port
        self.notify_port = self.moriio_config.notify_port
        self.node_hosts = self.moriio_config.node_hosts

        self.dp_rank_to_host: dict[int, str] = {}

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
            + get_port_offset(self.dp_rank, self.tp_rank, self.tp_size)
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
        # In-progress READ transfers: per request, keyed by layer name.
        self._recving_transfers: defaultdict[ReqId, dict[str, Any]] = defaultdict(dict)
        self._pending_read_plans: defaultdict[ReqId, dict[str, LayerTransferPlan]] = (
            defaultdict(dict)
        )
        # Values are (remote_host, remote_notify_port, transfer_id).
        self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str, str]] = {}
        self._recving_transfer_local_block_ids: dict[ReqId, set[int]] = {}
        self._invalid_block_ids: queue.Queue[set[int]] = queue.Queue()

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
        # Base remote engines ("host:handshake_port") whose full DP-rank set has
        # been eagerly handshaked AND TP-barriered. Gates
        # _eager_handshake_all_dp_ranks to fire once per remote engine (first
        # contact), not per request/step.
        self._eager_handshaked_engines: set[str] = set()

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
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}
        self.transfer_id_to_remote_tp_size: dict[TransferId, int] = {}
        self.transfer_id_to_completion_count: dict[TransferId, int] = {}
        self._read_completion_ids: defaultdict[TransferId, set[str]] = defaultdict(set)
        # READ-mode producer: buffer decode-side completion ids until
        # start_load_kv populates transfer_id_to_request_id. get_finished()
        # retries the buffered ids in insertion order and evicts malformed
        # notifications oldest-first at the cap.
        self._pending_unmapped_done_tids: dict[str, CompletionId | None] = {}

        # TODO: consider the integration of flashinfer or other backends.
        self.backend_name = backend.get_name()
        logger.debug("Detected attention backend %s", self.backend_name)

    def _remember_transfer_mapping(
        self, transfer_id: TransferId, request_id: ReqId
    ) -> None:
        self.transfer_id_to_request_id[transfer_id] = request_id
        self.request_id_to_transfer_id[request_id] = transfer_id
        external_request_id = _strip_vllm_request_suffix(request_id)
        if external_request_id != request_id:
            self.request_id_to_transfer_id.setdefault(external_request_id, transfer_id)

    def _forget_transfer_mapping(
        self, transfer_id: TransferId, request_id: ReqId | None = None
    ) -> None:
        if request_id is None:
            request_id = self.transfer_id_to_request_id.get(transfer_id)
        self.transfer_id_to_request_id.pop(transfer_id, None)
        self.transfer_id_to_remote_tp_size.pop(transfer_id, None)
        self.transfer_id_to_completion_count.pop(transfer_id, None)
        self._read_completion_ids.pop(transfer_id, None)
        if request_id is None:
            return
        with suppress(AttributeError):
            self._reqs_to_send.pop(request_id, None)
        self.request_id_to_transfer_id.pop(request_id, None)
        external_request_id = _strip_vllm_request_suffix(request_id)
        if external_request_id != request_id:
            mapped_transfer_id = self.request_id_to_transfer_id.get(external_request_id)
            if mapped_transfer_id == transfer_id:
                self.request_id_to_transfer_id.pop(external_request_id, None)

    def _drop_pending_unmapped_done_tid(
        self, transfer_id: TransferId, request_id: ReqId | None = None
    ) -> None:
        ids_to_drop = {transfer_id}
        if request_id is not None:
            ids_to_drop.add(request_id)
            ids_to_drop.add(_strip_vllm_request_suffix(request_id))

        for completion_id in list(self._pending_unmapped_done_tids):
            if completion_id in ids_to_drop:
                self._pending_unmapped_done_tids.pop(completion_id, None)
                continue
            if _read_completion_transfer_id(completion_id) == transfer_id:
                self._pending_unmapped_done_tids.pop(completion_id, None)
                continue
            match = _TRANSFER_ID_RE.search(completion_id)
            if match is not None and match.group(1) == transfer_id:
                self._pending_unmapped_done_tids.pop(completion_id, None)

    def _buffer_pending_unmapped_done_tid(self, completion_id: CompletionId) -> None:
        completion_key = _completion_id_text(completion_id)
        if completion_key in self._pending_unmapped_done_tids:
            return
        if len(self._pending_unmapped_done_tids) >= _MAX_PENDING_UNMAPPED_DONE_TIDS:
            oldest = next(iter(self._pending_unmapped_done_tids))
            self._pending_unmapped_done_tids.pop(oldest, None)
            logger.warning(
                "Dropping oldest pending READ completion id after buffer "
                "reached %d entries",
                _MAX_PENDING_UNMAPPED_DONE_TIDS,
            )
        self._pending_unmapped_done_tids[completion_key] = (
            completion_id if isinstance(completion_id, MoRIIOTransferAck) else None
        )

    def _translate_or_buffer_completion(
        self, completion_id: CompletionId, done_sending: set[str]
    ) -> None:
        resolved = self._resolve_completion_mapping(completion_id)
        if resolved is None:
            # Mapping not yet populated — keep pending, retry next tick.
            # Without this buffer the notification is lost and the producer's
            # KV blocks leak.
            self._buffer_pending_unmapped_done_tid(completion_id)
            return

        transfer_id, request_id = resolved
        ack = (
            completion_id
            if isinstance(completion_id, MoRIIOTransferAck)
            else MoRIIOTransferAck(transfer_id)
        )
        resolved_transfer_id = resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=self.world_size,
            live_transfer_ids=self.transfer_id_to_request_id.keys(),
            notification_counts=self._consumer_notification_counts,
            completed_transfer_ids=self._completed_consumer_notifications,
        )
        if resolved_transfer_id is None:
            return
        self._forget_transfer_mapping(resolved_transfer_id, request_id)
        done_sending.add(request_id)

    def _translate_or_buffer_read_completion(
        self, completion_id: CompletionId, done_sending: set[str]
    ) -> None:
        mapped, handled = self._pop_mapped_read_completion_id(completion_id)
        if mapped is not None:
            done_sending.add(mapped)
            with suppress(AttributeError):
                self._reqs_to_send.pop(mapped, None)
        elif not handled:
            self._buffer_pending_unmapped_done_tid(completion_id)

    def _resolve_completion_mapping(
        self, completion_id: CompletionId
    ) -> tuple[TransferId, ReqId] | None:
        completion_id = _completion_id_text(completion_id)
        request_id = self.transfer_id_to_request_id.get(completion_id)
        if request_id is not None:
            return completion_id, request_id

        transfer_id = _read_completion_transfer_id(completion_id)
        if transfer_id is not None:
            request_id = self.transfer_id_to_request_id.get(transfer_id)
            if request_id is not None:
                return transfer_id, request_id

        match = _TRANSFER_ID_RE.search(completion_id)
        if match is not None:
            transfer_id = match.group(1)
            request_id = self.transfer_id_to_request_id.get(transfer_id)
            if request_id is not None:
                return transfer_id, request_id

        request_keys = [completion_id]
        stripped_completion_id = _strip_vllm_request_suffix(completion_id)
        if stripped_completion_id != completion_id:
            request_keys.append(stripped_completion_id)
        for request_key in request_keys:
            transfer_id = self.request_id_to_transfer_id.get(request_key)
            if transfer_id is None:
                continue
            request_id = self.transfer_id_to_request_id.get(transfer_id)
            if request_id is None:
                self.request_id_to_transfer_id.pop(request_key, None)
                return None
            return transfer_id, request_id
        return None

    def _pop_mapped_completion_id(self, completion_id: CompletionId) -> ReqId | None:
        resolved = self._resolve_completion_mapping(completion_id)
        if resolved is None:
            return None
        transfer_id, request_id = resolved
        self._forget_transfer_mapping(transfer_id, request_id)
        return request_id

    def _pop_mapped_read_completion_id(
        self, completion_id: CompletionId
    ) -> tuple[ReqId | None, bool]:
        resolved = self._resolve_completion_mapping(completion_id)
        if resolved is None:
            return None, False
        transfer_id, request_id = resolved
        expected_count = self.transfer_id_to_completion_count.get(transfer_id, 1)
        if expected_count <= 1:
            self._forget_transfer_mapping(transfer_id, request_id)
            return request_id, True

        completion_key = _read_completion_key(completion_id)
        completion_ids = self._read_completion_ids[transfer_id]
        if completion_key in completion_ids:
            return None, True
        completion_ids.add(completion_key)
        if len(completion_ids) < expected_count:
            return None, True

        self._forget_transfer_mapping(transfer_id, request_id)
        return request_id, True

    def _pop_zero_quorum_read_completions(self, done_sending: set[str]) -> None:
        for req_id in list(self._reqs_to_send):
            transfer_id = self.request_id_to_transfer_id.get(req_id)
            if transfer_id is None:
                continue
            if self.transfer_id_to_completion_count.get(transfer_id) != 0:
                continue
            done_sending.add(req_id)
            self._reqs_to_send.pop(req_id, None)
            self._forget_transfer_mapping(transfer_id, req_id)

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
        event = torch.cuda.Event()
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
        # Use host:port format for http_address.
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
                        # The router routes by http_address; these sizes are kept
                        # for proxy compatibility.
                        "dp_size": self.moriio_config.dp_size,
                        "tp_size": self.moriio_config.tp_size,
                        # transfer_mode is included so the router can distinguish
                        # READ (prefill-then-decode, sequential) from WRITE (concurrent)
                        # scheduling.
                        "transfer_mode": self.mode.name,
                        "node_hosts": list(self.node_hosts),
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

    def _remote_tp_rank(self, remote_tp_size: int) -> int:
        """Map this local TP rank onto the remote TP layout for port addressing.

        The request's remote_dp_rank selects the remote DP rank that holds its KV;
        this helper only resolves the remote TP index. When remote TP size is 1,
        every local rank maps to tp0. Equal TP sizes preserve tp_rank. Wider
        remote TP layouts replicate MLA latent KV across remote TP ranks, so
        modulo selects a valid remote rank.
        """
        return self.tp_rank % max(1, int(remote_tp_size))

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

        # Each worker exposes a distinct metadata endpoint. Until MoRIIO has a
        # shared scheduler/coordinator socket, derive the endpoint from the
        # remote DP/TP rank so handshakes land on the process that owns the
        # transfer metadata.

        # Heterogeneous-TP port mapping: dial the remote's TP index, not our own
        # local tp_rank. For TP1/DP8 prefill ↔ TP8 decode this collapses every
        # decode rank onto the prefill rank's tp0 (remote_dp_rank picks the DP
        # rank). Identity for symmetric TP. See _remote_tp_rank.
        port_offset = get_port_offset(
            remote_dp_rank, self._remote_tp_rank(remote_tp_size), remote_tp_size
        )
        path = make_zmq_path("tcp", host, port + port_offset)
        logger.debug("handshake Querying metadata on path: %s", path)

        # Send query for the request.
        timeout_ms = int(self.moriio_config.handshake_timeout * 1000)
        with zmq_ctx(zmq.DEALER, path) as sock:
            # Bound the handshake so an unresponsive remote listener raises
            # HandshakeError instead of blocking this TP worker on recv().
            # LINGER=0 discards unsent data when the socket closes.
            sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            logger.debug("prepare send msg INSTAZNCE: %s", path)
            sock.send(MoRIIOConstants.GET_META_MSG)
            try:
                received_frame = sock.recv_multipart()
            except zmq.error.Again as e:
                raise HandshakeError(
                    f"MoRIIO handshake metadata recv timed out after "
                    f"{timeout_ms}ms on path {path} (remote dp rank "
                    f"{remote_dp_rank}); remote handshake listener unreachable"
                ) from e
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
            self.dp_rank_to_host[int(remote_dp_rank)] = host

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

            try:
                received_frame = sock.recv_multipart()
            except zmq.error.Again as e:
                raise HandshakeError(
                    f"MoRIIO handshake layer-metadata recv timed out after "
                    f"{timeout_ms}ms on path {path} (remote dp rank "
                    f"{remote_dp_rank})"
                ) from e
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

    def _pick_remote_host(self, meta: ReqMeta) -> str:
        """Resolve the per-worker peer host for multi-node TP prefill-decode."""
        return _pick_remote_rank_host(
            meta.remote_host, meta.remote_hosts, int(meta.tp_size), self.tp_rank
        )

    def _pick_host_for_dp_rank(self, meta: ReqMeta, dp_rank: int) -> str:
        return _pick_remote_rank_host(
            meta.remote_host,
            meta.remote_hosts,
            int(meta.tp_size),
            self.tp_rank,
            int(meta.remote_dp_size),
            dp_rank,
        )

    def _background_moriio_handshake(
        self, req_id: ReqId, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do MoRIIO handshake in background and add to _ready_requests when done.
        fut = None
        if remote_engine_id is not None:
            fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            port = int(meta.remote_handshake_port)
            tp_size = int(meta.tp_size)
            remote_dp_size = int(meta.remote_dp_size)

        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            logger.info("MoRIIO handshake done for request %s", req_id)
            self._ready_requests.put(entry)
            self.load_ready_flag[remote_engine_id] = True
            self.write_ready_flags[remote_engine_id] = True

        fut_list = []

        # In dp(prefill)<->dp(decode) communication, we require an all-to-all handshake.

        for cur_dp_rank in range(remote_dp_size):
            dp_engine_id = self.get_engine_name_with_dp(remote_engine_id, cur_dp_rank)
            host = self._pick_host_for_dp_rank(meta, cur_dp_rank)
            future = self._handshake_initiation_executor.submit(
                self._moriio_handshake, host, port, tp_size, dp_engine_id, cur_dp_rank
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
        assert first_geometry.block_size == self.block_size
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

        done_sending: set[str] = set()
        done_recving: set[str] = set()

        if self.is_producer:
            done_sending_raw = self.moriio_wrapper.pop_finished_req_ids()
            if self.mode == MoRIIOMode.READ:
                # READ-mode decode notifies prefill by transfer_id. Translate it
                # to the producer's request_id and keep unmapped notifications
                # buffered until start_load_kv syncs the mapping.
                pending_tids = self._pending_unmapped_done_tids
                self._pending_unmapped_done_tids = {}
                for tid, pending_ack in pending_tids.items():
                    self._translate_or_buffer_read_completion(
                        pending_ack if pending_ack is not None else tid,
                        done_sending,
                    )
                for completion_id in done_sending_raw:
                    if _completion_id_text(completion_id) in pending_tids:
                        continue
                    self._translate_or_buffer_read_completion(
                        completion_id, done_sending
                    )
                self._pop_zero_quorum_read_completions(done_sending)
                if self._pending_unmapped_done_tids:
                    logger.debug(
                        "get_finished (producer READ): %d tid(s) pending "
                        "transfer-id mapping",
                        len(self._pending_unmapped_done_tids),
                    )
            else:
                # WRITE mode: producer-local completions normally use scheduler
                # request_ids. Normalize transfer_ids as well because completion
                # delivery can race with start_load_kv.
                pending_tids = self._pending_unmapped_done_tids
                self._pending_unmapped_done_tids = {}
                for tid, pending_ack in pending_tids.items():
                    self._translate_or_buffer_completion(
                        pending_ack if pending_ack is not None else tid,
                        done_sending,
                    )
                for completion_id in done_sending_raw:
                    if _completion_id_text(completion_id) in pending_tids:
                        continue
                    self._translate_or_buffer_completion(completion_id, done_sending)
                if self._pending_unmapped_done_tids:
                    logger.debug(
                        "get_finished (producer WRITE): %d completion id(s) "
                        "pending request-id mapping",
                        len(self._pending_unmapped_done_tids),
                    )
        else:
            if self.mode == MoRIIOMode.WRITE:
                fresh = self.moriio_wrapper.pop_finished_write_req_ids()
                # Accumulate with any completions that arrived before their
                # transfer_id was registered in transfer_id_to_request_id.
                self._unmatched_write_completions |= fresh
                done_recving = self._unmatched_write_completions
            else:
                # READ mode: KV loads are synchronous (load_kv_async=False), so
                # requests go directly to RUNNING. We still call
                # _pop_done_transfers() to notify prefill and clean local state,
                # but skip done_recving for RUNNING requests.
                self._pop_done_transfers()

        # Translate consumer-side done_recving (transfer_ids reported by the
        # producer via send_notify in WRITE mode) back to the consumer's own
        # internal request_ids. Pop on success so the persistent worker map
        # (populated incrementally in start_load_kv) does not grow unbounded.
        translated_recving: set[str] = set()
        matched_xfer_ids: set[str] = set()
        for tid in done_recving:
            mapped = self._pop_mapped_completion_id(tid)
            if mapped is not None:
                translated_recving.add(mapped)
                matched_xfer_ids.add(tid)
        done_recving = translated_recving

        if self.mode == MoRIIOMode.WRITE and not self.is_producer:
            self._unmatched_write_completions -= matched_xfer_ids

        return done_sending, done_recving

    def _handle_failed_read_transfer_locked(
        self,
        req_id: ReqId,
        failed_status=None,
        error: Exception | None = None,
        record_invalid_blocks: bool = True,
    ) -> None:
        if failed_status is not None:
            message = failed_status.Message()
            code = failed_status.Code()
        else:
            message = str(error)
            code = "setup"
        invalid_block_ids = self._recving_transfer_local_block_ids.get(req_id, set())
        if record_invalid_blocks and invalid_block_ids:
            self._invalid_block_ids.put(set(invalid_block_ids))
            self._recving_transfer_local_block_ids[req_id] = set()
        logger.error(
            "RDMA transfer failed for request %s: %s (code=%s). Notifying "
            "prefill to free blocks%s.",
            req_id,
            message,
            code,
            " and marking local blocks invalid" if record_invalid_blocks else "",
        )
        callback_addr = self._recving_transfers_callback_addr.get(req_id)
        if callback_addr is not None:
            host, port, xfer_id = callback_addr
            try:
                self.moriio_wrapper.send_notify(
                    _make_read_completion_id(xfer_id, self.tp_rank), host, port
                )
            except Exception:
                logger.exception(
                    "Failed to send error notification for request %s; will retry",
                    req_id,
                )
                return
            self._forget_transfer_mapping(xfer_id, req_id)
        else:
            logger.warning(
                "No READ completion callback address for failed request %s",
                req_id,
            )
        self._recving_transfers.pop(req_id, None)
        self._pending_read_plans.pop(req_id, None)
        self._recving_transfers_callback_addr.pop(req_id, None)
        self._recving_transfer_local_block_ids.pop(req_id, None)

    @staticmethod
    def _is_sq_full_status(status) -> bool:
        """True if a MoRIIO transfer status is a transient RDMA send-queue-full
        rejection (retryable backpressure), not a terminal failure.

        The mori RDMA backend posts synchronously (the executor joins its worker
        before returning — executor.cpp:174-183 — and marks the status on the
        calling thread, backend_impl.cpp:1007), so an SQ-full rejection is a
        Failed() status the moment batch_read() returns. mori surfaces it as a
        generic ERR_RDMA_OP carrying "SQ full" in the message (no distinct code),
        so we match the message. Only meaningful once status.Failed() is True.
        """
        try:
            return bool(status.Failed()) and "SQ full" in (status.Message() or "")
        except Exception:
            return False

    @staticmethod
    def _read_status_active(status) -> bool:
        return not status.Succeeded() and not status.Failed()

    def _active_read_layer_count_locked(self) -> int:
        return sum(
            1
            for status_by_layer in self._recving_transfers.values()
            for status in status_by_layer.values()
            if self._read_status_active(status)
        )

    def _active_read_layers_for_req_locked(self, req_id: ReqId) -> int:
        status_by_layer = self._recving_transfers.get(req_id, {})
        return sum(
            1 for status in status_by_layer.values() if self._read_status_active(status)
        )

    def _per_transfer_read_cap(self) -> int:
        caps = [
            cap
            for cap in (
                self.moriio_config.max_inflight_per_transfer,
                self.moriio_config.max_dispatch_layers,
            )
            if cap > 0
        ]
        return min(caps) if caps else 0

    def _can_dispatch_read_plan_locked(self, req_id: ReqId) -> bool:
        active_layers = self._active_read_layers_for_req_locked(req_id)
        per_transfer_cap = self._per_transfer_read_cap()
        if per_transfer_cap > 0 and active_layers >= per_transfer_cap:
            return False

        global_cap = self.moriio_config.max_inflight_global
        return global_cap <= 0 or self._active_read_layer_count_locked() < global_cap

    def _take_dispatchable_read_plan(
        self, layer_name: str | None = None
    ) -> LayerTransferPlan | None:
        with self.moriio_wrapper.lock:
            for req_id, plans_by_layer in list(self._pending_read_plans.items()):
                if layer_name is None:
                    if not plans_by_layer:
                        continue
                    plan_layer_name = next(iter(plans_by_layer))
                else:
                    if layer_name not in plans_by_layer:
                        continue
                    plan_layer_name = layer_name

                if not self._can_dispatch_read_plan_locked(req_id):
                    continue

                plan = plans_by_layer.pop(plan_layer_name)
                if not plans_by_layer:
                    self._pending_read_plans.pop(req_id, None)
                return plan

        return None

    def _post_read_plan(self, plan: LayerTransferPlan) -> None:
        _sq_deadline = time.monotonic() + self.moriio_config.transfer_timeout
        _backoff = 0.001
        while True:
            try:
                transfer_status = self.moriio_wrapper.read_remote_data(
                    plan.transfer_sizes,
                    plan.transfer_local_offsets,
                    plan.transfer_remote_offsets,
                    plan.session,
                )
            except Exception as e:
                with self.moriio_wrapper.lock:
                    has_partial_status = bool(
                        self._recving_transfers.get(plan.request_id)
                    )
                    self._handle_failed_read_transfer_locked(
                        plan.request_id,
                        error=e,
                        record_invalid_blocks=has_partial_status,
                    )
                raise
            if not self._is_sq_full_status(transfer_status):
                break
            if time.monotonic() > _sq_deadline:
                logger.warning(
                    "MoRIIO READ send queue stayed full past transfer_timeout "
                    "for req %s layer %s; storing failed status (handled "
                    "non-fatally in wait_for_layer_load). Raise "
                    "VLLM_MORIIO_QP_PER_TRANSFER and/or "
                    "MORI_IO_SQ_BACKOFF_TIMEOUT_US if frequent.",
                    plan.request_id,
                    plan.layer_name,
                )
                break
            time.sleep(_backoff)
            _backoff = min(_backoff * 2, 0.05)

        with self.moriio_wrapper.lock:
            self._recving_transfers[plan.request_id][plan.layer_name] = transfer_status

    def _dispatch_pending_reads(self, layer_name: str | None = None) -> None:
        while True:
            plan = self._take_dispatchable_read_plan(layer_name)
            if plan is None:
                return
            self._post_read_plan(plan)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.is_producer or self.mode != MoRIIOMode.READ:
            return

        deadline = time.monotonic() + self.moriio_config.transfer_timeout
        while True:
            self._dispatch_pending_reads(layer_name)

            with self.moriio_wrapper.lock:
                pending_plan_req_ids = {
                    req_id
                    for req_id, plans_by_layer in self._pending_read_plans.items()
                    if layer_name in plans_by_layer
                }
                pending = [
                    (req_id, status_by_layer[layer_name])
                    for req_id, status_by_layer in self._recving_transfers.items()
                    if layer_name in status_by_layer
                ]

            if not pending and not pending_plan_req_ids:
                return

            still_running = False
            for req_id, status in pending:
                if status.Succeeded():
                    continue
                if status.Failed():
                    with self.moriio_wrapper.lock:
                        self._handle_failed_read_transfer_locked(req_id, status)
                    if self._is_sq_full_status(status):
                        # Treat sustained SQ-full as a request-local error: the
                        # blocks are already invalidated and prefill has been
                        # notified, so keep the worker alive and re-evaluate the
                        # remaining pending transfers.
                        logger.warning(
                            "MoRIIO READ send queue still full for req %s "
                            "layer %s after transfer_timeout; failed this "
                            "request (worker stays alive). Raise "
                            "VLLM_MORIIO_QP_PER_TRANSFER and/or "
                            "MORI_IO_SQ_BACKOFF_TIMEOUT_US if frequent.",
                            req_id,
                            layer_name,
                        )
                        continue
                    raise RuntimeError(
                        "MoRIIO READ transfer failed for "
                        f"request {req_id}, layer {layer_name}: "
                        f"{status.Message()} (code={status.Code()})"
                    )
                still_running = True

            if not still_running and not pending_plan_req_ids:
                self._dispatch_pending_reads()
                return

            if time.monotonic() > deadline:
                error = TimeoutError(
                    "Timed out waiting for MoRIIO READ transfer for "
                    f"layer {layer_name}; adjust with "
                    "kv_connector_extra_config.transfer_timeout"
                )
                with self.moriio_wrapper.lock:
                    req_ids = {req_id for req_id, _status in pending}
                    req_ids.update(pending_plan_req_ids)
                    for req_id in req_ids:
                        self._handle_failed_read_transfer_locked(req_id, error=error)
                raise error

            time.sleep(0.001)

    def _pop_done_transfers(self) -> set[str]:
        """Pop completed remote-read transfers and notify the producer.

        Sends the transfer_id (not the consumer's internal request_id) so the
        producer can translate it back to its own internal request_id; see
        get_finished() for the producer-side translation and the assign_request_id
        rationale.

        Returns an empty set because in READ mode the consumer scheduler does
        not track recv-completion (get_num_new_matched_tokens returns
        async=False, so requests never enter WAITING_FOR_REMOTE_KVS); reporting
        a recv-completion here would trip the scheduler assertion at
        _update_from_kv_xfer_finished. The downstream translation block in
        get_finished() therefore receives an empty set and is a no-op.
        """
        with self.moriio_wrapper.lock:
            to_remove = []
            for req_id, status_by_layer in list(self._recving_transfers.items()):
                statuses = list(status_by_layer.values())
                failed_status = next(
                    (status for status in statuses if status.Failed()), None
                )
                if (
                    statuses
                    and req_id not in self._pending_read_plans
                    and all(status.Succeeded() for status in statuses)
                ):
                    host, port, xfer_id = self._recving_transfers_callback_addr[req_id]
                    try:
                        self.moriio_wrapper.send_notify(
                            _make_read_completion_id(xfer_id, self.tp_rank),
                            host,
                            port,
                        )
                    except Exception:
                        logger.exception(
                            "MoRIIO READ completion notify failed for "
                            "request %s transfer %s; will retry",
                            req_id,
                            xfer_id,
                        )
                        continue
                    self._forget_transfer_mapping(xfer_id, req_id)
                    to_remove.append(req_id)
                elif failed_status is not None:
                    self._handle_failed_read_transfer_locked(req_id, failed_status)
                    # Do not add to done_req_ids: decode KV cache is incomplete.
                    # The request will expire via the normal request timeout.
            for req_id in to_remove:
                self._recving_transfers.pop(req_id, None)
                self._pending_read_plans.pop(req_id, None)
                self._recving_transfers_callback_addr.pop(req_id, None)
                self._recving_transfer_local_block_ids.pop(req_id, None)

            return set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        result: set[int] = set()
        while not self._invalid_block_ids.empty():
            try:
                result.update(self._invalid_block_ids.get_nowait())
            except queue.Empty:
                break
        return result

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
        waiting_write_engine_ids: set[EngineId] = set()
        written_req_ids: set[ReqId] = set()
        remote_engine_id = None

        for req_id, meta in metadata.reqs_to_save.items():
            # Gate on the DP rank that owns the remote allocation.
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )

            meta.remote_engine_id = remote_engine_id

            target_remote_engine_id = self.get_engine_name_with_dp(
                remote_engine_id, int(meta.remote_dp_rank)
            )
            if target_remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if target_remote_engine_id not in self._remote_agents:
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta
                        )
                        waiting_write_engine_ids.add(remote_engine_id)
                        continue
            self._write_blocks_for_req(req_id, meta, layer_name, kv_layer)
            written_req_ids.add(req_id)

        if remote_engine_id is None:
            return
        if waiting_write_engine_ids:
            _deadline = time.monotonic() + self.moriio_config.transfer_timeout
            while True:
                pending_engine_id = None
                for engine_id in waiting_write_engine_ids:
                    if engine_id not in self.write_ready_flags:
                        pending_engine_id = engine_id
                        break
                if pending_engine_id is None:
                    break
                if time.monotonic() > _deadline:
                    logger.warning(
                        "Timed out waiting for write_ready_flags for %s; "
                        "adjust with kv_connector_extra_config.transfer_timeout",
                        tuple(
                            engine_id
                            for engine_id in waiting_write_engine_ids
                            if engine_id not in self.write_ready_flags
                        ),
                    )
                    return
                time.sleep(0.001)

        if waiting_write_engine_ids or remote_engine_id in self.write_ready_flags:
            while True:
                try:
                    ready_req_id, ready_meta = self._ready_requests.get_nowait()
                except queue.Empty:
                    break
                if ready_req_id in written_req_ids:
                    continue
                self._write_blocks_for_req(
                    ready_req_id, ready_meta, layer_name, kv_layer
                )
                written_req_ids.add(ready_req_id)

    def get_engine_name_with_dp(self, engine_name, dp_rank):
        return f"{engine_name}_dp{dp_rank}"

    def _eager_handshake_all_dp_ranks(self, metadata: MoRIIOConnectorMetadata) -> None:
        """Handshake remote prefill DP ranks uniformly across local TP workers.

        Decode forward issues per-layer TP collectives, so handshake state needs
        to stay uniform across TP workers. The engine set comes from scheduler
        metadata shared across TP workers; each worker handshakes unresolved remote
        DP ranks, then enters the TP all-reduce vote together.

        Handshake exceptions are recorded before the collective and raised only
        after the vote, so all TP workers observe the same outcome.
        """
        import torch.distributed as dist

        # Distinct remote engines referenced this step, in metadata (==
        # scheduler) order so every TP worker iterates engines identically.
        engines: dict[str, ReqMeta] = {}
        for _req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )
            engines.setdefault(remote_engine_id, meta)

        for remote_engine_id, meta in engines.items():
            if remote_engine_id in self._eager_handshaked_engines:
                continue

            remote_dp_size = int(meta.remote_dp_size)
            port = int(meta.remote_handshake_port)
            tp_size = int(meta.tp_size)

            # Submit unresolved DP-rank handshakes while holding the lock; release
            # it before waits and the TP collective so stalled recv calls do not
            # block unrelated lock users.
            futures: list[tuple[str, Future[set[str]]]] = []
            with self._handshake_lock:
                for cur_dp_rank in range(remote_dp_size):
                    dp_engine_id = self.get_engine_name_with_dp(
                        remote_engine_id, cur_dp_rank
                    )
                    if dp_engine_id in self._remote_agents:
                        continue
                    host = self._pick_host_for_dp_rank(meta, cur_dp_rank)
                    fut = self._handshake_initiation_executor.submit(
                        self._moriio_handshake,
                        host,
                        port,
                        tp_size,
                        dp_engine_id,
                        cur_dp_rank,
                    )
                    futures.append((dp_engine_id, fut))

            # Join outside the lock. Bounded handshake errors are recorded here
            # and reported after the all-reduce.
            all_ok = True
            results: dict[str, set[str]] = {}
            for dp_engine_id, fut in futures:
                try:
                    results[dp_engine_id] = fut.result()
                except Exception:
                    logger.exception(
                        "Eager MoRIIO handshake failed for %s", dp_engine_id
                    )
                    all_ok = False

            with self._handshake_lock:
                for dp_engine_id, agents in results.items():
                    self._remote_agents[dp_engine_id] = agents

            # The CPU all-reduce is both the TP-uniform success vote and the
            # lockstep barrier. It blocks until all TP workers arrive, gives each
            # worker the same verdict, and stays off the model compute stream.
            logger.info(
                "Eager MoRIIO handshake: engine=%s dp_size=%d new_ranks=%d "
                "ok=%s tp_rank=%d",
                remote_engine_id,
                remote_dp_size,
                len(futures),
                all_ok,
                self.tp_rank,
            )
            vote = torch.tensor([1 if all_ok else 0], device="cpu", dtype=torch.int32)
            dist.all_reduce(vote, group=self.tp_group.cpu_group, op=dist.ReduceOp.MIN)
            if int(vote.item()) == 0:
                raise HandshakeError(
                    f"Eager MoRIIO handshake failed for {remote_engine_id} on "
                    f"at least one TP rank; failing this step fast to avoid a "
                    f"TP collective hang"
                )

            self._eager_handshaked_engines.add(remote_engine_id)

    def start_load_kv(self, metadata: MoRIIOConnectorMetadata):
        """
        Start loading by triggering non-blocking moriio_xfer.
        We check for these trnxs to complete in each step().
        """
        # Metadata carries only the scheduler-side mapping delta. Merge rather
        # than overwrite so the worker-side mapping survives after the
        # scheduler-side request_finished() unmaps a transfer_id. The producer
        # needs this entry to translate the consumer's transfer_id notification
        # (see get_finished) back to its own internal request_id, and that
        # notification can arrive several steps after request_finished.
        # get_finished() pops entries after a successful translation, so the
        # dict stays bounded.
        freed_transfer_ids = metadata.freed_transfer_ids
        freed_request_ids: set[ReqId] = set()
        for transfer_id in freed_transfer_ids:
            request_id = self.transfer_id_to_request_id.get(
                transfer_id
            ) or metadata.transfer_id_to_request_id.get(transfer_id)
            if request_id is not None:
                freed_request_ids.add(request_id)
                freed_request_ids.add(_strip_vllm_request_suffix(request_id))
            self._forget_transfer_mapping(transfer_id, request_id)
            self._drop_pending_unmapped_done_tid(transfer_id, request_id)
        self._unmatched_write_completions.difference_update(freed_transfer_ids)
        for transfer_id, remote_tp_size in getattr(
            metadata, "transfer_id_to_remote_tp_size", {}
        ).items():
            if transfer_id in freed_transfer_ids:
                continue
            self.transfer_id_to_remote_tp_size[transfer_id] = remote_tp_size
            self.transfer_id_to_completion_count[transfer_id] = _read_completion_quorum(
                remote_tp_size, self.tp_size, self.tp_rank
            )
        self._reqs_to_send.update(
            {
                req_id: deadline
                for req_id, deadline in getattr(metadata, "reqs_to_send", {}).items()
                if req_id not in freed_request_ids
            }
        )
        for transfer_id, req_id in metadata.transfer_id_to_request_id.items():
            if transfer_id in freed_transfer_ids:
                continue
            self._remember_transfer_mapping(transfer_id, req_id)
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

        # Eager all-rank handshake (TP-lockstep safe) before any read. Fires
        # once per remote engine; no-op on warm steps. Replaces the per-request
        # build-on-demand handshake that desynced the TP forward collective.
        self._eager_handshake_all_dp_ranks(metadata)

        wait_handshake_readd_req = False
        remote_engine_id = None

        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )
            meta.remote_engine_id = remote_engine_id
            target_remote_engine_id = self.get_engine_name_with_dp(
                remote_engine_id, int(meta.remote_dp_rank)
            )
            if target_remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if target_remote_engine_id not in self._remote_agents:
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
                while not self._ready_requests.empty():
                    self._read_blocks_for_req(*self._ready_requests.get_nowait())
                break
            else:
                break

    def wait_for_save(self, metadata: MoRIIOConnectorMetadata):
        if self.mode == MoRIIOMode.WRITE and self.is_producer:
            for layer_name, kv_layer in self.kv_caches.items():
                self.save_kv_layer(metadata, layer_name, kv_layer, None)
            self._writer.seal_pending_transfers()

    def _ensure_remote_dp_handshaked(self, meta: ReqMeta) -> None:
        """Build-on-demand handshake for the prefill DP rank this request reads from.

        With heterogeneous parallelism, a request can target any remote DP rank.
        If first-contact handshakes miss a rank, it has no cached metadata and
        _get_built_session would fail.

        Handshake the needed rank synchronously on first use. The result is cached
        in layer_name_to_remote_kv_cache_metadata and built_write_session, then
        reused thereafter; already-warmed ranks return immediately.
        """
        base_engine_id = str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
        dp_rank = int(meta.remote_dp_rank)
        dp_engine_id = self.get_engine_name_with_dp(base_engine_id, dp_rank)
        if dp_engine_id in self.layer_name_to_remote_kv_cache_metadata:
            return
        with self._handshake_lock:
            # Re-check under the lock — another worker step may have just
            # handshaked this rank.
            if dp_engine_id in self.layer_name_to_remote_kv_cache_metadata:
                return
            host = self._pick_host_for_dp_rank(meta, dp_rank)
            # Fallback only: eager handshakes normally cover every rank before
            # reads. Reaching this path means the rank was not cached yet.
            logger.warning(
                "MoRIIO fallback synchronous handshake for remote dp rank %d "
                "(%s) on the read path; eager handshake had not cached it",
                dp_rank,
                dp_engine_id,
            )
            self._remote_agents[dp_engine_id] = self._moriio_handshake(
                host,
                int(meta.remote_handshake_port),
                int(meta.tp_size),
                dp_engine_id,
                dp_rank,
            )

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        logger.debug(
            "Remote agent %s available, calling _read_blocks for req %s",
            meta.remote_engine_id,
            req_id,
        )
        actual_remote_host = self._pick_host_for_dp_rank(meta, int(meta.remote_dp_rank))
        self._read_blocks(
            request_id=req_id,
            transfer_id=meta.transfer_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            remote_host=actual_remote_host,
            remote_notify_port=meta.remote_notify_port,
            remote_dp_rank=meta.remote_dp_rank,
            remote_tp_size=int(meta.tp_size),
        )

    def _write_blocks_for_req(self, req_id: ReqId, meta: ReqMeta, layer_name, kv_layer):
        self.schedule_write_blocks(
            request_id=req_id,
            transfer_id=meta.transfer_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            layer_name=layer_name,
            kv_layer=kv_layer,
            remote_notify_port=meta.remote_notify_port,
            remote_ip=self._pick_remote_host(meta),
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
        remote_dp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> None:
        if self.mode == MoRIIOMode.WRITE:
            return

        # Use the prefill DP rank that actually computed the KV (forwarded by
        # the proxy via kv_transfer_params["remote_dp_rank"]). Hardcoding DP0
        # can read from a different rank's memory registration; per-DP ranks
        # may expose different num_blocks, so high block ids can exceed the
        # wrong rank's memory region.
        remote_dp_engine_id = self.get_engine_name_with_dp(
            dst_engine_id, int(remote_dp_rank)
        )
        sessions, remote_moriio_meta = self._get_built_session(remote_dp_engine_id)

        # Heterogeneous TP: target the remote TP index (tp0 for TP1 prefill).
        # Otherwise, the read-completion notify uses a port the producer does not own.
        notify_port = str(
            remote_notify_port
            + get_port_offset(
                int(remote_dp_rank),
                self._remote_tp_rank(int(remote_tp_size)),
                int(remote_tp_size),
            )
        )
        with self.moriio_wrapper.lock:
            self._recving_transfer_local_block_ids[request_id] = set(local_block_ids)
            self._recving_transfers_callback_addr[request_id] = (
                remote_host,
                notify_port,
                transfer_id,
            )
            for sess_idx, layer_name in enumerate(
                self.layer_name_to_local_kv_cache_metadata
            ):
                offs = self._compute_block_transfer_offsets(
                    layer_name,
                    local_block_ids,
                    remote_block_ids,
                    remote_moriio_meta,
                    remote_tp_size=remote_tp_size,
                )
                self._pending_read_plans[request_id][layer_name] = LayerTransferPlan(
                    request_id=request_id,
                    transfer_id=transfer_id,
                    layer_name=layer_name,
                    sess_idx=sess_idx,
                    transfer_local_offsets=offs[0],
                    transfer_remote_offsets=offs[1],
                    transfer_sizes=offs[2],
                    session=sessions[sess_idx],
                )

        self._dispatch_pending_reads()
