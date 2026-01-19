# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import logging
import math
import os
import queue
import sys
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import numpy as np
import torch
import zmq

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TpKVTopology,
    get_current_attn_backend,
    kv_postprocess_blksize_and_layout_on_receive,
    kv_postprocess_blksize_on_receive,
    kv_postprocess_layout_on_receive,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

TransferHandle = int
ReqId = str

#
# NIXL Connector Version
#
# Increment this version whenever there is an incompatible change to:
#   - NixlAgentMetadata schema
#   - kv_transfer_params schema or semantics
#   - NIXL transfer protocol or wire format
#   - KV cache memory layout or block organization
#   - Any other change that breaks P/D interoperability
#
# Version History:
#   1: Initial version with compatibility checking
#   2: Add remote_request_id to kv_transfer_params
#
NIXL_CONNECTOR_VERSION: int = 2

GET_META_MSG = b"get_meta_msg"

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    if "UCX_MEM_MMAP_HOOK_MODE" not in os.environ:
        # avoid a memory leak in UCX when using NIXL on some models
        # see: https://github.com/vllm-project/vllm/issues/24264
        if "nixl" in sys.modules or "rixl" in sys.modules:
            logger.warning(
                "NIXL was already imported, we can't disable UCX mmap hooks. "
                "Please set UCX_MEM_MMAP_HOOK_MODE to 'none' manually."
            )
        else:
            logger.info(
                "Setting UCX_MEM_MMAP_HOOK_MODE to 'none' to avoid a rare "
                "memory leak in UCX when using NIXL."
            )
            os.environ["UCX_MEM_MMAP_HOOK_MODE"] = "none"

    if not current_platform.is_rocm():
        from nixl._api import nixl_agent as NixlWrapper
        from nixl._bindings import nixlXferTelemetry
    else:
        from rixl._api import nixl_agent as NixlWrapper
        from rixl._bindings import nixlXferTelemetry

    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None
    nixlXferTelemetry = None


try:
    if not current_platform.is_rocm():
        from nixl._api import nixl_agent_config
    else:
        from rixl._api import nixl_agent_config
except ImportError:
    nixl_agent_config = None
    logger.warning("NIXL agent config is not available")

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": ("cpu",),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())


@dataclass
class NixlAgentMetadata:
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    block_lens: list[int]
    kv_cache_layout: str
    block_size: int


@dataclass
class NixlHandshakePayload(KVConnectorHandshakeMetadata):
    """
    Wrapper for NIXL handshake sent over the wire.

    Enables two-phase decoding for graceful compatibility checking:
    1. Decode NixlHandshakePayload to get compatibility_hash
    2. Compute local hash and compare
    3. Only if hashes match, decode agent_metadata_bytes

    This prevents decoder errors when NixlAgentMetadata schema is
    incompatible, allowing graceful failure with clear error message.
    """

    compatibility_hash: str
    agent_metadata_bytes: bytes  # NixlAgentMetadata encoded


def compute_nixl_compatibility_hash(
    vllm_config: VllmConfig, attn_backend_name: str
) -> str:
    """
    Compute compatibility hash for NIXL KV transfer.

    Hash only the factors that affect whether two NIXL instances can
    successfully transfer KV cache data.

    Factors included:
    - vLLM version and NIXL connector version
    - Model architecture (name, dtype, KV heads, layers)
    - KV cache format (dtype, sliding window)
    - Attention backend

    Note: Factors like tensor_parallel_size, block_size, and kv_cache_layout
    are validated at runtime in _validate_remote_agent_handshake and are not
    included in this hash to support heterogeneous deployments.

    Note - the set of factors are likely to evolve significantly over
    time to be more or less permissive.

    Returns:
        SHA-256 hex digest
    """
    from vllm import __version__ as vllm_version
    from vllm.config.utils import hash_factors

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config

    factors = {
        # Version compatibility
        "vllm_version": vllm_version,
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,
        # Model architecture - affects KV cache shape
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        # Attention backend and KV cache dtype affect memory layout
        "attn_backend_name": attn_backend_name,
        "cache_dtype": str(cache_config.cache_dtype),
    }

    compat_hash = hash_factors(factors)
    logger.debug(
        "NIXL compatibility hash: %s (model=%s, dtype=%s, num_kv_heads=%d, "
        "cache_dtype=%s, attn_backend=%s)",
        compat_hash,
        factors["model"],
        factors["dtype"],
        factors["num_kv_heads"],
        factors["cache_dtype"],
        attn_backend_name,
    )
    return compat_hash


@dataclass
class RemoteMeta:
    block_ids: list[int]
    host: str
    port: int
    engine_id: str
    request_id: str


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    # To be used when logical block size does not match the kernel block size
    local_physical_block_ids: list[int]
    tp_size: int
    remote: RemoteMeta | None = None


class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()

    def _add_new_req(
        self,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ) -> ReqMeta:
        return ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
        )

    def add_new_req_to_save(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ):
        self.reqs_to_save[request_id] = self._add_new_req(
            local_block_ids, kv_transfer_params
        )

    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ):
        req = self._add_new_req(local_block_ids, kv_transfer_params)
        req.remote = RemoteMeta(
            block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
        )
        self.reqs_to_recv[request_id] = req


class NixlConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: NixlConnectorScheduler | None = (
                NixlConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: NixlConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorker(vllm_config, self.engine_id)

    ############################################################
    # Class Methods
    ############################################################
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        use_mla = vllm_config.model_config.use_mla
        if use_mla:
            # return None when we have mla
            # as the layout should not matter in that case,
            # which fallback to the default behavior.
            return None
        logger.info_once(
            "NixlConnector setting KV cache layout to HND for better xfer performance."
        )
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
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

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        assert self.connector_worker is not None
        self.connector_worker.set_host_xfer_buffer_ops(copy_operation)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get block IDs that failed to load via NIXL."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            NixlKVConnectorStats(data=data)
            if data is not None
            else NixlKVConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return NixlPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """NixlConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        if self.connector_worker.use_host_buffer and self.connector_worker.copy_blocks:
            self.connector_worker.save_kv_to_host(self._connector_metadata)

    def shutdown(self):
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata


class NixlConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        assert vllm_config.kv_transfer_config is not None
        if current_platform.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = (
                vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
            )

        logger.info("Initializing NIXL Scheduler %s", engine_id)

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: threading.Thread | None = None
        self._encoded_xfer_handshake_metadata: dict[int, Any] = {}
        self._stop_event = threading.Event()

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, Request] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self._reqs_in_batch: set[ReqId] = set()
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

    def shutdown(self):
        self._stop_event.set()
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join()
            self._nixl_handshake_listener_t = None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        encoded_data: dict[int, bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        for tp_rank, rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            encoded_data[tp_rank] = encoder.encode(rank_metadata)
            logger.debug(
                "Tp rank %d: encoded NixlHandshakePayload size: %s bytes",
                tp_rank,
                str(len(encoded_data[tp_rank])),
            )
        self._encoded_xfer_handshake_metadata = encoded_data

        # Only start the listener when we have metadata to serve.
        if self._nixl_handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    encoded_data,
                    ready_event,
                    self._stop_event,
                    self.side_channel_port,
                ),
                daemon=True,
                name="nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[int, Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        port: int,
    ):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        # Listen for new requests for metadata.
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                # Decode the message which contains (GET_META_MSG, rank)
                msg, target_tp_rank = msgspec.msgpack.decode(msg)
                logger.debug(
                    "Received message for tp rank %s",
                    target_tp_rank,
                )
                if msg != GET_META_MSG:
                    logger.warning("Connection listener got unexpected message %s", msg)
                sock.send_multipart((identity, b"", encoded_data[target_tp_rank]))

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
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

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_decode"):
            self._reqs_in_batch.add(request.request_id)
        if self.use_host_buffer and params.get("do_remote_decode"):
            # NOTE: when accelerator is not directly supported by Nixl,
            # prefilled blocks need to be saved to host memory before transfer.
            self._reqs_need_save[request.request_id] = request
        elif params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(
                    p in params
                    for p in (
                        "remote_engine_id",
                        "remote_request_id",
                        "remote_host",
                        "remote_port",
                    )
                ):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the D worker. We need to call
                    # send_notif in _read_blocks to free the memory on the P.
                    local_block_ids = (
                        blocks.get_unhashed_block_ids()
                        if num_external_tokens > 0
                        else []
                    )
                    # Get unhashed blocks to pull from remote.
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
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        # NOTE: For the prefill side, there might be a chance that an early added
        # request is a chunked prefill, so we need to check if new blocks are added
        for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
            req_to_save = self._reqs_need_save.get(req_id)
            if req_to_save is None or new_block_id_groups is None:
                continue
            req = req_to_save

            assert req.kv_transfer_params is not None
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=new_block_id_groups[0],
                kv_transfer_params=req.kv_transfer_params,
            )
            assert scheduler_output.num_scheduled_tokens is not None
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            is_partial = (
                req.num_computed_tokens + num_scheduled_tokens
            ) < req.num_prompt_tokens
            if not is_partial:
                # For non-partial prefills, once new req_meta is scheduled, it
                # can be removed from _reqs_need_save.
                # For partial prefill case, we will retain the request in
                # _reqs_need_save until all blocks are scheduled with req_meta.
                # Therefore, only pop if `not is_partial`.
                self._reqs_need_save.pop(req_id)

        meta.reqs_to_send = self._reqs_need_send
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
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
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(request.request_id)
            # Clear _reqs_need_save if a request is aborted as partial prefill.
            self._reqs_need_save.pop(request.request_id, None)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            logger.debug(
                "NIXLConnector request_finished(%s) waiting for %d seconds "
                "for remote decode to fetch blocks",
                request.request_id,
                envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT,
            )
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT
            )

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
        )


class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        if vllm_config.kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set for NixlConnector")
        self.kv_transfer_config = vllm_config.kv_transfer_config

        self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config(
            "backends", ["UCX"]
        )

        # Agent.
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
        # Each UCX thread allocates UARs (doorbell pages) via DevX, and
        # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
        # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
        # initialization with "mlx5dv_devx_alloc_uar" errors.
        # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
        num_threads = vllm_config.kv_transfer_config.get_from_extra_config(
            "num_threads", 4
        )
        if nixl_agent_config is None:
            config = None
        else:
            # Enable telemetry by default for NIXL 0.7.1 and above.
            config = (
                nixl_agent_config(backends=self.nixl_backends, capture_telemetry=True)
                if len(non_ucx_backends) > 0
                else nixl_agent_config(num_threads=num_threads, capture_telemetry=True)
            )

        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), config)
        # Map of engine_id -> {rank0: agent_name0, rank1: agent_name1..}.
        self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)

        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.num_blocks = 0
        self.enable_permute_local_kv = False

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # cpu kv buffer for xfer
        # used when device memory can not be registered under nixl
        self.host_xfer_buffers: dict[str, torch.Tensor] = {}
        if self.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = self.kv_buffer_device == "cpu"

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        nixl_memory_type = current_platform.get_nixl_memory_type()
        if nixl_memory_type is None:
            if self.kv_buffer_device == "cuda":
                nixl_memory_type = "VRAM"
            elif self.kv_buffer_device == "cpu":
                nixl_memory_type = "DRAM"
        if nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.nixl_memory_type = nixl_memory_type

        # Note: host xfer buffer ops when use_host_buffer is True
        self.copy_blocks: CopyBlocksOp | None = None

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        self.device_id: int = 0
        # Current rank may pull from multiple remote TP workers.
        # EngineId, dict[int, list[int]] -> engine_id, tp_rank, base_addr_for_layer
        self.kv_caches_base_addr = defaultdict[EngineId, dict[int, list[int]]](dict)

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = 0

        # nixl_prepped_dlist_handle.
        self.src_xfer_handles_by_block_size: dict[int, int] = {}
        # Populated dynamically during handshake based on remote configuration.
        # Keep track of regions at different tp_ratio values. tp_ratio->handles
        self.src_xfer_handles_by_tp_ratio: dict[int, list[int]] = {}
        # Map of engine_id -> {tp_rank: nixl_prepped_dlist_handle (int)}.
        self.dst_xfer_side_handles = defaultdict[EngineId, dict[int, int]](dict)

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}
        # Set of requests that have been part of a batch, regardless of status.
        self._reqs_to_process: set[ReqId] = set()

        # invalid blocks from failed NIXL operations
        self._invalid_block_ids: set[int] = set()
        # requests that skipped transfer (handshake or transfer failures)
        self._failed_recv_reqs: set[ReqId] = set()

        # Handshake metadata of this worker for NIXL transfers.
        self.xfer_handshake_metadata: NixlHandshakePayload | None = None
        # Background thread for initializing new NIXL handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # NIXL is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator",
        )
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        # List of block window sizes for each layer for local attention
        self.block_window_per_layer: list[int | None] = []
        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        backend = get_current_attn_backend(vllm_config)

        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        self.host_buffer_kv_cache_layout = self.kv_cache_layout
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name
        )
        self.enforce_compat_hash = self.kv_transfer_config.get_from_extra_config(
            "enforce_handshake_compat", True
        )

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        # With heterogeneous TP, P must wait for all assigned D TP workers to
        # finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
        self.xfer_stats = NixlKVConnectorStats()

        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=backend,
        )
        self._physical_blocks_per_logical_kv_block = 1

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        """Do a NIXL handshake with a remote instance."""
        # When target instance TP > local TP, we need to perform multiple
        # handshakes. Do it in a single background job for simplicity.
        # Regardless, only handshake with the remote TP rank(s) that current
        # local rank will read from. Note that With homogeneous TP,
        # this happens to be the same single rank_i.
        p_remote_ranks = self.kv_topo.get_target_remote_ranks(remote_tp_size)
        remote_rank_to_agent_name = {}
        path = make_zmq_path("tcp", host, port)

        with zmq_ctx(zmq.REQ, path) as sock:
            for remote_rank in p_remote_ranks:
                logger.debug(
                    "Querying metadata on path: %s at remote tp rank %s",
                    path,
                    remote_rank,
                )

                start_time = time.perf_counter()
                # Send query for the request.
                msg = msgspec.msgpack.encode((GET_META_MSG, remote_rank))
                # Set receive timeout to 5 seconds to avoid hanging on dead server
                sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
                sock.send(msg)
                handshake_bytes = sock.recv()

                # Decode handshake payload to get compatibility hash
                handshake_decoder = msgspec.msgpack.Decoder(NixlHandshakePayload)
                try:
                    handshake_payload = handshake_decoder.decode(handshake_bytes)
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    raise RuntimeError(
                        f"Failed to decode NixlHandshakePayload. This likely indicates "
                        f"an incompatibility between connector version. Error: {e}"
                    ) from e

                got_metadata_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: get metadata took: %s",
                    got_metadata_time - start_time,
                )

                # Check compatibility hash BEFORE decoding agent metadata
                if (
                    self.enforce_compat_hash
                    and handshake_payload.compatibility_hash != self.compat_hash
                ):
                    raise RuntimeError(
                        f"NIXL compatibility hash mismatch. "
                        f"Local: {self.compat_hash}, "
                        f"Remote: {handshake_payload.compatibility_hash}. "
                        f"Prefill and decode instances have incompatible "
                        f"configurations. This may be due to: different vLLM versions,"
                        f" models, dtypes, KV cache layouts, attention backends, etc. "
                        f"Both instances must use identical configurations."
                        f"Disable this check using "
                        f'--kv-transfer-config \'{{"kv_connector_extra_config": '
                        f'{{"enforce_handshake_compat": false}}}}\''
                    )

                logger.info(
                    "NIXL compatibility check passed (hash: %s)",
                    handshake_payload.compatibility_hash,
                )

                # Decode agent metadata
                metadata_decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
                try:
                    metadata = metadata_decoder.decode(
                        handshake_payload.agent_metadata_bytes
                    )
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    # This should not happen if hash matched
                    raise RuntimeError(
                        f"Failed to decode NixlAgentMetadata. Error: {e}"
                    ) from e

                # Ensure engine id matches.
                if metadata.engine_id != expected_engine_id:
                    raise RuntimeError(
                        f"Remote NIXL agent engine ID mismatch. "
                        f"Expected {expected_engine_id},"
                        f"received {metadata.engine_id}."
                    )
                setup_agent_time = time.perf_counter()

                # Register Remote agent.
                remote_agent_name = self.add_remote_agent(
                    metadata, remote_rank, remote_tp_size
                )
                logger.debug(
                    "NIXL handshake: add agent took: %s",
                    setup_agent_time - got_metadata_time,
                )
                remote_rank_to_agent_name[remote_rank] = remote_agent_name
        return remote_rank_to_agent_name

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu)
        """
        xfer_buffers: dict[str, torch.Tensor] = {}
        inv_order = [0, 1, 3, 2, 4]
        try:
            for layer_name, kv_cache in kv_caches.items():
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
                permute_shape = False
                if (
                    self.kv_cache_layout == "NHD"
                    and self.vllm_config.kv_transfer_config is not None
                    and self.vllm_config.kv_transfer_config.enable_permute_local_kv
                ):
                    logger.info_once(
                        "'enable_permute_local_kv' flag is enabled while "
                        "device KV Layout is NHD. Init host buffer with"
                        " HND to better support Decode/Prefill TP_ratio > 1."
                    )
                    # Since NHD will not support Decode/Prefill TP_ratio > 1,
                    # we can leverage host_buffer for permute
                    self.host_buffer_kv_cache_layout = "HND"
                    kv_shape = (
                        tuple(kv_shape[i] for i in inv_order)
                        if not self.use_mla
                        else kv_shape
                    )
                    permute_shape = not self.use_mla

                xfer_buffers[layer_name] = torch.empty(
                    kv_shape, dtype=kv_dtype, device="cpu"
                )
                if permute_shape:
                    xfer_buffers[layer_name] = xfer_buffers[layer_name].permute(
                        inv_order
                    )
        except MemoryError as e:
            logger.error("NIXLConnectorWorker gets %s.", e)
            raise

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used."""
        # Set a no-op if the host buffer is not cpu.
        if self.kv_buffer_device != "cpu":
            return
        # Set a no-op if self.device_type is 'cpu'.
        if self.device_type == "cpu":
            return
        assert self.use_host_buffer
        self.copy_blocks = copy_operation

    def _log_failure(
        self,
        failure_type: str,
        req_id: str | None,
        msg: str = "",
        error: Exception | None = None,
        meta: ReqMeta | None = None,
        **extra_context,
    ):
        """Log transfer failure with structured context for easier debugging."""
        context: dict[str, Any] = {
            "failure_type": failure_type,
            "request_id": req_id,
            "engine_id": self.engine_id,
        }
        if meta is None and req_id is not None:
            # Try to get metadata from in progress transfers when not provided
            meta = self._recving_metadata.get(req_id)

        if meta and meta.remote:
            context.update(
                {
                    "remote_engine_id": meta.remote.engine_id,
                    "remote_request_id": meta.remote.request_id,
                    "remote_host": meta.remote.host,
                    "remote_port": meta.remote.port,
                    "num_local_blocks": len(meta.local_block_ids),
                    "num_remote_blocks": len(meta.remote.block_ids),
                    "local_block_ids_sample": meta.local_block_ids[:10],
                }
            )

        context.update(extra_context)
        if msg:
            failure_type = f"{failure_type}. {msg}"

        logger.error(
            "NIXL transfer failure: %s | Context: %s",
            failure_type,
            context,
            exc_info=error is not None,
            stacklevel=2,
        )

    def _background_nixl_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do NIXL handshake in background and add to _ready_requests when done.
        fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            assert meta.remote is not None
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                meta.remote.host,
                meta.remote.port,
                meta.tp_size,
                remote_engine_id,
            )
            self._handshake_futures[remote_engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=remote_engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception as e:
                        self._log_failure(
                            failure_type="handshake_setup_failed",
                            req_id=None,
                            error=e,
                            remote_engine_id=eid,
                        )

            fut.add_done_callback(done_callback)

        # check handshake success before proceeding with request
        def request_ready(f: Future[Any], entry=(req_id, meta)):
            try:
                # check if handshake succeeded
                f.result()
                self._ready_requests.put(entry)
            except Exception as e:
                # handshake failed - mark blocks as invalid
                self._log_failure(
                    failure_type="handshake_failed",
                    req_id=req_id,
                    error=e,
                    meta=meta,
                )
                if req_meta := self._recving_metadata.get(req_id):
                    self._invalid_block_ids.update(req_meta.local_block_ids)
                self._failed_recv_reqs.add(req_id)

        fut.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches), (
                f"host_buffer: {len(self.host_xfer_buffers)}, "
                f"kv_caches: {len(kv_caches)}"
            )
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}"
            )

        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s",
            self.use_mla,
            self.kv_buffer_device,
            self.use_host_buffer,
        )

        caches_data = []
        # With hybrid allocator, layers can share a kv cache tensor
        seen_base_addresses = []

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we can
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded NixlAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        # Conversely for FlashInfer, K and V are registered in the same region
        # to better exploit the memory layout (ie num_blocks is the first dim).
        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None

        # TODO (NickLucche): Get kernel_block_size in a cleaner way
        # NHD default "view" for non-MLA cache
        if self.device_type == "cpu":
            block_size_position = -2
        else:
            block_size_position = -2 if self.use_mla else -3

        # Enable different block lengths for different layers when MLA is used.
        self.block_len_per_layer = list[int]()
        self.slot_size_per_layer = list[int]()  # HD bytes in kv terms
        for layer_name, cache_or_caches in xfer_buffers.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                kernel_block_size = cache.shape[block_size_position]

                if self.block_size != kernel_block_size:
                    logger.info_once(
                        "User-specified logical block size (%s) does not match"
                        " physical kernel block size (%s). Using the latter. ",
                        self.block_size,
                        kernel_block_size,
                    )
                    self._physical_blocks_per_logical_kv_block = (
                        self.block_size // kernel_block_size
                    )
                    self.block_size = kernel_block_size
                    self._block_size[self.engine_id] = kernel_block_size

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.numel() * cache.element_size()

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )

                self.block_len_per_layer.append(
                    curr_tensor_size_bytes // self.num_blocks
                )
                self.slot_size_per_layer.append(
                    self.block_len_per_layer[-1] // self.block_size
                )

                if not self.use_mla:
                    # Different kv cache shape is not supported by HeteroTP
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All kv cache tensors must have the same size"
                    )
                # Need to make sure the device ID is non-negative for NIXL,
                # Torch uses -1 to indicate CPU tensors.
                self.device_id = max(cache.get_device(), 0)
                caches_data.append(
                    (base_addr, curr_tensor_size_bytes, self.device_id, "")
                )

        logger.debug(
            "Different block lengths collected: %s", set(self.block_len_per_layer)
        )
        assert len(self.block_len_per_layer) == len(seen_base_addresses)
        assert self.num_blocks != 0

        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
        self.num_regions = len(caches_data)
        self.num_layers = len(xfer_buffers.keys())

        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        logger.debug("Done registering descs")
        self._registered_descs.append(descs)

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        if self.kv_topo.is_kv_layout_blocks_first:
            for i in range(len(self.slot_size_per_layer)):
                assert self.slot_size_per_layer[i] % 2 == 0
                self.slot_size_per_layer[i] //= 2

            # NOTE (NickLucche) When FlashInfer is used, memory is registered
            # with joint KV for each block. This minimizes the overhead in
            # registerMem allowing faster descs queries. In order to be able to
            # split on kv_heads dim as required by heterogeneous TP, one must
            # be able to index K/V separately. Hence we double the number
            # of 'virtual' regions here and halve `block_len` below.
            self.num_regions *= 2

        # Register local/src descr for NIXL xfer.
        self.seen_base_addresses = seen_base_addresses
        self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
            self.register_local_xfer_handler(self.block_size)
        )

        # TODO(mgoin): Hybrid memory allocator is currently disabled for
        # models with local attention (Llama 4). Can remove this once enabled.
        if self.model_config.hf_config.model_type == "llama4":
            from transformers import Llama4TextConfig

            assert isinstance(self.model_config.hf_text_config, Llama4TextConfig)
            llama4_config = self.model_config.hf_text_config
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

        # After KV Caches registered, listen for new connections.
        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            kv_cache_layout=self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout,
            block_size=self.block_size,
        )
        # Wrap metadata in payload with hash for defensive decoding
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    def register_local_xfer_handler(
        self,
        block_size: int,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        """
        Function used for register local xfer handler with local block_size or
        Remote block_size.

        When local block_size is same as remote block_size, we use local block_size
        to register local_xfer_handler during init.

        When remote block size is less than local block size, we need to use
        register another local_xfer_handler using remote block len to ensure
        data copy correctness.
        """
        block_size_ratio = self.block_size // block_size
        blocks_data = []
        for i, base_addr in enumerate(self.seen_base_addresses):
            # The new block_len is using prefill block_len;
            # and num_blocks is multiple with N
            kv_block_len = (
                self.get_backend_aware_kv_block_len(layer_idx=i) // block_size_ratio
            )
            block_len_per_layer = self.block_len_per_layer[i] // block_size_ratio
            num_blocks = self.num_blocks * block_size_ratio
            for block_id in range(num_blocks):
                block_offset = block_id * block_len_per_layer
                addr = base_addr + block_offset
                # (addr, len, device id)
                blocks_data.append((addr, kv_block_len, self.device_id))

            if self.kv_topo.is_kv_layout_blocks_first:
                # Separate and interleave K/V regions to maintain the same
                # descs ordering. This is needed for selecting contiguous heads
                # when split across TP ranks.
                for block_id in range(num_blocks):
                    block_offset = block_id * block_len_per_layer
                    addr = base_addr + block_offset
                    # Register addresses for V cache (K registered first).
                    v_addr = addr + kv_block_len
                    blocks_data.append((v_addr, kv_block_len, self.device_id))
        logger.debug(
            "Created %s blocks for src engine %s and rank %s on device id %s",
            len(blocks_data),
            self.engine_id,
            self.tp_rank,
            self.device_id,
        )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        return self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs), blocks_data

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        """
        Add the remote NIXL agent and prepare the descriptors for reading cache
        blocks from remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i.
        The latter, in the case of D.world_size < P.world_size, requires that a
        local (D) TP worker reads from multiple remote (P) TP workers.
        Conversely, assuming D.world_size > P.world_size, two or more local TP
        workers will read from a single remote TP worker.

        Here's an example for the last case described above (non-MLA):

        rank_offset     p_remote_tp_rank
        (kv split no)
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Decoder TP workers                     Prefix TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2

        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]
        then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "HND" layout format.
        Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio
        first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
        along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.

        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" D TP workers.
        """  # noqa: E501
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank"
                "%s already exchanged metadata, skip handshake.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][remote_tp_rank]

        ### Register remote agent metadata
        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
        if engine_id not in self._block_size:
            self._block_size[engine_id] = nixl_agent_meta.block_size

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        # Create dst descs and xfer side handles. TP workers have same #blocks
        # so we only register once per engine_id.
        # Example:
        # block_size_ratio > 1:
        # remote:               | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|
        # local origin:|          0|          1|          8|         12|
        # local mapped:| 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
        block_size_ratio = self.kv_topo.block_size_ratio_from_engine_id(engine_id)

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        # Keep track of remote agent kv caches base addresses.
        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        # This is 1 when P and D `--tensor-parallel-size` match. Otherwise,
        # this is the ratio between the two sizes.
        tp_ratio = self.kv_topo.tp_ratio_from_engine_id(engine_id)

        # Handle tp_size>num_kv_heads: replicate KV cache.
        indexes_into_remote = (
            not self.kv_topo.replicates_kv_cache(engine_id) and tp_ratio > 0
        )

        logger.debug(
            "Registering remote agent (%s, rank %s) memory regions with tp_ratio %s",
            engine_id,
            remote_tp_rank,
            tp_ratio,
        )

        ### (Optional) Register local agent memory regions. MLA is not split.
        if (
            tp_ratio < 0
            and not self.use_mla
            and tp_ratio not in self.src_xfer_handles_by_tp_ratio
        ):
            # Remote tp_size > local tp_size: read from multiple remote ranks.
            # Logically "split" own regions into |tp_ratio| chunks. Mind that
            # we only do this once per remote tp_size (replica-friendly).
            self.src_xfer_handles_by_tp_ratio[tp_ratio] = []
            for i in range(-tp_ratio):
                blocks_data = []
                for memory_region in self.src_blocks_data:
                    addr, local_block_len, own_tp_rank = memory_region
                    # Computing block len layer by layer allows for different
                    # block sizes to be used.
                    remote_block_len = local_block_len // (-tp_ratio)
                    addr = addr + i * remote_block_len
                    blocks_data.append((addr, remote_block_len, own_tp_rank))
                descs = self.nixl_wrapper.get_xfer_descs(
                    blocks_data, self.nixl_memory_type
                )
                handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
                self.src_xfer_handles_by_tp_ratio[tp_ratio].append(handle)

        ### Register remote agent memory regions
        blocks_data = []
        # With homogeneous TP, D pulls the whole kv cache from corresponding
        # rank. With heterogeneous TP, prepare the descriptors by splitting the
        # P KV cache along kv_head dim, of D worker's kv_head size (D>P).
        # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].

        # Register all remote blocks, but only the corresponding kv heads.
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            # Read our whole local region size from remote.
            local_block_len = self.get_backend_aware_kv_block_len(layer_idx=i)
            remote_kv_block_len = local_block_len // block_size_ratio
            if block_size_ratio > 1:
                # using remote kv_block_len as transfer unit
                local_block_len = remote_kv_block_len

            if tp_ratio < 0 and not self.use_mla:
                # Remote tp is bigger: read a chunk of local region from remote
                local_block_len = local_block_len // (-tp_ratio)
            rank_offset = (
                self.tp_rank % tp_ratio * remote_kv_block_len
                if indexes_into_remote
                else 0
            )
            for block_id in range(nixl_agent_meta.num_blocks):
                block_offset = block_id * nixl_agent_meta.block_lens[i]
                # For each block, grab the heads chunk belonging to rank_i
                # of size remote_nheads // tp_ratio, which correspond to
                # self.block_len == remote_block_len//tp_ratio bytes.
                addr = base_addr + block_offset + rank_offset
                # (addr, len, device id)
                blocks_data.append((addr, local_block_len, nixl_agent_meta.device_id))

            if self.kv_topo.is_kv_layout_blocks_first:
                # With FlashInfer index V separately to allow head splitting.
                for block_id in range(nixl_agent_meta.num_blocks):
                    block_offset = block_id * nixl_agent_meta.block_lens[i]
                    addr = base_addr + block_offset + rank_offset
                    v_addr = addr + nixl_agent_meta.block_lens[i] // 2
                    blocks_data.append(
                        (v_addr, local_block_len, nixl_agent_meta.device_id)
                    )

        logger.debug(
            "Created %s blocks for dst engine %s with remote rank %s and local rank %s",
            len(blocks_data),
            engine_id,
            remote_tp_rank,
            self.tp_rank,
        )

        # Register with NIXL.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        self.dst_xfer_side_handles[engine_id][remote_tp_rank] = (
            self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        )

        if block_size_ratio > 1:
            # when prefill with smaller block_size, we need to init a
            # new handler with same block_len to match
            self.src_xfer_handles_by_block_size[nixl_agent_meta.block_size] = (
                self.register_local_xfer_handler(nixl_agent_meta.block_size)[0]
            )

        return remote_agent_name

    def _validate_remote_agent_handshake(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
    ):
        """
        Validate the remote agent handshake metadata ensuring the
        invariants hold true.
        """
        remote_engine_id = nixl_agent_meta.engine_id

        assert self._tp_size[remote_engine_id] == remote_tp_size

        tp_ratio = self.kv_topo.tp_ratio_from_engine_id(remote_engine_id)
        block_size_ratio = self.kv_topo.block_size_ratio_from_engine_id(
            remote_engine_id
        )
        # Num kv_heads > tp_size and P TP > D TP case, not supported
        assert not (tp_ratio < 0 and self.kv_topo.is_kv_replicated(remote_engine_id))

        kv_cache_layout = (
            self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout
        )
        if not self.use_mla and nixl_agent_meta.kv_cache_layout != kv_cache_layout:
            if (
                self.kv_transfer_config.enable_permute_local_kv
                and nixl_agent_meta.kv_cache_layout == "HND"
            ):
                logger.info(
                    "Remote is HND and local is NHD, enabled additional permute "
                    "on local device KV."
                )
                self.enable_permute_local_kv = True
            else:
                raise RuntimeError(
                    "Heterogeneous TP expects same kv_cache_layout. "
                    "Or enable experimental feature to use HND to NHD support by "
                    "setting 'enable_permute_local_kv'=True in --kv-transfer-config."
                )

        # Block len can only vary across layers when using MLA.
        remote_block_len = nixl_agent_meta.block_lens[0]
        if self.use_mla or self.kv_topo.is_kv_replicated(remote_engine_id):
            # With replicated KV cache, only the number of blocks can differ.
            for i in range(len(self.block_len_per_layer)):
                assert (
                    self.block_len_per_layer[i] // block_size_ratio
                    == nixl_agent_meta.block_lens[i]
                ), "KV cache sizes must match between P and D when replicated"
        else:
            # When MLA is not used, this is a list of the same block length
            for block_len in nixl_agent_meta.block_lens:
                assert block_len == remote_block_len, (
                    "All remote layers must have the same block size"
                )

            if tp_ratio > 0:
                # Remote tp is smaller: remote block_len size is bigger
                assert (
                    remote_block_len
                    == (self.block_len_per_layer[0] * tp_ratio) // block_size_ratio
                ), (
                    "Remote P worker KV layer cache must be of shape [2, N, "
                    "local_kv_heads*tp_ratio, page_size, head_dim] and same dtype."
                )  # noqa: E501
            else:
                assert block_size_ratio == 1, (
                    "Different local/remote block sizes are not supported when"
                    " P TP > D TP."
                )
                # Remote tp is bigger: remote block_len size is smaller
                assert remote_block_len == self.block_len_per_layer[0] // (-tp_ratio), (
                    "Remote P worker KV layer cache must be of shape [2, N, "
                    "local_kv_heads/tp_ratio, page_size, head_dim] and same dtype."
                )  # noqa: E501

        # TP workers that handhshake with same remote have same #blocks.
        assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks
        # Same number of regions/~layers.
        assert len(nixl_agent_meta.kv_caches_base_addr) == len(self.block_len_per_layer)

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        """copy recved kv from host buffer to device."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        local_block_ids = meta.local_physical_block_ids
        self.copy_blocks(
            self.host_xfer_buffers,
            self.device_kv_caches,
            local_block_ids,
            local_block_ids,
            "h2d",
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "synced recved kv of request[%s] to device kv buffer,"
                "local_block_ids: %s. ",
                req_id,
                ",".join(map(str, local_block_ids)),
            )

    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        """copy kv from device to host buffer."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        for req_id, meta in metadata.reqs_to_save.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "save_load_kv for request[%s] to host xfer buffer."
                    "local_block_ids: %s. ",
                    req_id,
                    ",".join(map(str, meta.local_physical_block_ids)),
                )
            # blocking
            self.copy_blocks(
                self.device_kv_caches,
                self.host_xfer_buffers,
                meta.local_physical_block_ids,
                meta.local_physical_block_ids,
                "d2h",
            )

    def post_process_device_kv_on_receive(
        self,
        block_size_ratio: int,
        block_ids_list: list[list[int]],
    ):
        """
        Post process device kv cache after receiving from remote.

        3 types of post processing supported:
            * kv_cache_postprocess_layout => convert from HND to NHD
            * kv_cache_postprocess_blksize => convert from small block size
              to large block size
            * kv_cache_postprocess_blksize_and_layout => convert from small
              block size to large block size and convert from HND to NHD

        """
        if len(self.device_kv_caches) == 0:
            return
        assert block_size_ratio >= 1, "Only nP < nD supported currently."
        if self.enable_permute_local_kv and block_size_ratio > 1:
            logger.debug(
                "Post-processing device kv cache on receive by converting "
                "block_size with %sx bigger and permuting layout from HND"
                " to NHD.",
                block_size_ratio,
            )
        elif self.enable_permute_local_kv:
            logger.debug(
                "Post-processing device kv cache on receive by permuting layout"
                "from HND to NHD."
            )
        else:
            logger.debug(
                "Post-processing device kv cache on receive by converting "
                "block_size with %sx bigger.",
                block_size_ratio,
            )

        split_k_and_v = not (self.use_mla or self.kv_topo.is_kv_layout_blocks_first)

        for block_ids in block_ids_list:
            indices = torch.tensor(block_ids, device=self.device_type, dtype=torch.long)

            for _, cache_or_caches in self.device_kv_caches.items():
                cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
                for cache in cache_list:
                    if self.enable_permute_local_kv and block_size_ratio > 1:
                        kv_postprocess_blksize_and_layout_on_receive(
                            cache, indices, block_size_ratio
                        )
                    elif self.enable_permute_local_kv:
                        kv_postprocess_layout_on_receive(cache, indices)
                    else:
                        kv_postprocess_blksize_on_receive(
                            cache, indices, block_size_ratio
                        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)

        # add requests that skipped transfer to done_recving
        done_recving.update(self._failed_recv_reqs)
        self._failed_recv_reqs.clear()

        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(done_sending),
                len(done_recving),
            )

        block_ids_for_blocksize_post_process = defaultdict(list)
        for req_id in done_recving:
            # clean up metadata for completed requests
            meta = self._recving_metadata.pop(req_id, None)
            assert meta is not None, f"{req_id} not found in recving_metadata list"
            assert meta.remote is not None
            if self.use_host_buffer:
                self.sync_recved_kv_to_device(req_id, meta)

            # post processing for heteroblocksize
            block_size_ratio = self.kv_topo.block_size_ratio_from_engine_id(
                meta.remote.engine_id
            )
            if not self.use_mla and (
                block_size_ratio > 1 or self.enable_permute_local_kv
            ):
                block_ids_for_blocksize_post_process[block_size_ratio].append(
                    meta.local_physical_block_ids
                )
        for (
            block_size_ratio,
            block_ids_list,
        ) in block_ids_for_blocksize_post_process.items():
            self.post_process_device_kv_on_receive(block_size_ratio, block_ids_list)

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        while self._reqs_to_send:
            req_id, expires = next(iter(self._reqs_to_send.items()))
            # Sorted dict, oldest requests are put first so we can exit early.
            if now < expires:
                break
            count = self.consumer_notification_counts_by_req.pop(req_id, 0)
            logger.warning(
                "Releasing expired KV blocks for request %s which were "
                "retrieved by %d decode worker(s) within %d seconds.",
                req_id,
                count,
                envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT,
            )
            self._reqs_to_process.remove(req_id)
            del self._reqs_to_send[req_id]
            done_sending.add(req_id)

        return done_sending, done_recving

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.
        """
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                req_id, tp_size = notif.decode("utf-8").rsplit(":", 1)
                if (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    logger.error(
                        "Potentially invalid KV blocks for "
                        "unrecognized request %s were retrieved by "
                        "a decode worker. They may have expired.",
                        req_id,
                    )
                    continue

                # NOTE: `tp_ratio` is the opposite when swapping local<>remote
                n_consumers = int(tp_size)
                tp_ratio = self.kv_topo.tp_ratio(n_consumers)

                # Number of reads *per producer* to wait for.
                # When remote D TP > local P TP we expect `tp_ratio` reads.
                consumers_per_producer = (
                    -tp_ratio if n_consumers > self.world_size else 1
                )

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    == consumers_per_producer
                ):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def _pop_done_transfers(self, transfers: dict[str, list[int]]) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = []
            for handle in handles:
                try:
                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                    if xfer_state == "DONE":
                        # Get telemetry from NIXL
                        res = self.nixl_wrapper.get_xfer_telemetry(handle)
                        self.xfer_stats.record_transfer(res)
                        self.nixl_wrapper.release_xfer_handle(handle)
                    elif xfer_state == "PROC":
                        in_progress.append(handle)
                        continue
                    else:
                        self._log_failure(
                            failure_type="transfer_failed",
                            msg="Marking blocks as invalid",
                            req_id=req_id,
                            xfer_state=xfer_state,
                        )
                        self._handle_failed_transfer(req_id, handle)
                except Exception as e:
                    self._log_failure(
                        failure_type="transfer_exception",
                        msg="Marking blocks as invalid",
                        req_id=req_id,
                        error=e,
                    )
                    self._handle_failed_transfer(req_id, handle)

            if not in_progress:
                # Only report request as completed when all transfers are done.
                done_req_ids.add(req_id)
                del transfers[req_id]
            else:
                transfers[req_id] = in_progress
        return done_req_ids

    def _handle_failed_transfer(self, req_id: str, handle: int):
        """
        Handle a failed transfer by marking all (logical) blocks as invalid and
        recording the failure.

        Args:
            req_id: The request ID.
            handle: The transfer handle.
        """
        # Use .get() here as the metadata cleanup is handled by get_finished()
        if meta := self._recving_metadata.get(req_id):
            self._invalid_block_ids.update(meta.local_block_ids)
        self.nixl_wrapper.release_xfer_handle(handle)
        self.xfer_stats.record_failed_transfer()

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            assert meta.remote is not None
            meta.remote.block_ids = self._logical_to_kernel_block_ids(
                meta.remote.block_ids
            )
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            # always store metadata for failure recovery
            self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Remove all requests that are not to be processed (eg aborted).
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        # Add to requests that are waiting to be read and track expiration.
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None
        remote_ranks = self.kv_topo.get_target_remote_ranks_from_engine_id(
            meta.remote.engine_id
        )
        tp_ratio = self.kv_topo.tp_ratio_from_engine_id(meta.remote.engine_id)
        # D may have to perform multiple reads from different remote ranks.
        for i, remote_rank in enumerate(remote_ranks):
            if self.use_mla and tp_ratio < 0 and i > 0:
                # MLA opt: when P TP > D TP, only a single read is executed for
                # the first remote rank (cache is duplicated)..
                break

            remote_block_size = self.kv_topo.remote_block_size[meta.remote.engine_id]
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                remote_rank,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if tp_ratio < 0 and not self.use_mla:
                assert remote_block_size == self.block_size
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][i]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                remote_rank
            ]
            self._read_blocks(
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_block_ids=meta.local_physical_block_ids,
                remote_block_ids=meta.remote.block_ids,
                remote_rank=remote_rank,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )

            if self.use_mla and tp_ratio < 0:
                # ..but we still need to notify the other remote ranks that we
                # have the blocks we need so they can update the request state.
                notif_id = f"{req_id}:{self.world_size}".encode()
                remote_agents = self._remote_agents[meta.remote.engine_id]
                for rank_to_notify, agent in remote_agents.items():
                    if rank_to_notify != remote_rank:
                        self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        remote_rank: int,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ):
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.
        """
        block_size_ratio = self.kv_topo.block_size_ratio_from_engine_id(dst_engine_id)
        if block_size_ratio > 1:
            local_block_ids = self.get_mapped_blocks(
                np.asarray(local_block_ids), block_size_ratio
            )
            if len(local_block_ids) > len(remote_block_ids):
                # NOTE:
                # get_mapped_blocks will always expand block_ids for n times.
                # ex:
                # prefill block_ids with block_size as 4:
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # Local decode block_ids with block_size as 16: [1, 2, 3]
                # expland ecode block_ids with get_mapped_blocks from [1, 2, 3] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                # Then we clip local to align with prefill
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                local_block_ids = local_block_ids[: len(remote_block_ids)]
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes. We should remove the staging
        # blocks until we are ready.

        # Number of D TP workers that will read from dst P. Propagate info
        # on notification so that dst worker can wait before freeing blocks.
        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()
            return

        # Partial prefix cache hit: just read uncomputed blocks.
        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        local_block_descs_ids: np.ndarray
        remote_block_descs_ids: np.ndarray

        if not self.block_window_per_layer:
            # Default case: assume global attention
            remote_block_descs_ids = self._get_block_descs_ids(
                dst_engine_id,
                remote_block_ids,
            )
            local_block_descs_ids = self._get_block_descs_ids(
                self.engine_id,
                local_block_ids,
                block_size_ratio=block_size_ratio,
            )
        else:
            # TODO(mgoin): remove this once we have hybrid memory allocator
            # Optimization for models with local attention (Llama 4)
            local_descs_list = []
            remote_descs_list = []
            for layer_idx, block_window in enumerate(self.block_window_per_layer):
                # For each layer:
                if block_window is None:
                    # If not chunked, we just use the
                    # full block lists (global attention)
                    layer_local_block_ids = local_block_ids
                    layer_remote_block_ids = remote_block_ids
                else:
                    # If chunked, get the last block_window blocks
                    layer_local_block_ids = local_block_ids[-block_window:]
                    layer_remote_block_ids = remote_block_ids[-block_window:]

                # Get descs ids for the layer.
                layer_local_desc_ids = self._get_block_descs_ids(
                    dst_engine_id,
                    layer_local_block_ids,
                    layer_idx,
                )
                layer_remote_desc_ids = self._get_block_descs_ids(
                    self.engine_id,
                    layer_remote_block_ids,
                    layer_idx,
                    block_size_ratio=block_size_ratio,
                )

                local_descs_list.append(layer_local_desc_ids)
                remote_descs_list.append(layer_remote_desc_ids)

            local_block_descs_ids = np.concatenate(local_descs_list)
            remote_block_descs_ids = np.concatenate(remote_descs_list)

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        # Prepare transfer with Nixl.
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            self._recving_transfers[request_id].append(handle)
        except Exception as e:
            # mark all (logical) blocks for this request as invalid
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            if meta := self._recving_metadata.get(request_id):
                self._invalid_block_ids.update(meta.local_block_ids)
            self.xfer_stats.record_failed_transfer()
            if handle is not None:
                self.nixl_wrapper.release_xfer_handle(handle)
            self._failed_recv_reqs.add(request_id)

    def get_mapped_blocks(self, block_ids, block_size_ratio):
        """
          Calculates the new set of block IDs by mapping every element
          in the (potentially sparse) input array.
          Example: block_ids=[0, 2], block_size_ratio=2
        get_mapped_blocks    0     1     [2     3]     4     5
              # remote is |h0-b0|h1-b0||h0-b1|h1-b1||h0-b1|h1-b1||
              # local is  |h0-b0......||h1-b0......||h2-b0........
        local_block_ids         0           [1]           2
        """
        if block_ids.size == 0:
            return np.array([], dtype=np.int64)

        start_ids = block_ids * block_size_ratio
        offsets = np.arange(block_size_ratio)
        mapped_2d = start_ids[:, None] + offsets[None, :]

        return mapped_2d.flatten().astype(np.int64)

    def _get_block_descs_ids(
        self,
        engine_id: str,
        block_ids: list[int],
        layer_idx: int | None = None,
        block_size_ratio: float | None = None,
    ) -> np.ndarray:
        """
        Get the descs ids for a set of block ids.
        If layer_idx is provided, we use the region_ids for the given layer.
        Otherwise, we use all regions.
        """
        if layer_idx is None:
            region_ids = np.arange(self.num_regions)
        else:
            assert layer_idx < self.num_layers
            if self.num_layers < self.num_regions:
                # If we have more regions than layers, we assume that
                # the regions are organized as [K0, V0, K1, V1, ...]
                # and we select K_i and V_i
                assert 2 * self.num_layers == self.num_regions
                region_ids = np.arange(2 * layer_idx, 2 * layer_idx + 2)
            else:
                # Otherwise, we assume we have MLA and select i-th layer
                assert self.num_layers == self.num_regions
                region_ids = np.arange(layer_idx, layer_idx + 1)

        num_blocks = self.dst_num_blocks[engine_id]
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)

        # Compute the desc ids for each block.
        region_ids = region_ids[:, None]
        block_ids = np.array(block_ids)[None, :]
        descs_ids = region_ids * num_blocks + block_ids
        return descs_ids.flatten()

    def _logical_to_kernel_block_ids(self, block_ids: list[int]) -> list[int]:
        """
        Convert logical block ids to kernel physical block ids.
        This is required when the logical block size (the one set by the user)
        does not match the one required by the attn backend.
        """
        if self._physical_blocks_per_logical_kv_block == 1:
            # Noop when physical and logical block sizes are the same
            return block_ids
        block_ids_np = np.array(block_ids)
        block_arange = np.arange(0, self._physical_blocks_per_logical_kv_block).reshape(
            1, -1
        )
        return BlockTable.map_to_kernel_blocks(
            block_ids_np, self._physical_blocks_per_logical_kv_block, block_arange
        ).tolist()

    def get_backend_aware_kv_block_len(self, layer_idx: int) -> int:
        """
        Get the block length for one K/V element (K and V have the same size).

        For FA and other backends, this is equal to the length of the whole
        block, as K and V are in separate regions.
        For FlashInfer, this is half the length of the whole block, as K and V
        share the same region.
        """
        if self.kv_topo.is_kv_layout_blocks_first:
            # For indexing only half (either just the K or V part).
            block_len = self.block_len_per_layer[layer_idx] // 2
        else:
            block_len = self.block_len_per_layer[layer_idx]
        return block_len

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """
        # Clear stats for next iteration
        if not self.xfer_stats.is_empty():
            return self.xfer_stats.clone_and_reset()
        return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Return and clear the set of block IDs that failed to load.

        This is called by the scheduler to identify blocks that need
        to be retried after a NIXL transfer failure.
        """
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown the connector worker."""
        self._handshake_initiation_executor.shutdown(wait=False)
        for handles in self._recving_transfers.values():
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._recving_transfers.clear()
        for handle in self.src_xfer_handles_by_block_size.values():
            self.nixl_wrapper.release_dlist_handle(handle)
        self.src_xfer_handles_by_block_size.clear()
        for handles in self.src_xfer_handles_by_tp_ratio.values():
            for handle in handles:
                self.nixl_wrapper.release_dlist_handle(handle)
        self.src_xfer_handles_by_tp_ratio.clear()
        for dst_xfer_side_handles in self.dst_xfer_side_handles.values():
            for dst_xfer_side_handle in dst_xfer_side_handles.values():
                self.nixl_wrapper.release_dlist_handle(dst_xfer_side_handle)
        self.dst_xfer_side_handles.clear()
        for remote_agents in self._remote_agents.values():
            for agent_name in remote_agents.values():
                self.nixl_wrapper.remove_remote_agent(agent_name)
        self._remote_agents.clear()
        for desc in self._registered_descs:
            self.nixl_wrapper.deregister_memory(desc)
        self._registered_descs.clear()


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[float]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
        }

    def record_transfer(self, res: nixlXferTelemetry):
        # Keep metrics units consistent with rest of the code: time us->s
        self.data["transfer_duration"].append(res.xferDuration / 1e6)
        self.data["post_duration"].append(res.postDuration / 1e6)
        self.data["bytes_transferred"].append(res.totalBytes)
        self.data["num_descriptors"].append(res.descCount)

    def record_failed_transfer(self):
        """Record a failed NIXL transfer operation."""
        self.data["num_failed_transfers"].append(1.0)

    def record_failed_notification(self):
        """Record a failed NIXL notification (send_notif)."""
        self.data["num_failed_notifications"].append(1.0)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return self.num_successful_transfers == 0

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging
        if self.is_empty():
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
            }

        xfer_time = np.asarray(self.data["transfer_duration"])
        post_time = np.asarray(self.data["post_duration"])
        # Convert to MB for CLI logging.
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n

        total_time_seconds = xfer_time.sum()
        throughput_mb_s = total_mb / total_time_seconds

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg post time (ms)": round(post_time.mean() * 1e3, 3),
            "P90 post time (ms)": round(np.percentile(post_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(descs.mean(), 1),
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class NixlPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
        nixl_histogram_xfer_time = self._histogram_cls(
            name="vllm:nixl_xfer_time_seconds",
            documentation="Histogram of transfer duration for NIXL KV Cache transfers.",
            buckets=buckets[1:],
            labelnames=labelnames,
        )
        self.nixl_histogram_xfer_time = self.make_per_engine(nixl_histogram_xfer_time)
        nixl_histogram_post_time = self._histogram_cls(
            name="vllm:nixl_post_time_seconds",
            documentation="Histogram of transfer post time for NIXL KV"
            " Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_post_time = self.make_per_engine(nixl_histogram_post_time)
        # uniform 2kb to 16gb range
        buckets = [2 ** (10 + i) for i in range(1, 25, 2)]
        nixl_histogram_bytes_transferred = self._histogram_cls(
            name="vllm:nixl_bytes_transferred",
            documentation="Histogram of bytes transferred per NIXL KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_bytes_transferred = self.make_per_engine(
            nixl_histogram_bytes_transferred
        )
        buckets = [
            10,
            20,
            30,
            50,
            75,
            100,
            200,
            400,
            1000,
            2000,
            4000,
            10000,
            20000,
            50000,
        ]
        nixl_histogram_num_descriptors = self._histogram_cls(
            name="vllm:nixl_num_descriptors",
            documentation="Histogram of number of descriptors per NIXL"
            "  KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_num_descriptors = self.make_per_engine(
            nixl_histogram_num_descriptors
        )
        counter_nixl_num_failed_transfers = self._counter_cls(
            name="vllm:nixl_num_failed_transfers",
            documentation="Number of failed NIXL KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_transfers = self.make_per_engine(
            counter_nixl_num_failed_transfers
        )
        counter_nixl_num_failed_notifications = self._counter_cls(
            name="vllm:nixl_num_failed_notifications",
            documentation="Number of failed NIXL KV Cache notifications.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_notifications = self.make_per_engine(
            counter_nixl_num_failed_notifications
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for prom_obj, list_item_key in zip(
            [
                self.nixl_histogram_xfer_time,
                self.nixl_histogram_post_time,
                self.nixl_histogram_bytes_transferred,
                self.nixl_histogram_num_descriptors,
            ],
            [
                "transfer_duration",
                "post_duration",
                "bytes_transferred",
                "num_descriptors",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)
        for counter_obj, counter_item_key in zip(
            [
                self.counter_nixl_num_failed_transfers,
                self.counter_nixl_num_failed_notifications,
            ],
            ["num_failed_transfers", "num_failed_notifications"],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)
