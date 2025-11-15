# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import logging
import math
import os
import queue
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
from vllm.attention import AttentionBackend
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
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
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

Transfer = tuple[int, float]  # (xfer_handle, start_time)
EngineId = str
ReqId = str

GET_META_MSG = b"get_meta_msg"

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    from nixl._bindings import nixlXferTelemetry

    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None
    nixlXferTelemetry = None


try:
    from nixl._api import nixl_agent_config
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
class NixlAgentMetadata(KVConnectorHandshakeMetadata):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    block_lens: list[int]
    attn_backend_name: str
    kv_cache_layout: str


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    # To be used when logical block size does not match the kernel block size
    local_physical_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    tp_size: int


class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
        save_to_host: bool = False,
    ):
        # save and load are mutually exclusive
        assert load_remote_cache ^ save_to_host
        _req = ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
        )
        if save_to_host:
            self.reqs_to_save[request_id] = _req
        if load_remote_cache:
            self.reqs_to_recv[request_id] = _req


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
        per_engine_labelvalues: dict[int, list[str]],
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
        attn_metadata: "AttentionMetadata",
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
            + vllm_config.parallel_config.data_parallel_rank
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
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}
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
            if not isinstance(rank_metadata, NixlAgentMetadata):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlAgentMetadata for "
                    "handshake metadata."
                )
            encoded_data[tp_rank] = encoder.encode(rank_metadata)
            logger.debug(
                "Tp rank %d: encoded NixlAgentMetadata size: %s bytes",
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

            # save all blocks
            block_ids = blocks.get_block_ids()[0]
            # TODO: skip the blocks that are already in the host xfer buffer.
            # Currently, the host xfer buffer block is 1-to-1 mapped to device
            # kv blocks, so host blocks won't be flushed as long as its device
            # block is not overwritten; and it will be safe to skip saving them
            # to host xfer buffer.
            if block_ids:
                self._reqs_need_save[request.request_id] = (request, block_ids)
        elif params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(
                    p in params
                    for p in ("remote_engine_id", "remote_host", "remote_port")
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
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                load_remote_cache=True,
                save_to_host=False,
            )

        for req_id, (req, block_ids) in self._reqs_need_save.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                load_remote_cache=False,
                save_to_host=True,
            )

        meta.reqs_to_send = self._reqs_need_send
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
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
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
        )


class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    @dataclass
    class TpKVTopology:
        """
        Helper class for tensor parallel and KV topology information for
        mapping between local and remote TP workers.
        """

        tp_size: int
        tp_rank: int
        remote_tp_size: dict[EngineId, int]
        is_mla: bool
        total_num_kv_heads: int
        attn_backend: type[AttentionBackend]

        def __post_init__(self):
            # Figure out whether the first dimension of the cache is K/V
            # or num_blocks. This is used to register the memory regions correctly.
            kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
            )
            # Non-MLA backends caches have 5 dims [2, num_blocks, H,N,D],
            # we just mock num_blocks to 1 for the dimension check below.
            self._is_kv_layout_blocks_first = (
                len(kv_cache_shape) == 5 and kv_cache_shape[0] == 1
            )

            attn_backend = AttentionBackendEnum[self.attn_backend.get_name()]
            self._use_pallas = attn_backend == AttentionBackendEnum.PALLAS

        @property
        def is_kv_layout_blocks_first(self) -> bool:
            return self._is_kv_layout_blocks_first

        @property
        def split_k_and_v(self) -> bool:
            # Whether to register regions for K and V separately (when present).
            return not (
                self.is_mla or self._use_pallas or self.is_kv_layout_blocks_first
            )

        def tp_ratio(
            self,
            remote_tp_size: int,
        ) -> int:
            """
            Calculate the tensor parallel ratio between local and remote TP.
            We can think of it as the number of local TP workers-per-remote TP
            workers. Local workers will read from the same remote TP worker in
            groups of size `tp_ratio`.
            """
            assert self.tp_size % remote_tp_size == 0, (
                f"Local tensor parallel size {self.tp_size} is not divisible "
                f"by remote tensor parallel size {remote_tp_size}."
            )
            return self.tp_size // remote_tp_size

        def tp_ratio_from_engine_id(
            self,
            remote_engine_id: EngineId,
        ) -> int:
            remote_tp_size = self.remote_tp_size[remote_engine_id]
            return self.tp_ratio(remote_tp_size)

        def is_kv_replicated(self, engine_id: EngineId) -> bool:
            """
            Whether the KV cache is replicated across TP workers due to the
            number of TP workers being greater than the number of KV heads.
            """
            tp_size = self.remote_tp_size[engine_id]
            return tp_size // self.total_num_kv_heads >= 1

        def replicates_kv_cache(self, remote_engine_id: EngineId) -> bool:
            # MLA is always replicated as the hidden dim can't be split.
            return self.is_mla or self.is_kv_replicated(remote_engine_id)

        def get_target_remote_rank(
            self,
            remote_tp_size: int,
        ) -> int:
            """
            Get the remote TP rank (on P) that the current local TP rank
            (on D) will read from.
            """
            tp_ratio = self.tp_ratio(remote_tp_size)
            return self.tp_rank // tp_ratio

        def get_target_remote_rank_from_engine_id(
            self,
            remote_engine_id: EngineId,
        ) -> int:
            remote_tp_size = self.remote_tp_size[remote_engine_id]
            return self.get_target_remote_rank(remote_tp_size)

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
        # TODO temporary, once nixl allows for telemetry flag in config
        # (next release), we can remove this env var.
        os.environ["NIXL_TELEMETRY_ENABLE"] = "1"

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
            config = (
                nixl_agent_config(backends=self.nixl_backends)
                if len(non_ucx_backends) > 0
                else nixl_agent_config(num_threads=num_threads)
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
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}
        self.device_id: int = 0

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = 0

        # nixl_prepped_dlist_handle.
        self.src_xfer_side_handle: int = 0
        # Map of engine_id -> nixl_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: dict[EngineId, int] = {}

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[Transfer]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}
        # Set of requests that have been part of a batch, regardless of status.
        self._reqs_to_process: set[ReqId] = set()

        # invalid blocks from failed NIXL operations
        self._invalid_block_ids: set[int] = set()
        # requests that skipped transfer (handshake or transfer failures)
        self._failed_recv_reqs: set[ReqId] = set()

        # Handshake metadata of this worker for NIXL transfers.
        self.xfer_handshake_metadata: NixlAgentMetadata | None = None
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

        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            use_mla=self.use_mla,
        )
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        self.host_buffer_kv_cache_layout = self.kv_cache_layout
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        # With heterogeneous TP, P must wait for all assigned D TP workers to
        # finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
        self.xfer_stats = NixlKVConnectorStats()

        self.kv_topo = self.TpKVTopology(
            tp_size=self.world_size,
            tp_rank=self.tp_rank,
            remote_tp_size=self._tp_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=backend,
        )
        self._use_pallas = self.kv_topo._use_pallas
        self._physical_blocks_per_logical_kv_block = 1

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        """Do a NIXL handshake with a remote instance."""

        start_time = time.perf_counter()

        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.

        # Handshake only with the remote TP rank that current local rank will
        # pull from. With homogeneous TP it happens to be the same rank_i.
        p_remote_rank = self.kv_topo.get_target_remote_rank(remote_tp_size)
        path = make_zmq_path("tcp", host, port)
        logger.debug(
            "Querying metadata on path: %s at remote tp rank %s", path, p_remote_rank
        )

        # Send query for the request.
        with zmq_ctx(zmq.REQ, path) as sock:
            msg = msgspec.msgpack.encode((GET_META_MSG, p_remote_rank))
            # Set receive timeout to 5 seconds to avoid hanging on dead server
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
            sock.send(msg)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.debug(
                "NIXL handshake: get metadata took: %s", got_metadata_time - start_time
            )

            # Ensure engine id matches.
            if metadata.engine_id != expected_engine_id:
                raise RuntimeError(
                    f"Remote NIXL agent engine ID mismatch. "
                    f"Expected {expected_engine_id},"
                    f"received {metadata.engine_id}."
                )

            # Register Remote agent.
            remote_agent_name = self.add_remote_agent(
                metadata, p_remote_rank, remote_tp_size
            )
            setup_agent_time = time.perf_counter()
            logger.debug(
                "NIXL handshake: add agent took: %s",
                setup_agent_time - got_metadata_time,
            )

        # Remote rank -> agent name.
        return {p_remote_rank: remote_agent_name}

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu)
        """
        xfer_buffers: dict[str, torch.Tensor] = {}
        try:
            for layer_name, kv_cache in kv_caches.items():
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
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
                    kv_shape = tuple(kv_shape[i] for i in [0, 1, 3, 2, 4])
                xfer_buffers[layer_name] = torch.empty(
                    kv_shape, dtype=kv_dtype, device="cpu"
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

    def _background_nixl_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do NIXL handshake in background and add to _ready_requests when done.
        fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                meta.remote_host,
                meta.remote_port,
                meta.tp_size,
                remote_engine_id,
            )
            self._handshake_futures[remote_engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=remote_engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            fut.add_done_callback(done_callback)

        # check handshake success before proceeding with request
        def request_ready(f: Future[Any], entry=(req_id, meta)):
            try:
                # check if handshake succeeded
                f.result()
                self._ready_requests.put(entry)
            except Exception:
                # handshake failed - mark blocks as invalid
                logger.exception(
                    "Handshake failed for request %s, marking blocks as invalid", req_id
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
        # Enable different block lengths for different layers when MLA is used.
        self.block_len_per_layer = list[int]()
        self.slot_size_per_layer = list[int]()  # HD bytes in kv terms
        self.device_id = self.tp_rank
        for layer_name, cache_or_caches in xfer_buffers.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if not self.use_host_buffer and current_platform.is_cuda_alike():
                    self.device_id = cache.device.index
                if base_addr in seen_base_addresses:
                    continue

                # TODO (NickLucche): Get kernel_block_size in a cleaner way
                # NHD default "view" for non-MLA cache
                kernel_block_size = cache.shape[-2] if self.use_mla else cache.shape[-3]

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
                # Torch uses -1 to indicate CPU tensors while NIXL uses explicit
                # memory type.
                self.device_id = max(cache.get_device(), 0)
                caches_data.append(
                    (base_addr, curr_tensor_size_bytes, self.device_id, "")
                )

        logger.debug(
            "Different block lengths collected: %s", set(self.block_len_per_layer)
        )
        assert len(self.block_len_per_layer) == len(seen_base_addresses)
        assert self.num_blocks != 0

        self.kv_caches_base_addr[self.engine_id] = seen_base_addresses
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
        blocks_data = []
        for i, base_addr in enumerate(seen_base_addresses):
            kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=i)
            # NOTE With heter-TP, more blocks are prepared than what are
            # needed as self.num_blocks >= nixl_agent_meta.num_blocks. We
            # could create fewer, but then _get_block_descs_ids needs to
            # select agent_meta.num_blocks instead of self.num_blocks for
            # local descr, and that makes handling regular flow less clean.
            for block_id in range(self.num_blocks):
                block_offset = block_id * self.block_len_per_layer[i]
                addr = base_addr + block_offset
                # (addr, len, device id)
                blocks_data.append((addr, kv_block_len, self.device_id))

            if self.kv_topo.is_kv_layout_blocks_first:
                # Separate and interleave K/V regions to maintain the same
                # descs ordering. This is needed for selecting contiguous heads
                # when split across TP ranks.
                for block_id in range(self.num_blocks):
                    block_offset = block_id * self.block_len_per_layer[i]
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
        self.src_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs
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
        self.xfer_handshake_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            device_id=self.device_id,
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            attn_backend_name=self.backend_name,
            kv_cache_layout=self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout,
        )

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
        The latter, assuming D.world_size > P.world_size, requires that two or
        more local TP worker share the xfer from a single TP worker.

        Here's an example (non-MLA case):

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

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        # Handle tp_size>num_kv_heads: replicate KV cache.
        replicates_kv_cache = self.kv_topo.replicates_kv_cache(engine_id)

        # Create dst descs and xfer side handles. TP workers have same #blocks
        # so we only register once per engine_id.
        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        # Keep track of remote agent kv caches base addresses.
        self.kv_caches_base_addr[engine_id] = nixl_agent_meta.kv_caches_base_addr

        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        # Number of D TP workers reading from a single P TP worker. This is
        # 1 when P and D `--tensor-parallel-size` match.
        tp_ratio = self.kv_topo.tp_ratio_from_engine_id(engine_id)

        ### Register remote agent memory regions
        blocks_data = []
        # With homogeneous TP, D pulls the whole kv cache from corresponding
        # rank. With heterogeneous TP, prepare the descriptors by splitting the
        # P KV cache along kv_head dim, of D worker's kv_head size (D>P).
        # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].

        # Register all remote blocks, but only the corresponding kv heads.
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=i)
            rank_offset = (
                self.tp_rank % tp_ratio * kv_block_len if not replicates_kv_cache else 0
            )
            for block_id in range(nixl_agent_meta.num_blocks):
                block_offset = block_id * nixl_agent_meta.block_lens[i]
                # For each block, grab the heads chunk belonging to rank_i
                # of size remote_nheads // tp_ratio, which correspond to
                # self.block_len == remote_block_len//tp_ratio bytes.
                addr = base_addr + block_offset + rank_offset
                # (addr, len, device id)
                blocks_data.append((addr, kv_block_len, nixl_agent_meta.device_id))

            if self.kv_topo.is_kv_layout_blocks_first:
                # With FlashInfer index V separately to allow head splitting.
                for block_id in range(nixl_agent_meta.num_blocks):
                    block_offset = block_id * nixl_agent_meta.block_lens[i]
                    addr = base_addr + block_offset + rank_offset
                    v_addr = addr + nixl_agent_meta.block_lens[i] // 2
                    blocks_data.append(
                        (v_addr, kv_block_len, nixl_agent_meta.device_id)
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
        self.dst_xfer_side_handles[engine_id] = self.nixl_wrapper.prep_xfer_dlist(
            remote_agent_name, descs
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
        # TODO We may eventually want to skip enforcing the same attn backend.
        assert nixl_agent_meta.attn_backend_name == self.backend_name

        tp_ratio = self.kv_topo.tp_ratio_from_engine_id(remote_engine_id)
        assert tp_ratio > 0, "Decode TP cannot be smaller than prefill TP"
        assert not self._use_pallas or tp_ratio == 1, (
            "TPU (pallas_v1) DOES NOT support heterogeneous TP yet."
        )
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
            assert self.block_len_per_layer == nixl_agent_meta.block_lens, (
                "KV cache sizes must match between P and D when replicated"
            )
            remote_block_size = remote_block_len // (self.slot_size_per_layer[0])
        else:
            # When MLA is not used, this is a list of the same block length
            for block_len in nixl_agent_meta.block_lens:
                assert block_len == remote_block_len, (
                    "All remote layers must have the same block size"
                )
            remote_block_size = remote_block_len // (
                self.slot_size_per_layer[0] * tp_ratio
            )
            if self.kv_topo.is_kv_layout_blocks_first:
                # With flashinfer, KV are sent in the same message.
                remote_block_size //= 2

            assert remote_block_len == self.block_len_per_layer[0] * tp_ratio, (
                "Remote P worker KV layer cache must be of shape [2, N, "
                "local_kv_heads*tp_ratio, block_size, head_dim] and same dtype."
            )

        assert self.block_size == remote_block_size, (
            "Remote P worker with different page/block size is not supported "
            f"{self.block_size=}, {remote_block_size=}"
        )

        # TP workers have same #blocks.
        assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks

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

    def permute_device_kv(self, block_ids: list[int]):
        """Transforms the layout of received KV cache blocks to the local format.

        This method corrects layout mismatches from direct memory copies by
        permuting the tensor dimensions.

        - **Source Layout:** `[num_blocks, n_kv_head, block_size, head_dim]`
        - **Target Layout:** `[num_blocks, block_size, n_kv_head, head_dim]`

        Args:
            block_ids: A list of block IDs to update and permute.

        Implementation:
        - x = blocks_to_update.reshape(src_shape) # view local kv with sender layout
        - permuted_blocks = x.permute(*inv_order) # transpose n_kv_heads, block_size
        - cache.index_copy_(0, indices, permuted_blocks) # copy permuted kv back

        """
        split_k_and_v = self.kv_topo.split_k_and_v
        inv_order = [0, 2, 1, 3]
        sample_cache = list(self.device_kv_caches.values())[0][0]
        target_shape = list(sample_cache.shape)
        target_shape[0] = -1
        src_shape = tuple(target_shape[i] for i in inv_order)
        indices = torch.tensor(block_ids, device=sample_cache.device)

        for _, cache_or_caches in self.device_kv_caches.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            for cache in cache_list:
                blocks_to_update = cache.index_select(0, indices)
                permuted_blocks = blocks_to_update.reshape(src_shape).permute(
                    *inv_order
                )
                cache.index_copy_(0, indices, permuted_blocks)

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

        block_ids_to_permute = []
        for req_id in done_recving:
            # clean up metadata for completed requests
            meta = self._recving_metadata.pop(req_id, None)
            assert meta is not None, f"{req_id} not found in recving_metadata list"
            if self.use_host_buffer:
                self.sync_recved_kv_to_device(req_id, meta)
            if self.enable_permute_local_kv:
                block_ids_to_permute += meta.local_physical_block_ids
        if len(block_ids_to_permute) > 0:
            self.permute_device_kv(block_ids_to_permute)

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
                req_id, tp_ratio = notif.decode("utf-8").rsplit(":", 1)
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

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if self.consumer_notification_counts_by_req[req_id] == int(tp_ratio):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def _pop_done_transfers(
        self, transfers: dict[str, list[tuple[int, float]]]
    ) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = False
            for handle, _xfer_stime in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    # Get telemetry from NIXL
                    res = self.nixl_wrapper.get_xfer_telemetry(handle)
                    self.xfer_stats.record_transfer(res)
                    self.nixl_wrapper.release_xfer_handle(handle)
                elif xfer_state == "PROC":
                    in_progress = True
                    continue
                else:
                    # transfer failed - mark blocks as invalid
                    logger.error(
                        "NIXL transfer failed for request %s with state %s. "
                        "Marking blocks as invalid.",
                        req_id,
                        xfer_state,
                    )
                    # mark all (logical)blocks for this request as invalid
                    if meta := self._recving_metadata.pop(req_id, None):
                        self._invalid_block_ids.update(meta.local_block_ids)
                    self._recving_metadata.pop(req_id, None)
                    self.nixl_wrapper.release_xfer_handle(handle)
                    self.xfer_stats.record_failed_transfer()
            if not in_progress:
                done_req_ids.add(req_id)
                del transfers[req_id]
        return done_req_ids

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            meta.remote_block_ids = self._logical_to_kernel_block_ids(
                meta.remote_block_ids
            )
            remote_engine_id = meta.remote_engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote_block_ids),
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
        logger.debug(
            "Remote agent %s available, calling _read_blocks for req %s",
            meta.remote_engine_id,
            req_id,
        )
        self._read_blocks(
            request_id=req_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_physical_block_ids,
            remote_block_ids=meta.remote_block_ids,
        )

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_engine_id: str,
        request_id: str,
    ):
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes. We should remove the staging
        # blocks until we are ready.

        # Number of D TP workers that will read from dst P. Propagate tp_ratio
        # on notification so that dst worker can wait before freeing blocks.
        tp_ratio = self.kv_topo.tp_ratio_from_engine_id(dst_engine_id)
        notif_id = f"{request_id}:{tp_ratio}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            remote_rank = self.kv_topo.get_target_remote_rank_from_engine_id(
                dst_engine_id
            )
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception:
                logger.exception(
                    "NIXL send_notif failed for request %s: "
                    "P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    request_id,
                )
                self.xfer_stats.record_failed_notification()
            return

        # Partial prefix cache hit: just read uncomputed blocks.
        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        # Get side handles.
        local_xfer_side_handle = self.src_xfer_side_handle
        remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        local_block_descs_ids: np.ndarray
        remote_block_descs_ids: np.ndarray
        if not self.block_window_per_layer:
            # Default case: assume global attention
            remote_block_descs_ids = self._get_block_descs_ids(
                dst_engine_id, remote_block_ids
            )
            local_block_descs_ids = self._get_block_descs_ids(
                self.engine_id, local_block_ids
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
                    self.engine_id, layer_local_block_ids, layer_idx
                )
                layer_remote_desc_ids = self._get_block_descs_ids(
                    dst_engine_id, layer_remote_block_ids, layer_idx
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
            self._recving_transfers[request_id].append((handle, time.perf_counter()))
        except Exception:
            logger.exception(
                "NIXL transfer setup/initiation failed for request %s. "
                "Marking blocks as invalid.",
                request_id,
            )
            # mark all (logical) blocks for this request as invalid
            if meta := self._recving_metadata.get(request_id):
                self._invalid_block_ids.update(meta.local_block_ids)
            self.xfer_stats.record_failed_transfer()
            if handle is not None:
                self.nixl_wrapper.release_xfer_handle(handle)
            self._failed_recv_reqs.add(request_id)

    def _get_block_descs_ids(
        self, engine_id: str, block_ids: list[int], layer_idx: int | None = None
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

    def get_backend_aware_kv_block_len(self, layer_idx: int):
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
            for handle, _ in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._recving_transfers.clear()
        if self.src_xfer_side_handle:
            self.nixl_wrapper.release_dlist_handle(self.src_xfer_side_handle)
            self.src_xfer_side_handle = 0
        for dst_xfer_side_handle in self.dst_xfer_side_handles.values():
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
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90) * 1e3, 3),
            "Avg post time (ms)": round(post_time.mean() * 1e3, 3),
            "P90 post time (ms)": round(np.percentile(post_time, 90) * 1e3, 3),
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
        per_engine_labelvalues: dict[int, list[str]],
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
