# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TpKVTopology,
    get_current_attn_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_local_first_rank,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    logger.warning(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    )
    TransferEngine = None

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)


_KVSERVE_TRACE_LOCK = threading.Lock()


def _trace_kvserve_transfer(event: str, **fields: Any) -> None:
    """Append an optional, per-transfer observability record.

    The Connector runs in a worker process whose stdout is often owned by a
    process manager. A separate JSONL trace makes compressed and raw paths
    independently auditable without changing the vLLM connector contract.
    """
    path = os.environ.get("KVSERVE_TRACE_PATH")
    if not path:
        return
    record = {
        "timestamp_ns": time.time_ns(),
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    try:
        with _KVSERVE_TRACE_LOCK, open(path, "a", encoding="utf-8") as trace:
            trace.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except OSError:
        logger.exception("Unable to write KVServe transfer trace to %s", path)


@dataclass(frozen=True)
class TransferRegion:
    base_addr: int
    block_len: int
    kv_block_len: int


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return the TP ratio used by heterogeneous TP transfer planning.

    Positive values mean one local rank maps into a larger remote KV region.
    Negative values mean one local rank must gather from multiple remote KV
    regions.
    """
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {local_tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size

    assert remote_tp_size % local_tp_size == 0, (
        f"Remote tensor parallel size {remote_tp_size} is not divisible "
        f"by local tensor parallel size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    is_kv_layout_blocks_first: bool,
) -> list[TransferRegion]:
    """Expand registered KV tensors into the regions transferred by Mooncake."""
    assert len(base_addrs) == len(block_lens), (
        "Mooncake transfer regions require matching numbers of base addresses "
        f"and block lengths, got {len(base_addrs)} and {len(block_lens)}."
    )
    regions: list[TransferRegion] = []
    for base_addr, block_len in zip(base_addrs, block_lens):
        kv_block_len = block_len // 2 if is_kv_layout_blocks_first else block_len
        regions.append(
            TransferRegion(
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
            )
        )
        if is_kv_layout_blocks_first:
            regions.append(
                TransferRegion(
                    base_addr=base_addr + kv_block_len,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                )
            )
    return regions


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank to one consumer-rank copy for heterogeneous TP."""
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len

    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )

    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len

    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _can_coalesce_block_transfers(
    local_region_block_len: int,
    remote_region_block_len: int,
    src_region_offset: int,
    dst_region_offset: int,
    transfer_len: int,
) -> bool:
    """Whether a contiguous block group can be emitted as one larger copy."""
    return (
        src_region_offset == 0
        and dst_region_offset == 0
        and transfer_len == local_region_block_len
        and transfer_len == remote_region_block_len
    )


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate transfer-region metadata for a fixed producer/consumer pair.

    This checks registered KV regions, not per-request block counts. A region
    corresponds to one registered KV tensor, or one K/V half after expansion
    for layouts that store K and V together.
    """
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    if producer_cache_replicated:
        return None

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )

    return None


def _get_tensor_dense_flag(tensor: torch.Tensor) -> bool | None:
    is_dense = getattr(tensor, "is_non_overlapping_and_dense", None)
    if callable(is_dense):
        return bool(is_dense())
    return None


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    # Compression is negotiated per request.  The data plane still uses the
    # same Mooncake write primitive; these addresses point to bounded decoder
    # staging buffers, never to Python-owned serialized tensors.
    compressed_reqs: dict[ReqId, bool] = msgspec.field(default_factory=dict)
    staging_addrs: dict[ReqId, int] = msgspec.field(default_factory=dict)
    staging_capacities: dict[ReqId, int] = msgspec.field(default_factory=dict)
    compression_wire_version: int = 0


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None
    # Descriptor-only response.  Body and auxiliary tensors remain in the
    # decoder staging buffer and are never put on this control channel.
    compressed_payloads: dict[ReqId, dict[str, Any]] = msgspec.field(
        default_factory=dict
    )


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0
    compressed: bool = False
    staging_slot: int = -1


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, self.engine_id)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            # This fallback mostly exists for unit tests that instantiate the
            # connector without a fully populated model config.
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        if vllm_config.model_config.use_mla:
            return None
        logger.info_once(
            "MooncakeConnector setting KV cache layout to HND for "
            "heterogeneous TP-safe KV transfer."
        )
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

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

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[int]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()

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
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert not self.is_kv_producer
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
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                local_block_ids = (
                    blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                )
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to PullReqMeta.
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()
            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()

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

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None

        assert not self.is_kv_consumer

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(params["transfer_id"])
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (request, block_ids)

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if TransferEngine is None:
            logger.error("Mooncake is not available")
            raise RuntimeError("Mooncake is not available")
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config

        self.engine = TransferEngine()
        self.hostname = get_ip()

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "mooncake_protocol", "rdma"
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_blocks = 0
        self.block_len_per_layer: list[int] = []
        self.seen_base_addresses: list[int] = []

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            raise ValueError(
                "Mooncake Transfer Engine does not support pipeline parallelism yet."
            )
        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}
        compression_config = kv_transfer_config.kv_connector_extra_config.get(
            "compression"
        )
        # Keep compatibility with the existing launch helper, which wraps the
        # actual CompressionManager spec in a runtime policy object.
        if isinstance(compression_config, dict) and "spec" in compression_config:
            compression_config = compression_config["spec"]
        self._compression_config = compression_config
        self._compression_adapter: Any | None = None
        self._compression_wire_version = 1
        self._compressed_payloads: dict[ReqId, dict[str, Any]] = {}
        self._staging_pool: Any | None = None
        self._workspace_pool: Any | None = None
        self._registered_addresses: list[int] = []
        self._shutdown = False
        self._staging_slot_bytes = int(
            kv_transfer_config.kv_connector_extra_config.get(
                "compression_staging_bytes", 256 * 1024 * 1024
            )
        )
        self._staging_slots = int(
            kv_transfer_config.kv_connector_extra_config.get(
                "compression_staging_slots", 2
            )
        )
        self._workspace_slot_bytes = int(
            kv_transfer_config.kv_connector_extra_config.get(
                "compression_workspace_bytes", self._staging_slot_bytes
            )
        )
        self._workspace_slots = int(
            kv_transfer_config.kv_connector_extra_config.get(
                "compression_workspace_slots", self._staging_slots
            )
        )
        from mooncake.kvserve.session import SessionBudget, TransferSessionManager

        self._session_manager = TransferSessionManager(
            SessionBudget(
                max_operations=int(
                    kv_transfer_config.kv_connector_extra_config.get(
                        "compression_max_operations", 128
                    )
                ),
                max_bytes=int(
                    kv_transfer_config.kv_connector_extra_config.get(
                        "compression_max_bytes", 2 * 1024 * 1024 * 1024
                    )
                ),
            )
        )

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                _, port = get_mooncake_bootstrap_addr(vllm_config)
                self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()
        # Guards finished_recving_reqs and _pending_compressed_recvs, which are
        # written from both the receiver loop thread and the worker main
        # thread (_apply_pending_compressed).  Without it, an add racing
        # fetch_finished_recving_reqs' read-then-clear is silently lost and
        # the request hangs in WAITING_FOR_REMOTE_KVS forever.
        self._recv_state_lock = threading.Lock()
        self._load_error_block_ids: set[int] = set()
        self._pending_compressed_recvs: dict[ReqId, tuple[dict[str, Any], PullReqMeta]] = {}
        self._inflight_compressed_buffers: dict[TransferId, list[torch.Tensor]] = {}

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        backend = get_current_attn_backend(vllm_config)
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=[backend],
        )

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

        if not self.is_kv_producer and self._compression_config:
            from mooncake.kvserve.staging import StagingPool

            self._staging_pool = StagingPool(
                slots=self._staging_slots,
                bytes_per_slot=self._staging_slot_bytes,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        # Producer-side bounded gather workspace: compression reads the KV
        # blocks into a preallocated slot instead of fresh dense tensors, so
        # peak producer memory is capped by slots * bytes_per_slot.
        if not self.is_kv_consumer and self._compression_config:
            from mooncake.kvserve.staging import StagingPool

            self._workspace_pool = StagingPool(
                slots=self._workspace_slots,
                bytes_per_slot=self._workspace_slot_bytes,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            # Destructors must never mask the original initialization/runtime
            # failure (and may run on a partially initialized worker).
            pass

    def shutdown(self):
        """Cleanup background threads on destruction."""
        if getattr(self, "_shutdown", False):
            return
        self._shutdown = True
        # Stop accepting new work before tearing down sockets.  All temporary
        # compressed buffers are owned by their transfer/session and can be
        # dropped only after the background loops have stopped.
        for transfer_id in list(getattr(self, "_inflight_compressed_buffers", {})):
            self._inflight_compressed_buffers.pop(transfer_id, None)
            if getattr(self, "_session_manager", None) is not None:
                self._session_manager.cleanup(transfer_id)
            if getattr(self, "_workspace_pool", None) is not None:
                self._workspace_pool.release(transfer_id)
        recv_state_lock = getattr(self, "_recv_state_lock", None)
        pending_recvs = getattr(self, "_pending_compressed_recvs", None)
        if pending_recvs:
            items = list(pending_recvs.items())
            if recv_state_lock is not None:
                with recv_state_lock:
                    pending_recvs.clear()
            else:
                pending_recvs.clear()
            for req_id, (_descriptor, pull_meta) in items:
                if getattr(self, "_staging_pool", None) is not None:
                    # Leases are keyed by transfer_id, not by the request id
                    # that keys _pending_compressed_recvs.
                    self._staging_pool.release(pull_meta.transfer_id)
        if getattr(self, "async_zmq_ctx", None) is not None:
            self.async_zmq_ctx.term()
        if not getattr(self, "is_kv_consumer", True):
            if getattr(self, "_sender_executor", None) is not None:
                self._sender_executor.shutdown(wait=False)
            if getattr(self, "sender_loop", None) is not None and self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if should_launch_bootstrap_server(self.vllm_config) and hasattr(
                self, "bootstrap_server"
            ):
                self.bootstrap_server.shutdown()
        if (not getattr(self, "is_kv_producer", True)
                and getattr(self, "receiver_loop", None) is not None
                and self.receiver_loop.is_running()):
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()
        # TransferEngine has no process-wide shutdown method in the Python
        # binding.  Memory registration is therefore explicitly balanced here.
        if getattr(self, "_registered_addresses", None):
            try:
                ret = self.engine.batch_unregister_memory(self._registered_addresses)
                if ret != 0:
                    logger.warning("Mooncake memory unregister returned %s", ret)
            except Exception:
                logger.exception("Failed to unregister Mooncake memory")
            finally:
                self._registered_addresses.clear()

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks(meta.remote_tp_size)
        if meta.remote_tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = (
                "This D tp_rank "
                f"{meta.remote_tp_rank} is not paired with P tp_rank "
                f"{self.tp_rank}; expected one of {remote_tp_ranks}."
            )
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        local_regions = self._get_transfer_regions(
            self.kv_caches_base_addr, self.block_len_per_layer
        )
        remote_regions = self._get_transfer_regions(
            meta.kv_caches_base_addr, meta.block_lens
        )
        validation_err = _validate_asymmetric_region_lengths(
            local_regions=local_regions,
            remote_regions=remote_regions,
            local_tp_size=self.tp_size,
            remote_tp_size=meta.remote_tp_size,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )
        if validation_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=validation_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[transfer_id] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

            (
                src_ptrs,
                dst_ptrs,
                lengths,
                err_reqs,
                err_msg,
                compressed_payloads,
            ) = await self._build_transfer_params(
                ready_reqs,
                meta,
                local_regions,
                remote_regions,
            )
            err_req_set = set(err_reqs)
            ok_ready_reqs = [
                (d_req_id, send_meta)
                for d_req_id, send_meta in ready_reqs
                if d_req_id not in err_req_set
            ]

            if src_ptrs:
                remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                ret_value = await self.sender_loop.run_in_executor(
                    self._sender_executor,
                    self._send_blocks,
                    remote_session,
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                )

                if ret_value != 0:
                    transfer_err_msg = f"Mooncake transfer engine returned {ret_value}"
                    err_msg = (
                        transfer_err_msg
                        if err_msg is None
                        else f"{err_msg}; {transfer_err_msg}"
                    )
                    err_reqs = list(err_reqs)
                    for d_req_id, send_meta in ok_ready_reqs:
                        _trace_kvserve_transfer(
                            "transfer_failed",
                            request_id=d_req_id,
                            transfer_id=send_meta.transfer_id,
                            error=transfer_err_msg,
                        )
                        err_reqs.append(d_req_id)
                        err_req_set.add(d_req_id)
                    ok_ready_reqs = []
                else:
                    for d_req_id, send_meta in ok_ready_reqs:
                        descriptor = compressed_payloads.get(d_req_id)
                        if descriptor is not None:
                            _trace_kvserve_transfer(
                                "compressed_transfer_completed",
                                request_id=d_req_id,
                                transfer_id=send_meta.transfer_id,
                                wire_bytes=descriptor["total_bytes"],
                                body_chunks=len(descriptor["body_chunks"]),
                                aux_chunks=len(descriptor["aux_chunks"]),
                            )

            for d_req_id, send_meta in ready_reqs:
                send_meta.sending -= 1

                if d_req_id in err_req_set:
                    self._inflight_compressed_buffers.pop(send_meta.transfer_id, None)
                    self._session_manager.cleanup(send_meta.transfer_id)
                    if self._workspace_pool is not None:
                        self._workspace_pool.release(send_meta.transfer_id)
                    continue

                send_meta.sent += 1
                if (
                    send_meta.sent == send_meta.need_send
                    and self.reqs_need_send.pop(send_meta.transfer_id, None) is not None
                ):
                    self.finished_sending_reqs.add(send_meta.p_req_id)
                    # All paired decoder ranks have completed their writes.
                    self._inflight_compressed_buffers.pop(send_meta.transfer_id, None)
                    self._session_manager.cleanup(send_meta.transfer_id)
                    if self._workspace_pool is not None:
                        self._workspace_pool.release(send_meta.transfer_id)

            response = MooncakeXferResponse(
                status=response_status,
                ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,
                err_reqs=err_reqs or None,
                err_msg=err_msg,
                compressed_payloads=compressed_payloads,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))

    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        logger.debug(
            "Mooncake request %s will be served by %d consumer TP workers: %s",
            send_meta.transfer_id,
            send_meta.need_send,
            remote_tp_ranks,
        )

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
        local_regions: list[TransferRegion],
        remote_regions: list[TransferRegion],
    ) -> tuple[list[int], list[int], list[int], list[ReqId], str | None]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        compressed_payloads: dict[ReqId, dict[str, Any]] = {}
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids = agent_meta.req_blocks[d_req_id]
            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                continue

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            if num_local_blocks < num_remote_blocks:
                logger.error(
                    "req %s: local blocks(%d) less than remote blocks(%d)!",
                    d_req_id,
                    num_local_blocks,
                    num_remote_blocks,
                )
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = "P num blocks less than D"
                continue
            if num_local_blocks > num_remote_blocks:
                local_block_ids = local_block_ids[-num_remote_blocks:]

            compression_supported = (
                self.tp_size == agent_meta.remote_tp_size
                and not self._producer_cache_is_replicated()
                and not self.use_mla
            )
            compression_requested = agent_meta.compressed_reqs.get(d_req_id, False)
            if compression_requested:
                from mooncake.kvserve.compression.wire import WIRE_VERSION

                if agent_meta.compression_wire_version != WIRE_VERSION:
                    # The decoder advertised a different wire version; stay
                    # on the native raw path (observable fallback).
                    logger.warning(
                        "Decoder %s speaks compressed wire version %d, we "
                        "speak %d; using raw transfer",
                        d_req_id,
                        agent_meta.compression_wire_version,
                        WIRE_VERSION,
                    )
                    compression_requested = False
                elif not compression_supported:
                    logger.warning(
                        "Compressed transfer unsupported for heterogeneous or "
                        "replicated TP request %s; using native raw transfer",
                        d_req_id,
                    )
                    compression_requested = False
            if compression_requested:
                try:
                    compressed = self._build_compressed_transfer(
                        request_id=d_req_id,
                        transfer_id=send_meta.transfer_id,
                        local_block_ids=local_block_ids,
                        staging_addr=agent_meta.staging_addrs[d_req_id],
                        staging_capacity=agent_meta.staging_capacities[d_req_id],
                    )
                except Exception as exc:
                    _trace_kvserve_transfer(
                        "raw_fallback",
                        request_id=d_req_id,
                        transfer_id=send_meta.transfer_id,
                        reason="compression_exception",
                        error=str(exc),
                    )
                    logger.exception(
                        "Compression failed for request %s; falling back to raw",
                        d_req_id,
                    )
                    compressed = None
                    if self._workspace_pool is not None:
                        self._workspace_pool.release(send_meta.transfer_id)
                    self._session_manager.cleanup(send_meta.transfer_id)
                    err_msg = f"compression fallback for {d_req_id}: {exc}"
                if compressed is not None:
                    body_ptrs, aux_ptrs, body_lens, aux_lens, descriptor, buffers = compressed
                    src_ptrs.extend(body_ptrs)
                    dst_ptrs.extend(
                        agent_meta.staging_addrs[d_req_id] + int(item["offset"])
                        for item in descriptor["body_chunks"]
                    )
                    lengths.extend(body_lens)
                    src_ptrs.extend(aux_ptrs)
                    dst_ptrs.extend(
                        agent_meta.staging_addrs[d_req_id] + int(item["offset"])
                        for item in descriptor["aux_chunks"]
                    )
                    lengths.extend(aux_lens)
                    compressed_payloads[d_req_id] = descriptor
                    self._inflight_compressed_buffers[send_meta.transfer_id] = buffers
                    logger.info(
                        "Compressed Mooncake transfer req=%s transfer=%s chunks=%d aux=%d bytes=%d",
                        d_req_id,
                        send_meta.transfer_id,
                        len(descriptor["body_chunks"]),
                        len(descriptor["aux_chunks"]),
                        descriptor["total_bytes"],
                    )
                    continue

                _trace_kvserve_transfer(
                    "raw_fallback",
                    request_id=d_req_id,
                    transfer_id=send_meta.transfer_id,
                    reason="compression_not_admitted",
                )

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            for local_region, remote_region in zip(local_regions, remote_regions):
                should_transfer, src_region_offset, dst_region_offset, transfer_len = (
                    self._get_sender_transfer_plan(
                        local_kv_block_len=local_region.kv_block_len,
                        remote_kv_block_len=remote_region.kv_block_len,
                        remote_tp_rank=agent_meta.remote_tp_rank,
                        remote_tp_size=agent_meta.remote_tp_size,
                    )
                )
                if not should_transfer:
                    # Replicated KV cache: only one producer rank in the TP group
                    # needs to send the actual bytes for this paired decoder rank.
                    # TODO: Account for replicated producer KV in
                    # get_target_remote_ranks() so we can avoid sending
                    # unnecessary ZMQ requests and remove this branch.
                    continue

                assert src_region_offset + transfer_len <= local_region.kv_block_len, (
                    "Computed source transfer region exceeds local KV block size."
                )
                assert dst_region_offset + transfer_len <= remote_region.kv_block_len, (
                    "Computed destination transfer region exceeds remote KV block size."
                )
                # Collapse one contiguous block group into a single larger
                # transfer descriptor when the per-block copy is identical.
                can_coalesce = _can_coalesce_block_transfers(
                    local_region_block_len=local_region.block_len,
                    remote_region_block_len=remote_region.block_len,
                    src_region_offset=src_region_offset,
                    dst_region_offset=dst_region_offset,
                    transfer_len=transfer_len,
                )

                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    if can_coalesce:
                        src_ptrs.append(
                            local_region.base_addr
                            + group_local_block_id[0] * local_region.block_len
                            + src_region_offset
                        )
                        dst_ptrs.append(
                            remote_region.base_addr
                            + group_remote_block_id[0] * remote_region.block_len
                            + dst_region_offset
                        )
                        lengths.append(transfer_len * len(group_local_block_id))
                    else:
                        for local_block_id, remote_block_id in zip(
                            group_local_block_id, group_remote_block_id
                        ):
                            src_ptrs.append(
                                local_region.base_addr
                                + local_block_id * local_region.block_len
                                + src_region_offset
                            )
                            dst_ptrs.append(
                                remote_region.base_addr
                                + remote_block_id * remote_region.block_len
                                + dst_region_offset
                            )
                            lengths.append(transfer_len)

                if local_region is local_regions[0]:
                    logger.debug(
                        "Mooncake transfer plan for request %s: local_tp=%d "
                        "remote_tp=%d remote_tp_rank=%d local_block_len=%d "
                        "remote_block_len=%d src_offset=%d dst_offset=%d "
                        "transfer_len=%d coalesce=%s",
                        d_req_id,
                        self.tp_size,
                        agent_meta.remote_tp_size,
                        agent_meta.remote_tp_rank,
                        local_region.block_len,
                        remote_region.block_len,
                        src_region_offset,
                        dst_region_offset,
                        transfer_len,
                        can_coalesce,
                    )

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                num_remote_blocks,
                remote_session,
            )

        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg, compressed_payloads

    def _get_compression_adapter(self):
        if not self._compression_config or self.use_mla:
            return None
        if self._compression_adapter is None:
            from mooncake.kvserve.compression.manager import KVCompressionAdapter

            self._compression_adapter = KVCompressionAdapter(
                compression_spec=self._compression_config,
                num_kv_heads=self.model_config.get_num_kv_heads(
                    self.vllm_config.parallel_config
                ),
                head_size=self.model_config.get_head_size(),
                model_name=os.path.basename(self.model_config.model.rstrip("/")),
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
        return self._compression_adapter

    def _gather_kv_blocks(
        self, block_ids: list[int], workspace: torch.Tensor
    ) -> torch.Tensor:
        """Gather requested HND blocks into the bounded producer workspace.

        Each layer is copied directly into a view of ``workspace`` (at most
        one temporary layer at a time) instead of building a fresh dense
        stack, so peak producer gather memory stays bounded by the pool.
        """
        if not block_ids:
            raise ValueError("cannot compress an empty block list")
        if not self.device_kv_caches:
            raise ValueError("no KV cache tensors are registered")

        split_k_and_v = self.kv_topo.split_k_and_v
        num_blocks = len(block_ids)
        first_entry = next(iter(self.device_kv_caches.values()))
        first_cache = first_entry[0] if split_k_and_v else first_entry
        # Non-split HND cache is block-major [blocks, 2, ...]; split caches
        # are [blocks, ...] each.  KVServe's contract is KV-major [2, ...].
        trailing_shape = (
            first_cache.shape[1:] if split_k_and_v else first_cache.shape[2:]
        )
        dtype = first_cache.dtype
        num_layers = len(self.device_kv_caches)
        buf_shape = (num_layers, 2, num_blocks, *trailing_shape)
        numel = 1
        for dim in buf_shape:
            numel *= dim
        nbytes = numel * dtype.itemsize
        if nbytes > workspace.numel():
            raise ValueError(
                f"compression workspace too small: need {nbytes} bytes, "
                f"have {workspace.numel()}"
            )
        buf = workspace[:nbytes].view(dtype).reshape(buf_shape)

        ids_by_device: dict[torch.device, torch.Tensor] = {}
        for layer_index, (layer_name, cache_or_caches) in enumerate(
            self.device_kv_caches.items()
        ):
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            device = cache_list[0].device
            ids = ids_by_device.setdefault(
                device, torch.tensor(block_ids, dtype=torch.long, device=device)
            )
            if split_k_and_v:
                if len(cache_list) != 2:
                    raise ValueError(
                        f"expected K/V cache pair for layer {layer_name}, "
                        f"got {len(cache_list)} caches"
                    )
                for kv_index, cache in enumerate(cache_list):
                    buf[layer_index, kv_index].copy_(cache.index_select(0, ids))
            else:
                cache = cache_list[0]
                if cache.ndim < 2 or cache.shape[1] != 2:
                    raise ValueError(
                        f"unsupported KV cache shape for layer {layer_name}: "
                        f"{tuple(cache.shape)}"
                    )
                buf[layer_index].copy_(cache.index_select(0, ids).transpose(0, 1))
        return buf

    def _build_compressed_transfer(
        self,
        *,
        request_id: str,
        transfer_id: str,
        local_block_ids: list[int],
        staging_addr: int,
        staging_capacity: int,
    ):
        from mooncake.kvserve.compression.wire import build_wire

        adapter = self._get_compression_adapter()
        if adapter is None:
            return None
        from mooncake.kvserve.session import BudgetExceeded

        if self._workspace_pool is None:
            _trace_kvserve_transfer(
                "raw_fallback",
                request_id=request_id,
                transfer_id=transfer_id,
                reason="workspace_pool_unavailable",
            )
            logger.warning(
                "Compression workspace pool not configured; using raw transfer"
            )
            return None
        lease = self._workspace_pool.reserve(transfer_id)
        deadline = time.monotonic() + 300
        while lease is None:
            if time.monotonic() > deadline:
                _trace_kvserve_transfer(
                    "raw_fallback",
                    request_id=request_id,
                    transfer_id=transfer_id,
                    reason="workspace_exhausted_timeout",
                )
                logger.warning(
                    "Compression workspace exhausted for %s after 300s timeout; using raw transfer",
                    transfer_id,
                )
                return None
            logger.warning(
                "Compression workspace exhausted for %s, waiting for slot...",
                transfer_id,
            )
            time.sleep(0.1)
            lease = self._workspace_pool.reserve(transfer_id)

        admitted = False
        while not admitted:
            if time.monotonic() > deadline:
                _trace_kvserve_transfer(
                    "raw_fallback",
                    request_id=request_id,
                    transfer_id=transfer_id,
                    reason="compression_budget_exhausted_timeout",
                )
                logger.warning(
                    "Compression budget exhausted for %s after 300s timeout; using raw transfer",
                    transfer_id,
                )
                self._workspace_pool.release(transfer_id)
                return None
            try:
                self._session_manager.admit(
                    transfer_id,
                    request_id,
                    reserved_bytes=staging_capacity + lease.capacity,
                    reserved_ops=1,
                )
                admitted = True
            except BudgetExceeded:
                logger.warning(
                    "Compression budget exhausted for %s, waiting for slot...",
                    transfer_id,
                )
                time.sleep(0.1)
        _trace_kvserve_transfer(
            "compression_admitted",
            request_id=request_id,
            transfer_id=transfer_id,
            workspace_bytes=lease.capacity,
            staging_bytes=staging_capacity,
        )
        stacked = self._gather_kv_blocks(local_block_ids, lease.tensor)
        original_bytes = stacked.numel() * stacked.element_size()
        compressed = adapter.compress(stacked, request_id)
        if compressed is None:
            _trace_kvserve_transfer(
                "raw_fallback",
                request_id=request_id,
                transfer_id=transfer_id,
                reason="compression_manager_returned_none",
            )
            self._workspace_pool.release(transfer_id)
            self._session_manager.cleanup(transfer_id)
            return None
        wire = build_wire(
            compressed,
            max_chunk_bytes=max(1, staging_capacity // 2),
            transfer_id=transfer_id,
            request_id=request_id,
        )
        # Gather/compress/contiguous kernels were merely enqueued on this
        # thread's CUDA stream.  The Transfer Engine reads these buffers via
        # RDMA, which bypasses stream ordering, so the writes must be complete
        # before any pointer below is exposed to it.
        torch.cuda.current_stream().synchronize()
        body_ptrs: list[int] = []
        aux_ptrs: list[int] = []
        body_lens: list[int] = []
        aux_lens: list[int] = []
        body_specs: list[dict[str, Any]] = []
        aux_specs: list[dict[str, Any]] = []
        offset = 0
        for tensor in wire.body_chunks:
            nbytes = tensor.numel() * tensor.element_size()
            # nvcomp requires 256-byte aligned input addresses; pad offset
            # so every chunk in the staging buffer starts on an aligned boundary.
            misalign = offset % 256
            if misalign:
                offset += 256 - misalign
            body_ptrs.append(tensor.data_ptr())
            body_lens.append(nbytes)
            body_specs.append({"offset": offset, "nbytes": nbytes, "shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")})
            offset += nbytes
        for tensor in wire.aux_tensors:
            nbytes = tensor.numel() * tensor.element_size()
            # Align to this tensor's element_size before writing its offset
            alignment = max(tensor.element_size(), 256)
            misalign = offset % alignment
            if misalign:
                offset += alignment - misalign
            aux_ptrs.append(tensor.data_ptr())
            aux_lens.append(nbytes)
            aux_specs.append({"offset": offset, "nbytes": nbytes, "shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")})
            offset += nbytes
        if offset > staging_capacity:
            self._workspace_pool.release(transfer_id)
            self._session_manager.cleanup(transfer_id)
            return None
        descriptor = dict(wire.meta)
        descriptor.update(
            {
                "body_chunks": body_specs,
                "aux_chunks": aux_specs,
                "total_bytes": offset,
                "block_ids": list(local_block_ids),
                # Explicit config agreement: the decoder verifies this against
                # its own adapter fingerprint before decompressing.
                "compression_fingerprint": adapter.fingerprint,
            }
        )
        wire.validate(expected_transfer_id=transfer_id, expected_request_id=request_id)
        _trace_kvserve_transfer(
            "compressed_wire_built",
            request_id=request_id,
            transfer_id=transfer_id,
            original_bytes=original_bytes,
            wire_bytes=offset,
            body_bytes=sum(body_lens),
            aux_bytes=sum(aux_lens),
            body_chunks=len(body_specs),
            aux_chunks=len(aux_specs),
        )
        return body_ptrs, aux_ptrs, body_lens, aux_lens, descriptor, wire.body_chunks + wire.aux_tensors

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value == 0:
            logger.debug(
                "Sending to %s done, took %s",
                remote_session,
                time.perf_counter() - start_time,
            )
        return ret_value

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []
        self.block_len_per_layer = []

        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            logger.debug(
                "registering layer %s with %d cache tensor(s)",
                layer_name,
                len(cache_list),
            )

            for cache in cache_list:
                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.nbytes

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]
                assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                assert curr_tensor_size_bytes % self.num_blocks == 0, (
                    "Mooncake expects each kv cache tensor size to be "
                    "divisible by the number of blocks."
                )
                self.block_len_per_layer.append(
                    curr_tensor_size_bytes // self.num_blocks
                )

                kernel_block_size = cache.shape[-2 if self.use_mla else -3]
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(curr_tensor_size_bytes)

        self.kv_caches_base_addr = seen_base_addresses
        self.seen_base_addresses = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")
        self._registered_addresses.extend(kv_data_ptrs)

        if self._staging_pool is not None:
            staging_tensors = self._staging_pool.buffers()
            staging_ret = self.engine.batch_register_memory(
                [tensor.data_ptr() for tensor in staging_tensors],
                [tensor.numel() for tensor in staging_tensors],
            )
            if staging_ret != 0:
                raise RuntimeError("Mooncake compressed staging registration failed.")
            self._registered_addresses.extend(tensor.data_ptr() for tensor in staging_tensors)
            logger.info(
                "Registered %d compressed staging buffers (%d bytes each)",
                len(staging_tensors),
                self._staging_slot_bytes,
            )

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_lens=%s",
            self.num_blocks,
            self.block_len_per_layer,
        )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        with self._recv_state_lock:
            finished_recving_reqs = self.finished_recving_reqs
            self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]

        return finished_sending_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return finished_sending_reqs or None, finished_recving_reqs or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed = self._load_error_block_ids
        self._load_error_block_ids = set()
        return failed

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        compressed_reqs: dict[ReqId, bool] = {}
        staging_addrs: dict[ReqId, int] = {}
        staging_capacities: dict[ReqId, int] = {}
        if self._staging_pool is not None:
            for req_id, pull_meta in pull_metas.items():
                if not pull_meta.local_block_ids:
                    continue
                lease = self._staging_pool.reserve(pull_meta.transfer_id)
                deadline = time.monotonic() + 300
                while lease is None:
                    if time.monotonic() > deadline:
                        logger.warning(
                            "Compressed staging exhausted for %s after 300s timeout; using raw transfer",
                            req_id,
                        )
                        break
                    logger.warning(
                        "Compressed staging exhausted for %s, waiting for slot...",
                        req_id,
                    )
                    await asyncio.sleep(0.1)
                    lease = self._staging_pool.reserve(pull_meta.transfer_id)
                if lease is None:
                    continue
                pull_meta.compressed = True
                pull_meta.staging_slot = lease.slot
                compressed_reqs[req_id] = True
                staging_addrs[req_id] = lease.address
                staging_capacities[req_id] = lease.capacity
                _trace_kvserve_transfer(
                    "compressed_staging_reserved",
                    request_id=req_id,
                    transfer_id=pull_meta.transfer_id,
                    staging_slot=lease.slot,
                    staging_bytes=lease.capacity,
                )
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
            compressed_reqs=compressed_reqs,
            staging_addrs=staging_addrs,
            staging_capacities=staging_capacities,
            compression_wire_version=(1 if compressed_reqs else 0),
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during transferring kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        self._fail_pull_metas(pull_metas, response.err_msg or "transfer error")
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            self._fail_pull_metas(pull_metas, str(e))
            return

    def _fail_pull_metas(
        self, pull_metas: dict[ReqId, PullReqMeta], reason: str
    ) -> None:
        logger.error("Marking %d KV receives failed: %s", len(pull_metas), reason)
        for req_id, pull_meta in pull_metas.items():
            self._load_error_block_ids.update(pull_meta.local_block_ids)
            if self._staging_pool is not None:
                self._staging_pool.release(pull_meta.transfer_id)
            with self._recv_state_lock:
                self.finished_recving_reqs.add(req_id)

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
            pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                descriptor = response.compressed_payloads.get(req_id)
                if descriptor is not None:
                    _trace_kvserve_transfer(
                        "compressed_wire_received",
                        request_id=req_id,
                        transfer_id=pull_meta.transfer_id,
                        wire_bytes=descriptor["total_bytes"],
                        body_chunks=len(descriptor["body_chunks"]),
                        aux_chunks=len(descriptor["aux_chunks"]),
                    )
                    with self._recv_state_lock:
                        self._pending_compressed_recvs[req_id] = (descriptor, pull_meta)
                else:
                    # Producer selected deterministic raw fallback (for
                    # example compression admission or capability failure).
                    if pull_meta.compressed and self._staging_pool is not None:
                        self._staging_pool.release(pull_meta.transfer_id)
                    _trace_kvserve_transfer(
                        "raw_received",
                        request_id=req_id,
                        transfer_id=pull_meta.transfer_id,
                        reason="producer_raw_fallback",
                    )
                    with self._recv_state_lock:
                        self.finished_recving_reqs.add(pull_meta.d_req_id)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )
            for req_id in response.err_reqs:
                pull_meta = pull_metas.get(req_id)
                if pull_meta is not None:
                    self._load_error_block_ids.update(pull_meta.local_block_ids)
                    if self._staging_pool is not None:
                        self._staging_pool.release(pull_meta.transfer_id)
                    with self._recv_state_lock:
                        self.finished_recving_reqs.add(req_id)

    def _apply_pending_compressed(self) -> None:
        with self._recv_state_lock:
            pending = list(self._pending_compressed_recvs.items())
        if not pending:
            return
        adapter = self._get_compression_adapter()
        if adapter is None:
            return
        from mooncake.kvserve.compression.wire import (
            restore_from_wire,
            wire_views_from_buffer,
        )

        for req_id, (descriptor, pull_meta) in pending:
            lease = self._staging_pool.get(pull_meta.transfer_id) if self._staging_pool else None
            if lease is None:
                continue
            try:
                # Explicit producer/decoder configuration agreement: the
                # fingerprint of the producer's effective compression config
                # must match ours.  A mismatch is a deterministic load error,
                # never silent decompression with the wrong pipeline.
                remote_fingerprint = descriptor.get("compression_fingerprint")
                if (remote_fingerprint is not None
                        and remote_fingerprint != adapter.fingerprint):
                    raise ValueError(
                        f"compression config mismatch: producer "
                        f"{remote_fingerprint} != decoder {adapter.fingerprint}"
                    )
                wire = wire_views_from_buffer(
                    lease.tensor,
                    descriptor,
                    expected_transfer_id=pull_meta.transfer_id,
                    expected_request_id=req_id,
                )
                restored = restore_from_wire(wire)
                stacked = adapter.decompress(restored)
                if stacked is None:
                    raise RuntimeError("decompression returned no tensor")
                ids_by_device: dict[torch.device, torch.Tensor] = {}
                cache_names = list(self.device_kv_caches)
                if len(cache_names) != stacked.shape[0]:
                    raise RuntimeError("compressed layer count does not match KV cache")
                for layer_index, layer_name in enumerate(cache_names):
                    cache_or_caches = self.device_kv_caches[layer_name]
                    cache_list = cache_or_caches if self.kv_topo.split_k_and_v else [cache_or_caches]
                    device = cache_list[0].device
                    ids = ids_by_device.setdefault(
                        device,
                        torch.tensor(pull_meta.local_block_ids, dtype=torch.long, device=device),
                    )
                    source = stacked[layer_index].to(device=device)
                    if self.kv_topo.split_k_and_v:
                        for kv_index, cache in enumerate(cache_list):
                            cache.index_copy_(0, ids, source[kv_index].contiguous())
                    else:
                        cache_list[0].index_copy_(0, ids, source.transpose(0, 1).contiguous())
                _trace_kvserve_transfer(
                    "compressed_decompressed_scattered",
                    request_id=req_id,
                    transfer_id=pull_meta.transfer_id,
                    wire_bytes=descriptor["total_bytes"],
                    restored_bytes=stacked.numel() * stacked.element_size(),
                    layers=stacked.shape[0],
                )
                with self._recv_state_lock:
                    self.finished_recving_reqs.add(req_id)
            except Exception as exc:
                _trace_kvserve_transfer(
                    "compressed_decompression_failed",
                    request_id=req_id,
                    transfer_id=pull_meta.transfer_id,
                    error=str(exc),
                )
                logger.exception("Failed to apply compressed KV for %s", req_id)
                self._load_error_block_ids.update(pull_meta.local_block_ids)
                with self._recv_state_lock:
                    self.finished_recving_reqs.add(req_id)
            finally:
                if self._staging_pool is not None:
                    self._staging_pool.release(pull_meta.transfer_id)
                with self._recv_state_lock:
                    self._pending_compressed_recvs.pop(req_id, None)

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_queries[remote_bootstrap_addr].set()
        del self._pending_bootstrap_queries[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks_from_engine_id(
            remote_engine_id
        )
        count = len(remote_tp_ranks)
        logger.debug(
            "Receiving Mooncake KV for engine %s from producer TP ranks %s",
            remote_engine_id,
            remote_tp_ranks,
        )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id)
            if send_meta:
                assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        # A compressed transfer is marked finished only after this worker has
        # rebuilt the GPU wire views, decompressed, and scattered into the
        # decoder KV blocks.  This prevents Scheduler from observing a false
        # remote-KV completion one step early.
        self._apply_pending_compressed()
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )

    def _producer_cache_is_replicated(self) -> bool:
        return self.kv_topo.replicates_kv_cache(self.engine_id)

    def _get_transfer_regions(
        self, base_addrs: list[int], block_lens: list[int]
    ) -> list[TransferRegion]:
        return _expand_transfer_regions(
            base_addrs=base_addrs,
            block_lens=block_lens,
            is_kv_layout_blocks_first=self.kv_topo.is_kv_layout_blocks_first,
        )

    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[bool, int, int, int]:
        return _compute_sender_transfer_plan(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            local_kv_block_len=local_kv_block_len,
            remote_kv_block_len=remote_kv_block_len,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )

    def _log_debug_cache_registration(
        self, layer_name: str, cache: torch.Tensor
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Mooncake register view layer=%s shape=%s stride=%s "
            "storage_offset=%d contiguous=%s dense=%s data_ptr=%d",
            layer_name,
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.storage_offset(),
            cache.is_contiguous(),
            _get_tensor_dense_flag(cache),
            cache.data_ptr(),
        )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    #
    # In internal LB mode,
    # only the real global first rank need to launch the bootstrap server.
    return is_local_first_rank() and (
        parallel_config.local_engines_only or parallel_config.data_parallel_index == 0
    )


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
