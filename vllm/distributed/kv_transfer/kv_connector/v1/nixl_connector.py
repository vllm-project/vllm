# SPDX-License-Identifier: Apache-2.0
import contextlib
import math
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import msgspec
import torch
import zmq
from typing_extensions import Optional

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.sampling_params import KVTransferParams
from vllm.utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

GET_META_MSG = b"get_meta_msg"

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None


class NixlAgentMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    engine_id: str
    agent_metadata: bytes
    # Base addr for each layer for KVs
    # NOTE: we will need another list for TP>1
    kv_caches_base_addr: list[int]
    num_blocks: int


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str


class NixlConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: KVTransferParams,
    ):
        assert request_id not in self.requests
        assert kv_transfer_params.remote_engine_id is not None
        assert kv_transfer_params.remote_block_ids is not None

        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.remote_block_ids,
            remote_engine_id=kv_transfer_params.remote_engine_id,
            remote_host=kv_transfer_params.remote_host,
            remote_port=kv_transfer_params.remote_port,
        )


class NixlConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        self.engine_id = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler : Optional[NixlConnectorScheduler] = \
                NixlConnectorScheduler(vllm_config, str(self.engine_id))
            self.connector_worker: Optional[NixlConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorker(str(self.engine_id))

    ############################################################
    # Scheduler Side Methods
    ############################################################
    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> int:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 block_ids: list[int],
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, block_ids, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """NixlConnector does not save explicitly."""
        return

    def wait_for_save(self):
        """NixlConnector does not save explicitly."""
        return


class NixlConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        logger.info("Initializing NIXL Scheduler %s", engine_id)

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int]]] = {}

    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> int:
        """For remote prefill, allocate for all tokens."""

        # NOTE: this function is called in the WAITING loop.
        # So we should only have full blocks of computed tokens.
        assert num_computed_tokens % self.block_size == 0

        if request.do_remote_prefill:
            rounded_num_prompt_tokens = round_down(
                len(request.prompt_token_ids), self.block_size)
            return max(rounded_num_prompt_tokens - num_computed_tokens, 0)

        return 0

    def update_state_after_alloc(self, request: "Request",
                                 block_ids: list[int],
                                 num_external_tokens: int):
        if request.do_remote_prefill and num_external_tokens > 0:
            self._reqs_need_recv[request.request_id] = (request, block_ids)

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
            )

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()

        return meta


class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, engine_id: str):
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Agent.
        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), None)
        # Map of engine_id -> list[agent_names] (1 per rank).
        self._remote_agents: dict[str, list[str]] = {}

        # Metadata.
        self.engine_id = engine_id
        self.rank = 0

        # KV Caches and nixl tracking data.
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Map of engine_id -> kv_caches_base_addr
        # For Local: base addr for *this* rank, each layer for K,V
        # For Remote: base addr for *each* rank, each layer for K,V
        # KV_CACHES_ADDR_TYPE = Union[list[tuple[int, int]],
        #                             list[list[tuple[int, int]]]]
        self.kv_caches_base_addr: dict[str, list[int]] = {}

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0

        # Map of tp_mult -> nixl_prepped_dlist_handle (int).
        self.src_xfer_side_handles: dict[int, int] = {}
        # Map of engine_id -> map[tp_mult -> nixl_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: defaultdict[str,
                                                dict[int,
                                                     int]] = defaultdict(dict)
        # Map of engine_id -> num_blocks.
        self.dst_num_blocks: dict[str, int] = {}
        self._registered_descs: list[Any] = []

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_transfers: dict[str, list[Any]] = defaultdict(list[Any])

        # Background thread for establishing new connections.
        self._nixl_handshake_listener_t: Optional[threading.Thread] = None

    @staticmethod
    def _nixl_handshake_listener(metadata: NixlAgentMetadata,
                                 ready_event: threading.Event):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach like an ETCD server in the future.

        # NOTE(rob): to support heterogeneous TP, we will have to
        # move this into the scheduler rather than worker, since
        # each rank needs the metadata of all other ranks (whereas
        # in this setup, each rank only gets one other rank's meta.

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded NixlAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        port = envs.VLLM_NIXL_SIDE_CHANNEL_PORT
        with zmq_ctx(zmq.ROUTER, f"tcp://{host}:{port}") as sock:
            ready_event.set()
            while True:
                identity, _, msg = sock.recv_multipart()
                if msg != GET_META_MSG:
                    logger.warning(
                        "Connection listener got unexpected message %s", msg)
                sock.send_multipart((identity, b"", encoded_data))

    def _nixl_handshake(self, host: str, port: int):
        """Do a NIXL handshake with a remote instance."""

        start_time = time.perf_counter()
        with zmq_ctx(zmq.REQ, f"tcp://{host}:{port}") as sock:
            # Send query for the request.
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()

            # Register Remote agent.
            self.add_remote_agent(metadata)
            setup_agent_time = time.perf_counter()

            logger.debug("NIXL handshake: get metadata took: %s",
                         got_metadata_time - start_time)
            logger.debug("NIXL handshake: add agent took: %s",
                         setup_agent_time - got_metadata_time)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        first_layer_name = next(iter(kv_caches))
        first_kv_cache = kv_caches[first_layer_name]

        # [2 (k and v), num_blocks, ...]
        # TODO(tms): num_blocks will be in a different spot for MLA.
        num_blocks = first_kv_cache.shape[1]
        kv_elem_size = first_kv_cache[0].element_size()
        # TODO(tms): self.block_len needs to be per-layer for sliding window,
        # hybrid attn, etc
        self.block_len = kv_elem_size * math.prod(first_kv_cache.shape[-3:])

        logger.debug("Per layer kv cache size: %s", first_kv_cache[0].shape)
        self.num_blocks = num_blocks
        self.dst_num_blocks[self.engine_id] = num_blocks
        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        caches_data = []

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we can
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded NixlAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        for layer_name in kv_caches:
            for cache in kv_caches[layer_name]:
                base_addr = cache.data_ptr()
                region_len = num_blocks * self.block_len
                caches_data.append((base_addr, region_len, self.rank, ""))
                kv_caches_base_addr.append(base_addr)
        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
        self.num_regions = len(caches_data)

        descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs)
        logger.debug("Done registering descs")

        self._registered_descs.append(descs)

        # After KV Caches registered, listen for new connections.
        metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
        )
        ready_event = threading.Event()
        self._nixl_handshake_listener_t = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(metadata, ready_event),
            daemon=True,
            name="nixl_handshake_listener")
        import os
        if os.getenv("SKIP", None) != "1":
            self._nixl_handshake_listener_t.start()
            ready_event.wait()

    def add_remote_agent(self, nixl_agent_meta: NixlAgentMetadata, tp_idx=0):
        engine_id = nixl_agent_meta.engine_id
        if engine_id in self._remote_agents:
            return

        num_blocks = nixl_agent_meta.num_blocks
        logger.debug("Adding remote agent %s %s", engine_id, str(num_blocks))

        agent_names = [
            self.nixl_wrapper.add_remote_agent(nixl_agent_meta.agent_metadata)
        ]

        self._remote_agents[engine_id] = agent_names
        self.kv_caches_base_addr[
            engine_id] = nixl_agent_meta.kv_caches_base_addr

        # NOTE: once we support heterogeneous TP, we will need maintain the
        # src for each TP multiplier.
        # NOTE(rob): Dynamo only supports D TP size > P TP size.
        # https://github.com/vllm-project/vllm/pull/16124/files#diff-876efa5533f5dcff3fba850e8684a47d53c112e287988957c115b11691374f4bR331 # noqa: E501
        # Create descs and xfer side handles.
        tp_multiplier = 1
        dst_block_len = self.block_len // tp_multiplier
        if tp_multiplier not in self.src_xfer_side_handles:
            # Create descs and xfer side handles.
            blocks_data = []
            for base_addr in self.kv_caches_base_addr[self.engine_id]:
                for block_id in range(self.num_blocks):
                    block_offset = block_id * self.block_len
                    for i in range(tp_multiplier):
                        tp_multiplier_offset = tp_idx * dst_block_len
                        blocks_data.append(
                            (base_addr + block_offset + tp_multiplier_offset,
                             dst_block_len, self.rank))
            logger.debug("Created %s blocks for src engine %s and rank %s",
                         len(blocks_data), self.engine_id, self.rank)

            # Register with NIXL.
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
            self.src_xfer_side_handles[tp_multiplier] = (
                self.nixl_wrapper.prep_xfer_dlist("", descs))

        # create dst xfer side handles
        self.dst_num_blocks[engine_id] = num_blocks
        blocks_data = []
        for base_addr in self.kv_caches_base_addr[engine_id]:
            for block_id in range(num_blocks):
                block_offset = block_id * dst_block_len
                blocks_data.append((base_addr + block_offset, dst_block_len,
                                    self.rank * tp_multiplier))
        logger.debug("Created %s blocks for dst engine %s and rank %s",
                     len(blocks_data), engine_id, self.rank)
        # Register with NIXL.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
        self.dst_xfer_side_handles[engine_id][tp_idx] = (
            self.nixl_wrapper.prep_xfer_dlist(
                self._remote_agents[engine_id][self.rank * tp_multiplier +
                                               tp_idx], descs))

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Get requests that are done sending or recving."""
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "get_finished: %s requests done sending "
                "and %s requests done recving", len(done_sending),
                len(done_recving))
        return done_sending, done_recving

    def _get_new_notifs(self) -> set[str]:
        """Get req_ids which got a remote xfer message."""

        notified_req_ids: set[str] = set()
        # TODO: handle the TP case (N notifies for TP=N).
        # See: vllm/worker/worker_base.py L476 in DynamoPR.
        for req_ids in self.nixl_wrapper.get_new_notifs().values():
            for req_id in req_ids:
                assert req_id not in notified_req_ids
                notified_req_ids.add(req_id.decode("utf-8"))
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
            running_reqs = []
            for handle in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    # TODO ptarasiewicz: why abort is throwing errors?
                    # self.nixl_wrapper.release_xfer_handle(handle)
                    continue
                if xfer_state == "PROC":
                    running_reqs.append(handle)
                else:
                    raise RuntimeError("Transfer failed with state %s",
                                       xfer_state)
            if len(running_reqs) == 0:
                done_req_ids.add(req_id)
                del transfers[req_id]
            else:
                transfers[req_id] = running_reqs
        return done_req_ids

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.requests.items():
            # NOTE: this is non-blocking
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
                meta.remote_engine_id, len(meta.local_block_ids),
                len(meta.remote_block_ids))
            self._read_blocks(
                request_id=req_id,
                dst_engine_id=meta.remote_engine_id,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                remote_host=meta.remote_host,
                remote_port=meta.remote_port,
            )

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_host: str,
        remote_port: int,
        dst_engine_id: str,
        request_id: str,
    ):
        # NOTE(rob): this takes ~2s. We need to get this off the hotpath.
        if dst_engine_id not in self._remote_agents:
            self._nixl_handshake(remote_host, remote_port)

        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes. We should remove the staging
        # blocks until we are ready.

        # NOTE(rob): we could potentially do the rearranging during the load_kv!

        # Note(tms): The remote_block_ids only contain full computed blocks,
        # while the local_block_ids are all blocks allocated for this request,
        # so truncate the local_block_ids to account for this.
        del local_block_ids[len(remote_block_ids):]
        assert len(local_block_ids) == len(remote_block_ids)

        # NOTE(rob): this can cause the remote blocks to not be freed?
        if len(local_block_ids) == 0:
            return

        # TODO: support TP multipliers.
        tp_multiplier = 1
        remote_block_descs_ids = self._get_block_descs_ids(
            dst_engine_id, "all", remote_block_ids)
        local_xfer_side_handle = self.src_xfer_side_handles[tp_multiplier]

        # Read the data from the remote.
        for i in range(tp_multiplier):
            local_block_descs_ids = self._get_block_descs_ids(
                self.engine_id,
                "all",
                local_block_ids,
                i=None,  #TODO: Enable both tp_multiplier and staging_ranges.
                tp_multiplier=tp_multiplier,
                staging_ranges=None)
            assert len(local_block_descs_ids) == len(remote_block_descs_ids)
            remote_xfer_side_handle = self.dst_xfer_side_handles[
                dst_engine_id][i]

            # NOTE(rob): we use the request_id as the notify msg, so we
            # must use the same request_id in both the p and d workers.
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=request_id.encode("utf-8"),
            )

            # Call transfer to begin the async transfer
            # We will check this is done in the next forward pass.
            self.nixl_wrapper.transfer(handle)
            self._recving_transfers[request_id].append(handle)

    def _get_block_descs_ids(self,
                             engine_id,
                             region_ids,
                             block_ids,
                             i=None,
                             tp_multiplier=1,
                             staging_ranges=None):

        if region_ids == "all":
            region_ids = range(self.num_regions)
        if block_ids == "all":
            block_ids = range(self.num_blocks)

        descs_ids = []

        if i is not None:
            raise NotImplementedError("Prefill and Decode instances must have "
                                      "the same TP size.")

        num_blocks = self.dst_num_blocks[engine_id]
        for reg_id in region_ids:
            for block_id in block_ids:
                descs_ids.append(reg_id * num_blocks + block_id)
        return descs_ids


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]

        if socket_type == zmq.ROUTER:
            socket = ctx.socket(zmq.ROUTER)
            socket.bind(addr)
        elif socket_type == zmq.REQ:
            socket = ctx.socket(zmq.REQ)
            socket.connect(addr)
        else:
            raise ValueError(f"Unexpected socket type: {socket_type}")

        yield socket
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
