# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific scheduler-side logic for the NIXL connector."""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    PUSH_TRIGGER_MSG,
    REGISTER_BLOCKS_MSG,
    NixlHandshakePayload,
    RemoteMeta,
    ReqId,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
    push_trigger_addr,
    zmq_ctx,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class NixlPushConnectorScheduler(NixlBaseConnectorScheduler):
    """Push-specific scheduler logic (WRITE-based KV transfer)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Storage for registered blocks from D nodes (for push-based transfer)
        # Maps remote_request_id -> registration_data
        self._registered_blocks: dict[str, dict[str, Any]] = {}
        self._registered_blocks_lock = threading.Lock()

        # Track block_ids for finished requests (for scenario 2)
        # Maps request_id -> block_ids
        self._finished_request_blocks: dict[ReqId, BlockIds] = {}

        # Queue for push dispatch work items.
        self._push_dispatch_queue: queue.Queue[tuple[str, dict[str, Any]]] = (
            queue.Queue()
        )
        self._push_dispatcher_t: threading.Thread | None = None

        # ZMQ PUSH sockets for sending push triggers to worker processes.
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._push_trigger_paths = [
            push_trigger_addr(engine_id, rank) for rank in range(self._tp_size)
        ]
        self._push_trigger_socks: list[zmq.Socket | None] = [
            None for _ in range(self._tp_size)
        ]

    def shutdown(self):
        super().shutdown()
        # Clean up push trigger sockets
        for sock in self._push_trigger_socks:
            if sock is not None:
                sock.close(linger=0)
        self._push_trigger_socks = [None for _ in range(self._tp_size)]
        if self._push_dispatcher_t is not None:
            self._push_dispatcher_t.join(timeout=2)
            self._push_dispatcher_t = None

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """In push mode, D doesn't pull — it registers blocks and waits.

        However, we still need to handle the do_remote_prefill case
        where D needs to know how many tokens will be pushed.
        """
        params = request.kv_transfer_params
        logger.debug(
            "NixlPushConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote (pushed by P).
            token_ids = request.prompt_token_ids or []
            actual = self._mamba_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        """In push mode, D registers its blocks with P instead of pulling."""
        params = request.kv_transfer_params
        logger.debug(
            "NixlPushConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_decode") or (
            params.get("do_remote_prefill") and self.is_bidirectional_kv_xfer_enabled
        ):
            self._reqs_in_batch.add(request.request_id)
        if self.use_host_buffer and params.get("do_remote_decode"):
            self._reqs_need_save[request.request_id] = request
        elif params.get("do_remote_prefill") or (
            params.get("do_remote_decode")
            and self.is_bidirectional_kv_xfer_enabled
            and not params.get("_remote_blocks_processed")
        ):
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
                    unhashed_local_block_ids: BlockIds = (
                        blocks.get_unhashed_block_ids_all_groups()
                        if num_external_tokens > 0
                        else ()
                    )
                    local_block_ids = self.get_sw_clipped_blocks(
                        unhashed_local_block_ids
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
            elif num_external_tokens > 0:
                # There are external tokens but the remote block ids
                # are not provided. This is a KV push mode where D node
                # registers blocks with P node
                logger.info(
                    "KV PUSH mode: D node registering blocks for request %s",
                    request.request_id,
                )
                local_block_ids = blocks.get_unhashed_block_ids_all_groups()
                local_block_ids = self.get_sw_clipped_blocks(local_block_ids)

                # Create ReqMeta for registration.
                meta = ReqMeta(
                    local_block_ids=local_block_ids,
                    local_physical_block_ids=local_block_ids,
                    tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                )
                meta.remote = RemoteMeta(
                    block_ids=(),  # Not used for push mode
                    engine_id=params["remote_engine_id"],
                    request_id=params["remote_request_id"],
                    host=params["remote_host"],
                    port=params["remote_port"],
                )

                # Register blocks with P node.
                success, p_block_size, p_tp_size = self._register_blocks_with_prefill(
                    request.request_id,
                    meta,
                )

                if not success:
                    logger.error(
                        "Failed to register blocks with P node for request %s",
                        request.request_id,
                    )
                else:
                    if p_block_size is not None:
                        params["remote_block_size"] = p_block_size
                    if p_tp_size is not None:
                        params["remote_tp_size"] = p_tp_size

                # Still add to reqs_need_recv to track the request
                self._reqs_need_recv[request.request_id] = (
                    request,
                    (),  # Empty because we're waiting for push
                )
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False
            params["_remote_blocks_processed"] = True

    def request_finished(
        self,
        request: Request,
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Push-mode request_finished: stores blocks and triggers push dispatch."""
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NixlPushConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        is_p_node = bool(params.get("do_remote_decode"))
        is_d_node = not is_p_node

        self._stop_heartbeat(request.request_id)

        if params.get("do_remote_prefill"):
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None

        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            self._reqs_not_processed.add(request.request_id)
            self._reqs_need_save.pop(request.request_id, None)
            return False, None

        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        if delay_free_blocks:
            request_kv_blocks_ttl = self._kv_lease_duration
            if is_d_node:
                request_kv_blocks_ttl = self.decoder_kv_blocks_ttl
            logger.debug(
                "NixlPushConnector request_finished(%s) waiting for %d seconds "
                "before releasing blocks",
                request.request_id,
                request_kv_blocks_ttl,
            )
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + request_kv_blocks_ttl
            )

            block_ids = self.get_sw_clipped_blocks(block_ids)
            remote_num_tokens = request.num_computed_tokens

            # Track clipped block_ids for push scenario 2
            # (when D registers after request_finished).
            self._finished_request_blocks[request.request_id] = block_ids

            # Scenario 1: Check if D node has already registered
            # blocks for push-based transfer
            registration_data = self.pop_registered_blocks(request.request_id)
            if registration_data:
                logger.info(
                    "Scenario 1: D node already registered blocks for "
                    "request %s, P node KV ready, enqueuing push dispatch",
                    request.request_id,
                )
                self._push_dispatch_queue.put((request.request_id, registration_data))
            else:
                logger.debug(
                    "Scenario 2: D node hasn't registered yet for "
                    "request %s, storing %d blocks for push when "
                    "registration arrives",
                    request.request_id,
                    len(block_ids),
                )

        return delay_free_blocks, dict(
            do_remote_prefill=is_p_node,
            do_remote_decode=is_d_node,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_num_tokens=remote_num_tokens,
        )

    def pop_registered_blocks(self, request_id: str) -> dict[str, Any] | None:
        """Get and remove registered block data for a specific request."""
        with self._registered_blocks_lock:
            data = self._registered_blocks.pop(request_id, None)
            if data is not None:
                return data

            # Fuzzy match by base request ID
            base_id = get_base_request_id(request_id)
            for reg_id in list(self._registered_blocks.keys()):
                if get_base_request_id(reg_id) == base_id:
                    logger.info(
                        "Fuzzy-matched registered blocks: "
                        "request_finished ID %s matched registration ID %s "
                        "(base: %s)",
                        request_id,
                        reg_id,
                        base_id,
                    )
                    return self._registered_blocks.pop(reg_id)
            return None

    def _register_blocks_with_prefill(
        self,
        req_id: str,
        meta: ReqMeta,
    ) -> tuple[bool, int | None, int | None]:
        """Register D node's allocated block IDs with P node for push-based
        transfer.

        Args:
            req_id: Local request ID
            meta: Request metadata containing remote info and local block IDs

        Returns:
            (success, p_block_size, p_tp_size) tuple
        """

        assert meta.remote is not None
        remote_request_id = meta.remote.request_id
        remote_host = meta.remote.host
        remote_port = meta.remote.port

        try:
            path = make_zmq_path("tcp", remote_host, remote_port)
            logger.debug(
                "Registering blocks with P node at %s "
                "for request %s (remote_request_id: %s)",
                path,
                req_id,
                remote_request_id,
            )

            num_blocks = sum(len(group) for group in meta.local_physical_block_ids)
            if num_blocks == 0:
                logger.warning(
                    "No blocks to register with P node for request %s",
                    req_id,
                )
                return False, None, None

            registration_data = {
                "request_id": remote_request_id,
                "decode_request_id": req_id,
                "decode_engine_id": self.engine_id,
                "decode_tp_size": meta.tp_size,
                "local_block_ids": meta.local_physical_block_ids,
                "num_blocks": num_blocks,
                # D node's actual IP for P to handshake and send
                "remote_host": self.side_channel_host,
                "remote_port": self.side_channel_port,
            }

            with zmq_ctx(zmq.REQ, path) as sock:
                # Send registration request
                msg = msgspec.msgpack.encode((REGISTER_BLOCKS_MSG, registration_data))
                sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
                sock.send(msg)

                # Wait for acknowledgment
                ack_bytes = sock.recv()
                ack_msg = msgspec.msgpack.decode(ack_bytes)

                if ack_msg.get("status") == "success":
                    p_block_size = ack_msg.get("block_size")
                    p_tp_size = ack_msg.get("tp_size")
                    logger.info(
                        "Successfully registered %s blocks with P node "
                        "for request %s (P block_size=%s, P tp_size=%s)",
                        registration_data["num_blocks"],
                        req_id,
                        p_block_size,
                        p_tp_size,
                    )
                    return True, p_block_size, p_tp_size
                else:
                    logger.error(
                        "P node rejected block registration for request %s: %s",
                        req_id,
                        ack_msg.get("error", "Unknown error"),
                    )
                    return False, None, None

        except zmq.Again:
            logger.error(
                "Timeout registering blocks with P node for "
                "request %s. P node may not be responding.",
                req_id,
            )
            return False, None, None
        except Exception as e:
            logger.exception(
                "Failed to register blocks with P node for request %s: %s",
                req_id,
                e,
            )
            return False, None, None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """Start handshake listener (with push support) and push dispatcher."""
        encoded_data: dict[int, bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        for tp_rank, rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlPushConnectorScheduler expects NixlHandshakePayload "
                    "for handshake metadata."
                )
            encoded_data[tp_rank] = encoder.encode(rank_metadata)

        # Start the listener with push-specific args
        if self._nixl_handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    encoded_data,
                    ready_event,
                    self._stop_event,
                    self.side_channel_host,
                    self.side_channel_port,
                ),
                kwargs={
                    "scheduler_instance": self,
                    "registered_blocks": self._registered_blocks,
                    "registered_blocks_lock": self._registered_blocks_lock,
                },
                daemon=True,
                name="nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()

        # Start the push dispatcher thread
        if self._push_dispatcher_t is None:
            self._push_dispatcher_t = threading.Thread(
                target=self._push_dispatcher_loop,
                daemon=True,
                name="nixl_push_dispatcher",
            )
            self._push_dispatcher_t.start()

    def _send_push_trigger(
        self,
        request_id: str,
        block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Send a push trigger to all TP workers via ZMQ TCP (non-blocking)."""
        ctx = zmq.Context.instance()
        for rank in range(self._tp_size):
            if self._push_trigger_socks[rank] is None:
                sock = ctx.socket(zmq.PUSH)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(self._push_trigger_paths[rank])
                self._push_trigger_socks[rank] = sock
                logger.info(
                    "Scheduler connected PUSH socket to %s (rank %d)",
                    self._push_trigger_paths[rank],
                    rank,
                )

        payload = msgspec.msgpack.encode(
            (
                PUSH_TRIGGER_MSG,
                {
                    "request_id": request_id,
                    "block_ids": block_ids,
                    "registration_data": registration_data,
                },
            )
        )
        sent_count = 0
        for rank in range(self._tp_size):
            sock = self._push_trigger_socks[rank]
            assert sock is not None
            try:
                sock.send(payload, zmq.NOBLOCK)
                sent_count += 1
            except zmq.Again:
                logger.warning(
                    "Push trigger send buffer full for request %s rank %d, dropping",
                    request_id,
                    rank,
                )
        logger.info(
            "Sent push trigger for request %s (%d blocks) via TCP to %d/%d workers",
            request_id,
            len(block_ids),
            sent_count,
            self._tp_size,
        )

    def _push_dispatcher_loop(self):
        """Background thread that processes push dispatch work items."""
        logger.info("Push dispatcher thread started")
        while not self._stop_event.is_set():
            try:
                request_id, registration_data = self._push_dispatch_queue.get(
                    timeout=0.5
                )
            except queue.Empty:
                continue

            finished_req_id = None
            if request_id in self._finished_request_blocks:
                finished_req_id = request_id
            else:
                base_id = get_base_request_id(request_id)
                for rid in list(self._finished_request_blocks):
                    if get_base_request_id(rid) == base_id:
                        logger.info(
                            "Dispatcher fuzzy-matched registration ID %s "
                            "to finished request ID %s (base: %s)",
                            request_id,
                            rid,
                            base_id,
                        )
                        finished_req_id = rid
                        break

            if finished_req_id is None:
                continue

            block_ids = self._finished_request_blocks.get(finished_req_id)
            if block_ids is not None:
                logger.info(
                    "Push dispatcher: triggering push for request %s "
                    "(%d blocks) via TCP",
                    finished_req_id,
                    len(block_ids),
                )
                self._send_push_trigger(
                    finished_req_id,
                    block_ids,
                    registration_data,
                )
                self._finished_request_blocks.pop(finished_req_id, None)
            else:
                logger.error(
                    "Dispatcher: request %s matched but no block_ids found",
                    finished_req_id,
                )
        logger.info("Push dispatcher thread stopped")

    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[int, Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
        registered_blocks: dict[str, dict[str, Any]] | None = None,
        registered_blocks_lock: threading.Lock | None = None,
        scheduler_instance: NixlPushConnectorScheduler | None = None,
    ):
        """Background thread for NIXL handshakes + REGISTER_BLOCKS_MSG.

        Handles both GET_META_MSG (pull mode handshake) and
        REGISTER_BLOCKS_MSG (push mode block registration from D nodes).
        """
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

                try:
                    decoded_msg = msgspec.msgpack.decode(msg)
                except Exception as e:
                    logger.warning("Failed to decode message: %s", e)
                    continue

                if not isinstance(decoded_msg, (tuple, list)) or len(decoded_msg) < 2:
                    logger.warning(
                        "Connection listener got malformed message: "
                        "type=%s, content=%s",
                        type(decoded_msg),
                        decoded_msg,
                    )
                    continue

                msg_type = decoded_msg[0]

                if msg_type == GET_META_MSG:
                    target_tp_rank = decoded_msg[1]
                    logger.debug(
                        "Received GET_META_MSG for tp rank %s",
                        target_tp_rank,
                    )
                    if target_tp_rank not in encoded_data:
                        logger.error(
                            "No metadata available for tp rank %s. Available ranks: %s",
                            target_tp_rank,
                            list(encoded_data.keys()),
                        )
                        continue
                    sock.send_multipart((identity, b"", encoded_data[target_tp_rank]))

                elif msg_type == REGISTER_BLOCKS_MSG:
                    # Handle block registration from D node
                    registration_data = decoded_msg[1] if len(decoded_msg) > 1 else {}

                    # Validate required fields
                    request_id = registration_data.get("request_id")
                    missing = [
                        k
                        for k in (
                            "request_id",
                            "decode_engine_id",
                            "decode_tp_size",
                            "local_block_ids",
                            "remote_host",
                            "remote_port",
                        )
                        if not registration_data.get(k)
                    ]
                    if missing:
                        logger.warning(
                            "Rejecting REGISTER_BLOCKS_MSG: missing fields %s",
                            missing,
                        )
                        ack_msg = {
                            "status": "error",
                            "error": f"Missing required fields: {missing}",
                        }
                        sock.send_multipart(
                            (identity, b"", msgspec.msgpack.encode(ack_msg))
                        )
                        continue

                    assert isinstance(request_id, str)

                    logger.info(
                        "Received REGISTER_BLOCKS_MSG from D node "
                        "for request %s, decode_engine_id: %s, "
                        "num_blocks: %s",
                        request_id,
                        registration_data.get("decode_engine_id"),
                        registration_data.get("num_blocks"),
                    )

                    # Send ACK with P's block_size and tp_size
                    assert scheduler_instance is not None
                    ret_ack_msg: dict[str, str | int] = {
                        "status": "success",
                        "block_size": scheduler_instance.block_size,
                        "tp_size": (
                            scheduler_instance.vllm_config.parallel_config.tensor_parallel_size
                        ),
                    }
                    sock.send_multipart(
                        (identity, b"", msgspec.msgpack.encode(ret_ack_msg))
                    )

                    # Store the registration data
                    assert registered_blocks_lock is not None
                    assert registered_blocks is not None
                    with registered_blocks_lock:
                        registered_blocks[request_id] = registration_data
                    logger.info(
                        "Stored registration data for request %s with %s blocks",
                        request_id,
                        registration_data.get("num_blocks"),
                    )

                    # Enqueue for the push dispatcher thread to handle
                    # scenario-2 check without blocking this listener.
                    scheduler_instance._push_dispatch_queue.put(
                        (request_id, registration_data)
                    )

                else:
                    logger.warning(
                        "Connection listener got unexpected message type: %s",
                        msg_type,
                    )
