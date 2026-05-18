# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector."""

import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineTransferInfo,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_TRIGGER_MSG,
    NixlConnectorMetadata,
    RemoteMeta,
    ReqId,
    ReqMeta,
    TransferHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    push_trigger_addr,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Push-specific state
        self._sending_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        self._push_stop_event = threading.Event()
        self._push_listener_thread: threading.Thread | None = None
        self._push_trigger_path = push_trigger_addr(engine_id, self.tp_rank)

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        """Register KV caches and start the push listener."""
        super().register_kv_caches(kv_caches)
        # Start the push trigger listener so the scheduler can send
        # push triggers to this worker via ZMQ TCP.
        self._start_push_listener()

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """D-side: store metadata and wait for P to push blocks."""
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            assert meta.remote is not None
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv (push) for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            # Store metadata for failure recovery and completion tracking
            self._recving_metadata[req_id] = meta

            # In push mode, D doesn't handshake with P. Register
            # P's engine info in transfer_topo so post-processing
            # (block_size_ratio, etc.) works when the push completes.
            if (
                self.transfer_topo is not None
                and meta.remote_block_size is not None
                and remote_engine_id not in self.transfer_topo._engines
            ):
                self.transfer_topo.register_remote_engine(
                    remote_engine_id=remote_engine_id,
                    info=EngineTransferInfo(
                        remote_tp_size=meta.tp_size or self.world_size,
                        remote_block_size=meta.remote_block_size,
                        remote_block_len=meta.remote_block_size * self.block_size,
                        remote_physical_blocks_per_logical=1,
                    ),
                )
            logger.info(
                "Push mode: D node waiting for P to push blocks for request %s",
                req_id,
            )

        # Track batch membership and expiration (same as pull)
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            assert req_id not in self._reqs_to_send

        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

        self._send_heartbeats(metadata)

    def _start_push_listener(self) -> None:
        """Start the background thread that listens for push triggers
        from the scheduler via ZMQ TCP."""
        if self._push_listener_thread is not None:
            return
        ready = threading.Event()
        self._push_listener_thread = threading.Thread(
            target=self._push_listener_loop,
            args=(ready,),
            daemon=True,
            name="nixl-push-listener",
        )
        self._push_listener_thread.start()
        ready.wait()
        logger.info(
            "Push listener thread started on %s",
            self._push_trigger_path,
        )

    def _push_listener_loop(self, ready: threading.Event) -> None:
        """Poll the ZMQ PULL socket for push triggers and call
        start_push_kv directly on this thread."""
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVTIMEO, 500)  # 500ms poll
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(self._push_trigger_path)
        ready.set()

        while not self._push_stop_event.is_set():
            try:
                raw = sock.recv()
            except zmq.Again:
                continue
            except Exception:
                if self._push_stop_event.is_set():
                    break
                logger.exception("Push listener recv error")
                continue

            try:
                decoded = msgspec.msgpack.decode(raw)
                if not isinstance(decoded, (tuple, list)) or len(decoded) < 2:
                    logger.warning(
                        "Push listener got malformed message: %s",
                        type(decoded),
                    )
                    continue
                msg_type, msg = decoded[0], decoded[1]
                if msg_type != PUSH_TRIGGER_MSG:
                    logger.warning(
                        "Push listener got unexpected message type: %s",
                        msg_type,
                    )
                    continue
                request_id = msg["request_id"]
                block_ids = msg["block_ids"]
                registration_data = msg["registration_data"]
            except Exception:
                logger.exception("Failed to decode push trigger message")
                continue

            logger.info(
                "Push listener received trigger for request %s (%d blocks)",
                request_id,
                len(block_ids),
            )
            self.start_push_kv(request_id, block_ids, registration_data)

        sock.close()
        logger.info("Push listener thread stopped")

    def start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Start push-based KV transfer from P worker to D node."""
        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        remote_host = registration_data["remote_host"]
        remote_port = registration_data["remote_port"]
        decode_request_id = registration_data.get("decode_request_id", request_id)
        if not local_block_ids:
            logger.warning(
                "No local blocks to push for request %s",
                request_id,
            )
            return

        logger.info(
            "Processing kv push request %s to D node %s: "
            "pushing %d local blocks to %d remote blocks",
            request_id,
            decode_engine_id,
            len(local_block_ids),
            len(remote_block_ids),
        )

        # Handshake with D node (one-time per engine_id)
        if decode_engine_id not in self._remote_agents:
            logger.info(
                "No remote agent info for D node %s, performing push handshake",
                decode_engine_id,
            )
            try:
                remote_tp_size = registration_data["decode_tp_size"]
                remote_agents = self._nixl_handshake(
                    remote_host,
                    remote_port,
                    remote_tp_size,
                    decode_engine_id,
                )
                with self._handshake_lock:
                    self._remote_agents[decode_engine_id] = remote_agents
                logger.info(
                    "Push handshake with D node %s complete, got %d remote agents",
                    decode_engine_id,
                    len(remote_agents),
                )
            except Exception:
                logger.exception(
                    "Failed to handshake with D node %s for push request %s",
                    decode_engine_id,
                    request_id,
                )
                return

        # Convert logical block IDs to physical/kernel block IDs
        block_ids_grouped: BlockIds = local_block_ids
        if local_block_ids and not isinstance(local_block_ids[0], (list, tuple)):
            block_ids_grouped = (list(local_block_ids),)
        physical_block_ids = self._logical_to_kernel_block_ids(block_ids_grouped)

        # Handle remote block IDs
        remote_ids_grouped: BlockIds = remote_block_ids
        if remote_block_ids and not isinstance(remote_block_ids[0], (list, tuple)):
            remote_ids_grouped = (list(remote_block_ids),)
        physical_remote_block_ids = self._logical_to_kernel_block_ids(
            remote_ids_grouped
        )

        logger.info(
            "start_push_kv block shapes: local_groups=%d local_blocks=%s, "
            "remote_groups=%d remote_blocks=%s",
            len(physical_block_ids),
            [len(g) for g in physical_block_ids],
            len(physical_remote_block_ids),
            [len(g) for g in physical_remote_block_ids],
        )

        # Initiate WRITE transfer(s) to D rank(s).
        push_meta = ReqMeta(
            local_block_ids=physical_block_ids,
            local_physical_block_ids=physical_block_ids,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=physical_remote_block_ids,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=decode_request_id,
            ),
        )
        self._xfer_blocks_for_req(
            req_id=request_id,
            meta=push_meta,
        )

    def _xfer_blocks_for_req(self, req_id: str, meta: ReqMeta):
        """Issue WRITE transfers to one or more remote TP ranks."""
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(local_block_ids)
        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=[
                    list(local_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
                remote_block_ids=[
                    list(remote_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
            )
            for rank in plan.all_source_ranks
        ]

        if self.use_mla and tp_ratio < 0:
            assert len(read_specs) == 1

        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _xfer_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            if tp_ratio < 0 and not self.use_mla:
                assert remote_block_size == self.block_size
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][i]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )

        if self.use_mla and tp_ratio < 0 and read_specs:
            notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != read_specs[0].remote_rank:
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ):
        """Post a WRITE point-to-point xfer request."""
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            assert not self._is_hma_required
            local_block_ids0 = local_block_ids[0] if local_block_ids else []
            remote_block_ids0 = remote_block_ids[0]
            local_block_ids_mapped = self.get_mapped_blocks(
                np.asarray(local_block_ids0), block_size_ratio
            ).tolist()
            if len(local_block_ids_mapped) > len(remote_block_ids0):
                local_block_ids_mapped = local_block_ids_mapped[
                    : len(remote_block_ids0)
                ]
            local_block_ids = [local_block_ids_mapped] if local_block_ids_mapped else []
            remote_block_ids = [remote_block_ids0]

        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        if len(local_block_ids) == 0:
            logger.warning("No blocks to push for request %s", request_id)
            return

        # Align per-group block counts for push.
        local_block_ids = list(local_block_ids)
        remote_block_ids = list(remote_block_ids)
        for i in range(min(len(local_block_ids), len(remote_block_ids))):
            num_local = len(local_block_ids[i])
            num_remote = len(remote_block_ids[i])
            if num_local > num_remote:
                local_block_ids[i] = local_block_ids[i][:num_remote]
            elif num_local < num_remote:
                remote_block_ids[i] = remote_block_ids[i][:num_local]

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            # Track push WRITE handles so P can free blocks once done.
            self._sending_transfers[request_id].append(handle)
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, handle)

    def _get_new_notifs(self) -> set[str]:
        """Handle notifications for push mode.

        On D-side: receives notification that P finished writing.
        On P-side: receives notification that D finished reading (pull compat).
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                msg = notif.decode("utf-8")

                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                req_id, tp_size = msg.rsplit(":", 1)

                # Push mode: D receives notification that P finished writing.
                if req_id in self._recving_metadata and (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    logger.info(
                        "Received push completion notification for request %s",
                        req_id,
                    )
                    # Create empty handle list so _pop_done_transfers
                    # immediately marks it as done.
                    _ = self._recving_transfers[req_id]
                    continue

                # P-side: receives notification that D finished reading.
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

                n_consumers = int(tp_size)
                tp_ratio = self.transfer_topo.tp_ratio(n_consumers)
                consumers_per_producer = (
                    -tp_ratio if n_consumers > self.world_size else 1
                )

                self.consumer_notification_counts_by_req[req_id] += 1
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    == consumers_per_producer
                ):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Override to also track push WRITE transfer completions."""
        done_sending, done_recving = super().get_finished()

        # Push mode: check if outgoing WRITE transfers have completed.
        done_pushing = self._pop_done_transfers(self._sending_transfers)
        for req_id in done_pushing:
            logger.info(
                "Push WRITE transfer completed for request %s, freeing blocks on P",
                req_id,
            )
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self.consumer_notification_counts_by_req.pop(req_id, None)
            done_sending.add(req_id)

        return done_sending, done_recving

    def shutdown(self):
        """Shutdown push listener and base worker."""
        self._push_stop_event.set()
        if self._push_listener_thread is not None:
            self._push_listener_thread.join(timeout=2)
            self._push_listener_thread = None
        # Clean up sending transfers
        for handles in self._sending_transfers.values():
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._sending_transfers.clear()
        super().shutdown()
