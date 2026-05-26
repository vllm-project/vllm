# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector.

P2P communication uses NIXL notifications as the sole inter-node channel.
D workers send registration notifications to P workers, and P workers
initiate WRITE transfers when matched with finished blocks received from
the P scheduler via ``build_connector_meta``.

The P engine is kept stepping by the ``has_pending_push_work`` scheduler
hook, so notifications are always polled on the main thread.
"""

from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
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
    get_base_request_id,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic.

    D-side: sends registration notifications to P workers via NIXL after
    handshaking.  P-side: receives registrations, matches with finished
    blocks from the scheduler, and initiates WRITE transfers.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Push-specific state
        self._sending_transfers = defaultdict[ReqId, list[TransferHandle]](list)

        # P-side: finished request blocks received from scheduler metadata.
        self._push_finished_blocks: dict[ReqId, BlockIds] = {}
        # P-side: D registrations awaiting matching finished blocks.
        self._pending_d_registrations: dict[ReqId, dict[str, Any]] = {}

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """Process metadata from the scheduler.

        D-side: sends NIXL registration notifications to P workers.
        P-side: accumulates finished blocks and matches with pending
        registrations.
        """
        # --- D-side: process reqs_to_recv (wait for P to push) ---
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
            self._recving_metadata[req_id] = meta
            logger.info(
                "Push mode: D node waiting for P to push blocks for request %s",
                req_id,
            )

        # --- D-side: send registration notifications to P via NIXL ---
        for req_id, reg_data in metadata.push_registrations.items():
            self._send_registration_to_p(req_id, reg_data)

        # --- P-side: accumulate newly finished blocks ---
        for req_id, block_ids in metadata.push_finished_blocks.items():
            self._push_finished_blocks[req_id] = block_ids

            # Check for a pending D registration (scenario 1: D registered
            # before P finished).
            matched_reg = self._pop_matching_registration(req_id)
            if matched_reg is not None:
                logger.info(
                    "Scenario 1: matched D registration with "
                    "finished blocks for request %s, initiating WRITE",
                    req_id,
                )
                self._push_finished_blocks.pop(req_id, None)
                self.start_push_kv(req_id, block_ids, matched_reg)

        # --- Common: batch tracking and expiration ---
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            assert req_id not in self._reqs_to_send

        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

        self._send_heartbeats(metadata)

    # ------------------------------------------------------------------ #
    #  D-side: registration notification helpers                          #
    # ------------------------------------------------------------------ #

    def _send_registration_to_p(self, req_id: str, reg_data: dict[str, Any]) -> None:
        """Handshake with P (if needed) then send registration notification."""
        remote_engine_id = reg_data["remote_engine_id"]
        remote_host = reg_data["remote_host"]
        remote_port = reg_data["remote_port"]
        remote_tp_size = reg_data["remote_tp_size"]

        fut = self._ensure_handshake(
            remote_engine_id, remote_host, remote_port, remote_tp_size
        )
        if fut is None:
            self._do_send_reg_notif(req_id, reg_data)
        else:

            def _on_handshake(
                f: Future[dict[int, str]],
                rid: str = req_id,
                rd: dict[str, Any] = reg_data,
            ) -> None:
                try:
                    f.result()
                    self._do_send_reg_notif(rid, rd)
                except Exception as e:
                    self._log_failure(
                        failure_type="push_reg_handshake_failed",
                        req_id=rid,
                        error=e,
                    )
                    self._handle_failed_transfer(rid, None)

            fut.add_done_callback(_on_handshake)

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        """Encode and send the registration notification to all P workers."""
        remote_engine_id = reg_data["remote_engine_id"]
        payload = msgspec.msgpack.encode(reg_data)
        notif_msg = PUSH_REG_NOTIF_PREFIX + payload

        agents = self._remote_agents.get(remote_engine_id)
        if not agents:
            logger.error(
                "No remote agents for engine %s, cannot send registration "
                "for request %s",
                remote_engine_id,
                req_id,
            )
            self._handle_failed_transfer(req_id, None)
            return

        for rank, agent_name in agents.items():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_msg)
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_notif_failed",
                    req_id=req_id,
                    error=e,
                    remote_rank=rank,
                )

        logger.info(
            "Sent push registration notification for request %s to %d "
            "P workers on engine %s",
            req_id,
            len(agents),
            remote_engine_id,
        )

    # ------------------------------------------------------------------ #
    #  P-side: matching helpers                                           #
    # ------------------------------------------------------------------ #

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        """Find and remove a pending D registration matching *request_id*."""
        data = self._pending_d_registrations.pop(request_id, None)
        if data is not None:
            return data
        base_id = get_base_request_id(request_id)
        for reg_id in list(self._pending_d_registrations):
            if get_base_request_id(reg_id) == base_id:
                logger.info(
                    "Fuzzy-matched registration %s to finished blocks %s (base: %s)",
                    reg_id,
                    request_id,
                    base_id,
                )
                return self._pending_d_registrations.pop(reg_id)
        return None

    def _pop_matching_finished_blocks(
        self, request_id: str
    ) -> tuple[str, BlockIds] | None:
        """Find and remove finished blocks matching *request_id*."""
        blocks = self._push_finished_blocks.pop(request_id, None)
        if blocks is not None:
            return request_id, blocks
        base_id = get_base_request_id(request_id)
        for fin_id in list(self._push_finished_blocks):
            if get_base_request_id(fin_id) == base_id:
                logger.info(
                    "Fuzzy-matched finished blocks %s to registration %s (base: %s)",
                    fin_id,
                    request_id,
                    base_id,
                )
                return fin_id, self._push_finished_blocks.pop(fin_id)
        return None

    # ------------------------------------------------------------------ #
    #  WRITE transfer logic (P-side, largely unchanged)                   #
    # ------------------------------------------------------------------ #

    def start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Start push-based KV transfer from P worker to D node."""
        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        decode_host = registration_data["decode_host"]
        decode_port = registration_data["decode_port"]
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
                    decode_host,
                    decode_port,
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

        Processes three kinds of notifications:
        - PUSH_REG: registration from D worker (P-side)
        - HB: heartbeat for lease renewal
        - req_id:tp_size: transfer completion
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                # P-side: D worker registration notification.
                if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                    reg_payload = notif[len(PUSH_REG_NOTIF_PREFIX) :]
                    reg_data = msgspec.msgpack.decode(reg_payload)
                    p_request_id = reg_data["request_id"]

                    # Check for matching finished blocks (scenario 2:
                    # P finished before D registered).
                    match = self._pop_matching_finished_blocks(p_request_id)
                    if match is not None:
                        fin_id, block_ids = match
                        logger.info(
                            "Scenario 2: matched finished blocks for "
                            "request %s with D registration, "
                            "initiating WRITE",
                            fin_id,
                        )
                        self.start_push_kv(fin_id, block_ids, reg_data)
                    else:
                        self._pending_d_registrations[p_request_id] = reg_data
                        logger.debug(
                            "Stored D registration for request %s, "
                            "awaiting P finished blocks",
                            p_request_id,
                        )
                    continue

                msg = notif.decode("utf-8")

                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                req_id, tp_size = msg.rsplit(":", 1)

                # D receives notification that P finished writing.
                if req_id in self._recving_metadata and (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    logger.info(
                        "Received push completion notification for request %s",
                        req_id,
                    )
                    _ = self._recving_transfers[req_id]
                    continue

                # P-side: D finished reading.
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

        # Check if outgoing WRITE transfers have completed.
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
        """Shutdown: clean up sending transfers."""
        for handles in self._sending_transfers.values():
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._sending_transfers.clear()
        super().shutdown()
