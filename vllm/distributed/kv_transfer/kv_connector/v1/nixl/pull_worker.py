# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import time
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Slack (seconds) subtracted from D's exported block-expiry deadline on the turn-2
# readback, absorbing clock-offset error and read latency.
_KV_BLOCKS_EXPIRY_SAFETY_MARGIN = 5.0


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-specific (READ) worker logic."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            # Remote block IDs are kept logical here; expanded in
            # _read_blocks_for_req using the remote engine's phys ratio.
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
        # Deadlines are stamped with the scheduler process's perf_counter,
        # which is not comparable to ours when the worker runs in another
        # process on another node (perf_counter epochs differ by boot time).
        # Rebase the remaining TTL onto our clock; broadcast latency only
        # lengthens the lease, which is the safe direction. A cross-node
        # epoch gap larger than the TTL otherwise expires the lease on
        # arrival and the blocks are freed before D reads them.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        # Send heartbeats to P-side engines to keep KV blocks alive while
        # requests sit in the D scheduler WAITING queue.
        self._send_heartbeats(metadata)

    def _is_turn2_read_expired(self, meta: ReqMeta) -> bool:
        """Whether D's cached blocks for this turn-2 readback have (nearly) expired."""
        assert meta.remote is not None
        blocks_expiry_time = meta.remote.blocks_expiry_time
        # Deadline may be absent (router may not forward it) -> read as usual.
        if blocks_expiry_time is None or not meta.local_physical_block_ids:
            return False
        clock_offset = self._engine_clock_offset[meta.remote.engine_id]
        deadline = blocks_expiry_time - clock_offset
        return time.perf_counter() + _KV_BLOCKS_EXPIRY_SAFETY_MARGIN >= deadline

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()

        if self._bidirectional_kv_xfer_enabled and self._is_turn2_read_expired(meta):
            logger.warning(
                "Declining expired remote read for %s from engine %s.",
                req_id,
                engine_id,
            )
            self.xfer_stats.record_kv_expired_req()
            self._handle_failed_transfer(req_id, None)
            return

        if any(len(group) > 0 for group in meta.local_block_ids):
            # The scheduler waits for finished_recving from *every* worker.
            # Under DCP a rank's slice can legitimately come out empty when its
            # interleaved positions fall past the end of the sequence. _read_blocks
            # then takes the notify-only path without registering a transfer.
            # Seed the entry so this rank still reports completion.
            self._recving_transfers.setdefault(req_id, [])

        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        remote_logical_block_ids = meta.remote.block_ids
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            remote_logical_block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        num_groups = len(meta.local_block_ids)
        dcp_active = self.dcp_size > 1 or remote_info.remote_dcp_size > 1

        def group_ids(block_ids: BlockIds, rank: int) -> list[list[int]]:
            return [
                list(block_ids[g]) if rank in plan.source_ranks_per_group[g] else []
                for g in range(num_groups)
            ]

        read_specs = []
        for rank in plan.all_source_ranks:
            if dcp_active:
                # DCP interleaves at block granularity, so slicing happens
                # here on logical blocks, before kernel-block expansion.
                local_ids = group_ids(meta.local_block_ids, rank)
                remote_ids = group_ids(remote_logical_block_ids, rank)
                for g in range(num_groups):
                    if not local_ids[g]:
                        continue
                    # Prefix cache hit may lead to skip some of the remote reads
                    # TODO (NickLucche) consider unifying prefix cache handling on
                    # logical blocks here for both dcp and non-dcp
                    local_ids[g], remote_ids[g] = self._apply_dcp_prefix_caching(
                        local_ids[g],
                        remote_ids[g],
                        remote_rank=rank,
                        local_dcp_size=self.dcp_size,
                        local_dcp_rank=self.dcp_rank,
                        remote_dcp_size=remote_info.remote_dcp_size,
                        local_num_computed_blocks=meta.local_num_computed_blocks[g],
                    )
                local_physical_ids = self._logical_to_kernel_block_ids(
                    local_ids, self._physical_blocks_per_logical_kv_block
                )
                remote_physical_ids = self._logical_to_kernel_block_ids(
                    remote_ids, remote_info.remote_physical_blocks_per_logical
                )
            else:
                # No DCP realignment needed: reuse the already-expanded full
                # physical lists instead of re-deriving them from logical ids.
                local_physical_ids = group_ids(meta.local_physical_block_ids, rank)
                remote_physical_ids = group_ids(meta.remote.block_ids, rank)
            read_specs.append(
                ReadSpec(
                    remote_rank=rank,
                    local_block_ids=local_physical_ids,
                    remote_block_ids=remote_physical_ids,
                )
            )

        # D may have to perform multiple reads from different remote ranks.
        # Pure MLA reads once because its cache is replicated. Hybrid
        # MLA+SSM still needs one read per SSM source rank. With DCP, pure
        # MLA may also read from multiple ranks (disjoint token slices).
        if self.use_mla and tp_ratio < 0 and not self._has_mamba and not dcp_active:
            assert len(read_specs) == 1

        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if tp_ratio < 0 and (not self.use_mla or len(read_specs) > 1):
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                expected_consumers=plan.local_consumers,
            )

        if self.use_mla and tp_ratio < 0 and len(read_specs) == 1:
            # ..but we still need to notify the other remote ranks that we
            # have the blocks we need so they can update the request state.
            # Same thing for DCP (tp_size == dcp_size), so the raw tp_ratio already
            # reflects whether any remote replica is left unchosen.
            notif_id = f"{meta.remote.request_id}:{plan.local_consumers}".encode()
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != (0, read_specs[0].remote_rank):
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
        expected_consumers: int,
    ):
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes.

        # Number of local workers that will notify this producer worker.
        # Propagate on notification so dst worker can wait before freeing.
        notif_id = f"{remote_request_id}:{expected_consumers}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if not any(len(group) > 0 for group in local_block_ids):
            # A full prefix cache hit is indicated with an empty list.
            agent_name = self._remote_agents[dst_engine_id][(0, remote_rank)]
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

        assert (
            len(remote_block_ids)
            == len(local_block_ids)
            == len(self.kv_cache_config.kv_cache_groups)
        )
        if not (self.dcp_size > 1 or remote_info.remote_dcp_size > 1):
            # DCP-active reads were already trimmed (and DCP-realigned, when
            # sizes mismatch) in _read_blocks_for_req, at logical granularity.
            remote_physical_per_logical = remote_info.remote_physical_blocks_per_logical
            local_block_ids, remote_block_ids = self._apply_prefix_caching(
                local_block_ids, remote_block_ids, remote_physical_per_logical
            )

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

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
            self._handle_failed_transfer(request_id, handle)

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP or DCP
        scenario), wait for all consumers to be done pulling.

        Also handles heartbeat notifications ("HB:req1,req2,...") by
        extending the lease on the referenced requests.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                msg = notif.decode("utf-8")

                # Handle heartbeat messages from D-side.
                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                req_id, expected_consumers = msg.rsplit(":", 1)
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

                # Every reader of this req_id reports the same count (it's
                # derived from aggregate topology, not the specific rank),
                # so repeated notifications never disagree on it.
                self.expected_consumer_notifications_by_req[req_id] = int(
                    expected_consumers
                )

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    == self.expected_consumer_notifications_by_req[req_id]
                ):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    del self.expected_consumer_notifications_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids
