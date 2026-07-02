# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import time
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    RemoteWorkerKey,
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
                meta.local_block_ids
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
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

        # Send heartbeats to P-side engines to keep KV blocks alive while
        # requests sit in the D scheduler WAITING queue.
        self._send_heartbeats(metadata)

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)
        remote_worker_keys = (
            self.transfer_topo.get_target_remote_worker_keys_from_engine_id(engine_id)
        )
        logical_local_block_ids = meta.local_block_ids
        logical_remote_block_ids = meta.remote.block_ids
        local_block_offset = (
            meta.local_num_computed_tokens // self._local_logical_block_size()
        ) * self.dcp_size

        launched_read = False
        for remote_worker_key in remote_worker_keys:
            remote_tp_rank, remote_dcp_rank = remote_worker_key
            if len(logical_local_block_ids) == 0:
                local_block_ids = []
                remote_block_ids = []
            else:
                local_logical_ids, remote_logical_ids = (
                    self.transfer_topo.get_matched_blocks(
                        logical_local_block_ids,
                        logical_remote_block_ids,
                        remote_info.remote_dcp_size,
                        remote_dcp_rank,
                        local_block_offset,
                    )
                )
                local_block_ids = self._logical_to_kernel_block_ids(local_logical_ids)
                remote_block_ids = self._logical_to_remote_kernel_block_ids(
                    remote_logical_ids,
                    remote_info.remote_physical_blocks_per_logical,
                )

            if any(len(group) > 0 for group in local_block_ids):
                launched_read = True

            spec = ReadSpec(
                remote_rank=remote_tp_rank,
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
            )
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote worker %s with remote block size %s for req %s",
                meta.remote.engine_id,
                remote_worker_key,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if tp_ratio < 0 and not self.use_mla:
                assert remote_block_size == self.block_size
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][
                    remote_tp_rank - self.tp_rank * (-tp_ratio)
                ]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            # Destination handle: remote_engine_id -> (tp_rank, dcp_rank) -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                remote_worker_key
            ]
            expected_consumers = self.transfer_topo.calculate_local_consumer_count(
                engine_id,
                remote_worker_key,
            )

            self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                remote_worker_key=remote_worker_key,
                expected_consumers=expected_consumers,
            )

        if not launched_read:
            self._done_recving_without_xfer.add(req_id)

    def _local_logical_block_size(self) -> int:
        return (
            self.block_size * self._physical_blocks_per_logical_kv_block * self.dcp_size
        )

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
        remote_worker_key: RemoteWorkerKey | None = None,
        expected_consumers: int | None = None,
    ):
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        if remote_worker_key is None:
            remote_worker_key = (remote_rank, 0)
        if expected_consumers is None:
            expected_consumers = self.world_size
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids
        # Number of local workers that will notify this producer worker.
        notif_id = f"{remote_request_id}:{expected_consumers}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if not any(len(group) > 0 for group in local_block_ids):
            # A full prefix cache hit is indicated with an empty list.
            agent_name = self._remote_agents[dst_engine_id][remote_worker_key]
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
                    remote_worker_key=remote_worker_key,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()
            return

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            # TODO (NickLucche) assume HMA is off. Change to handle multiple KV groups.
            assert not self._is_hma_required
            local_block_ids0 = local_block_ids[0] if local_block_ids else []
            remote_block_ids0 = remote_block_ids[0]
            local_block_ids_mapped = self.get_mapped_blocks(
                np.asarray(local_block_ids0), block_size_ratio
            ).tolist()
            if len(local_block_ids_mapped) > len(remote_block_ids0):
                # NOTE:
                # get_mapped_blocks will always expand block_ids for n times.
                # ex:
                # prefill block_ids with block_size as 4:
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # Local decode block_ids with block_size as 16: [1, 2, 3]
                # expanded decode block_ids with get_mapped_blocks from [1, 2, 3] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                # Then we clip local to align with prefill
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                local_block_ids_mapped = local_block_ids_mapped[
                    : len(remote_block_ids0)
                ]
            local_block_ids = [local_block_ids_mapped] if local_block_ids_mapped else []
            remote_block_ids = [remote_block_ids0]
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes.

        assert (
            len(remote_block_ids)
            == len(local_block_ids)
            == len(self.kv_cache_config.kv_cache_groups)
        )
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
                remote_worker_key=remote_worker_key,
            )
            self._handle_failed_transfer(request_id, handle)

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.

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

                consumers_per_producer = int(expected_consumers)
                self.expected_consumer_notifications_by_req[req_id] = max(
                    consumers_per_producer,
                    self.expected_consumer_notifications_by_req.get(req_id, 0),
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
