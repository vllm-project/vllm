# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import math
import time
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds, EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
    _LocalDescs,
    _RemoteStage,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HB_NOTIF_PREFIX,
    NixlConnectorMetadata,
    ReqId,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import _is_attention_spec
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
        # Registration generation of each remote engine's loaded agent.
        self._remote_registration_generations: dict[EngineId, int] = {}

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        # Heartbeats keep P's blocks alive while requests wait on D. Sent first so
        # replicated-PCP ranks > 0 (which return early) and this step's reads see
        # this round's renewals.
        self._send_heartbeats(metadata)

        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_group
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
            # Full local hits and aborted cleanup only notify P; no recv is awaited.
            # On a full local hit:
            # - Notification failure must not fail the request: its KV is local.
            # - Receive completion must not be reported: num_external_tokens == 0,
            #   so the scheduler never entered WAITING_FOR_REMOTE_KVS to begin with.
            # Aborted cleanup requests have already been removed from the scheduler.
            if meta.awaiting_kvs or any(meta.local_block_ids):
                self._recving_metadata[req_id] = meta
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        if self.pcp_rank > 0 and not self.pcp_dcp_sharded:
            # Replicated-KV PCP: only PCP rank 0 serves the KV, so this rank
            # has nothing to send. Report the requests as sent right away so
            # the scheduler-side aggregation (world_size workers, and any
            # sibling connector inside a MultiConnector) still completes.
            self._replicated_pcp_done_sending.update(metadata.reqs_to_send)
            return

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

    def _is_remote_read_expired(self, meta: ReqMeta) -> bool:
        """Whether the remote's lease on the blocks has (nearly) run out, after
        which they may hold other requests' KV. Uses the exported deadline, rebased
        by the clock offset and extended by our heartbeats' renewals."""
        remote = meta.remote
        assert remote is not None
        clock_offset = self._engine_clock_offset.get(remote.engine_id)
        # No read for a full local hit; no deadline (older peer): read as before.
        if (
            not meta.local_physical_block_ids
            or remote.blocks_expiry_time is None
            or clock_offset is None
        ):
            return False
        deadline = max(
            remote.blocks_expiry_time - clock_offset,
            self._remote_leases.get((remote.engine_id, remote.request_id), -math.inf),
        )
        return time.perf_counter() + self._lease_safety_margin >= deadline

    def _get_lease_lapsed_recvs(self, done_recving: set[ReqId]) -> set[ReqId]:
        """Completed reads that may have overlapped the remote's lease expiry,
        so the blocks could have been recycled while being read."""
        lapsed: set[ReqId] = set()
        for req_id in done_recving:
            meta = self._recving_metadata.get(req_id)
            if meta is None or meta.remote is None:
                continue
            # Completion is observed no earlier than it happened: conservative.
            if self._is_remote_read_expired(meta):
                logger.debug(
                    "Lease of %s (remote %s on engine %s) may have run out "
                    "before its read completed",
                    req_id,
                    meta.remote.request_id,
                    meta.remote.engine_id,
                )
                self.xfer_stats.record_declined_expired_read()
                lapsed.add(req_id)
        return lapsed

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        if engine_id in self._invalid_remote_engines:
            self._handle_failed_transfer(
                req_id, None, self._recv_failures, record_failed_transfer=False
            )
            return
        generation = meta.remote.registration_generation
        if self._remote_registration_generations.get(engine_id, 0) < generation:
            # The remote registered its KV cache again, eg after a sleep.
            if engine_id in self._remote_agents:
                self._cleanup_remote_engine(engine_id, log_eviction=False)
            self._remote_registration_generations[engine_id] = generation
        if engine_id not in self._remote_agents:
            # Also re-handshakes queued requests that outlived invalid-peer cleanup.
            self._background_nixl_handshake(req_id, engine_id, meta)
            return
        if self._is_remote_read_expired(meta):
            # Warn once per engine; each decline is counted and debug-logged.
            logger.warning_once(
                "Declining remote KV reads from engine %s whose lease ran out "
                "before the read (router held the request longer than "
                "kv_lease_duration?); kv_load_failure_policy applies. Counted "
                "in vllm:nixl_num_declined_expired_reads.",
                engine_id,
            )
            logger.debug(
                "Declined expired remote read for %s (remote %s on engine %s)",
                req_id,
                meta.remote.request_id,
                engine_id,
            )
            self.xfer_stats.record_declined_expired_read()
            # Reported separately from transport failures, so only the state
            # cleanup side of _handle_failed_transfer runs here.
            self._handle_failed_transfer(
                req_id, None, self._recv_failures, record_failed_transfer=False
            )
            return

        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()

        if any(len(group) > 0 for group in meta.local_block_ids):
            # The scheduler waits for finished_recving from *every* worker.
            # Under DCP a rank's slice can legitimately come out empty when its
            # interleaved positions fall past the end of the sequence. _read_blocks
            # then takes the notify-only path without registering a transfer.
            # Seed the entry so this rank still reports completion.
            self._recving_transfers.setdefault(req_id, [])

        source_ranks = self.source_ranks[engine_id]
        all_source_ranks = sorted(set().union(*source_ranks))
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        # Local ranks reading each remote rank, all of which must notify it.
        num_consumers = self.transfer_topo.dcp_consumer_count(
            remote_info.remote_tp_size, remote_info.remote_dcp_size
        )
        # One stage per PP rank of the remote; a PP=1 remote is a single stage.
        stages = self._remote_stages[engine_id]

        dcp_active = self.dcp_size > 1 or remote_info.remote_dcp_size > 1
        different_block_sizes = (
            self.block_size != remote_info.remote_block_size
            or remote_info.remote_physical_blocks_per_logical
            != self._physical_blocks_per_logical_kv_block
        )
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(meta.local_block_ids)

        def group_ids(block_ids: BlockIds, rank: int) -> list[list[int]]:
            return [
                list(block_ids[g]) if rank in source_ranks[g] else []
                for g in range(num_groups)
            ]

        if (
            local_block_ids
            and different_block_sizes
            and not dcp_active
            and meta.remote.num_tokens is None
        ):
            # P's block size is only known after the handshake, so this is the
            # earliest point to reject a pull the blocks can't be paired for.
            logger.error_once(
                "Declining remote reads from engine %s: remote_num_tokens is "
                "required when P/D block sizes differ (local block_size=%s, "
                "remote block_size=%s, local/remote physical_per_logical=%s/%s).",
                engine_id,
                self.block_size,
                remote_info.remote_block_size,
                self._physical_blocks_per_logical_kv_block,
                remote_info.remote_physical_blocks_per_logical,
            )
            self._handle_failed_transfer(
                req_id, None, self._recv_failures, record_failed_transfer=False
            )
            return

        remote_block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids, self._remote_physical_blocks_per_group[engine_id]
        )
        meta.remote.block_ids = remote_block_ids
        # The remote cache group each of ours pairs with (`_pair_regions`).
        remote_group_ids = self.remote_group_ids[engine_id]
        # Our blocks of each cache group and the remote's they are read from,
        # per remote rank.
        reads: dict[int, tuple[BlockIds, BlockIds]] = {}
        if not local_block_ids:
            # A full prefix cache hit reads none: _read_blocks just notifies P.
            reads = {rank: ([], []) for rank in all_source_ranks}
        elif not (dcp_active or different_block_sizes):
            # Both sides hold the same blocks: read the remote's past our
            # prefix cache hit.
            local_block_ids, remote_block_ids = self._apply_prefix_caching(
                decode_block_ids=local_block_ids,
                prefill_block_ids=remote_block_ids,
                decode_physical_per_logical=self._physical_blocks_per_logical_kv_block,
                prefill_physical_per_logical=(
                    remote_info.remote_physical_blocks_per_logical
                ),
                prefill_group_ids=remote_group_ids,
            )
            reads = {
                rank: (
                    group_ids(local_block_ids, rank),
                    group_ids(remote_block_ids, rank),
                )
                for rank in all_source_ranks
            }
        else:
            # DCP and different block sizes pair blocks by their position in
            # the request (`_pair_blocks`).
            transfer_group_ids = self.kv_cache_config.transfer_group_ids
            covered = [range(0)] * num_groups
            # The blocks past the last token are only left out when the
            # receive post-process zeroes them.
            num_tokens = meta.remote.num_tokens if not dcp_active else None

            def blocks_read_from(rank: int, group: int) -> tuple[list[int], list[int]]:
                """Our blocks of a cache group and the remote rank's blocks
                they are read from."""
                remote_group = remote_group_ids[group]
                local = list(local_block_ids[group])
                remote = list(remote_block_ids[remote_group])
                ranks = source_ranks[group]
                # Every remote rank holds a DCP-replicated group whole: read
                # it from one.
                replicated = dcp_active and group in self._dcp_replicated_groups
                if rank not in ranks or (
                    replicated and rank != ranks[self.tp_rank % len(ranks)]
                ):
                    return [], []
                # A hybrid model's decode may hold one more state block than
                # its prefill, for the final token it recomputes:
                # `_apply_prefix_caching` pairs state blocks.
                if not _is_attention_spec(self._group_spec_types[group]):
                    covered[group] = range(len(local))
                    return local, remote
                assert not replicated or len(local) <= len(remote), (
                    f"Group {group}: decode needs {len(local)} blocks, "
                    f"prefill sent {len(remote)}"
                )
                # DCP shares out a request's blocks from its first, so the
                # blocks in our prefix cache have to be counted.
                counts_from_first_token = dcp_active and not replicated
                local, remote, covered[group] = self._pair_blocks(
                    engine_id,
                    rank,
                    group,
                    local,
                    remote_group,
                    remote,
                    num_computed_blocks=(
                        meta.local_num_computed_blocks[transfer_group_ids[group]]
                        if counts_from_first_token
                        else None
                    ),
                    num_tokens=num_tokens,
                )
                # Only a DCP-replicated group is read whole. Otherwise the
                # remote rank may lack blocks we hold: a DCP rank allocates
                # past the last token, and with different block sizes the
                # receive post-process zeroes what no read covers.
                if covered[group].start and replicated:
                    raise ValueError("Remote KV pages do not cover the requested range")
                return local, remote

            for rank in all_source_ranks:
                blocks = [blocks_read_from(rank, group) for group in range(num_groups)]
                local_ids: BlockIds = [local for local, _ in blocks]
                remote_ids: BlockIds = [remote for _, remote in blocks]
                if not dcp_active:
                    # Attention blocks are paired already; this pairs the state
                    # blocks.
                    local_ids, remote_ids = self._apply_prefix_caching(
                        decode_block_ids=local_ids,
                        prefill_block_ids=remote_ids,
                        decode_physical_per_logical=(
                            self._physical_blocks_per_logical_kv_block
                        ),
                        prefill_physical_per_logical=(
                            remote_info.remote_physical_blocks_per_logical
                        ),
                    )
                reads[rank] = (local_ids, remote_ids)
            if not dcp_active:
                meta.covered_sub_blocks = covered

        # D may have to perform multiple reads from different remote ranks.
        for rank, (local_ids, remote_ids) in reads.items():
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                rank,
                remote_info.remote_block_size,
                req_id,
            )
            # Read every stage of the remote rank; each holds a slice of the
            # layers when the remote is pipeline-parallel.
            for stage in stages.values():
                local_block_descs_ids, remote_block_descs_ids = self._compute_desc_ids(
                    stage, rank, local_ids, remote_ids
                )
                # Once a read routes the request to failure reporting, the
                # scheduler may free and reuse its blocks, so no sibling READs
                # may be posted (and P must not be notified).
                if not self._read_blocks(
                    remote_rank=rank,
                    local_block_descs_ids=local_block_descs_ids,
                    remote_block_descs_ids=remote_block_descs_ids,
                    request_id=req_id,
                    dst_engine_id=meta.remote.engine_id,
                    remote_request_id=meta.remote.request_id,
                    expected_consumers=num_consumers,
                    awaiting_kvs=meta.awaiting_kvs,
                    stage=stage,
                ):
                    return

        # Remote ranks we handshook with but read nothing from (e.g. replicas of
        # replicated KV) still need to hear that we have the blocks we need, so
        # they can update the request state. Every PP stage of a chosen rank
        # was already read and notified.
        read_ranks = {
            (stage.pp_rank, rank) for rank in reads for stage in stages.values()
        }
        notif_id = f"{meta.remote.request_id}:{num_consumers}".encode()
        remote_agents = self._remote_agents[meta.remote.engine_id]
        for rank_to_notify, agent in remote_agents.items():
            if rank_to_notify in read_ranks:
                continue
            try:
                self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="Remote rank will not update request state. "
                    "This may indicate network issues.",
                    req_id=req_id,
                    error=e,
                    dst_engine_id=meta.remote.engine_id,
                    remote_rank=rank_to_notify[1],
                    remote_agent_name=agent,
                )
                self.xfer_stats.record_failed_notification()

    def _read_blocks(
        self,
        remote_rank: int,
        local_block_descs_ids: np.ndarray,
        remote_block_descs_ids: np.ndarray,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        expected_consumers: int,
        awaiting_kvs: bool,
        stage: _RemoteStage,
    ) -> bool:
        """Post a READ point-to-point xfer request from a single local worker to
        one PP stage of a single remote worker.

        Returns True when the read was posted (or was unnecessary), False
        when the request was routed to failure reporting — the caller must
        not post further transfers for it.
        """
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

        agent_name = self._remote_agents[dst_engine_id][(stage.pp_rank, remote_rank)]
        # Full prefix cache hit, or none of the blocks are in a cache group the
        # remote stage holds layers of: just notify P that we have the blocks
        # we need.
        if not len(local_block_descs_ids):
            # A full prefix cache hit is indicated with an empty list.
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
            # Report even on notification failure: the KV is already local, and
            # an unreported parked request would hold its blocks forever.
            # Notify-only recvs must stay unreported (scheduler asserts).
            if awaiting_kvs:
                self._recving_transfers.setdefault(request_id, [])
            return True

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.
        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        # Prepare transfer with Nixl.
        remote_xfer_side_handle = stage.xfer_side_handles[remote_rank]
        handle = None
        try:
            local_descs = stage.local_descs
            assert local_descs is not None
            if self._mixed_mem_types:
                self._read_blocks_mixed(
                    request_id=request_id,
                    local_descs=local_descs,
                    remote_xfer_side_handle=remote_xfer_side_handle,
                    local_block_descs_ids=local_block_descs_ids,
                    remote_block_descs_ids=remote_block_descs_ids,
                    notif_agent=self._remote_agents[dst_engine_id][(0, remote_rank)],
                    notif_id=notif_id,
                )
                return True
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_descs.handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            self._recving_transfers[request_id].append(handle)
            return True
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Deferring failure reporting until outstanding transfers finish",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            if not self._handle_failed_transfer(
                request_id, handle, self._recv_failures, error=e
            ):
                assert handle is not None
                self._recving_transfers[request_id].append(handle)
            return False

    def _read_blocks_mixed(
        self,
        request_id: str,
        local_descs: _LocalDescs,
        remote_xfer_side_handle: int,
        local_block_descs_ids: np.ndarray,
        remote_block_descs_ids: np.ndarray,
        notif_agent: str,
        notif_id: bytes,
    ) -> None:
        """Split a READ across the local DRAM and device descriptor lists."""
        desc_is_dram, desc_pos = local_descs.desc_is_dram, local_descs.desc_pos
        assert desc_is_dram is not None and desc_pos is not None
        local_dram_handle = local_descs.dram_handle
        local_ids = np.asarray(local_block_descs_ids)
        remote_ids = np.asarray(remote_block_descs_ids)
        is_dram = desc_is_dram[local_ids]

        if self._skip_dram_xfer:
            # The node's host leader reads the DRAM (shared host pool) part for
            # every local rank; keep only the device part. Register the request
            # even if nothing is left, so it still completes and notifies.
            keep = ~is_dram
            local_ids = local_ids[keep]
            remote_ids = remote_ids[keep]
            is_dram = is_dram[keep]
            self._recving_transfers.setdefault(request_id, [])
        assert local_dram_handle is not None or not is_dram.any()
        reads = (
            (is_dram, local_dram_handle),
            (~is_dram, local_descs.handle),
        )
        handles: list[int] = []
        try:
            for mask, local_handle in reads:
                if mask.any():
                    handles.append(
                        self.nixl_wrapper.make_prepped_xfer(
                            "READ",
                            local_handle,
                            desc_pos[local_ids[mask]],
                            remote_xfer_side_handle,
                            remote_ids[mask],
                        )
                    )
        except Exception:
            for handle in handles:
                if not self._try_release_xfer_handle(request_id, handle):
                    self._recving_transfers[request_id].append(handle)
            raise

        self._pending_recv_notifs.setdefault(request_id, []).append(
            (notif_agent, notif_id)
        )
        for i, handle in enumerate(handles):
            try:
                self.nixl_wrapper.transfer(handle)
            except Exception:
                for unstarted in handles[i:]:
                    if not self._try_release_xfer_handle(request_id, unstarted):
                        self._recving_transfers[request_id].append(unstarted)
                raise
            self._recving_transfers[request_id].append(handle)

    def _get_new_notifs(self) -> set[str]:
        """Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP or DCP
        scenario), wait for all consumers to be done pulling.

        Also handles heartbeat notifications (HB_NOTIF_PREFIX + "req1,req2") by
        extending the lease on the referenced requests.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                msg = notif.decode("utf-8")

                # Handle heartbeat messages from D-side.
                if msg.startswith(HB_NOTIF_PREFIX):
                    self._handle_heartbeat(msg[len(HB_NOTIF_PREFIX) :])
                    continue

                req_id, expected_consumers = msg.rsplit(":", 1)
                if (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    self.xfer_stats.record_notification_after_expiry()
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
