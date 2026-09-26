# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import os
import time
from collections.abc import Iterator
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HeartbeatInfo,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_receiver import (
    BackendEvent,
    NixlPullReceiver,
    NotifyOnlyTerminal,
    ReadJob,
    ReceiveRetired,
    ReceiveTerminal,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import NixlKVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Slack (seconds) subtracted from D's exported block-expiry deadline on the turn-2
# readback, absorbing clock-offset error and read latency.
_KV_BLOCKS_EXPIRY_SAFETY_MARGIN = 5.0


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-specific (READ) worker logic."""

    _background_receiver_enabled = False
    _receiver: NixlPullReceiver | None = None

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self._receiver: NixlPullReceiver | None = None
        super().__init__(vllm_config, engine_id, kv_cache_config)
        enabled = self.kv_transfer_config.get_from_extra_config(
            "background_receiver", False
        )
        if not isinstance(enabled, bool):
            raise ValueError("background_receiver must be a boolean")
        self._background_receiver_enabled = enabled
        if enabled and (
            self.pp_size != 1
            or self.pcp_size != 1
            or self.dcp_size != 1
            or self._mixed_mem_types
            or self._is_csa_linear
            or self.use_host_buffer
            or self.kv_buffer_device != "cuda"
            or self._bidirectional_kv_xfer_enabled
            or self.kv_transfer_config.enable_permute_local_kv
            or self.enable_heterogeneous_attn_post_process
            or self._has_mamba
        ):
            raise ValueError(
                "background_receiver requires PP1/PCP1/DCP1, direct CUDA, "
                "unidirectional pull and matching attention cache layouts"
            )
        self._receiver_keys: dict[str, tuple[str, int]] = {}
        self._receiver_stats = NixlKVConnectorStats()
        self._receiver_published_sequence = 0

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        self._start_receiver()

    def _start_receiver(self):
        if not self._background_receiver_enabled or self._receiver is not None:
            return
        cfg = self.kv_transfer_config.get_from_extra_config
        max_history = cfg("background_receiver_max_seen_requests", 100000)
        max_notify = cfg("background_receiver_max_notify_only", max_history)
        if max_notify < max_history:
            raise ValueError(
                "background_receiver_max_notify_only must cover the request "
                "lifetime budget (background_receiver_max_seen_requests)"
            )
        if self._mixed_mem_types or self._transfer_layer_names:
            raise ValueError(
                "background_receiver does not support mixed memory or "
                "layer-name descriptor routing"
            )
        self._receiver = NixlPullReceiver(
            _PullReceiverBackend(self),
            max_receives=cfg("background_receiver_max_pending", 64),
            # Full prefix hits and pre-admission aborts cannot wait on receive
            # credits. Reserve one notification slot for every lifetime identity.
            max_notify_only=max_notify,
            max_controls=cfg("background_receiver_max_controls", 256),
            max_history=max_history,
            fatal_handler=_PullReceiverBackend.fatal,
        )
        self._receiver.start()

    def _publish_receiver_metadata(self, metadata):
        receiver = self._receiver
        assert receiver is not None
        receiver.check_health()
        if metadata.receiver_heartbeat_version is not None:
            receiver.publish_snapshot(
                metadata.receiver_heartbeat_version,
                deepcopy(metadata.heartbeat_by_engine),
            )
        if (
            metadata.reqs_in_batch
            or metadata.reqs_not_processed
            or metadata.reqs_to_send
        ):
            sequence = self._receiver_published_sequence + 1
            # Commit before making the command visible. A failed publication
            # is process-fatal, so no uncommitted sequence can be reused.
            self._receiver_published_sequence = sequence
            receiver.publish_control(
                (
                    "lifecycle",
                    sequence,
                    metadata.scheduler_clock,
                    frozenset(metadata.reqs_in_batch),
                    frozenset(metadata.reqs_not_processed),
                    tuple(metadata.reqs_to_send.items()),
                )
            )
        for req_id, meta in metadata.reqs_to_recv.items():
            if meta.receiver_generation <= 0:
                raise RuntimeError(
                    "Receiver requires generation-aware scheduler metadata"
                )
            key = (req_id, meta.receiver_generation)
            existing = self._receiver_keys.get(req_id)
            if existing is not None and existing != key:
                raise RuntimeError("Request ID reused before receiver retirement")
            # Only logical block metadata is copied here. Descriptor expansion
            # and native construction run exclusively on the receiver.
            if receiver.submit(
                ReadJob(key, deepcopy(meta), not meta.receiver_is_async)
            ):
                self._receiver_keys[req_id] = key
        receiver.check_health()

    def get_finished(self, finished_req_ids: set[str] | None = None):
        if self._receiver is None:
            return super().get_finished()
        receiver = self._receiver
        receiver.check_health()
        for req_id in finished_req_ids or ():
            if (key := self._receiver_keys.get(req_id)) is not None:
                receiver.cancel(key)
        sent, received = set(), set()
        for event in receiver.drain_results(limit=128):
            if isinstance(event, ReceiveTerminal):
                # Enabled shapes have no device conversion/finalization work.
                # Retain allocation credit until the owner acknowledges retirement.
                receiver.finalized(event.job.key)
            elif isinstance(event, ReceiveRetired):
                req_id = event.job.key[0]
                self._receiver_keys.pop(req_id, None)
                received.add(req_id)
            elif isinstance(event, NotifyOnlyTerminal):
                self._receiver_keys.pop(event.job.key[0], None)
            elif isinstance(event, BackendEvent):
                done_sending, stats = event.payload
                sent.update(done_sending)
                if stats is not None:
                    self._receiver_stats.aggregate(stats)
        receiver.check_health()
        return sent, received

    def get_transfer_results(
        self, finished_req_ids: set[str] | None = None
    ) -> KVConnectorTransferResults:
        if self._receiver is None:
            return super().get_transfer_results()
        sent, received = self.get_finished(finished_req_ids)
        return KVConnectorTransferResults(
            finished_sending=sent,
            finished_recving=received,
        )

    def get_kv_connector_stats(self):
        if self._receiver is None:
            return super().get_kv_connector_stats()
        self._receiver.check_health()
        if not self._receiver_stats.is_empty():
            return self._receiver_stats.clone_and_reset()
        return None

    def get_block_ids_with_load_errors(self):
        if self._receiver is None:
            return super().get_block_ids_with_load_errors()
        # All uncertain transport errors fail the worker. In particular, HMA
        # must never enter the core's single-cache-group invalid-block handler.
        self._receiver.check_health()
        return set()

    def shutdown(self):
        receiver = getattr(self, "_receiver", None)
        if receiver is None:
            return super().shutdown()
        receiver.shutdown(wait=False)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self.get_finished()
            if receiver.shutdown(wait=True, timeout=0.01):
                return
        _PullReceiverBackend.fatal(
            RuntimeError("Receiver shutdown did not quiesce within 30 seconds")
        )

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        if self._background_receiver_enabled:
            if self._receiver is None:
                raise RuntimeError("Receiver cache registration has not completed")
            self._publish_receiver_metadata(metadata)
            return
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
            # Full local hits and aborted cleanup only notify P; no recv is awaited.
            # On a full local hit:
            # - Notification failure must not fail the request: its KV is local.
            # - Receive completion must not be reported: num_external_tokens == 0,
            #   so the scheduler never entered WAITING_FOR_REMOTE_KVS to begin with.
            # Aborted cleanup requests have already been removed from the scheduler.
            if meta.awaiting_kvs or any(meta.local_block_ids):
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
            # KV expiry is reported separately from transport failures, so only
            # the state cleanup side of _handle_failed_transfer runs here.
            self._handle_failed_transfer(
                req_id, None, self._recv_failures, record_failed_transfer=False
            )
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

        dcp_active = self.dcp_size > 1 or remote_info.remote_dcp_size > 1
        local_block_ids = meta.local_physical_block_ids
        remote_region_groups = self.dst_region_group_ids[engine_id]
        local_region_groups = self.region_group_ids or remote_region_groups
        if not local_block_ids:
            # Region expansion cannot index empty groups. Pass empty specs to
            # _read_blocks so its existing cache-hit notification path runs.
            read_specs = [
                ReadSpec(remote_rank=rank, local_block_ids=[], remote_block_ids=[])
                for rank in plan.all_source_ranks
            ]
        elif local_region_groups != remote_region_groups:
            if not self.use_mla or self._has_mamba:
                raise NotImplementedError(
                    "Different NIXL cache-group layouts are only supported for "
                    "pure MLA models"
                )
            if self.block_size != remote_info.remote_block_size:
                raise NotImplementedError(
                    "Region-mapped NIXL transfers require matching physical block sizes"
                )
            remote_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.remote.block_ids,
                remote_info.remote_physical_blocks_per_logical,
            )
            remote_by_region = self._block_ids_by_region(
                remote_physical_block_ids, remote_region_groups
            )
            local_by_region = self._block_ids_by_region(
                local_block_ids, local_region_groups
            )
            num_computed_blocks = None
            num_remote_blocks = None
            if (
                meta.remote.num_tokens is not None
                and meta.local_num_computed_blocks
                and all(group >= 0 for group in local_region_groups)
                and all(group >= 0 for group in remote_region_groups)
            ):
                transfer_groups = self.kv_cache_config.transfer_group_ids
                num_computed_blocks = [
                    meta.local_num_computed_blocks[transfer_groups[group]]
                    * self._physical_blocks_per_logical_kv_block
                    for group in local_region_groups
                ]
                num_remote_blocks = cdiv(
                    meta.remote.num_tokens, remote_info.remote_block_size
                )
            elif any(local_by_region) and (
                dcp_active
                or remote_info.remote_physical_blocks_per_logical
                != self._physical_blocks_per_logical_kv_block
            ):
                raise NotImplementedError(
                    "Region-mapped pulls with DCP or different logical block sizes "
                    "require remote_num_tokens, per-group prefix counts "
                    "and unshared regions"
                )
            read_specs = [
                ReadSpec(
                    rank,
                    *self._apply_prefix_caching_by_region(
                        local_by_region,
                        remote_by_region,
                        num_computed_blocks=num_computed_blocks,
                        num_remote_blocks=num_remote_blocks,
                        remote_rank=rank,
                        remote_dcp_size=remote_info.remote_dcp_size,
                    ),
                    block_ids_by_region=True,
                )
                for rank in plan.all_source_ranks
            ]
            meta.region_blocks_to_zero = [
                list(blocks[sum(len(spec.local_block_ids[r]) for spec in read_specs) :])
                for r, blocks in enumerate(local_by_region)
            ]
        else:
            remote_logical_block_ids = meta.remote.block_ids
            meta.remote.block_ids = self._logical_to_kernel_block_ids(
                remote_logical_block_ids,
                remote_info.remote_physical_blocks_per_logical,
            )
            num_groups = len(meta.local_block_ids)

            def group_ids(block_ids: BlockIds, rank: int) -> list[list[int]]:
                return [
                    list(block_ids[g]) if rank in plan.source_ranks_per_group[g] else []
                    for g in range(num_groups)
                ]

            read_specs = []
            for rank in plan.all_source_ranks:
                if dcp_active:
                    local_ids = group_ids(meta.local_block_ids, rank)
                    remote_ids = group_ids(remote_logical_block_ids, rank)
                    for g in range(num_groups):
                        if not local_ids[g] or not _is_attention_spec(
                            self._group_spec_types[g]
                        ):
                            continue
                        local_ids[g], remote_ids[g] = self._apply_dcp_prefix_caching(
                            local_ids[g],
                            remote_ids[g],
                            remote_rank=rank,
                            local_dcp_size=self.dcp_size,
                            local_dcp_rank=self.dcp_rank,
                            remote_dcp_size=remote_info.remote_dcp_size,
                            local_num_computed_blocks=(
                                meta.local_num_computed_blocks[g]
                            ),
                        )
                    local_physical_ids = self._logical_to_kernel_block_ids(
                        local_ids, self._physical_blocks_per_logical_kv_block
                    )
                    remote_physical_ids = self._logical_to_kernel_block_ids(
                        remote_ids,
                        remote_info.remote_physical_blocks_per_logical,
                    )
                else:
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
                local_dram_handle = (
                    self._dram_src_handles_by_tp_ratio[split_key][i]
                    if self._mixed_mem_types
                    else None
                )
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]
                local_dram_handle = (
                    self._dram_src_handles_by_block_size[remote_block_size]
                    if self._mixed_mem_types
                    else None
                )

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            # Once a read routes the request to failure reporting, the
            # scheduler may free and reuse its blocks, so no sibling READs
            # may be posted (and P must not be notified).
            if not self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                local_dram_handle=local_dram_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                expected_consumers=plan.local_consumers,
                awaiting_kvs=meta.awaiting_kvs,
            ):
                return

        if self.use_mla and tp_ratio < 0 and len(read_specs) == 1:
            # ..but we still need to notify the other remote ranks that we
            # have the blocks we need so they can update the request state.
            # Same thing for DCP (tp_size == dcp_size), so the raw tp_ratio already
            # reflects whether any remote replica is left unchosen.
            notif_id = f"{meta.remote.request_id}:{plan.local_consumers}".encode()
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != (0, read_specs[0].remote_rank):
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
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        local_dram_handle: int | None,
        remote_xfer_side_handle: int,
        expected_consumers: int,
        awaiting_kvs: bool,
    ) -> bool:
        """Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.

        Returns True when the read was posted (or was unnecessary), False
        when the request was routed to failure reporting — the caller must
        not post further transfers for it.
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
            if read_spec.block_ids_by_region:
                raise NotImplementedError(
                    "Region-mapped NIXL transfers require matching physical block sizes"
                )
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
            # Report even on notification failure: the KV is already local, and
            # an unreported parked request would hold its blocks forever.
            # Notify-only recvs must stay unreported (scheduler asserts).
            if awaiting_kvs:
                self._recving_transfers.setdefault(request_id, [])
            return True

        if not read_spec.block_ids_by_region:
            assert (
                len(remote_block_ids)
                == len(local_block_ids)
                == len(self.kv_cache_config.transfer_groups)
            )
            if not (self.dcp_size > 1 or remote_info.remote_dcp_size > 1):
                local_block_ids, remote_block_ids = self._apply_prefix_caching(
                    decode_block_ids=local_block_ids,
                    prefill_block_ids=remote_block_ids,
                    decode_physical_per_logical=(
                        self._physical_blocks_per_logical_kv_block
                    ),
                    prefill_physical_per_logical=(
                        remote_info.remote_physical_blocks_per_logical
                    ),
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
            region_num_blocks=(self.dst_region_num_blocks.get(dst_engine_id) or None),
            region_group_ids=(
                list(range(self.num_regions))
                if read_spec.block_ids_by_region
                else (self.dst_region_group_ids.get(dst_engine_id) or None)
            ),
            uses_region_group_mapping=(
                self.num_regions > 1
                if read_spec.block_ids_by_region
                else self.dst_uses_region_group_mapping[dst_engine_id]
            ),
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            region_num_blocks=(self.dst_region_num_blocks.get(self.engine_id) or None),
            region_group_ids=(
                list(range(self.num_regions))
                if read_spec.block_ids_by_region
                else (self.region_group_ids or None)
            ),
            uses_region_group_mapping=(
                self.num_regions > 1
                if read_spec.block_ids_by_region
                else self._uses_region_group_mapping
            ),
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        # Prepare transfer with Nixl.
        handle = None
        try:
            if self._mixed_mem_types:
                self._read_blocks_mixed(
                    request_id=request_id,
                    local_block_size_key=remote_info.remote_block_size,
                    local_device_handle=local_xfer_side_handle,
                    local_dram_handle=local_dram_handle,
                    remote_xfer_side_handle=remote_xfer_side_handle,
                    local_block_descs_ids=local_block_descs_ids,
                    remote_block_descs_ids=remote_block_descs_ids,
                    notif_agent=self._remote_agents[dst_engine_id][(0, remote_rank)],
                    notif_id=notif_id,
                )
                return True
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
                request_id, handle, self._recv_failures
            ):
                assert handle is not None
                self._recving_transfers[request_id].append(handle)
            return False

    def _read_blocks_mixed(
        self,
        request_id: str,
        local_block_size_key: int,
        local_device_handle: int,
        local_dram_handle: int | None,
        remote_xfer_side_handle: int,
        local_block_descs_ids: np.ndarray,
        remote_block_descs_ids: np.ndarray,
        notif_agent: str,
        notif_id: bytes,
    ) -> None:
        """Split a READ across the local DRAM and device descriptor lists."""
        desc_is_dram = self._desc_is_dram_by_block_size[local_block_size_key]
        desc_pos = self._desc_pos_by_block_size[local_block_size_key]
        local_ids = np.asarray(local_block_descs_ids)
        remote_ids = np.asarray(remote_block_descs_ids)
        is_dram = desc_is_dram[local_ids]

        assert local_dram_handle is not None
        reads = (
            (is_dram, local_dram_handle),
            (~is_dram, local_device_handle),
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


@dataclass
class _PullTransfer:
    handle: Any
    request_id: str
    descriptors: int
    generation: int = 0
    remote_rank: int = 0


class _PullReceiverBackend:
    """Native adapter; every method except construction runs on the owner."""

    def __init__(self, worker: NixlPullConnectorWorker):
        self.worker = worker
        self.handshakes: dict[str, tuple[Future, int]] = {}
        self.heartbeats: dict[str, HeartbeatInfo] = {}
        self.birth_sequence: dict[str, int] = {}
        self.retired_producers: set[str] = set()
        self.pending_notifications: list[tuple[str, int, int]] = []
        self.applied_sequence = 0
        self.next_heartbeat = 0.0
        self.next_stats = 0.0
        cfg = worker.kv_transfer_config.get_from_extra_config
        self.max_controls = cfg("background_receiver_max_controls", 256)
        self.max_history = cfg("background_receiver_max_seen_requests", 100000)
        self.heartbeat_interval = cfg("background_receiver_heartbeat_interval", 5.0)
        self.max_heartbeat_targets = cfg(
            "background_receiver_max_heartbeat_targets", 4096
        )

    def initialize(self):
        current_platform.set_device(self.worker.device_id)

    @staticmethod
    def fatal(error):
        # Uncertain DMA must not enter normal Python shutdown: native handle
        # __del__ methods can cancel/free on whichever thread runs finalizers.
        try:
            message = f"NIXL background receiver fatal; hard worker exit: {error}\n"
            os.write(2, message.encode(errors="replace")[:8192])
        finally:
            # Diagnostic failures must not allow normal Python finalization.
            os._exit(1)

    def _validate_peer(self, metadata: NixlAgentMetadata, tp_size: int):
        w = self.worker
        if (
            tp_size != w.world_size
            or metadata.dcp_size != 1
            or metadata.pcp_size != 1
            or metadata.block_size != w.block_size
            or metadata.physical_blocks_per_logical_kv_block
            != w._physical_blocks_per_logical_kv_block
            or metadata.kv_cache_layout != w.kv_cache_layout
            or metadata.attn_backend_name != w.backend_name
            or metadata.block_lens != w.block_len_per_layer
            or metadata.block_strides != w.block_stride_per_layer
            or (metadata.region_group_ids or []) != w.region_group_ids
            or metadata.region_members != w.region_members
            or any(mem != "VRAM" for mem in metadata.region_mem_types or [])
            or len(metadata.kv_caches_base_addr) != len(w.block_len_per_layer)
            or metadata.ssm_sizes != w._mamba_ssm_size
        ):
            raise ValueError("Unsupported background receiver peer cache geometry")

    def _peer(self, engine_id, host, port, tp_size, pp_size=1):
        w = self.worker
        if tp_size != w.world_size or pp_size != 1:
            raise ValueError("background_receiver requires homogeneous TP and PP1")
        if engine_id in w._remote_agents:
            w._engine_last_active[engine_id] = time.perf_counter()
            return True
        entry = self.handshakes.get(engine_id)
        if entry is None:
            if len(self.handshakes) >= self.max_controls:
                raise RuntimeError("Receiver handshake capacity exhausted")
            future = w._handshake_initiation_executor.submit(
                w._nixl_handshake,
                host,
                port,
                tp_size,
                engine_id,
                remote_pp_size=pp_size,
                fetch_only=True,
            )
            self.handshakes[engine_id] = (future, tp_size)
            return False
        future, expected_tp = entry
        if expected_tp != tp_size:
            raise ValueError("Peer topology changed during handshake")
        if not future.done():
            return False
        metadata_by_rank, offset = future.result()
        assert w.transfer_topo is not None
        expected_ranks = {
            (0, rank) for rank in w.transfer_topo.handshake_target_ranks(tp_size)
        }
        if set(metadata_by_rank) != expected_ranks:
            raise ValueError("Receiver handshake rank set mismatch")
        for metadata in metadata_by_rank.values():
            if metadata.engine_id != engine_id:
                raise ValueError("Receiver peer identity mismatch")
            self._validate_peer(metadata, tp_size)
        names = {}
        for (pp_rank, tp_rank), metadata in metadata_by_rank.items():
            names[(pp_rank, tp_rank)] = w.add_remote_agent(metadata, tp_rank, tp_size)
        w._remote_agents[engine_id] = names
        w._engine_clock_offset[engine_id] = offset
        w._engine_last_active[engine_id] = time.perf_counter()
        del self.handshakes[engine_id]
        return True

    def ready(self, job):
        meta = job.metadata
        remote = meta.remote
        if remote is None:
            raise ValueError("Receive job lacks remote metadata")
        if meta.dcp_size != 1:
            raise ValueError("background_receiver requires DCP1 peers")
        return self._peer(
            remote.engine_id, remote.host, remote.port, meta.tp_size, meta.pp_size
        )

    def transfers(self, job) -> Iterator[_PullTransfer]:
        w = self.worker
        meta = job.metadata
        remote = meta.remote
        assert remote is not None and w.transfer_topo is not None
        local_blocks = w._logical_to_kernel_block_ids(
            meta.local_block_ids, w._physical_blocks_per_logical_kv_block
        )
        info = w.transfer_topo.get_engine_info(remote.engine_id)
        remote_blocks = w._logical_to_kernel_block_ids(
            remote.block_ids, info.remote_physical_blocks_per_logical
        )
        plan = w.tp_mappings[remote.engine_id]
        if len(plan.all_source_ranks) != 1:
            raise ValueError("Homogeneous receiver requires one source per TP rank")
        for rank in plan.all_source_ranks:
            local: BlockIds = [
                list(group) if rank in plan.source_ranks_per_group[g] else []
                for g, group in enumerate(local_blocks)
            ]
            other: BlockIds = [
                list(group) if rank in plan.source_ranks_per_group[g] else []
                for g, group in enumerate(remote_blocks)
            ]
            if len(local) != len(other) or len(local) != len(
                w.kv_cache_config.transfer_groups
            ):
                raise ValueError("Receive KV group count mismatch")
            local, other = w._apply_prefix_caching(
                local,
                other,
                w._physical_blocks_per_logical_kv_block,
                info.remote_physical_blocks_per_logical,
            )
            remote_descs = w._compute_desc_ids(
                block_ids=other,
                dst_num_blocks=w.dst_num_blocks[remote.engine_id],
                block_size_ratio=None,
                physical_blocks_per_logical=info.remote_physical_blocks_per_logical,
                region_num_blocks=w.dst_region_num_blocks.get(remote.engine_id) or None,
                region_group_ids=w.dst_region_group_ids.get(remote.engine_id) or None,
                uses_region_group_mapping=w.dst_uses_region_group_mapping[
                    remote.engine_id
                ],
            )
            local_descs = w._compute_desc_ids(
                block_ids=local,
                dst_num_blocks=w.dst_num_blocks[w.engine_id],
                block_size_ratio=1,
                physical_blocks_per_logical=w._physical_blocks_per_logical_kv_block,
                region_num_blocks=w.dst_region_num_blocks.get(w.engine_id) or None,
                region_group_ids=w.region_group_ids or None,
                uses_region_group_mapping=w._uses_region_group_mapping,
            )
            if len(local_descs) != len(remote_descs) or not len(local_descs):
                raise ValueError(
                    "Actual receive requires matching nonempty descriptors"
                )
            handle = w.nixl_wrapper.make_prepped_xfer(
                "READ",
                w.src_xfer_handles_by_block_size[info.remote_block_size],
                local_descs,
                w.dst_xfer_side_handles[remote.engine_id][rank],
                remote_descs,
                notif_msg=f"{remote.request_id}:1".encode(),
            )
            # Yield into the owner's registry before it posts this handle.
            yield _PullTransfer(
                handle,
                job.key[0],
                len(local_descs),
                job.key[1],
                rank,
            )

    def post(self, transfer):
        return self.worker.nixl_wrapper.transfer(transfer.handle)

    def poll(self, transfer):
        return self.worker.nixl_wrapper.check_xfer_state(transfer.handle)

    def release(self, transfer):
        w = self.worker
        try:
            telemetry = w.nixl_wrapper.get_xfer_telemetry(transfer.handle)
            w.xfer_stats.record_transfer(telemetry)
        except Exception:
            # Transport is already DONE. Missing telemetry is not a KV failure.
            logger.warning("Receiver telemetry unavailable", exc_info=True)
        w.nixl_wrapper.release_xfer_handle(transfer.handle)

    def notify_without_read(self, job):
        w = self.worker
        remote = job.metadata.remote
        assert remote is not None
        for name in w._remote_agents[remote.engine_id].values():
            w.nixl_wrapper.send_notif(name, notif_msg=f"{remote.request_id}:1".encode())

    def control(self, message):
        if isinstance(message, dict):
            if (
                sum(len(h.req_ids) for h in message.values())
                > self.max_heartbeat_targets
            ):
                raise RuntimeError("Receiver heartbeat membership capacity exhausted")
            self.heartbeats = message
            return
        kind, sequence, clock, added, removed, expiries = message
        if kind != "lifecycle" or sequence != self.applied_sequence + 1:
            raise RuntimeError("Producer lifecycle publication out of order")
        w = self.worker
        for req_id in added:
            if req_id in self.retired_producers:
                raise RuntimeError("Producer request ID reused after retirement")
            self.birth_sequence.setdefault(req_id, sequence)
            w._reqs_to_process.add(req_id)
        for req_id in removed:
            if req_id in w._reqs_to_send:
                raise RuntimeError("Aborted producer still has an exported lease")
            w._reqs_to_process.discard(req_id)
            self._retire_producer(req_id)
        now = time.perf_counter()
        for req_id, expiry in expiries:
            if req_id in w._reqs_to_process:
                w._reqs_to_send[req_id] = now + (expiry - clock) if clock else expiry
        self.applied_sequence = sequence
        if len(self.birth_sequence) + len(self.retired_producers) > self.max_history:
            raise RuntimeError("Producer identity history capacity exhausted")

    def _retire_producer(self, req_id):
        self.retired_producers.add(req_id)
        self.birth_sequence.pop(req_id, None)

    def tick(self, active_jobs):
        w = self.worker
        now = time.perf_counter()
        pinned_engines = set(self.heartbeats) | set(self.handshakes)
        for job in active_jobs:
            pinned_engines.add(job.metadata.remote.engine_id)
        if now >= self.next_heartbeat:
            targets = deepcopy(self.heartbeats)
            for job in active_jobs:
                meta = job.metadata
                remote = meta.remote
                heartbeat = targets.setdefault(
                    remote.engine_id,
                    HeartbeatInfo(
                        req_ids=set(),
                        host=remote.host,
                        port=remote.port,
                        tp_size=meta.tp_size,
                        pp_size=meta.pp_size,
                    ),
                )
                heartbeat.req_ids.add(remote.request_id)
            for engine_id, heartbeat in targets.items():
                if self._peer(
                    engine_id,
                    heartbeat.host,
                    heartbeat.port,
                    heartbeat.tp_size,
                    heartbeat.pp_size,
                ):
                    msg = ("HB:" + ",".join(sorted(heartbeat.req_ids))).encode()
                    for name in w._remote_agents[engine_id].values():
                        w.nixl_wrapper.send_notif(name, notif_msg=msg)
            self.next_heartbeat = now + self.heartbeat_interval

        notifications = w.nixl_wrapper.get_new_notifs()
        publication_fence = w._receiver_published_sequence
        for messages in notifications.values():
            for raw in messages:
                msg = raw.decode("utf-8")
                if msg.startswith("HB:"):
                    w._handle_heartbeat(msg[3:])
                    continue
                req_id, tp_size = msg.rsplit(":", 1)
                if req_id not in self.retired_producers:
                    self.pending_notifications.append(
                        (req_id, int(tp_size), publication_fence)
                    )
        sent, pending = set(), []
        for req_id, tp_size, fence in self.pending_notifications:
            if req_id in self.retired_producers:
                continue
            if tp_size != 1:
                raise ValueError("Unexpected receiver notification consumer count")
            birth = self.birth_sequence.get(req_id)
            if birth is None:
                if self.applied_sequence >= fence:
                    raise RuntimeError(
                        "Unknown producer notification lacks publication fence"
                    )
                pending.append((req_id, tp_size, fence))
                continue
            if birth > fence:
                raise RuntimeError("Stale notification predates producer lifecycle")
            w._reqs_to_process.discard(req_id)
            w._reqs_to_send.pop(req_id, None)
            self._retire_producer(req_id)
            sent.add(req_id)
        if len(pending) > self.max_controls:
            raise RuntimeError("Unmatched notification capacity exhausted")
        self.pending_notifications = pending
        for req_id, expiry in list(w._reqs_to_send.items()):
            if now >= expiry:
                w._reqs_to_process.discard(req_id)
                del w._reqs_to_send[req_id]
                self._retire_producer(req_id)
                w.xfer_stats.record_kv_expired_req()
                sent.add(req_id)

        if w._engine_ttl > 0:
            for engine_id, last_active in list(w._engine_last_active.items()):
                if (
                    engine_id not in pinned_engines
                    and now - last_active > w._engine_ttl
                ):
                    w._cleanup_remote_engine(engine_id)
        stats = None
        if now >= self.next_stats and not w.xfer_stats.is_empty():
            stats = w.xfer_stats.clone_and_reset()
            self.next_stats = now + 1.0
        if sent or stats is not None:
            yield (frozenset(sent), stats)

    def retire(self, job):
        pass

    def shutdown(self):
        # Executor fetches own no native resources. Its late results are discarded.
        self.worker._handshake_initiation_executor.shutdown(
            wait=True, cancel_futures=True
        )
        self.handshakes.clear()
        NixlBaseConnectorWorker.shutdown(self.worker)
        # Destroy the native agent on its owner, after all descriptors are gone.
        del self.worker.nixl_wrapper
