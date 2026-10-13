# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disaggregated prefill (NIXL, pull mode) under simulated execution.

Simulated execution loads this module's `NixlConnector` in place of the real
one. Its scheduler side is the real `NixlPullConnectorScheduler`: the prefill
instance holds a finished request's blocks under a lease, the decode instance
waits in WAITING_FOR_REMOTE_KVS, and the kv_transfer_params between them are
the real ones. Its worker side, `SimulatedNixlWorker`, moves no data and
needs no NIXL. Workers of engines in one process (see
`vllm.v1.worker.simulated.disaggregated`) find each other by engine id; served
instances (`vllm serve`, the real clock, tensor parallelism) reach each
other over ZMQ at the prefill instance's side-channel address, which the
real kv_transfer_params carry: a decode rank asks the prefill ranks the
real connector maps it to, whatever the two tensor parallelisms, to book
its reads.
"""

import heapq
import math
import threading
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import zmq

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    _group_kv_bytes_per_block,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.simulated.timing import Clock, RealClock

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.worker.simulated.kv_transfer import TransferRate

logger = init_logger(__name__)

# Simulated NIXL workers of the engines in this process, by engine id.
_workers: "weakref.WeakValueDictionary[str, SimulatedNixlWorker]" = (
    weakref.WeakValueDictionary()
)


def simulated_nixl_worker(engine_id: str) -> "SimulatedNixlWorker":
    """The simulated NIXL worker of an engine in this process."""
    return _workers[engine_id]


# How long a decode rank waits for a served prefill rank to book its read.
_BOOKING_TIMEOUT_MS = 10_000


def _rank_port(side_channel_port: int, tp_rank: int) -> int:
    """Where a served prefill rank books reads. A stride of 100 keeps the
    ports of data-parallel instances, which are consecutive, apart."""
    return side_channel_port + 100 * (1 + tp_rank)


@dataclass(order=True)
class _Event:
    time: float
    req_id: str = field(compare=False)


class SimulatedNixlWorker:
    """The NIXL worker of one engine, as a timing model.

    A decode instance reads a request's KV from its prefill instance when the
    scheduler hands it over, from the prefill ranks the real connector maps
    each decode rank to. A request's reads run concurrently, each taking
    `TransferRate.seconds()` of its bytes; reads of different requests queue
    on each rank's link. When they complete, the request is reported in
    finished_recving, and each prefill rank, once all the decode ranks
    reading it have notified it, reports the request in finished_sending and
    so frees its blocks. Leases never expire: a simulated decode instance
    always reads or releases the blocks, which is what the real one's
    heartbeats keep the lease alive for (the scheduler sends them on the
    wall clock, which would make a simulation depend on the host's speed).
    A booking lost to a prefill instance that stopped answering therefore
    holds that rank's blocks."""

    # Attributes the NixlConnector reads off its worker.
    pcp_rank = 0
    pcp_dcp_sharded = False
    use_host_buffer = False
    copy_blocks = None
    xfer_handshake_metadata = None

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
    ):
        assert vllm_config.kv_transfer_config is not None
        self.tp_rank, self.tp_size = 0, 1
        if model_parallel_is_initialized():
            if get_pp_group().world_size > 1:
                raise NotImplementedError(
                    "Simulated disaggregated prefill supports no pipeline parallelism."
                )
            self.tp_rank = get_tp_group().rank_in_group
            self.tp_size = get_tp_group().world_size
        self.engine_id = engine_id
        self.is_producer = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        self.use_mla = vllm_config.model_config.use_mla
        self.num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        # Per transfer group, in the order of ReqMeta.local_block_ids.
        groups = kv_cache_config.kv_cache_groups
        self.bytes_per_block = [
            _group_kv_bytes_per_block(groups[group_id])
            for group_id in kv_cache_config.transfer_group_ids
        ]
        self.clock: Clock | None = None
        self.rate: TransferRate | None = None
        self.link_free_at = 0.0
        # Decode side: reads in flight, and reads that could not be booked.
        self._recving: list[_Event] = []
        self._failed: set[str] = set()
        # Prefill side: read notifications, and the requests whose blocks may
        # still be read.
        self._notifs: list[_Event] = []
        self._reqs_to_process: set[str] = set()
        # Notifications a request's blocks wait for, and those received.
        self._expected: dict[str, int] = {}
        self._notified: dict[str, int] = {}
        # When the link was free for each request being read, so that the
        # reads of several decode ranks run concurrently.
        self._read_floor: dict[str, float] = {}
        # A served prefill rank books reads on a thread: guards its link and
        # notifications.
        self._lock = threading.Lock()
        self._zmq: zmq.Context | None = None
        self._peers: dict[tuple[str, int], zmq.Socket] = {}
        self._server: threading.Thread | None = None
        self._stop = threading.Event()
        if engine_id in _workers:
            raise ValueError(f"Two simulated engines share engine_id {engine_id}.")
        _workers[engine_id] = self

    def configure(self, clock: Clock, rate: "TransferRate") -> None:
        self.clock = clock
        self.rate = rate
        if self.is_producer and isinstance(clock, RealClock):
            # A served prefill instance: decode instances book reads over ZMQ.
            # Bind here, so that a taken port fails startup.
            port = _rank_port(self.side_channel_port, self.tp_rank)
            sock = self._context().socket(zmq.REP)
            sock.bind(make_zmq_path("tcp", envs.VLLM_NIXL_SIDE_CHANNEL_HOST, port))
            self._server = threading.Thread(
                target=self._serve, args=(sock,), daemon=True, name="sim-nixl-bookings"
            )
            self._server.start()
            logger.info(
                "Simulated NIXL rank %d books reads on port %d", self.tp_rank, port
            )

    def _context(self) -> zmq.Context:
        if self._zmq is None:
            self._zmq = zmq.Context()
        return self._zmq

    def _serve(self, sock: zmq.Socket) -> None:
        try:
            while not self._stop.is_set():
                if not sock.poll(100):
                    continue
                reply = None
                try:
                    req_id, num_bytes, delay_s, num_readers = sock.recv_json()
                    if not (
                        isinstance(req_id, str)
                        and isinstance(num_readers, int)
                        and num_readers >= 1
                        and all(
                            isinstance(v, int | float) and math.isfinite(v) and v >= 0
                            for v in (num_bytes, delay_s)
                        )
                    ):
                        raise ValueError(f"bad booking {req_id!r}")
                    assert self.clock is not None
                    now = self.clock.now()
                    end = self.book_read(
                        req_id, int(num_bytes), now + delay_s, num_readers
                    )
                    reply = end - now
                except Exception:
                    logger.warning("Rejected a KV read booking", exc_info=True)
                # A REP socket must answer before it can receive again.
                sock.send_json(reply)
        finally:
            sock.close(linger=0)

    def book_read(
        self, req_id: str, num_bytes: int, earliest: float, num_readers: int = 1
    ) -> float:
        """Prefill side: book a read of `num_bytes` of `req_id`'s KV on this
        rank's link, starting no earlier than `earliest`; it notifies this
        rank when it ends, which is returned. The request's blocks are freed
        once all `num_readers` decode ranks reading this rank notified it.
        Reads of one request by several decode ranks run concurrently (each at
        the measured per-read rate); those of different requests one after
        another."""
        assert self.rate is not None
        with self._lock:
            floor = self._read_floor.setdefault(req_id, self.link_free_at)
            end = earliest
            if num_bytes:
                end = max(earliest, floor) + self.rate.seconds(num_bytes)
                self.link_free_at = max(self.link_free_at, end)
            self._expected[req_id] = num_readers
            heapq.heappush(self._notifs, _Event(end, req_id))
        return end

    def _read_plan(self, remote_tp: int) -> list[tuple[int, bool]] | None:
        """The prefill ranks this decode rank reads a request's KV from, as the
        real pull connector maps tensor-parallel ranks, each with whether it
        is read or only notified; None for layouts the real connector
        rejects. Reads split this rank's KV evenly. (The real mapping,
        `compute_tp_mapping`, needs the handshake's attention backends and
        block layout, which simulated instances do not exchange.)"""
        local, d = self.tp_size, self.tp_rank
        if max(remote_tp, local) % min(remote_tp, local):
            return None
        if local >= remote_tp:
            return [(d * remote_tp // local, True)]
        group = range(d * (remote_tp // local), (d + 1) * (remote_tp // local))
        if self.use_mla:
            # One copy of the replicated latent; the other holders are notified.
            return [(r, r == group[0]) for r in group]
        if remote_tp > self.num_kv_heads:
            return None
        return [(r, True) for r in group]

    def _notified_ranks(self, remote_tp: int) -> list[int]:
        """The prefill ranks this decode rank notifies of a request's reads:
        those the real mapping pairs it with, which for every layout covers
        each prefill rank."""
        local, d = self.tp_size, self.tp_rank
        if local >= remote_tp:
            return [d * remote_tp // local]
        return [r for r in range(remote_tp) if r * local // remote_tp == d]

    def _num_readers(self, remote_tp: int, rank: int) -> int:
        """How many decode ranks notify prefill rank `rank` of a request."""
        local = self.tp_size
        if local >= remote_tp:
            return sum(1 for d in range(local) if d * remote_tp // local == rank)
        return 1

    def _book_remote_read(
        self,
        host: str,
        port: int,
        rank: int,
        req_id: str,
        num_bytes: int,
        delay_s: float,
        num_readers: int,
    ) -> float | None:
        """Book a read at a prefill rank in another process; returns how long
        from now it ends, or None if the rank does not book it in time."""
        address = (host, _rank_port(port, rank))
        sock = self._peers.pop(address, None)
        if sock is None:
            sock = self._context().socket(zmq.REQ)
            sock.connect(make_zmq_path("tcp", *address))
        reply = None
        try:
            sock.send_json([req_id, num_bytes, delay_s, num_readers])
            if sock.poll(_BOOKING_TIMEOUT_MS):
                reply = sock.recv_json()
                self._peers[address] = sock
        except zmq.ZMQError:
            logger.warning("Booking a KV read at %s:%d failed", *address, exc_info=True)
        if address not in self._peers:
            # A REQ socket that got no answer cannot send again.
            sock.close(linger=0)
        return float(reply) if isinstance(reply, int | float) else None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        pass

    def set_host_xfer_buffer_ops(self, copy_operation: Any) -> None:
        pass

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        assert self.clock is not None and self.rate is not None
        now = self.clock.now()
        for req_id, meta in metadata.reqs_to_recv.items():
            remote = meta.remote
            assert remote is not None
            # A full local prefix hit, or an aborted request, only notifies.
            awaited = meta.awaiting_kvs or any(meta.local_block_ids)
            num_bytes = sum(
                len(ids) * size
                for ids, size in zip(meta.local_block_ids, self.bytes_per_block)
            )
            plan = self._read_plan(meta.tp_size)
            failed = plan is None
            if plan is None:
                # Layouts the real connector rejects: notify the prefill ranks
                # this rank stands for, so they free the blocks, and fail the
                # request, which is then prefilled here.
                logger.warning(
                    "Failing %s: prefill tensor parallelism %d, decode %d.",
                    req_id,
                    meta.tp_size,
                    self.tp_size,
                )
                plan = [(r, False) for r in self._notified_ranks(meta.tp_size)]
            share = num_bytes // max(1, sum(reads for _, reads in plan))
            producer = _workers.get(remote.engine_id)
            if producer is None and not isinstance(self.clock, RealClock):
                raise ValueError(
                    f"No simulated engine {remote.engine_id} to read {req_id}'s "
                    "KV from."
                )
            # A request's reads from several prefill ranks run concurrently;
            # those of different requests one after another on this link.
            earliest = max(now, self.link_free_at)
            end = now
            for rank, reads in plan:
                read_bytes = share if reads and not failed else 0
                start = earliest if read_bytes else now
                if producer is not None:
                    assert rank == 0, "in-process engines have one rank"
                    read_end = producer.book_read(
                        remote.request_id,
                        read_bytes,
                        start,
                        self._num_readers(meta.tp_size, rank),
                    )
                else:
                    duration = self._book_remote_read(
                        remote.host,
                        remote.port,
                        rank,
                        remote.request_id,
                        read_bytes,
                        start - now,
                        self._num_readers(meta.tp_size, rank),
                    )
                    if duration is None:
                        failed = True
                        continue
                    read_end = now + duration
                if read_bytes:
                    self.link_free_at = max(self.link_free_at, read_end)
                end = max(end, read_end)
            if failed:
                if awaited:
                    self._failed.add(req_id)
                continue
            if awaited:
                heapq.heappush(self._recving, _Event(end, req_id))

        self._reqs_to_process.update(metadata.reqs_in_batch)
        self._reqs_to_process.difference_update(metadata.reqs_not_processed)

    def get_transfer_results(self) -> KVConnectorTransferResults:
        assert self.clock is not None
        now = self.clock.now()
        done_recving = set()
        while self._recving and self._recving[0].time <= now:
            done_recving.add(heapq.heappop(self._recving).req_id)
        # As the real worker, report failures as finished too.
        failed, self._failed = self._failed, set()
        done_recving |= failed
        done_sending = set()
        with self._lock:
            while self._notifs and self._notifs[0].time <= now:
                req_id = heapq.heappop(self._notifs).req_id
                # As the real worker, drop reads of requests no longer tracked.
                if req_id not in self._reqs_to_process:
                    self._expected.pop(req_id, None)
                    self._read_floor.pop(req_id, None)
                    continue
                self._notified[req_id] = self._notified.get(req_id, 0) + 1
                if self._notified[req_id] >= self._expected.get(req_id, 1):
                    self._reqs_to_process.remove(req_id)
                    del self._notified[req_id]
                    self._expected.pop(req_id, None)
                    self._read_floor.pop(req_id, None)
                    done_sending.add(req_id)
        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed,
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        results = self.get_transfer_results()
        return results.finished_sending, results.finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def get_kv_connector_stats(self) -> None:
        return None

    def next_completion(self) -> float | None:
        with self._lock:
            times = [q[0].time for q in (self._recving, self._notifs) if q]
        return min(times, default=None)

    def shutdown(self) -> None:
        if _workers.get(self.engine_id) is self:
            del _workers[self.engine_id]
        self._stop.set()
        if self._server is not None:
            self._server.join()
        for sock in self._peers.values():
            sock.close(linger=0)
        if self._zmq is not None:
            self._zmq.term()


class NixlConnector(NixlPullConnector):
    """The pull-mode NixlConnector with a simulated worker side. It keeps the
    real connector's name, which configuration checks look for."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlPullConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        else:
            self.worker = SimulatedNixlWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = self.worker  # type: ignore[assignment]

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.worker.start_load_kv(self._connector_metadata)

    def set_xfer_handshake_metadata_pp_aware(self, metadata: dict) -> None:
        """No handshake: simulated workers find each other in process, or
        book reads at the prefill ranks' own ports."""
