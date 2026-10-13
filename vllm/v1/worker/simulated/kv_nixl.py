# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disaggregated prefill (NIXL) under simulated execution.

Simulated execution loads this module's `NixlConnector` (pull mode) and
`NixlPushConnector` in place of the real ones; push mode, for served
instances, also lets a prefill instance use pipeline parallelism (see
`SimulatedNixlPushWorker`). The rest describes pull mode.

Its scheduler side is the real `NixlPullConnectorScheduler`: the prefill
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
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPushConnector as RealNixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
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
from vllm.v1.worker.simulated.kv_transfer import PostCost, read_runs
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


def _rank_port(side_channel_port: int, rank: int) -> int:
    """Where a served rank (pp_rank * tp_size + tp_rank) takes bookings. The
    stride of 1000 keeps apart the ports of instances whose side-channel
    ports differ by less, such as data-parallel instances (consecutive) and
    the usual prefill/decode pairs (100 apart)."""
    return side_channel_port + 10_000 + 1000 * rank


def _is_finite(*values: Any) -> bool:
    return all(
        isinstance(v, int | float) and math.isfinite(v) and v >= 0 for v in values
    )


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

    # Whether a prefill instance may use pipeline parallelism.
    supports_pp = False

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
        self.pp_rank, self.pp_size = 0, 1
        if model_parallel_is_initialized():
            self.tp_rank = get_tp_group().rank_in_group
            self.tp_size = get_tp_group().world_size
            self.pp_rank = get_pp_group().rank_in_group
            self.pp_size = get_pp_group().world_size
        self.engine_id = engine_id
        self.is_producer = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        if self.pp_size > 1 and not (self.supports_pp and self.is_producer):
            raise NotImplementedError(
                "Simulated disaggregated prefill supports pipeline parallelism "
                "only on prefill instances of the NixlPushConnector."
            )
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
        self.post = PostCost()
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
        # Bookings may be sent from the worker and the booking thread, one at
        # a time per peer.
        self._call_lock = threading.Lock()
        self._peer_locks: dict[tuple[str, int], threading.Lock] = {}
        self._server: threading.Thread | None = None
        self._stop = threading.Event()
        if engine_id in _workers:
            raise ValueError(f"Two simulated engines share engine_id {engine_id}.")
        _workers[engine_id] = self

    def configure(
        self, clock: Clock, rate: "TransferRate", post: PostCost | None = None
    ) -> None:
        self.clock = clock
        self.rate = rate
        self.post = post or PostCost()
        if self._serves() and isinstance(clock, RealClock):
            # A served instance: peers book transfers over ZMQ. Bind here, so
            # that a taken port fails startup.
            port = _rank_port(
                self.side_channel_port, self.pp_rank * self.tp_size + self.tp_rank
            )
            if port > 65535:
                raise ValueError(
                    f"Simulated NIXL booking port {port} is out of range; lower "
                    "VLLM_NIXL_SIDE_CHANNEL_PORT."
                )
            sock = self._context().socket(zmq.REP)
            sock.bind(make_zmq_path("tcp", envs.VLLM_NIXL_SIDE_CHANNEL_HOST, port))
            self._server = threading.Thread(
                target=self._serve, args=(sock,), daemon=True, name="sim-nixl-bookings"
            )
            self._server.start()
            logger.info(
                "Simulated NIXL rank (pp %d, tp %d) takes bookings on port %d",
                self.pp_rank,
                self.tp_rank,
                port,
            )

    def _serves(self) -> bool:
        """Whether peers in other processes book transfers with this rank."""
        return self.is_producer

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
                    reply = self._handle(sock.recv_json())
                except Exception:
                    logger.warning("Rejected a KV transfer booking", exc_info=True)
                # A REP socket must answer before it can receive again.
                sock.send_json(reply)
        finally:
            sock.close(linger=0)

    def _handle(self, message: Any) -> Any:
        """Answer a booking from a peer in another process."""
        op, req_id, num_bytes, delay_s, num_readers = message
        if not (
            op == "read"
            and isinstance(req_id, str)
            and isinstance(num_readers, int)
            and num_readers >= 1
            and _is_finite(num_bytes, delay_s)
        ):
            raise ValueError(f"bad booking {message!r}")
        assert self.clock is not None
        now = self.clock.now()
        return self.book_read(req_id, int(num_bytes), now + delay_s, num_readers) - now

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
        answered, reply = self._call(
            host, port, rank, ["read", req_id, num_bytes, delay_s, num_readers]
        )
        return float(reply) if answered and isinstance(reply, int | float) else None

    def _call(
        self, host: str, port: int, rank: int, message: list[Any]
    ) -> tuple[bool, Any]:
        """Send a booking to a rank in another process; returns whether it
        answered in time, and its answer."""
        address = (host, _rank_port(port, rank))
        with self._call_lock:
            peer_lock = self._peer_locks.setdefault(address, threading.Lock())
        # One peer at a time per address: a slow peer delays no other.
        with peer_lock:
            sock = self._peers.pop(address, None)
            if sock is None:
                sock = self._context().socket(zmq.REQ)
                sock.connect(make_zmq_path("tcp", *address))
            answered, reply = False, None
            try:
                sock.send_json(message)
                if sock.poll(_BOOKING_TIMEOUT_MS):
                    answered, reply = True, sock.recv_json()
                    self._peers[address] = sock
            except Exception:
                answered = False
                logger.warning(
                    "A KV transfer booking at %s:%d failed", *address, exc_info=True
                )
            if not answered:
                # A REQ socket that got no answer cannot send again.
                sock.close(linger=0)
        return answered, reply

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
            if self.post != PostCost() and num_bytes and not failed:
                # Posting the reads holds the engine thread; they start once
                # posted.
                runs = read_runs(meta.local_block_ids, remote.block_ids)
                self.clock.sleep_until(now + self.post.seconds(1, runs))
                now = self.clock.now()
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


@dataclass
class _Incoming:
    """A decode rank's push-mode request: the writes it waits for (unknown
    until its own scheduler step, as writes may arrive first), those
    notified, and when the last ends."""

    expected: int | None = None
    count: int = 0
    end: float = 0.0


class SimulatedNixlPushWorker(SimulatedNixlWorker):
    """The NIXL worker of one served engine in push mode, as a timing model.

    A decode instance registers each request with its prefill instance as
    soon as it is scheduled, from every decode rank to the prefill ranks the
    real connector pairs it with, in every pipeline stage; it also forwards
    its scheduler's heartbeats, which extend the prefill instance's leases. A
    prefill rank writes a request's KV once it has both finished the request
    and received the registration: its own stage's layers and heads, split
    over the decode ranks it maps to, concurrently, after the writes of
    earlier requests on its link. Each write notifies its decode rank when it
    ends; the decode rank reports the request in finished_recving after one
    write per prefill stage and source rank, and the prefill rank reports it
    in finished_sending when its own writes end, or when the lease of a
    request that was never registered (a full prefix hit on the decode
    instance) expires. Layouts the real connector rejects fail the request
    on the decode instance."""

    supports_pp = True

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        lease_s = vllm_config.kv_transfer_config.get_from_extra_config(
            "kv_lease_duration", 30
        )
        self.lease_extension_s = lease_s * 2 // 3
        # Decode side, by request id.
        self._incoming: dict[str, _Incoming] = {}
        # Prefill side, by base request id: finished requests and
        # registrations awaiting each other, and how many more copies of a
        # matched registration other decode ranks still send. By request id:
        # when its writes end, and the lease of an unregistered one.
        self._finished: dict[str, tuple[str, list[int]]] = {}
        self._registrations: dict[str, dict[str, Any]] = {}
        self._late_registrations: dict[str, int] = {}
        self._writes_out: dict[str, float] = {}
        self._leases: dict[str, float] = {}

    def configure(
        self, clock: Clock, rate: "TransferRate", post: PostCost | None = None
    ) -> None:
        if not isinstance(clock, RealClock):
            raise NotImplementedError(
                "The simulated NixlPushConnector needs served instances."
            )
        # A decode rank posts no reads in push mode.
        super().configure(clock, rate)

    def _serves(self) -> bool:
        return True

    def _handle(self, message: Any) -> Any:
        assert self.clock is not None
        op = message[0]
        if op == "reg":
            _, reg = message
            if not (
                isinstance(reg["request_id"], str)
                and isinstance(reg["host"], str)
                and isinstance(reg["port"], int)
                and isinstance(reg["tp_size"], int)
                and reg["tp_size"] >= 1
                and isinstance(reg["blocks"], list)
                and all(isinstance(n, int) and n >= 0 for n in reg["blocks"])
            ):
                raise ValueError(f"bad registration {reg!r}")
            base = get_base_request_id(reg["request_id"])
            with self._lock:
                late = self._late_registrations.get(base)
                if late is not None:
                    # Another decode rank's copy of a matched registration.
                    if late <= 1:
                        del self._late_registrations[base]
                    else:
                        self._late_registrations[base] = late - 1
                    return True
                self._registrations[base] = reg
                writes = self._match(base)
            self._notify_writes(writes)
            return True
        if op == "write":
            _, req_id, duration = message
            if not (isinstance(req_id, str) and _is_finite(duration)):
                raise ValueError(f"bad write notification {message!r}")
            with self._lock:
                incoming = self._incoming.setdefault(req_id, _Incoming())
                incoming.count += 1
                incoming.end = max(incoming.end, self.clock.now() + duration)
            return True
        if op == "heartbeat":
            _, req_ids = message
            if not (
                isinstance(req_ids, list) and all(isinstance(r, str) for r in req_ids)
            ):
                raise ValueError(f"bad heartbeat {message!r}")
            bases = {get_base_request_id(r) for r in req_ids}
            deadline = self.clock.now() + self.lease_extension_s
            with self._lock:
                for req_id, lease in self._leases.items():
                    if get_base_request_id(req_id) in bases:
                        self._leases[req_id] = max(lease, deadline)
            return True
        raise ValueError(f"unknown booking {op!r}")

    def _layout_supported(self, prefill_tp: int) -> bool:
        """Whether the real push connector accepts these tensor-parallel
        layouts (decode ranks with replicated heads would get no write)."""
        decode_tp = self.tp_size
        if max(prefill_tp, decode_tp) % min(prefill_tp, decode_tp):
            return False
        return self.use_mla or decode_tp <= max(prefill_tp, self.num_kv_heads)

    def _write_plan(self, decode_tp: int) -> list[tuple[int, float]]:
        """The decode ranks this prefill rank writes a request's KV to, each
        with its share of this rank's bytes, as the real push connector maps
        ranks (decode instances reject other layouts)."""
        local, p = self.tp_size, self.tp_rank
        if local >= decode_tp:
            return [(p * decode_tp // local, 1.0)]
        ratio = decode_tp // local
        share = 1.0 if self.use_mla else 1.0 / ratio
        return [(p * ratio + i, share) for i in range(ratio)]

    def _match(self, base: str) -> list[tuple[str, int, int, str, float]]:
        """Under the lock: start the writes of a request that is both finished
        and registered; returns the notifications to send."""
        if base not in self._finished or base not in self._registrations:
            return []
        assert self.clock is not None and self.rate is not None
        req_id, blocks = self._finished.pop(base)
        reg = self._registrations.pop(base)
        self._leases.pop(req_id, None)
        # Decode ranks beyond the first that register with this rank.
        if reg["tp_size"] > self.tp_size:
            self._late_registrations[base] = reg["tp_size"] // self.tp_size - 1
        num_bytes = sum(
            min(n, m) * size
            for n, m, size in zip(blocks, reg["blocks"], self.bytes_per_block)
        )
        now = self.clock.now()
        start = max(now, self.link_free_at)
        end = now
        writes = []
        for rank, share in self._write_plan(reg["tp_size"]):
            write_end = start + self.rate.seconds(int(num_bytes * share))
            end = max(end, write_end)
            writes.append(
                (reg["host"], reg["port"], rank, reg["request_id"], write_end - now)
            )
        self.link_free_at = max(self.link_free_at, end)
        self._writes_out[req_id] = end
        return writes

    def _notify_writes(self, writes: list[tuple[str, int, int, str, float]]) -> None:
        for host, port, rank, req_id, duration in writes:
            answered, ok = self._call(host, port, rank, ["write", req_id, duration])
            if not (answered and ok is True):
                logger.warning("Decode rank %d did not take write of %s", rank, req_id)

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        assert self.clock is not None
        now = self.clock.now()
        # Decode side: wait for one write per prefill stage and source rank.
        failed = set()
        for req_id, meta in metadata.reqs_to_recv.items():
            if not self._layout_supported(meta.tp_size):
                logger.warning(
                    "Failing %s: prefill tensor parallelism %d, decode %d.",
                    req_id,
                    meta.tp_size,
                    self.tp_size,
                )
                failed.add(req_id)
                continue
            expected = max(1, meta.pp_size) * max(1, meta.tp_size // self.tp_size)
            with self._lock:
                self._incoming.setdefault(req_id, _Incoming()).expected = expected
        for req_id, reg in metadata.push_registrations.items():
            if req_id in failed:
                continue
            message = [
                "reg",
                {
                    "request_id": req_id,
                    "host": reg["decode_host"],
                    "port": reg["decode_port"],
                    "tp_size": reg["decode_tp_size"],
                    "blocks": [len(ids) for ids in reg["local_block_ids"]],
                },
            ]
            prefill_tp = reg["remote_tp_size"]
            if not self._register(reg, prefill_tp, message):
                # As after a failed handshake: fail the request.
                failed.add(req_id)
        with self._lock:
            for req_id in failed:
                self._incoming.pop(req_id, None)
        self._failed |= failed
        for heartbeat in metadata.heartbeat_by_engine.values():
            message = ["heartbeat", sorted(heartbeat.req_ids)]
            for pp_rank in range(heartbeat.pp_size):
                for tp_rank in self._notified_ranks(heartbeat.tp_size):
                    rank = pp_rank * heartbeat.tp_size + tp_rank
                    self._call(heartbeat.host, heartbeat.port, rank, message)
        # Prefill side: finished requests write once registered.
        writes = []
        with self._lock:
            for req_id, deadline in metadata.reqs_to_send.items():
                if req_id not in self._writes_out:
                    # Deadlines are stamped with the scheduler's perf_counter.
                    self._leases[req_id] = now + deadline - metadata.scheduler_clock
            for req_id, block_ids in metadata.push_finished_blocks.items():
                base = get_base_request_id(req_id)
                self._finished[base] = (req_id, [len(ids) for ids in block_ids])
                writes += self._match(base)
        self._notify_writes(writes)

    def _register(
        self, reg: dict[str, Any], prefill_tp: int, message: list[Any]
    ) -> bool:
        """Register a request with every prefill stage's paired ranks."""
        for pp_rank in range(reg["remote_pp_size"]):
            for tp_rank in self._notified_ranks(prefill_tp):
                answered, ok = self._call(
                    reg["remote_host"],
                    reg["remote_port"],
                    pp_rank * prefill_tp + tp_rank,
                    message,
                )
                if not (answered and ok is True):
                    return False
        return True

    def get_transfer_results(self) -> KVConnectorTransferResults:
        assert self.clock is not None
        now = self.clock.now()
        failed, self._failed = self._failed, set()
        done_recving = set(failed)
        done_sending = set()
        with self._lock:
            for req_id, incoming in list(self._incoming.items()):
                if (
                    incoming.expected is not None
                    and incoming.count >= incoming.expected
                    and incoming.end <= now
                ):
                    done_recving.add(req_id)
                    del self._incoming[req_id]
            for req_id, end in list(self._writes_out.items()):
                if end <= now:
                    done_sending.add(req_id)
                    del self._writes_out[req_id]
            for req_id, deadline in list(self._leases.items()):
                if deadline <= now:
                    done_sending.add(req_id)
                    del self._leases[req_id]
                    self._finished.pop(get_base_request_id(req_id), None)
        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed,
        )

    def next_completion(self) -> float | None:
        with self._lock:
            times = [
                i.end
                for i in self._incoming.values()
                if i.expected is not None and i.count >= i.expected
            ]
            times += self._writes_out.values()
            times += self._leases.values()
        return min(times, default=None)


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


class NixlPushConnector(RealNixlPushConnector):
    """The NixlPushConnector with a simulated worker side, for served
    instances. It keeps the real connector's name, which configuration
    checks look for."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlPushConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        else:
            self.worker = SimulatedNixlPushWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = self.worker  # type: ignore[assignment]

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.worker.start_load_kv(self._connector_metadata)

    def set_xfer_handshake_metadata_pp_aware(self, metadata: dict) -> None:
        """No handshake: simulated ranks book transfers at each other's ports."""
