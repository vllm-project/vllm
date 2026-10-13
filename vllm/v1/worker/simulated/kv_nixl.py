# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disaggregated prefill (NIXL) under simulated execution.

The connectors here are the real pull- and push-mode NIXL connectors with a
simulated worker side, which moves no data and needs no NIXL: the scheduler
side, leases and kv_transfer_params are the real ones. Engines in one process
(`vllm.v1.worker.simulated.disaggregated`) find each other's workers by engine
id; served instances book transfers over ZMQ at ports derived from the
prefill instance's side-channel port.
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
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector as RealNixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPushConnector as RealNixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
    get_representative_spec_type,
    kv_lease_extension,
    side_channel_port,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    group_kv_bytes_per_block,
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
    from vllm.v1.worker.simulated.kv_transfer import TransferRate

logger = init_logger(__name__)

# Simulated NIXL workers of the engines in this process, by engine id.
_workers: "weakref.WeakValueDictionary[str, SimulatedNixlWorker]" = (
    weakref.WeakValueDictionary()
)


def simulated_nixl_worker(engine_id: str) -> "SimulatedNixlWorker":
    """The simulated NIXL worker of an engine in this process."""
    return _workers[engine_id]


_BOOKING_TIMEOUT_MS = 10_000
_BOOKING_PORT_OFFSET = 10_000
# Larger than the side-channel port gaps between DP ranks and P/D instances.
_BOOKING_PORT_STRIDE = 1000


def _rank_port(side_channel_port: int, rank: int) -> int:
    """The booking port of served rank `pp_rank * tp_size + tp_rank`."""
    return side_channel_port + _BOOKING_PORT_OFFSET + _BOOKING_PORT_STRIDE * rank


def _warn_layout(req_id: str, prefill_tp: int, decode_tp: int) -> None:
    logger.warning(
        "Failing %s: prefill tensor parallelism %d, decode %d.",
        req_id,
        prefill_tp,
        decode_tp,
    )


def _is_finite(*values: Any) -> bool:
    return all(
        isinstance(v, int | float) and math.isfinite(v) and v >= 0 for v in values
    )


@dataclass(order=True)
class _Event:
    time: float
    req_id: str = field(compare=False)


class SimulatedNixlWorker:
    """The pull-mode NIXL worker of one engine, as a timing model.

    Each decode rank reads from the prefill ranks the real TP mapping assigns
    it; a request's reads run concurrently, and reads of different requests
    queue on each rank's link. A prefill rank frees a request's blocks once
    all its readers have notified it. Leases never expire, since heartbeats
    run on the wall clock: a booking lost to an unresponsive prefill rank
    holds its blocks."""

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
        self.is_producer = vllm_config.kv_transfer_config.is_kv_producer
        if self.pp_size > 1 and not (self.supports_pp and self.is_producer):
            raise NotImplementedError(
                "Simulated disaggregated prefill supports pipeline parallelism "
                "only on prefill instances of the NixlPushConnector."
            )
        self.use_mla = vllm_config.model_config.use_mla
        self.num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        self.block_size = vllm_config.cache_config.block_size
        self._group_spec_types = tuple(
            get_representative_spec_type(g.kv_cache_spec)
            for g in kv_cache_config.transfer_groups
        )
        self.side_channel_port = side_channel_port(vllm_config.parallel_config)
        # Per transfer group, in the order of ReqMeta.local_block_ids.
        groups = kv_cache_config.kv_cache_groups
        self.bytes_per_block = [
            group_kv_bytes_per_block(groups[group_id])
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
                self._after_reply()
        finally:
            sock.close(linger=0)

    def _after_reply(self) -> None:
        """Booking-thread work that must not delay a booking's answer."""

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
        """Prefill side: book a read of `req_id`'s KV on this rank's link,
        starting no earlier than `earliest`; returns when it ends. The blocks
        are freed once all `num_readers` decode ranks have read."""
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

    @property
    def _topology(self) -> TransferTopology:
        """The local layout, all the real connector's rank mapping needs."""
        return TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            is_mamba=False,
            total_num_kv_heads=1 if self.use_mla else self.num_kv_heads,
            attn_backends=[],
        )

    def _paired_ranks(self, remote_tp: int) -> list[int]:
        """The remote ranks the real connector handshakes with; none if the
        two tensor parallelisms do not divide each other."""
        if max(remote_tp, self.tp_size) % min(remote_tp, self.tp_size):
            return []
        return self._topology.handshake_target_ranks(remote_tp)

    def _mapping(self, remote_tp: int) -> TPMapping | None:
        """The real connector's rank mapping to a remote instance of
        `remote_tp` ranks; None for layouts it rejects."""
        if not self._paired_ranks(remote_tp):
            return None
        if not self.use_mla and self.tp_size < remote_tp > self.num_kv_heads:
            # Replicated remote heads (NixlBaseWorker's handshake check).
            return None
        return compute_tp_mapping(self._topology, remote_tp, self._group_spec_types)

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
            # Read from the source ranks and notify the other paired ones; a
            # rejected layout fails the request and only notifies.
            mapping = self._mapping(meta.tp_size)
            failed = mapping is None
            if mapping is None:
                _warn_layout(req_id, meta.tp_size, self.tp_size)
            sources = mapping.all_source_ranks if mapping else ()
            plan = [(r, r in sources) for r in self._paired_ranks(meta.tp_size)]
            readers = mapping.local_consumers if mapping else 1
            share = num_bytes // max(1, sum(reads for _, reads in plan))
            producer = _workers.get(remote.engine_id)
            if producer is None and not isinstance(self.clock, RealClock):
                raise ValueError(
                    f"No simulated engine {remote.engine_id} to read {req_id}'s "
                    "KV from."
                )
            if (
                (self.post.per_read_s or self.post.per_run_s)
                and num_bytes
                and not failed
            ):
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
                        remote.request_id, read_bytes, start, readers
                    )
                else:
                    duration = self._book_remote_read(
                        remote.host,
                        remote.port,
                        rank,
                        remote.request_id,
                        read_bytes,
                        start - now,
                        readers,
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
                    self._notified.pop(req_id, None)
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
            self._server = None
        for sock in self._peers.values():
            sock.close(linger=0)
        self._peers.clear()
        if self._zmq is not None:
            self._zmq.term()
            self._zmq = None


@dataclass
class _Incoming:
    """A decode rank's push-mode request: the writes it waits for (unknown
    until its own scheduler step, as writes may arrive first), those
    notified, and when the last ends."""

    expected: int | None = None
    count: int = 0
    end: float = 0.0


class SimulatedNixlPushWorker(SimulatedNixlWorker):
    """The push-mode NIXL worker of one served engine, as a timing model.

    Decode ranks register each scheduled request with their paired prefill
    ranks in every pipeline stage, and forward heartbeats. A prefill rank
    writes its share of a request's KV once the request is both finished and
    registered; the decode rank has it after one write per prefill stage and
    source rank. A request never registered (a full prefix hit) is freed when
    its lease expires."""

    supports_pp = True

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.lease_extension_s = kv_lease_extension(vllm_config.kv_transfer_config)
        # Decode side, by request id.
        self._incoming: dict[str, _Incoming] = {}
        # Prefill side, by base request id: finished requests and
        # registrations awaiting each other, and how many more copies of a
        # matched registration other decode ranks still send. By request id:
        # when its writes end, and the lease of an unregistered one.
        self._finished: dict[str, tuple[str, list[int]]] = {}
        self._registrations: dict[str, dict[str, Any]] = {}
        self._registration_copies: dict[str, int] = {}
        self._late_registrations: dict[str, int] = {}
        # Write notifications a registration started, sent once it is answered
        # (booking thread only).
        self._unsent_writes: list[tuple[str, int, int, str, float]] = []
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
                if base in self._registrations:
                    # Another decode rank's copy, before the request finished.
                    self._registration_copies[base] += 1
                    return True
                self._registrations[base] = reg
                self._registration_copies[base] = 1
                self._unsent_writes += self._match(base)
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

    def _after_reply(self) -> None:
        writes, self._unsent_writes = self._unsent_writes, []
        self._notify_writes(writes)

    def _layout_supported(self, prefill_tp: int) -> bool:
        """Whether every decode rank gets a write: no replicated decode heads."""
        if not self._paired_ranks(prefill_tp):
            return False
        return self.use_mla or self.tp_size <= max(prefill_tp, self.num_kv_heads)

    def _write_plan(self, decode_tp: int) -> list[tuple[int, float]]:
        """The decode ranks this prefill rank writes to, each with its share of
        the bytes, as the real push worker maps them; MLA's replicated latent
        goes to every paired decode rank."""
        mapping = self._mapping(decode_tp)
        assert mapping is not None
        if self.use_mla and self.tp_size < decode_tp:
            return [(r, 1.0) for r in self._paired_ranks(decode_tp)]
        ranks = mapping.all_source_ranks
        return [(r, 1.0 if self.use_mla else 1.0 / len(ranks)) for r in ranks]

    def _match(self, base: str) -> list[tuple[str, int, int, str, float]]:
        """Under the lock: start the writes of a request that is both finished
        and registered; returns the notifications to send."""
        if base not in self._finished or base not in self._registrations:
            return []
        assert self.clock is not None and self.rate is not None
        req_id, blocks = self._finished.pop(base)
        reg = self._registrations.pop(base)
        self._leases.pop(req_id, None)
        # Copies of the registration from the other decode ranks this rank
        # writes to that are still to come.
        late = max(1, reg["tp_size"] // self.tp_size) - self._registration_copies.pop(
            base, 1
        )
        if late > 0:
            self._late_registrations[base] = late
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
                _warn_layout(req_id, meta.tp_size, self.tp_size)
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
                for tp_rank in self._paired_ranks(heartbeat.tp_size):
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
            for tp_rank in self._paired_ranks(prefill_tp):
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


class NixlConnector(RealNixlPullConnector):
    """The pull-mode NixlConnector with a simulated worker side."""

    worker_cls = SimulatedNixlWorker  # type: ignore[assignment]

    def set_xfer_handshake_metadata_pp_aware(self, metadata: dict) -> None:
        """No handshake: simulated workers find or book each other directly."""


class NixlPushConnector(RealNixlPushConnector):
    """The NixlPushConnector with a simulated worker side."""

    worker_cls = SimulatedNixlPushWorker  # type: ignore[assignment]

    def set_xfer_handshake_metadata_pp_aware(self, metadata: dict) -> None:
        """No handshake: simulated ranks book transfers at each other's ports."""


NixlPullConnector = NixlConnector

# The connector names (vllm.distributed.kv_transfer.kv_connector.factory)
# whose worker side this module simulates.
SIMULATED_CONNECTORS = ("NixlConnector", "NixlPullConnector", "NixlPushConnector")
