# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side of the ECCPUConnector.

Single class, two role branches:

- **Producer**: binds a ZMQ ROUTER, runs a listener thread that ingests
  `XferReq` messages, posts NIXL WRITEs against its NIXL-registered
  mmap region, and fires `XferAck`s back from the main thread after
  polling completion each step. Carries a two-phase
  `_pending_save` / `_encodings` book so the listener can only serve
  mm_hashes whose bytes have definitely landed in the mmap.
- **Consumer**: lazy DEALER pool keyed by `(peer_host, peer_port)` plus
  a local `_encoding_map` of outstanding transfers, a `_ready` set of
  arrived-but-not-yet-loaded mm_hashes, and a
  `_to_free_after_load` deferred-free list that closes the loop one
  step after the worker has copied the bytes out of the mmap.

On nodes configured with `ec_role="ec_both"` both branches run in the
same instance (the role strings live in `ECTransferConfig.ec_role`; the
SCHEDULER/WORKER split is a separate process-level enum).

Locking
-------
The producer runs two threads: the main scheduler thread and a ZMQ
listener thread (`_run_listener`). Two locks are involved:

- `self._lock` guards the three producer-side fields the listener
  touches: `_encodings`, `_in_flight`, and `_pending_nacks`. All other
  producer fields (`_pending_save`, `_pending_new_encodings`,
  `_remote_peers`) are accessed by exactly one thread and need no
  synchronization.
- `self._region` has its own internal lock for the free pool and per-
  block ref counts.

When both locks are held, the order is **scheduler lock first, region
lock second**. Three sites hold them together: `_handle_xfer_req`
(lookup + pin), `_sweep_completions` (lookup + unpin), and
`_alloc_with_lru_eviction` (try_free + `_encodings` mutation, in one
sweep). The eviction sweep must be atomic against the listener's
lookup-then-pin: if `try_free` succeeded but `_encodings` still mapped
the mm_hash, the listener could pin blocks we just put on the free
list. Holding the scheduler lock across the whole sweep closes that
window without changing lock order.
"""

import contextlib
import threading
import uuid
from math import ceil
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
from pybase64 import b64decode, b64encode

from vllm import envs
from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import (
    EC_CONNECTOR_VERSION,
    ECCPUConnectorMetadata,
    XferAck,
    XferReq,
    compute_ec_compatibility_hash,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    PeerEntry,
    build_block_descs,
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.version import __version__ as VLLM_VERSION

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# NIXL memory-type tag for host DRAM-backed regions.
_NIXL_DRAM = "DRAM"


class _RemotePeer:
    """Producer-side cache of per-consumer NIXL state.

    Populated on first `XferReq` from a given consumer. The remote xfer
    dlist must be prepared on the same NIXL agent that will issue the
    WRITE, so the producer scheduler owns this state. Accessed only
    from the listener thread; needs no synchronization.
    """

    __slots__ = ("agent_name", "metadata_bytes", "remote_xfer_handle")

    def __init__(
        self,
        agent_name: str,
        metadata_bytes: bytes,
        remote_xfer_handle: int,
    ) -> None:
        self.agent_name = agent_name
        self.metadata_bytes = metadata_bytes
        self.remote_xfer_handle = remote_xfer_handle


class ECCPUScheduler:
    """Scheduler delegate for the ECCPUConnector."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        self._vllm_config = vllm_config
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._ec_config = ec_config

        self._is_producer: bool = ec_config.is_ec_producer
        self._is_consumer: bool = ec_config.is_ec_consumer

        # ----- Shared (both roles) --------------------------------------
        hidden_dim = vllm_config.model_config.get_inputs_embeds_size()
        dtype = vllm_config.model_config.dtype
        self._element_size = torch.empty(0, dtype=dtype).element_size()
        self._hidden_dim = hidden_dim
        self._block_size_bytes = hidden_dim * self._element_size
        self._num_blocks = int(ec_config.get_from_extra_config("num_ec_blocks", 256))

        # Hard-require NIXL on both roles.
        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError(
                "ECCPUConnector requires NIXL; "
                "install the `nixl` package or set a different ec_connector."
            )

        # Mirror the KV NIXL side (worker.py:293): the vLLM-level engine_id
        # doubles as the NIXL agent name, so peers can identify this
        # instance from the UUID in their config without a second handshake.
        # `engine_id` is typed `str | None` on the config but
        # `ECTransferConfig.__post_init__` always populates it.
        assert ec_config.engine_id is not None
        self._engine_id: str = ec_config.engine_id
        self._nixl = NixlWrapper(
            self._engine_id,
            nixl_agent_config(num_threads=1),
        )
        self._agent_metadata: bytes = self._nixl.get_agent_metadata()

        # Same mmap file as the worker(s); O_CREAT|O_EXCL race handled
        # inside ECSharedRegion.
        self._region = ECSharedRegion(
            instance_id=self._engine_id,
            num_blocks=self._num_blocks,
            block_size_bytes=self._block_size_bytes,
        )

        # Register the mmap with NIXL for RDMA. `register_memory` takes
        # a list of `(addr, size, device_id)` reg descs; we register the
        # whole region as a single contiguous DRAM chunk.
        reg_descs = self._nixl.get_reg_descs(
            [(self._region.base_ptr, self._region.total_size_bytes, 0)],
            _NIXL_DRAM,
        )
        self._nixl.register_memory(reg_descs)
        self._registered_reg_descs = reg_descs

        # Per-block xfer descs: one descriptor per block so that a WRITE
        # to block indices `[i1, i2, ...]` becomes
        # `make_prepped_xfer(local_handle, [i1, i2], remote_handle, [r1, r2])`.
        self._block_descs_list = build_block_descs(
            self._region.base_ptr,
            self._num_blocks,
            self._block_size_bytes,
            device_id=0,
        )
        xfer_descs = self._nixl.get_xfer_descs(self._block_descs_list, _NIXL_DRAM)
        self._local_xfer_handle: int = self._nixl.prep_xfer_dlist(
            "NIXL_INIT_AGENT", xfer_descs
        )
        # Serialized form of our block descs, shipped to peers on first
        # contact so they can prepare a remote dlist addressing us.
        self._mem_descriptor_bytes: bytes = serialize_mem_descriptor(
            self._block_descs_list
        )

        self._compat_hash: str = compute_ec_compatibility_hash(
            vllm_version=VLLM_VERSION,
            model=str(vllm_config.model_config.model),
            dtype=str(dtype),
            block_size_bytes=self._block_size_bytes,
        )

        # ZMQ context shared by ROUTER + DEALERs.
        self._zmq_ctx = zmq.Context.instance()

        # ----- Producer-only --------------------------------------------
        # Main-thread-only state. No lock.
        self._pending_save: dict[str, list[int]] = {}
        # mm_hash -> size_bytes, populated from update_state_after_alloc
        # between scheduling passes; drained by build_connector_meta.
        self._pending_new_encodings: dict[str, int] = {}
        # Listener-thread-only state. No lock.
        self._remote_peers: dict[str, _RemotePeer] = {}

        # Lock-guarded state (shared between main and listener threads).
        # Fields below this comment are read/written under `self._lock`.
        # When this lock is held together with the region's internal
        # lock, this one is acquired first.
        self._lock = threading.Lock()
        self._encodings: dict[str, list[int]] = {}
        self._in_flight: dict[str, tuple[bytes, str, Any]] = {}
        self._pending_nacks: list[tuple[bytes, str]] = []

        self._router: zmq.Socket | None = None
        self._listener_t: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._peer_host = envs.VLLM_EC_SIDE_CHANNEL_HOST
        self._peer_port = envs.VLLM_EC_SIDE_CHANNEL_PORT

        if self._is_producer:
            self._router = make_zmq_socket(
                ctx=self._zmq_ctx,
                path=make_zmq_path(
                    scheme="tcp",
                    host=self._peer_host,
                    port=self._peer_port,
                ),
                socket_type=zmq.ROUTER,
                bind=True,
            )
            self._router.setsockopt(zmq.RCVTIMEO, 1000)
            self._listener_t = threading.Thread(
                target=self._run_listener,
                name="ec-nixl-listener",
                daemon=True,
            )
            self._listener_t.start()

        # ----- Consumer-only --------------------------------------------
        self._encoding_map: dict[str, list[int]] = {}
        self._ready: set[str] = set()
        self._to_free_after_load: list[str] = []
        self._peer_pool: dict[tuple[str, int], PeerEntry] = {}

        # Reusable msgspec codecs. Encoder is type-agnostic so one
        # instance handles both XferReq and XferAck sends.
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_req_decoder = msgspec.msgpack.Decoder(XferReq)
        self._xfer_ack_decoder = msgspec.msgpack.Decoder(XferAck)

    # ==========================================================================
    # Producer role
    # ==========================================================================

    def _run_listener(self) -> None:
        """Ingest XferReqs from the ROUTER, post NIXL WRITEs.

        The listener thread never writes to the ROUTER — all ACK / NACK
        sends happen on the main thread inside `_sweep_completions`
        (ZMQ sockets are not safe for concurrent send/recv from
        different threads).
        """
        assert self._router is not None
        while not self._stop_event.is_set():
            try:
                identity, _, payload = self._router.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ContextTerminated:
                return
            except Exception:
                logger.exception("ec: listener recv failed")
                continue
            try:
                req = self._xfer_req_decoder.decode(payload)
            except (msgspec.DecodeError, msgspec.ValidationError):
                logger.warning("ec: dropped malformed XferReq")
                continue
            self._handle_xfer_req(identity, req)

    def _handle_xfer_req(self, identity: bytes, req: XferReq) -> None:
        if req.connector_version != EC_CONNECTOR_VERSION:
            logger.warning(
                "ec: version mismatch req=%d local=%d, NACKing",
                req.connector_version,
                EC_CONNECTOR_VERSION,
            )
            with self._lock:
                self._pending_nacks.append((identity, req.mm_hash))
            return

        # `_compat_hash` is set in __init__ and immutable thereafter;
        # no lock needed.
        if req.compatibility_hash != self._compat_hash:
            with self._lock:
                self._pending_nacks.append((identity, req.mm_hash))
            return

        # Look up the encoding and pin its blocks atomically against
        # eviction on the main thread. The region's lock is acquired
        # inside `pin()`, nested under `self._lock`. Order: scheduler
        # lock first, region lock second (matches `_sweep_completions`,
        # the other site that nests these two).
        with self._lock:
            block_indices = self._encodings.get(req.mm_hash)
            if block_indices is None:
                self._pending_nacks.append((identity, req.mm_hash))
                return
            self._region.pin(block_indices)

        # Peer wireup and NIXL post happen lock-free — add_remote_agent
        # can take several ms on first contact and must not block
        # build_connector_meta on the main thread.
        try:
            peer = self._ensure_remote_peer(req)
            handle = self._post_nixl_write(peer, block_indices, req.dst_block_indices)
        except Exception:
            logger.exception(
                "ec: failed to post NIXL WRITE for mm_hash=%s", req.mm_hash
            )
            self._region.unpin(block_indices)
            with self._lock:
                self._pending_nacks.append((identity, req.mm_hash))
            return

        xfer_id = str(uuid.uuid4())
        with self._lock:
            self._in_flight[xfer_id] = (identity, req.mm_hash, handle)

    def _ensure_remote_peer(self, req: XferReq) -> _RemotePeer:
        """Look up (or create) the producer-side NIXL state for this consumer.

        Keyed by `consumer_agent_name` — the consumer's NIXL-level UUID.
        If the consumer was seen before with the same metadata we reuse
        the prepared remote dlist; if the metadata changed we tear down
        the old entry (the consumer restarted or re-initialized).

        Listener-thread only; no lock.
        """
        existing = self._remote_peers.get(req.consumer_agent_name)
        if (
            existing is not None
            and existing.metadata_bytes == req.consumer_nixl_metadata
        ):
            return existing
        if existing is not None:
            try:
                self._nixl.remove_remote_agent(existing.agent_name)
            except Exception:
                logger.warning(
                    "ec: remove_remote_agent failed for %s",
                    existing.agent_name,
                    exc_info=True,
                )

        agent_name = self._nixl.add_remote_agent(req.consumer_nixl_metadata)
        remote_blocks = deserialize_mem_descriptor(req.consumer_mem_descriptor)
        remote_xfer_descs = self._nixl.get_xfer_descs(remote_blocks, _NIXL_DRAM)
        remote_xfer_handle = self._nixl.prep_xfer_dlist(agent_name, remote_xfer_descs)
        peer = _RemotePeer(
            agent_name=agent_name,
            metadata_bytes=req.consumer_nixl_metadata,
            remote_xfer_handle=remote_xfer_handle,
        )
        self._remote_peers[req.consumer_agent_name] = peer
        return peer

    def _post_nixl_write(
        self,
        peer: _RemotePeer,
        local_block_indices: list[int],
        remote_block_indices: list[int],
    ) -> Any:
        """Post an async NIXL WRITE and return the handle.

        Completion signalling is via our own ZMQ XferAck, not NIXL
        notifs — we want explicit success/failure semantics and NIXL
        notifs are fire-and-forget post-DONE.
        """
        if len(local_block_indices) != len(remote_block_indices):
            raise ValueError(
                "ec: local/remote block count mismatch "
                f"({len(local_block_indices)} vs {len(remote_block_indices)})"
            )
        handle = self._nixl.make_prepped_xfer(
            "WRITE",
            self._local_xfer_handle,
            local_block_indices,
            peer.remote_xfer_handle,
            remote_block_indices,
            notif_msg=b"",
        )
        self._nixl.transfer(handle)
        return handle

    def _sweep_completions(self) -> None:
        """Poll in-flight xfer state and emit acks.

        NIXL polling runs lock-free so the listener thread can keep
        accepting XferReqs in parallel; state mutations and ROUTER sends
        re-acquire the lock / stay on this (main) thread.

        Branches on `check_xfer_state` mirror the KV NIXL connector
        (`kv_transfer/.../nixl/worker.py:_pop_done_transfers`):
        DONE → success, PROC → still in flight, anything else → failure.
        Handles are released inline on terminal states (DONE or
        failure), not deferred to the lock-held phase.
        """
        # (1) Snapshot.
        with self._lock:
            to_poll = [
                (xfer_id, handle) for xfer_id, (_, _, handle) in self._in_flight.items()
            ]

        # (2) Poll lock-free; release handles inline on terminal states.
        # outcomes maps xfer_id → ok; PROC entries are not added.
        outcomes: dict[str, bool] = {}
        for xfer_id, handle in to_poll:
            try:
                state = self._nixl.check_xfer_state(handle)
            except Exception:
                logger.exception(
                    "ec: check_xfer_state raised for xfer_id=%s; treating as failure",
                    xfer_id,
                )
                outcomes[xfer_id] = False
                self._release_xfer_handle(handle)
                continue
            if state == "DONE":
                outcomes[xfer_id] = True
                self._release_xfer_handle(handle)
            elif state == "PROC":
                continue
            else:
                logger.warning(
                    "ec: NIXL xfer in unexpected state %r for xfer_id=%s; "
                    "treating as failure",
                    state,
                    xfer_id,
                )
                outcomes[xfer_id] = False
                self._release_xfer_handle(handle)

        # (3) Apply outcomes + drain queued NACKs under the lock.
        ok_routes: list[tuple[bytes, str]] = []
        fail_routes: list[tuple[bytes, str]] = []
        with self._lock:
            for xfer_id, ok in outcomes.items():
                entry = self._in_flight.pop(xfer_id, None)
                if entry is None:
                    continue
                identity, mm_hash, _handle = entry
                blocks = self._encodings.get(mm_hash)
                if blocks is not None:
                    # Nested region lock under scheduler lock. Same
                    # order as `_handle_xfer_req`.
                    self._region.unpin(blocks)
                (ok_routes if ok else fail_routes).append((identity, mm_hash))
            queued_nacks = self._pending_nacks
            self._pending_nacks = []

        # (4) ROUTER sends, main thread only.
        self._send_xfer_acks(queued_nacks, ok=False)
        self._send_xfer_acks(ok_routes, ok=True)
        self._send_xfer_acks(fail_routes, ok=False)

    def _release_xfer_handle(self, handle: Any) -> None:
        """Release a NIXL xfer handle, swallowing failures.

        Bookkeeping-only — a failure here leaks a NIXL handle but does
        not affect correctness, so we log at debug.
        """
        try:
            self._nixl.release_xfer_handle(handle)
        except Exception:
            logger.debug("ec: release_xfer_handle failed", exc_info=True)

    def _send_xfer_acks(self, routes: list[tuple[bytes, str]], ok: bool) -> None:
        if not routes:
            return
        assert self._router is not None
        for identity, mm_hash in routes:
            try:
                payload = self._encoder.encode(XferAck(mm_hash=mm_hash, ok=ok))
                self._router.send_multipart([identity, b"", payload])
            except Exception:
                logger.exception(
                    "ec: failed to send XferAck mm_hash=%s ok=%s", mm_hash, ok
                )

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """Emit `ec_transfer_params` for each of this request's mm_features
        the producer currently has blocks for.

        Hits on either `_encodings` (serveable right now) or
        `_pending_save` (serveable after next step's promotion) — the
        consumer's XferReq may land in the intermediate window; a NACK
        in that window falls back to local encode cleanly (§5.5).
        """
        if not self._is_producer:
            return False, None
        params: dict[str, dict[str, Any]] = {}
        nixl_meta_b64 = b64encode(self._agent_metadata).decode("ascii")
        # Snapshot the shared `_encodings` set once under the lock;
        # `_pending_save` is main-thread only, no lock needed.
        with self._lock:
            in_encodings = set(self._encodings)
        for feature in request.mm_features:
            mm_hash = feature.mm_hash or feature.identifier
            if mm_hash not in in_encodings and mm_hash not in self._pending_save:
                continue
            size_bytes = (
                feature.mm_position.length * self._hidden_dim * self._element_size
            )
            params[mm_hash] = {
                "peer_host": self._peer_host,
                "peer_port": self._peer_port,
                "size_bytes": size_bytes,
                "nixl_agent_metadata_b64": nixl_meta_b64,
            }
        return False, (params or None)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Hook point called by the scheduler after it allocates KV/encoder
        cache for this request.

        Producer side: stash this feature's `(mm_hash, size_bytes)` so
        `build_connector_meta` will allocate CPU blocks for it this step.
        Consumer side: no-op — the consumer drives allocs from
        `ensure_cache_available`, which runs earlier in the scheduling
        loop than `update_state_after_alloc`.

        Main-thread only. The `_encodings` check needs the lock; the
        rest is single-threaded state.
        """
        if not self._is_producer:
            return
        feature = request.mm_features[index]
        mm_hash = feature.mm_hash or feature.identifier
        if mm_hash in self._pending_save or mm_hash in self._pending_new_encodings:
            return
        with self._lock:
            if mm_hash in self._encodings:
                return
        size_bytes = feature.mm_position.length * self._hidden_dim * self._element_size
        self._pending_new_encodings[mm_hash] = size_bytes

    # ==========================================================================
    # Consumer role
    # ==========================================================================

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """Decide whether `request` can run this step, allocating and
        sending any XferReqs needed.

        Returns True if the request is waiting on at least one in-flight
        transfer and should be deferred; False if every feature is
        locally-cached, remote-ready, or absent-from-announcement (and
        therefore falls through to vLLM's local encode).
        """
        if not self._is_consumer:
            return False
        self._drain_acks()
        params: dict[str, dict[str, Any]] = (
            getattr(request, "ec_transfer_params", None) or {}
        )
        pending = False
        for feature in request.mm_features:
            pos = feature.mm_position
            if pos.offset + pos.length <= num_computed_tokens:
                continue
            mm_hash = feature.mm_hash or feature.identifier
            if self.has_cache_item(mm_hash):
                continue
            if mm_hash in self._encoding_map:
                pending = True
                continue
            info = params.get(mm_hash)
            if info is None:
                # Not announced by any producer — fall through to local
                # encode.
                continue
            # Verify the announced size matches the size derivable from
            # this feature's mm_position locally. The compat hash already
            # protects vllm_version/model/dtype/block_size; this guards
            # against a mismatch within otherwise-compatible peers (bug
            # in the producer or in the orchestrator's announcement).
            expected_size = pos.length * self._hidden_dim * self._element_size
            announced_size = int(info.get("size_bytes", -1))
            if announced_size != expected_size:
                logger.warning(
                    "ec: announced size_bytes=%d for mm_hash=%s does not "
                    "match locally-derived %d (length=%d, hidden_dim=%d, "
                    "element_size=%d); falling back to local encode",
                    announced_size,
                    mm_hash,
                    expected_size,
                    pos.length,
                    self._hidden_dim,
                    self._element_size,
                )
                continue
            try:
                self._alloc_and_start_xfer(mm_hash, info, expected_size)
            except Exception:
                logger.exception(
                    "ec: failed to start xfer for mm_hash=%s; "
                    "falling back to local encode",
                    mm_hash,
                )
                continue
            pending = True
        return pending

    def _alloc_and_start_xfer(
        self, mm_hash: str, info: dict[str, Any], size_bytes: int
    ) -> None:
        n_blocks = max(1, ceil(size_bytes / self._block_size_bytes))
        # AllocationError propagates — operator sized the region too
        # small for the active request fleet.
        indices = self._region.alloc(n_blocks)
        try:
            peer = self._get_or_add_peer(info)
            req = XferReq(
                mm_hash=mm_hash,
                dst_block_indices=indices,
                consumer_agent_name=self._engine_id,
                consumer_nixl_metadata=self._agent_metadata,
                consumer_mem_descriptor=self._mem_descriptor_bytes,
                compatibility_hash=self._compat_hash,
            )
            # 3-frame DEALER -> ROUTER envelope:
            # send_multipart([b"", payload]) so the peer's ROUTER
            # `recv_multipart` returns [identity, b"", payload] — matches
            # the symmetric ROUTER->DEALER reply in `_send_xfer_acks`.
            peer.dealer.send_multipart([b"", self._encoder.encode(req)])
        except Exception:
            self._region.free(indices)
            raise
        # Only record in-flight after the XferReq has left the socket —
        # an entry in `_encoding_map` commits us to waiting for an ack.
        self._encoding_map[mm_hash] = indices

    def _drain_acks(self) -> None:
        """Non-blocking drain of XferAcks from every open DEALER.

        Producer ROUTER sends `[identity, b"", payload]`; the DEALER
        strips the identity, so we receive `[b"", payload]` and take the
        last frame.
        """
        for peer in self._peer_pool.values():
            while True:
                try:
                    frames = peer.dealer.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                except Exception:
                    logger.exception("ec: DEALER recv failed")
                    break
                if len(frames) != 2 or frames[0] != b"":
                    logger.warning(
                        "ec: dropped malformed XferAck envelope "
                        "(expected [b'', payload], got %d frames)",
                        len(frames),
                    )
                    continue
                try:
                    ack = self._xfer_ack_decoder.decode(frames[1])
                except (msgspec.DecodeError, msgspec.ValidationError):
                    logger.warning("ec: dropped malformed XferAck")
                    continue
                if ack.mm_hash not in self._encoding_map:
                    continue
                if ack.ok:
                    self._ready.add(ack.mm_hash)
                else:
                    indices = self._encoding_map.pop(ack.mm_hash)
                    self._region.free(indices)

    def _get_or_add_peer(self, info: dict[str, Any]) -> PeerEntry:
        # Remote NIXL agents are added here on first contact and removed
        # only on peer-metadata change (producer restart) or shutdown —
        # a NACK does NOT tear the agent down. Under stable deployments
        # the agent count is bounded by the producer fleet; unbounded
        # peer churn will accumulate agents until shutdown. See §7 of
        # the design doc (future: periodic peer-pool GC).
        host = info["peer_host"]
        port = int(info["peer_port"])
        key = (host, port)
        metadata = b64decode(info["nixl_agent_metadata_b64"])
        existing = self._peer_pool.get(key)
        if existing is not None and existing.metadata_bytes == metadata:
            return existing
        if existing is not None:
            try:
                self._nixl.remove_remote_agent(existing.agent_name)
            except Exception:
                logger.warning(
                    "ec: remove_remote_agent failed for %s",
                    existing.agent_name,
                    exc_info=True,
                )
            try:
                existing.dealer.close(linger=0)
            except Exception:
                logger.warning("ec: close stale DEALER failed", exc_info=True)
            self._peer_pool.pop(key, None)

        dealer = make_zmq_socket(
            ctx=self._zmq_ctx,
            path=make_zmq_path(scheme="tcp", host=host, port=port),
            socket_type=zmq.DEALER,
            bind=False,
        )
        agent_name = self._nixl.add_remote_agent(metadata)
        entry = PeerEntry(
            dealer=dealer,
            agent_name=agent_name,
            metadata_bytes=metadata,
        )
        self._peer_pool[key] = entry
        return entry

    def has_cache_item(self, identifier: str) -> bool:
        """Consumer-side cache-existence check.

        True iff the bytes for `identifier` are in our local mmap, either
        because `_drain_acks` just promoted them (worker will copy this
        step) or because the worker already copied them last step and
        we have not yet freed the blocks.
        """
        if not self._is_consumer:
            return False
        return identifier in self._ready or identifier in self._to_free_after_load

    # ==========================================================================
    # Shared
    # ==========================================================================

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        """Produce this step's scheduler → worker metadata.

        Producer branch: promote last step's `_pending_save` to
        `_encodings` (the worker has completed its save_caches by now, so
        those bytes are safe to serve), then allocate for newly scheduled
        encodings and park them in `_pending_save`.

        Consumer branch: free blocks the worker finished reading last
        step, drain any new acks, then list arrived mm_hashes in
        `meta.loads` and stage them for next-step free.
        """
        meta = ECCPUConnectorMetadata()

        if self._is_producer:
            # (a) Promote last step's pending saves into the shared
            # `_encodings` set so the listener can serve them. Snapshot
            # `_pending_save` (main-thread only, no lock) and merge it
            # into `_encodings` under the lock.
            to_promote = self._pending_save
            self._pending_save = {}
            with self._lock:
                self._encodings.update(to_promote)

            # (b) Drain newly scheduled encodings and allocate for each.
            pending_new = list(self._pending_new_encodings.items())
            self._pending_new_encodings = {}
            for mm_hash, size_bytes in pending_new:
                # Skip if already known. `_pending_save` is single-thread
                # so we can read it freely; `_encodings` needs the lock.
                if mm_hash in self._pending_save:
                    continue
                with self._lock:
                    already_known = mm_hash in self._encodings
                if already_known:
                    continue
                n_blocks = max(1, ceil(size_bytes / self._block_size_bytes))
                indices = self._alloc_with_lru_eviction(n_blocks)
                self._pending_save[mm_hash] = indices
                meta.saves[mm_hash] = indices

            # (c) Drive completion polling / ack sending.
            self._sweep_completions()

        if self._is_consumer:
            # (a) Free blocks handed off to the worker last step.
            #
            # Caveat: if a second request for the same mm_hash arrives
            # while the mm_hash is still in `_to_free_after_load`,
            # `has_cache_item` returns True (so ensure_cache_available
            # does not re-fetch) but the blocks are freed here one step
            # later — correctness depends on `encoder_cache[mm_hash]`
            # surviving on the GPU. If the GPU-side encoder_cache has
            # evicted the entry, that second request will see a miss.
            # Narrow window; accepted MVP behavior.
            for mm_hash in self._to_free_after_load:
                if mm_hash in self._encoding_map:
                    self._region.free(self._encoding_map.pop(mm_hash))
            self._to_free_after_load = []

            # (b) Drain any fresh ack arrivals.
            self._drain_acks()

            # (c) Hand arrived mm_hashes to the worker; stage for free.
            for mm_hash in list(self._ready):
                arrived = self._encoding_map.get(mm_hash)
                if arrived is None:
                    # Stale ack entry — drop.
                    self._ready.discard(mm_hash)
                    continue
                meta.loads[mm_hash] = arrived
                self._ready.discard(mm_hash)
                self._to_free_after_load.append(mm_hash)

        return meta

    def _alloc_with_lru_eviction(self, n_blocks: int) -> list[int]:
        """Producer-side LRU wrap around `_region.alloc`.

        If the pool is short, iterate `_encodings` in insertion order
        (LRU) and try to free each via `region.try_free` — which
        atomically skips entries whose blocks are currently pinned by
        an in-flight WRITE. Retry `alloc` after each successful eviction
        and return as soon as it succeeds. `_pending_save` entries are
        never candidates here — their blocks are actively being filled
        this step by the worker's save_caches.

        The free-and-pop must be atomic against the listener's
        `_encodings.get(mm_hash) → pin(indices)` sequence, otherwise the
        listener can pin blocks we just put on the free list. We hold
        `self._lock` across the whole sweep; lock order is scheduler →
        region (matches the listener), so no deadlock.
        """
        try:
            return self._region.alloc(n_blocks)
        except AllocationError:
            pass

        with self._lock:
            # Iterate over a key snapshot so we can mutate `_encodings`
            # mid-loop. The dict's insertion order is the LRU order.
            for mm_hash in list(self._encodings.keys()):
                indices = self._encodings[mm_hash]
                if not self._region.try_free(indices):
                    # Pinned by an in-flight WRITE; leave it in
                    # `_encodings` so listener lookups still resolve.
                    continue
                del self._encodings[mm_hash]
                try:
                    return self._region.alloc(n_blocks)
                except AllocationError:
                    continue

        raise AllocationError(
            f"ECSharedRegion exhausted: cannot satisfy {n_blocks} blocks "
            f"even after evicting all unpinned encodings — operator "
            f"under-sized num_ec_blocks for the active workload"
        )

    def shutdown(self) -> None:
        if self._is_producer and self._listener_t is not None:
            self._stop_event.set()
            self._listener_t.join(timeout=5)
            if self._router is not None:
                try:
                    self._router.close(linger=0)
                except Exception:
                    logger.debug("ec: router close failed", exc_info=True)

        if self._is_consumer:
            for entry in self._peer_pool.values():
                try:
                    entry.dealer.close(linger=0)
                except Exception:
                    logger.warning("ec: failed to close peer dealer", exc_info=True)
            self._peer_pool.clear()

        # NIXL cleanup — best-effort; we're on the teardown path.
        # Listener thread is joined by now (producer) or never started
        # (consumer-only), so no synchronization is needed.
        try:
            for _xfer_id, (_, _, handle) in list(self._in_flight.items()):
                with contextlib.suppress(Exception):
                    self._nixl.release_xfer_handle(handle)
            self._in_flight.clear()
        except Exception:
            logger.debug("ec: release in_flight failed", exc_info=True)

        for peer in list(self._remote_peers.values()):
            try:
                self._nixl.remove_remote_agent(peer.agent_name)
            except Exception:
                logger.debug(
                    "ec: remove_remote_agent failed for %s",
                    peer.agent_name,
                    exc_info=True,
                )
        self._remote_peers.clear()

        try:
            self._nixl.deregister_memory(self._registered_reg_descs)
        except Exception:
            logger.debug("ec: deregister_memory failed", exc_info=True)

        try:
            self._region.cleanup()
        except Exception:
            logger.debug("ec: region cleanup failed", exc_info=True)
