# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side of the ECCPUConnector.

Single class, two role branches:

- **Producer**: binds a ZMQ ROUTER and owns a single router thread
  (`_run_router`) that ingests `XferReq` messages, posts NIXL WRITEs
  against its NIXL-registered mmap region, polls completions on every
  loop iteration, and sends `XferAck`s back to peers via the same
  ROUTER socket.
- **Consumer**: lazy ZMQ DEALER pool keyed by `(peer_host, peer_port)` plus
  a local `_remote_encodings` of outstanding transfers, a `_ready` set of
  arrived-but-not-yet-loaded mm_hashes, a `_loaded` mmap cache that keeps
  completed transfer blocks alive for local re-copy, and a `_pending_reload`
  set tracking which `_loaded` entries were requested this step.

Locking
-------
The producer runs two threads: the main scheduler thread and the
router thread (`_run_router`). NIXL state — `_in_flight`, the
`_handle_xfer_req` ↔ `_sweep_completions` pair, and ROUTER socket I/O
— lives entirely on the router thread, so completion polling no longer
depends on the LLM scheduler running. Two locks are involved:

- `self._lock` guards the three producer-side datastructs that both
  threads touch: `_local_encodings`, `_in_flight`, and `_pending_nacks`.
- `self._region` has its own internal lock for the free pool and per-
  block ref counts.

When both locks are held, the order is **scheduler lock first, region
lock second**. Three sites hold them together: `_handle_xfer_req`
(lookup + pin) and `_sweep_completions` (lookup + unpin) on the router
thread, and `_producer_fifo_alloc` (try_free + `_local_encodings`
mutation, in one sweep) on the main thread. The eviction sweep must be
atomic against the router thread's lookup-then-pin: if `try_free`
succeeded but `_local_encodings` still mapped the mm_hash, the router
thread could pin blocks we just put on the free list. Holding the
scheduler lock across the whole sweep closes that window without
changing lock order.
"""

import contextlib
import threading
import uuid
from math import ceil
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from pybase64 import b64decode, b64encode
from zmq.utils.monitor import recv_monitor_message

from vllm import envs
from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import (
    EC_CONNECTOR_VERSION,
    ECCPUConnectorMetadata,
    XferAck,
    XferReq,
    compute_ec_compatibility_hash,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    ConsumerPeer,
    PeerAddr,
    ProducerPeer,
    build_block_descs,
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
    setup_ec_region,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
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

# ZMTP heartbeat for consumer DEALERs.
_HEARTBEAT_IVL_MS = 2000
_HEARTBEAT_TIMEOUT_MS = 4000
_HEARTBEAT_TTL_MS = 8000  # 2 × TIMEOUT


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
        # Hard-require NIXL on both roles.
        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError(
                "ECCPUConnector requires NIXL; "
                "install the `nixl` package or set a different ec_connector."
            )

        # engine_id doubles as the NIXL agent name
        assert ec_config.engine_id is not None
        self._engine_id: str = ec_config.engine_id
        self._nixl = NixlWrapper(
            self._engine_id,
            nixl_agent_config(num_threads=1, capture_telemetry=True),
        )

        # Build the mmap region + derived layout.
        layout = setup_ec_region(vllm_config)
        self._region = layout.region
        self._hidden_dim = layout.hidden_dim
        self._element_size = layout.element_size
        self._block_size_bytes = layout.block_size_bytes
        self._num_blocks = layout.num_blocks

        # Register the whole mmap as a single DRAM region bound to UCX.
        # Reg descs are `(addr, size, device_id, name)` 4-tuples.
        reg_descs = self._nixl.get_reg_descs(
            [(self._region.base_ptr, self._region.total_size_bytes, 0, "")],
            _NIXL_DRAM,
        )
        self._nixl.register_memory(reg_descs, backends=["UCX"])
        self._registered_reg_descs = reg_descs

        # Snapshot agent metadata after register_memory so peers see our
        # registered DRAM region; otherwise their `prep_xfer_dlist` against us
        # has no backend that handles DRAM_SEG.
        self._agent_metadata: bytes = self._nixl.get_agent_metadata()

        # Per-block xfer descs: one descriptor per block
        self._block_descs_list = build_block_descs(
            self._region.base_ptr,
            self._num_blocks,
            self._block_size_bytes,
            device_id=0,
        )
        xfer_descs = self._nixl.get_xfer_descs(self._block_descs_list, _NIXL_DRAM)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
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
            dtype=str(layout.dtype),
            block_size_bytes=self._block_size_bytes,
        )

        # ZMQ context shared by ROUTER + DEALERs.
        self._zmq_ctx = zmq.Context()

        # Reusable msgspec codecs — both roles need them.
        # Encoder is type-agnostic; one instance covers XferReq and XferAck.
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_req_decoder = msgspec.msgpack.Decoder(XferReq)
        self._xfer_ack_decoder = msgspec.msgpack.Decoder(XferAck)

        # ----- Producer-only: three-phase encoding pipeline ----------------
        #
        # Encodings move through three stages across two scheduling steps:
        #
        #   _pending_new_encodings  (mm_hash → size_bytes)
        #       Populated by update_state_after_alloc() during the scheduler
        #       loop. Holds encodings the GPU worker is about to compute this
        #       step but for which no CPU blocks have been allocated yet.
        #       Drained at the start of build_connector_meta().
        #
        #   _pending_save  (mm_hash → block_indices)
        #       Populated by build_connector_meta() after draining
        #       _pending_new_encodings. CPU blocks are allocated here and
        #       handed to the worker via meta.saves; the worker writes to
        #       them during the current step. Main-thread only, no lock.
        #
        #   _local_encodings  (mm_hash → block_indices)  [lock-guarded]
        #       Promoted from _pending_save at the start of the *next*
        #       build_connector_meta() call, once the worker has finished
        #       writing. The router thread reads this to serve consumer
        #       XferReqs. Guarded by self._lock.
        #
        # Allocation is deferred to build_connector_meta() (not done in
        # update_state_after_alloc()) so that _producer_fifo_alloc always
        # runs *after* last step's _pending_save has been promoted into
        # _local_encodings. This gives LRU eviction the fullest possible
        # candidate set; allocating earlier would make last step's saves
        # invisible to eviction while the pool is still full.
        if self._is_producer:
            self._pending_new_encodings: dict[str, int] = {}
            self._pending_save: dict[str, list[int]] = {}
            # Router-thread-only state. No lock.
            self._remote_peers: dict[str, ProducerPeer] = {}

            # Lock-guarded state (shared between main and router thread).
            self._lock = threading.Lock()
            self._local_encodings: dict[str, list[int]] = {}
            self._in_flight: dict[str, tuple[bytes, str, Any]] = {}
            self._pending_nacks: list[tuple[bytes, str]] = []

            self._router: zmq.Socket | None = None
            self._router_t: threading.Thread | None = None
            self._stop_event = threading.Event()
            self._peer_host = envs.VLLM_EC_SIDE_CHANNEL_HOST
            self._peer_port = envs.VLLM_EC_SIDE_CHANNEL_PORT

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
            self._router_t = threading.Thread(
                target=self._run_router,
                name="ec-nixl-router",
                daemon=True,
            )
            self._router_t.start()

        # ----- Consumer-only --------------------------------------------
        if self._is_consumer:
            # Value is (block_indices, peer_key) while the XferReq is in flight,
            # or None once tombstoned (NACK or peer down).
            self._remote_encodings: dict[
                str, tuple[list[int], tuple[str, int]] | None
            ] = {}
            self._ready: set[str] = set()
            # Completed transfers whose mmap blocks are still held as a local
            # cache. Subsequent requests for the same mm_hash are re-served
            # with a local mmap→GPU re-copy rather than a producer round-trip.
            # Evicted in insertion order under allocation pressure by
            # _consumer_fifo_alloc, skipping entries in _pending_reload.
            self._loaded: dict[str, list[int]] = {}
            # mm_hashes from _loaded that were requested this step; cleared at
            # the end of build_connector_meta.
            self._pending_reload: set[str] = set()
            self._peer_pool: dict[tuple[str, int], ConsumerPeer] = {}

    # ==========================================================================
    # Producer role
    # ==========================================================================

    def _run_router(self) -> None:
        """Ingest XferReqs, post NIXL WRITEs, poll completions, send acks.

        The router thread owns the ROUTER socket exclusively (ZMQ rule:
        one thread per socket) and is the single owner of the
        producer-side NIXL state machine: it accepts XferReqs, posts
        WRITEs, polls in-flight completions on each loop iteration, and
        sends XferAcks directly via ROUTER. Decoupled from LLM scheduling
        so that a producer which has finished its encode and gone idle
        still advances completions for any in-flight transfers.
        """
        assert self._router is not None
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        while not self._stop_event.is_set():
            # Tighter poll while transfers are in flight so completion
            # latency tracks NIXL progress rather than the 1s idle tick.
            with self._lock:
                in_flight = bool(self._in_flight)
            timeout_ms = 5 if in_flight else 1000
            try:
                events = dict(poller.poll(timeout=timeout_ms))
            except zmq.ContextTerminated:
                return
            except Exception:
                logger.exception("ec: router poll failed")
                continue
            if self._router in events:
                try:
                    identity, _, payload = self._router.recv_multipart(
                        flags=zmq.NOBLOCK
                    )
                except zmq.Again:
                    pass
                except zmq.ContextTerminated:
                    return
                except Exception:
                    logger.exception("ec: router recv failed")
                else:
                    try:
                        req = self._xfer_req_decoder.decode(payload)
                    except (msgspec.DecodeError, msgspec.ValidationError):
                        logger.warning("ec: dropped malformed XferReq")
                    else:
                        self._handle_xfer_req(identity, req)
            try:
                self._sweep_completions()
            except Exception:
                logger.exception("ec: sweep_completions failed")

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

        if req.compatibility_hash != self._compat_hash:
            with self._lock:
                self._pending_nacks.append((identity, req.mm_hash))
            return

        # Look up the encoding and pin its blocks atomically against
        # eviction on the main thread. Matches `_sweep_completions`.
        with self._lock:
            block_indices = self._local_encodings.get(req.mm_hash)
            if block_indices is None:
                self._pending_nacks.append((identity, req.mm_hash))
                return
            self._region.pin(block_indices)

        logger.debug(
            "ec: transfer requested mm_hash=%s consumer=%s n_blocks=%d",
            req.mm_hash,
            req.consumer_agent_name,
            len(block_indices),
        )
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

    def _ensure_remote_peer(self, req: XferReq) -> ProducerPeer:
        """Look up (or create) the producer-side NIXL state for this consumer.

        Keyed by `consumer_agent_name` — the consumer's NIXL-level UUID.
        If the consumer was seen before with the same metadata we reuse
        the prepared remote dlist; if the metadata changed we tear down
        the old entry (the consumer restarted or re-initialized).

        """
        existing = self._remote_peers.get(req.consumer_agent_name)
        if (
            existing is not None
            and existing.nixl_metadata_bytes == req.consumer_nixl_metadata
        ):
            return existing
        if existing is not None:
            try:
                self._nixl.remove_remote_agent(existing.nixl_agent_name)
            except Exception:
                logger.warning(
                    "ec: remove_remote_agent failed for %s",
                    existing.nixl_agent_name,
                    exc_info=True,
                )

        agent_name = self._nixl.add_remote_agent(req.consumer_nixl_metadata)
        remote_blocks = deserialize_mem_descriptor(req.consumer_mem_descriptor)
        remote_xfer_descs = self._nixl.get_xfer_descs(remote_blocks, _NIXL_DRAM)
        remote_xfer_handle = self._nixl.prep_xfer_dlist(agent_name, remote_xfer_descs)
        peer = ProducerPeer(
            nixl_agent_name=agent_name,
            nixl_metadata_bytes=req.consumer_nixl_metadata,
            nixl_xfer_handle=remote_xfer_handle,
        )
        self._remote_peers[req.consumer_agent_name] = peer
        return peer

    def _post_nixl_write(
        self,
        peer: ProducerPeer,
        local_block_indices: list[int],
        remote_block_indices: list[int],
    ) -> Any:
        """Post an async NIXL WRITE and return the handle.

        Completion signalling is via our ZMQ XferAck.
        """
        # For now, we assume same block size:
        if len(local_block_indices) != len(remote_block_indices):
            raise ValueError(
                "ec: local/remote block count mismatch "
                f"({len(local_block_indices)} vs {len(remote_block_indices)})"
            )
        handle = self._nixl.make_prepped_xfer(
            "WRITE",
            self._local_xfer_handle,
            local_block_indices,
            peer.nixl_xfer_handle,
            remote_block_indices,
            notif_msg=b"",
        )
        self._nixl.transfer(handle)
        return handle

    def _sweep_completions(self) -> None:
        """Poll in-flight xfer state and emit acks.

        NIXL polling runs lock-free; acks and NACKs are sent directly
        via the ROUTER socket (router thread only — no inproc relay).

        Branches on `check_xfer_state` mirror the KV NIXL connector
        (`kv_transfer/.../nixl/worker.py:_pop_done_transfers`):
        DONE → success, PROC → still in flight, anything else → failure.
        Handles are released inline on terminal states (DONE or
        failure), not deferred to the lock-held phase.
        """
        in_flight_snapshot = []
        # (1) Snapshot.
        with self._lock:
            in_flight_snapshot = [
                (xfer_id, handle) for xfer_id, (_, _, handle) in self._in_flight.items()
            ]

        # (2) Poll lock-free; release handles inline on terminal states.
        # outcomes maps xfer_id → ok; PROC entries are not added.
        outcomes: dict[str, bool] = {}
        for xfer_id, handle in in_flight_snapshot:
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
                logger.debug("ec: transfer complete xfer_id=%s", xfer_id)
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

        # (3) Apply outcomes, unpin memory and drain queued NACKs.
        ok_routes: list[tuple[bytes, str]] = []
        fail_routes: list[tuple[bytes, str]] = []
        with self._lock:
            for xfer_id, ok in outcomes.items():
                entry = self._in_flight.pop(xfer_id, None)
                if entry is None:
                    continue
                identity, mm_hash, _handle = entry
                blocks = self._local_encodings.get(mm_hash)
                if blocks is not None:
                    self._region.unpin(blocks)
                (ok_routes if ok else fail_routes).append((identity, mm_hash))
            queued_nacks = self._pending_nacks
            self._pending_nacks = []

        # (4) ROUTER sends, main thread only.
        self._send_xfer_acks(queued_nacks, ok=False)
        self._send_xfer_acks(ok_routes, ok=True)
        self._send_xfer_acks(fail_routes, ok=False)

    def _release_xfer_handle(self, handle: Any) -> None:
        """Release a NIXL xfer handle

        Bookkeeping-only — a failure here leaks a NIXL handle but does
        not affect correctness, so we log at error.
        """
        try:
            self._nixl.release_xfer_handle(handle)
        except Exception:
            logger.exception("ec: release_xfer_handle failed")

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

        Hits on either `_local_encodings` (serveable right now) or
        `_pending_save` (serveable after next step's promotion) — the
        consumer's XferReq may land in the intermediate window; a NACK
        in that window falls back to local encode cleanly (§5.5).
        """
        if not self._is_producer:
            return False, None
        params: dict[str, dict[str, Any]] = {}
        nixl_meta_b64 = b64encode(self._agent_metadata).decode("ascii")
        # Snapshot the shared `_local_encodings` set once under the lock;
        # `_pending_save` is main-thread only, no lock needed.
        with self._lock:
            local_encodings_snapshot = set(self._local_encodings)
        for feature in request.mm_features:
            mm_hash = feature.mm_hash or feature.identifier
            if (
                mm_hash not in local_encodings_snapshot
                and mm_hash not in self._pending_save
            ):
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

        Main-thread only. The `_local_encodings` check needs the lock; the
        rest is single-threaded state.
        """
        if not self._is_producer:
            return
        feature = request.mm_features[index]
        mm_hash = feature.mm_hash or feature.identifier
        if mm_hash in self._pending_save or mm_hash in self._pending_new_encodings:
            return
        with self._lock:
            if mm_hash in self._local_encodings:
                return
        size_bytes = feature.mm_position.length * self._hidden_dim * self._element_size
        self._pending_new_encodings[mm_hash] = size_bytes
        logger.debug("ec: save scheduled mm_hash=%s size_bytes=%d", mm_hash, size_bytes)

    # ==========================================================================
    # Consumer role
    # ==========================================================================

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """Decide whether `request` can run this step, allocating and
        sending any XferReqs needed.

        Returns False if the request is waiting on at least one in-flight
        transfer and should be deferred; True if every feature is
        locally-cached or absent-from-announcement (and
        therefore falls through to vLLM's local encode).
        """
        if not self._is_consumer:
            return True
        params: dict[str, dict[str, Any]] = (
            getattr(request, "ec_transfer_params", None) or {}
        )
        if not params:
            return True
        pending = False
        for feature in request.mm_features:
            pos = feature.mm_position
            # Make sure the requested mm object is relevant:
            if pos.offset + pos.length <= num_computed_tokens:
                continue
            mm_hash = feature.mm_hash or feature.identifier
            if self.has_cache_item(mm_hash):
                if mm_hash in self._loaded:
                    self._pending_reload.add(mm_hash)
                continue
            if mm_hash in self._remote_encodings:
                if self._remote_encodings[mm_hash] is None:
                    # NACK tombstone: drop it, fall through to local
                    # encode. One-shot consumption is safe — the v1
                    # scheduler does not revisit a request after we
                    # return True, so `_alloc_and_start_xfer` cannot
                    # re-fire within this request.
                    del self._remote_encodings[mm_hash]
                    continue
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
        return not pending

    def _alloc_and_start_xfer(
        self, mm_hash: str, info: dict[str, Any], size_bytes: int
    ) -> None:
        n_blocks = max(1, ceil(size_bytes / self._block_size_bytes))
        indices = self._consumer_fifo_alloc(n_blocks)
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
            peer.zmq_dealer.send_multipart([b"", self._encoder.encode(req)])
        except Exception:
            self._region.free(indices)
            raise
        logger.debug(
            "ec: load requested mm_hash=%s n_blocks=%d peer=%s:%d",
            mm_hash,
            n_blocks,
            info["peer_host"],
            int(info["peer_port"]),
        )
        # Only record in-flight after the XferReq has left the socket —
        # an entry in `_remote_encodings` commits us to waiting for an ack.
        host = info["peer_host"]
        port = int(info["peer_port"])
        peer_addr: PeerAddr = (host, port)
        self._remote_encodings[mm_hash] = (indices, peer_addr)

    def _drain_acks(self) -> None:
        """Non-blocking drain of XferAcks from every open DEALER.

        Producer ROUTER sends `[identity, b"", payload]`; the DEALER
        strips the identity, so we receive `[b"", payload]` and take the
        last frame.
        """
        for addr in self._poll_dead_peers():
            self.on_peer_down(addr)

        for peer in self._peer_pool.values():
            while True:
                try:
                    frames = peer.zmq_dealer.recv_multipart(flags=zmq.NOBLOCK)
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
                if ack.mm_hash not in self._remote_encodings:
                    continue
                entry = self._remote_encodings[ack.mm_hash]
                if entry is None:
                    # Already tombstoned by a prior NACK or peer-down; ignore.
                    continue
                indices, _ = entry
                if ack.ok:
                    self._ready.add(ack.mm_hash)
                    logger.debug("ec: load arrived mm_hash=%s", ack.mm_hash)
                else:
                    # NACK: free the consumer-side blocks and leave a
                    # tombstone for `ensure_cache_available` to consume on
                    # its next call. Consuming the tombstone there flips
                    # the request from deferred to runnable so vLLM's local
                    # encode path covers this feature (§5.5).
                    self._region.free(indices)
                    self._remote_encodings[ack.mm_hash] = None

    def _get_or_add_peer(self, info: dict[str, Any]) -> ConsumerPeer:
        # Remote NIXL agents are added here on first contact and removed
        # only on peer-metadata change (producer restart) or shutdown —
        # a NACK does NOT tear the agent down.
        host = info["peer_host"]
        port = int(info["peer_port"])
        key = (host, port)
        metadata = b64decode(info["nixl_agent_metadata_b64"])
        existing = self._peer_pool.get(key)
        if existing is not None and existing.nixl_metadata_bytes == metadata:
            return existing
        if existing is not None:
            self.on_peer_down(key)

        # If the producer restarted on a new IP but kept the same port and
        # NIXL agent name, the old pool entry sits under a different key.
        # Evict it now so add_remote_agent doesn't hit NIXL_ERR_NOT_ALLOWED
        # due to the agent name already being registered.
        for stale_key in [k for k in self._peer_pool if k[1] == port and k[0] != host]:
            logger.info(
                "ec: evicting stale peer %s (same port %d, new peer %s)",
                stale_key,
                port,
                host,
            )
            self.on_peer_down(stale_key)

        dealer = make_zmq_socket(
            ctx=self._zmq_ctx,
            path=make_zmq_path(scheme="tcp", host=host, port=port),
            socket_type=zmq.DEALER,
            bind=False,
        )
        dealer.setsockopt(zmq.HEARTBEAT_IVL, _HEARTBEAT_IVL_MS)
        dealer.setsockopt(zmq.HEARTBEAT_TIMEOUT, _HEARTBEAT_TIMEOUT_MS)
        dealer.setsockopt(zmq.HEARTBEAT_TTL, _HEARTBEAT_TTL_MS)

        monitor_addr = f"inproc://ec-peer-mon-{host}-{port}"
        dealer.monitor(monitor_addr, zmq.EVENT_DISCONNECTED)
        mon = self._zmq_ctx.socket(zmq.PAIR)
        mon.connect(monitor_addr)

        agent_name = self._nixl.add_remote_agent(metadata)
        entry = ConsumerPeer(
            zmq_dealer=dealer,
            nixl_agent_name=agent_name,
            nixl_metadata_bytes=metadata,
            zmq_monitor=mon,
        )
        self._peer_pool[key] = entry
        return entry

    def _poll_dead_peers(self) -> list[PeerAddr]:
        dead: list[PeerAddr] = []
        for addr, peer in list(self._peer_pool.items()):
            if peer.zmq_monitor is None:
                continue
            try:
                while True:
                    evt = recv_monitor_message(peer.zmq_monitor, flags=zmq.NOBLOCK)
                    if evt["event"] == zmq.EVENT_DISCONNECTED:
                        dead.append(addr)
                        break
            except zmq.Again:
                pass
            except Exception:
                logger.warning(
                    "ec: monitor poll failed for addr=%s", addr, exc_info=True
                )
        return dead

    def on_peer_down(self, addr: PeerAddr) -> None:
        peer = self._peer_pool.pop(addr, None)
        if peer is None:
            return
        if peer.zmq_monitor is not None:
            try:
                peer.zmq_dealer.disable_monitor()
            except Exception:
                logger.warning(
                    "ec: disable_monitor failed addr=%s", addr, exc_info=True
                )
            try:
                peer.zmq_monitor.close(linger=0)
            except Exception:
                logger.warning("ec: close monitor failed addr=%s", addr, exc_info=True)
        try:
            self._nixl.remove_remote_agent(peer.nixl_agent_name)
        except Exception:
            logger.warning(
                "ec: remove_remote_agent failed for %s",
                peer.nixl_agent_name,
                exc_info=True,
            )
        try:
            peer.zmq_dealer.close(linger=0)
        except Exception:
            logger.warning("ec: close DEALER failed addr=%s", addr, exc_info=True)

        n = 0
        for mm_hash, entry in list(self._remote_encodings.items()):
            if entry is None or entry[1] != addr:
                continue
            indices, _ = entry
            if mm_hash not in self._ready:
                self._region.free(indices)
                self._remote_encodings[mm_hash] = None
                n += 1
        logger.info(
            "ec: peer down addr=%s agent=%s tombstoned=%d",
            addr,
            peer.nixl_agent_name,
            n,
        )

    def has_cache_item(self, identifier: str) -> bool:
        """Consumer-side cache-existence check.

        True iff the bytes for `identifier` are in our local mmap, either
        because `_drain_acks` just promoted them (worker will copy this
        step) or because the worker already copied them last step and
        we have not yet freed the blocks.
        """
        if not self._is_consumer:
            return False
        return identifier in self._loaded or identifier in self._ready

    # ==========================================================================
    # Shared
    # ==========================================================================

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        """Produce this step's scheduler → worker metadata.

        Producer branch: promote last step's `_pending_save` to
        `_local_encodings` (the worker has completed its save_caches by now, so
        those bytes are safe to serve), then allocate for newly scheduled
        encodings and park them in `_pending_save`.

        Consumer branch: drain new acks, hand arrived mm_hashes to the
        worker via `meta.loads`, and keep their mmap blocks alive in
        `_loaded` as a local cache. Subsequent requests for the same
        mm_hash are re-served with a local mmap→GPU re-copy. Blocks are
        only freed when evicted under allocation pressure.
        """
        meta = ECCPUConnectorMetadata()

        if self._is_producer:
            # (a) Promote last step's pending saves into the shared
            # `_local_encodings` set so the router thread can serve them. Snapshot
            # `_pending_save` and merge it into `_local_encodings`
            to_promote = self._pending_save
            self._pending_save = {}
            with self._lock:
                self._local_encodings.update(to_promote)

            # (b) Drain newly scheduled encodings and allocate for each.
            pending_new = list(self._pending_new_encodings.items())
            self._pending_new_encodings = {}
            for mm_hash, size_bytes in pending_new:
                # Skip if already known.
                if mm_hash in self._pending_save:
                    continue
                with self._lock:
                    already_known = mm_hash in self._local_encodings
                if already_known:
                    continue
                n_blocks = max(1, ceil(size_bytes / self._block_size_bytes))
                indices = self._producer_fifo_alloc(n_blocks)
                self._pending_save[mm_hash] = indices
                meta.saves[mm_hash] = indices
                logger.debug(
                    "ec: save allocated mm_hash=%s n_blocks=%d blocks=%s",
                    mm_hash,
                    n_blocks,
                    indices,
                )

        if self._is_consumer:
            # (a) Drain any fresh ack arrivals.
            self._drain_acks()

            # (b) Hand newly-arrived mm_hashes to the worker; move to loaded
            #     cache. Blocks stay allocated so that subsequent requests for
            #     the same mm_hash are re-served with a local mmap→GPU re-copy
            #     instead of a producer round-trip.
            for mm_hash in list(self._ready):
                entry = self._remote_encodings.pop(mm_hash, None)
                if entry is None:
                    # Stale ack entry — drop.
                    continue
                indices, _ = entry
                meta.loads[mm_hash] = indices
                self._loaded[mm_hash] = indices
                logger.debug("ec: load issued mm_hash=%s blocks=%s", mm_hash, indices)
            self._ready.clear()

            # (c) Re-serve cached entries requested this step via a local
            #     mmap→GPU re-copy, bypassing the producer entirely.
            for mm_hash in self._pending_reload:
                if mm_hash not in meta.loads:
                    blocks = self._loaded.get(mm_hash)
                    if blocks is not None:
                        meta.loads[mm_hash] = blocks
                        logger.debug(
                            "ec: cache hit mm_hash=%s blocks=%s", mm_hash, blocks
                        )
            self._pending_reload = set()

        return meta

    def _evict_and_alloc(
        self,
        n_blocks: int,
        cache: dict[str, list[int]],
        *,
        skip_pinned: bool = False,
        protected: set[str] | None = None,
    ) -> list[int] | None:
        """Evict entries from `cache` in insertion order until `alloc` succeeds.

        Called under `self._lock` by the producer (which must skip pinned
        blocks via `try_free`) and without a lock by the consumer (which
        can always free). Entries whose mm_hash appears in `protected`
        are skipped — the consumer uses this to keep `_pending_reload`
        blocks alive until `build_connector_meta` has consumed them.

        Returns the allocated block list, or None if the cache was
        exhausted without satisfying the request.
        """
        for mm_hash in list(cache.keys()):
            if protected is not None and mm_hash in protected:
                continue
            indices = cache[mm_hash]
            if skip_pinned:
                if not self._region.try_free(indices):
                    continue
            else:
                self._region.free(indices)
            del cache[mm_hash]
            try:
                return self._region.alloc(n_blocks)
            except AllocationError:
                continue
        return None

    def _producer_fifo_alloc(self, n_blocks: int) -> list[int]:
        """Producer-side LRU wrap around `_region.alloc`.

        If the pool is short, sweep `_local_encodings` in insertion order,
        skipping pinned blocks via `try_free`. Held under `self._lock` so
        the eviction is atomic against the router thread's lookup-then-pin.
        """
        try:
            return self._region.alloc(n_blocks)
        except AllocationError:
            pass

        with self._lock:
            result = self._evict_and_alloc(
                n_blocks, self._local_encodings, skip_pinned=True
            )
        if result is not None:
            return result

        raise AllocationError(
            f"ECSharedRegion exhausted: cannot satisfy {n_blocks} blocks "
            f"even after evicting all unpinned encodings — operator "
            f"under-sized num_ec_blocks for the active workload"
        )

    def _consumer_fifo_alloc(self, n_blocks: int) -> list[int]:
        """Consumer-side FIFO wrap around `_region.alloc`.

        If the pool is short, evict the oldest `_loaded` cache entries
        (mmap→GPU already done) until alloc succeeds, skipping any
        mm_hash in `_pending_reload` — those blocks are promised to this
        step's `meta.loads` and freeing them would silently drop the
        request from the worker's load list. Future requests for an
        evicted mm_hash will re-fetch from the producer.
        """
        try:
            return self._region.alloc(n_blocks)
        except AllocationError:
            pass

        result = self._evict_and_alloc(
            n_blocks, self._loaded, protected=self._pending_reload
        )
        if result is not None:
            return result

        raise AllocationError(
            f"ECSharedRegion exhausted: cannot satisfy {n_blocks} blocks "
            f"even after evicting all evictable cache entries — operator "
            f"under-sized num_ec_blocks for the active workload"
        )

    def shutdown(self) -> None:
        if self._is_producer and self._router_t is not None:
            self._stop_event.set()
            self._router_t.join(timeout=5)
            if self._router is not None:
                try:
                    self._router.close(linger=0)
                except Exception:
                    logger.debug("ec: router close failed", exc_info=True)

        if self._is_consumer:
            for addr in list(self._peer_pool):
                self.on_peer_down(addr)

        # NIXL cleanup — best-effort; we're on the teardown path.
        # Router thread is joined by now (producer) or never started
        # (consumer-only), so no synchronization is needed.
        if self._is_producer:
            try:
                for _xfer_id, (_, _, handle) in list(self._in_flight.items()):
                    with contextlib.suppress(Exception):
                        self._nixl.release_xfer_handle(handle)
                self._in_flight.clear()
            except Exception:
                logger.debug("ec: release in_flight failed", exc_info=True)

            for peer in list(self._remote_peers.values()):
                try:
                    self._nixl.remove_remote_agent(peer.nixl_agent_name)
                except Exception:
                    logger.debug(
                        "ec: remove_remote_agent failed for %s",
                        peer.nixl_agent_name,
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

        try:
            self._zmq_ctx.destroy(linger=0)
        except Exception:
            logger.debug("ec: zmq context destroy failed", exc_info=True)
