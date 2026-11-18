# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base scheduler-side logic for the NIXL connector."""

import contextlib
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    clip_ssm_state_blocks,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    LOOKUP_PIN_MSG,
    NIXL_CONNECTOR_VERSION,
    HeartbeatInfo,
    LookupPinReply,
    LookupPinRequest,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    PeerMissReason,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.peer_lookup_client import (
    PeerLookupClient,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.peer_pins import (
    PeerPinServer,
    miss_reply,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.engine import PeerLookupInfo
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

_LISTENER_POLL_MS = 1000


@dataclass
class PeerLookupStats:
    """Consumer-side peer lookup counters (scheduler thread only)."""

    lookups: int = 0
    full_hits: int = 0
    partial_hits: int = 0
    # PeerMissReason values, plus the local reasons "timeout", "bad_reply",
    # "send_failed", "closed" and "local_precondition".
    misses: dict[str, int] = field(default_factory=dict)
    # Lookups cancelled because the request went away.
    cancelled: int = 0
    # Hits that load nothing (the local hit covers them, or the load is below
    # the recompute threshold): A's pin is freed by a notify-only recv.
    notify_only_releases: int = 0
    # Pins whose lookup was cancelled, released on the peer / left to expire.
    orphans_released: int = 0
    orphans_unreleased: int = 0

    def record_miss(self, reason: str) -> None:
        self.misses[reason] = self.misses.get(reason, 0) + 1


@dataclass(frozen=True)
class PeerEndpoint:
    """What the worker needs to handshake with a peer, from request params."""

    engine_id: EngineId
    tp_size: int | None
    dcp_size: int = 1
    pp_size: int = 1


class PeerLookupChannel:
    """Hand-off between the side-channel listener thread and the scheduler.

    The listener puts decoded lookups on `inbox`; the scheduler thread serves
    them and puts `(identity, reply)` on `outbox`, then calls `wake` so the
    listener, which owns the ROUTER socket, sends them. Nothing here touches
    the block pool or the pin server.
    """

    def __init__(self) -> None:
        self.inbox: queue.Queue[tuple[bytes, LookupPinRequest]] = queue.Queue()
        self.outbox: queue.Queue[tuple[bytes, LookupPinReply]] = queue.Queue()
        # Whether this engine answers lookups at all.
        self.enabled = False
        # Thread-safe callback that wakes an idle engine loop.
        self.wakeup: Callable[[], None] | None = None
        self.wake_r, self.wake_w = os.pipe()
        os.set_blocking(self.wake_r, False)
        os.set_blocking(self.wake_w, False)
        self._closed = False

    def wake(self) -> None:
        """Tell the listener that the outbox has replies."""
        try:
            os.write(self.wake_w, b"\x01")
        except BlockingIOError:
            pass  # The pipe is full, so a wake-up is already pending.
        except OSError:
            pass  # Closed during shutdown.

    def drain_wake_pipe(self) -> None:
        try:
            while os.read(self.wake_r, 4096):
                pass
        except (BlockingIOError, OSError):
            pass

    def notify_engine(self) -> None:
        callback = self.wakeup
        if callback is None:
            return
        try:
            callback()
        except Exception:
            logger.warning("Engine wake-up callback failed", exc_info=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in (self.wake_r, self.wake_w):
            with contextlib.suppress(OSError):
                os.close(fd)


class NixlBaseConnectorScheduler:
    """Base implementation of Scheduler side methods shared by pull and push."""

    # Emitted in kv_transfer_params so an external router can distinguish a
    # pull (READ) producer from a push (WRITE) one. Overridden by the push
    # scheduler.
    _TRANSFER_MODE: str = "pull"

    # Class-level defaults so a scheduler built without __init__ can shut down.
    _lookup_client: PeerLookupClient | None = None
    _peer_lookup_timeout: float = 2.0

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        parallel_config = vllm_config.parallel_config
        # TP1 PCP+DCP exposes its DCP shards as transfer ranks.
        self.transfer_tp_size = max(
            parallel_config.tensor_parallel_size,
            parallel_config.decode_context_parallel_size,
        )
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.kv_cache_config = kv_cache_config
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        assert vllm_config.kv_transfer_config is not None
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 30
            )
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._heartbeat_interval = self._kv_lease_duration // 6
        if current_platform.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = (
                vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
            )
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            # Also handle unlikely SW-only model case instead of checking num_groups>1.
            and any(
                not issubclass(g.manager_cls, FullAttentionManager)
                for g in kv_cache_config.transfer_groups
            )
        )
        self._has_mamba = kv_cache_config.has_mamba_layers
        self._bounded_replay = any(
            g.prefix_replay_tokens > 0 for g in kv_cache_config.kv_cache_groups
        )

        logger.info("Initializing NIXL Scheduler %s", engine_id)
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            logger.info("Hybrid Memory Allocator is enabled with NIXL")

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Payloads the listener serves; a republish after a wake-up updates them.
        self._handshake_data: dict[tuple[int, int], bytes] = {}
        self._registration_generation = 0

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[
            ReqId, tuple[Request, BlockIds, tuple[int, ...], bool]
        ] = {}
        self._reqs_need_save: dict[ReqId, Request] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self._reqs_in_batch: set[ReqId] = set()
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

        # Heartbeat tracking: requests needing periodic lease-renewal heartbeats to
        # remote P-side, stored as ready-to-send HeartbeatInfo grouped by remote engine
        self._heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}
        # Reverse lookup: local req_id -> (engine_id, remote_req_id) for O(1) removal
        self._heartbeat_req_engine: dict[ReqId, tuple[EngineId, ReqId]] = {}
        self._last_heartbeat_time: float = 0.0

        # Gather Sliding Window sizes for each kv cache group (if any) in number of
        # blocks per KV cache group. This is used to clip the local attention window.
        sw_sizes_tokens: list[tuple[int, int]] = [
            (p.sliding_window, g.block_size)
            if isinstance(p := g.manager_config.params, SlidingWindowManager.Params)
            else (0, self.block_size)
            for g in kv_cache_config.transfer_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to conservatively
        # account for boundary overlap eg window isn't fully aligned with blocks.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

        # Trailing scratch slots that mamba managers co-allocate per request
        # for speculative decoding; None for non-SSM groups.
        self._ssm_spec_blocks = [
            g.manager_config.params.num_speculative_blocks
            if isinstance(g.manager_config.params, MambaManager.Params)
            else None
            for g in kv_cache_config.transfer_groups
        ]

        # Threshold to decide whether to compute kv cache locally
        # or pull from a remote node: minimum number of remote
        # tokens to amortize the xfer latencies
        self.kv_recompute_threshold: int = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_recompute_threshold", 64
            )
        )

        # Peer prefix lookup-and-pin (pull mode only): lets other engines ask
        # whether this one caches a prefix and pin it for a remote read.
        extra = vllm_config.kv_transfer_config.get_from_extra_config
        self._peer_lookup_enabled = bool(extra("peer_lookup", False))
        self._peer_pin_max_fraction = float(extra("peer_pin_max_fraction", 0.25))
        self._peer_pin_min_free_fraction = float(
            extra("peer_pin_min_free_fraction", 0.05)
        )
        self._peer_pin_serve_timeout = float(extra("peer_lookup_serve_timeout", 5.0))
        self._peer_lookup_timeout = float(extra("peer_lookup_timeout", 2.0))
        self._lookup_client: PeerLookupClient | None = None
        self.peer_lookup_stats = PeerLookupStats()
        # (host, port) of a peer's side channel -> how to reach its engine; filled
        # when a lookup is submitted, used to release pins of cancelled lookups.
        self._peer_endpoints: dict[tuple[str, int], PeerEndpoint] = {}
        self._peer_pins: PeerPinServer | None = None
        self._compatibility_hash: str | None = None
        self._lookup = PeerLookupChannel()
        self._lookup.enabled = (
            self._peer_lookup_enabled and self._TRANSFER_MODE == "pull"
        )

        # Bi-directional KV transfer feature supports KV block
        # transfers from D node to P node
        self.is_bidirectional_kv_xfer_enabled = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )
        self.decoder_kv_blocks_ttl = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "decoder_kv_blocks_ttl", 480
            )
        )

        if self.is_bidirectional_kv_xfer_enabled and self.kv_recompute_threshold > 0:
            logger.info(
                "Bidirectional KV transfer is enabled and the kv "
                "recompute threshold is set to %d tokens."
                "KV blocks on D are released after a TTL of %d seconds.",
                self.kv_recompute_threshold,
                self.decoder_kv_blocks_ttl,
            )

    def get_peer_lookup_info(self) -> "PeerLookupInfo | None":
        """How peers reach this rank for a lookup, or None if it serves none.

        None unless lookups are served (`peer_lookup` set, pull mode) and the
        layout hash is known: a peer cannot validate an empty fingerprint. The
        hash is set by `set_xfer_handshake_metadata` during engine setup, before
        the ready message that carries this record is built.
        """
        if not self._lookup.enabled or not self._compatibility_hash:
            return None
        from vllm.v1.engine import PeerLookupInfo

        parallel_config = self.vllm_config.parallel_config
        kv_transfer_config = self.vllm_config.kv_transfer_config
        assert kv_transfer_config is not None and kv_transfer_config.kv_connector
        return PeerLookupInfo(
            enabled=True,
            engine_id=self.engine_id,
            side_channel_host=self.side_channel_host,
            side_channel_port=self.side_channel_port,
            # Same values pull_scheduler puts in the kv_transfer_params it
            # exports, which is what a consumer must send back.
            tp_size=self.transfer_tp_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            connector=kv_transfer_config.kv_connector,
            connector_version=str(NIXL_CONNECTOR_VERSION),
            layout_fingerprint=self._compatibility_hash,
        )

    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        """Create the peer pin server once the scheduler's cache manager exists."""
        if not self._peer_lookup_enabled or self._TRANSFER_MODE != "pull":
            return
        if not kv_cache_manager.supports_prefix_pinning():
            logger.info(
                "Peer prefix lookup is enabled but this KV cache layout cannot "
                "pin prefixes by hash; every lookup will be refused."
            )
        self._peer_pins = PeerPinServer(
            kv_cache_manager,
            block_size=self.block_size,
            hash_algo=self.vllm_config.cache_config.prefix_caching_hash_algo,
            lease_duration=self._kv_lease_duration,
            max_pinned_fraction=self._peer_pin_max_fraction,
            min_free_fraction=self._peer_pin_min_free_fraction,
            max_pending_seconds=self._peer_pin_serve_timeout,
        )
        if self._compatibility_hash is not None:
            self._peer_pins.set_compatibility_hash(self._compatibility_hash)

    def on_step_processed(self) -> None:
        """One scheduled step's output has been processed."""
        if self._peer_pins is not None:
            self._peer_pins.on_step_processed()

    def serve_peer_lookup(self, token: Any, request: LookupPinRequest) -> None:
        """Scheduler thread only. The reply comes from take_peer_lookup_replies."""
        assert self._peer_pins is not None
        self._peer_pins.serve(token, request)

    def take_peer_lookup_replies(self) -> list[tuple[Any, LookupPinReply]]:
        if self._peer_pins is None:
            return []
        return self._peer_pins.take_ready_replies()

    def set_engine_wakeup(self, callback: Callable[[], None]) -> None:
        """Store the thread-safe callback that wakes an idle engine loop."""
        self._lookup.wakeup = callback

    def _drain_peer_lookups(self) -> None:
        """Serve the lookups the listener queued. Scheduler thread only."""
        inbox = self._lookup.inbox
        while True:
            try:
                identity, request = inbox.get_nowait()
            except queue.Empty:
                return
            if self._peer_pins is None:
                # Not bound to a cache manager (yet): nothing to look up in.
                self._lookup.outbox.put(
                    (identity, miss_reply(PeerMissReason.IDENTITY_MISMATCH))
                )
                self._lookup.wake()
                continue
            self.serve_peer_lookup(identity, request)

    def _flush_peer_lookup_replies(self) -> None:
        """Hand ready replies to the listener. Scheduler thread only."""
        replies = self.take_peer_lookup_replies()
        for identity, reply in replies:
            self._lookup.outbox.put((identity, reply))
        if replies:
            self._lookup.wake()

    def _get_lookup_client(self) -> PeerLookupClient:
        """The consumer-side lookup client, created on first use."""
        if self._lookup_client is None:
            self._lookup_client = PeerLookupClient(self._peer_lookup_timeout)
        return self._lookup_client

    def shutdown(self):
        if self._lookup_client is not None:
            self._lookup_client.close()
            self._lookup_client = None
        self._stop_event.set()
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join()
            self._nixl_handshake_listener_t = None
        self._lookup.close()

    def on_new_request(self, request: "Request") -> None:
        """Track a request that may need heartbeats."""
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_decode"):
            self._truncate_request_for_prefill(request)

        # NOTE (NickLucche) This excludes request meant for P, ie heartbeats are
        # effectively disabled for Bidirectional KV transfer.
        if params is None or not params.get("do_remote_prefill"):
            return
        self._register_heartbeat(request.request_id, params)

    def _register_heartbeat(self, req_id: ReqId, params: dict[str, Any]) -> None:
        """Start lease-renewal heartbeats for the remote blocks named in `params`."""
        # Only track if all required remote fields are present.
        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        dcp_size = params.get("dcp_size", 1)
        pp_size = params.get("pp_size", 1)
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return
        if remote_engine_id not in self._heartbeat_by_engine:
            self._heartbeat_by_engine[remote_engine_id] = HeartbeatInfo(
                req_ids=set(),
                host=host,
                port=port,
                tp_size=tp_size,
                dcp_size=dcp_size,
                pp_size=pp_size,
            )
        info = self._heartbeat_by_engine[remote_engine_id]
        info.req_ids.add(remote_request_id)
        expiry = params.get("remote_blocks_expiry_time")
        extension = params.get("remote_lease_extension")
        if expiry is not None and extension:
            info.leases[remote_request_id] = (expiry, extension)
        self._heartbeat_req_engine[req_id] = (
            remote_engine_id,
            remote_request_id,
        )

    def _stop_heartbeat(self, req_id: ReqId) -> None:
        """Remove *req_id* from heartbeat tracking (if tracked)."""
        if key := self._heartbeat_req_engine.pop(req_id, None):
            engine_id, remote_id = key
            if info := self._heartbeat_by_engine.get(engine_id):
                info.req_ids.discard(remote_id)
                info.leases.pop(remote_id, None)
                if not info.req_ids:
                    # Clean up empty engines so we don't leak a key when remote dies.
                    del self._heartbeat_by_engine[engine_id]

    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        """Clip a request's block lists down to the transferable blocks.

        Sliding-window groups keep only the in-window tail: the KV cache
        manager allocates blocks for the entire sequence length and cleans up
        out-of-window blocks only prior to the `request_finished_all_groups`
        hook.

        SSM groups keep only their state-bearing slot: the trailing
        speculative scratch slots and everything before the running state
        (null placeholders and the previous step's superseded state) go.

        Use this at every block-id exchange point. Pass `clip_ssm=False`
        for per-step partial lists (host-buffer save), where the SSM strip
        does not apply.
        """
        if len(block_ids) == 0:
            # No blocks to clip, e.g. a full prefix cache hit.
            return block_ids
        block_ids = self.kv_cache_config.select_transfer_block_ids(block_ids)
        if not self._is_hma_required:
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        clipped = []
        for i, blocks in enumerate(block_ids):
            if n_sw := self.blocks_per_sw[i]:
                blocks = blocks[-n_sw:]
            elif (
                clip_ssm
                and blocks
                and (n_spec_blocks := self._ssm_spec_blocks[i]) is not None
            ):
                blocks = clip_ssm_state_blocks(blocks, n_spec_blocks)
            clipped.append(blocks)
        return tuple(clipped)

    def set_xfer_handshake_metadata(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.

        """
        encoder = msgspec.msgpack.Encoder()
        for (pp_rank, tp_rank), rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            encoded = encoder.encode(rank_metadata)
            if self._handshake_data.get((pp_rank, tp_rank)) != encoded:
                # Peers must handshake again to load the changed registration.
                self._registration_generation += 1
            self._handshake_data[(pp_rank, tp_rank)] = encoded
            # All ranks share the compatibility hash.
            self._compatibility_hash = rank_metadata.compatibility_hash
            if self._peer_pins is not None:
                self._peer_pins.set_compatibility_hash(rank_metadata.compatibility_hash)
            logger.debug(
                "PP rank %d, TP rank %d: encoded NixlHandshakePayload size: %s bytes",
                pp_rank,
                tp_rank,
                str(len(self._handshake_data[(pp_rank, tp_rank)])),
            )

        # Only start the listener when we have metadata to serve.
        if self._nixl_handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    self._handshake_data,
                    ready_event,
                    self._stop_event,
                    self.side_channel_host,
                    self.side_channel_port,
                    self._lookup,
                ),
                daemon=True,
                name="nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[tuple[int, int], Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
        lookup: PeerLookupChannel | None = None,
    ):
        """Background thread for NIXL handshakes and peer prefix lookups.

        Two messages arrive on the ROUTER socket, both msgpack:

        - `(GET_META_MSG, pp_rank, tp_rank)`: answered with the handshake
          payload and this process's clock, unchanged.
        - `(LOOKUP_PIN_MSG, <encoded LookupPinRequest>)`: handed to the
          scheduler thread through `lookup.inbox`; the reply comes back
          through `lookup.outbox` and is sent from here as
          `(identity, b"", <encoded LookupPinReply>)`.

        This thread owns the socket and never touches the block pool or the
        pin server. A malformed or unknown message is logged and skipped: it
        must not end the thread.

        NIXL_CONNECTOR_VERSION is deliberately not bumped for the lookup
        message, since a bump would fail the handshake of every mixed-version
        pair, stock P/D included. Listeners are robust to unknown messages
        instead, and a lookup must only be sent to an engine that was started
        with `peer_lookup` (any other engine answers with a miss, or an old
        one not at all).
        """
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            if lookup is not None:
                poller.register(lookup.wake_r, zmq.POLLIN)
            ready_event.set()
            while not stop_event.is_set():
                try:
                    events = dict(poller.poll(_LISTENER_POLL_MS))
                except zmq.ZMQError:
                    if stop_event.is_set():
                        break
                    logger.warning("Connection listener poll failed", exc_info=True)
                    time.sleep(0.01)
                    continue
                if lookup is not None and lookup.wake_r in events:
                    lookup.drain_wake_pipe()
                if lookup is not None:
                    # Always, not only on a wake-up: a reply is never stranded.
                    NixlBaseConnectorScheduler._send_lookup_replies(sock, lookup)
                if sock not in events:
                    continue
                while True:
                    try:
                        frames = sock.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        NixlBaseConnectorScheduler._handle_listener_message(
                            sock, frames, encoded_data, lookup
                        )
                    except Exception:
                        logger.warning(
                            "Connection listener dropped a bad message", exc_info=True
                        )

    @staticmethod
    def _send_lookup_replies(sock: zmq.Socket, lookup: PeerLookupChannel) -> None:
        while True:
            try:
                identity, reply = lookup.outbox.get_nowait()
            except queue.Empty:
                return
            try:
                sock.send_multipart(
                    (identity, b"", msgspec.msgpack.encode(reply)), flags=zmq.NOBLOCK
                )
            except zmq.ZMQError:
                logger.warning("Could not send a peer lookup reply", exc_info=True)

    @staticmethod
    def _handle_listener_message(
        sock: zmq.Socket,
        frames: list[bytes],
        encoded_data: dict[tuple[int, int], Any],
        lookup: PeerLookupChannel | None,
    ) -> None:
        if len(frames) != 3:
            logger.warning("Connection listener got %d frames, expected 3", len(frames))
            return
        identity, _, raw = frames
        decoded = msgspec.msgpack.decode(raw)
        msg = decoded[0]
        if msg == GET_META_MSG:
            _, target_pp_rank, target_tp_rank = decoded
            logger.debug(
                "Received message for pp rank %s, tp rank %s",
                target_pp_rank,
                target_tp_rank,
            )
            # Echo our perf_counter so P can estimate the clock offset.
            # perf_counter is only comparable within a process, so this
            # listener must run in the same process that stamps the block
            # expiry deadline (`_reqs_need_send`).
            ts = msgspec.msgpack.encode(time.perf_counter())
            sock.send_multipart(
                (identity, b"", encoded_data[(target_pp_rank, target_tp_rank)], ts)
            )
        elif msg == LOOKUP_PIN_MSG:
            request = msgspec.msgpack.decode(decoded[1], type=LookupPinRequest)
            if lookup is None or not lookup.enabled:
                sock.send_multipart(
                    (
                        identity,
                        b"",
                        msgspec.msgpack.encode(
                            miss_reply(PeerMissReason.IDENTITY_MISMATCH)
                        ),
                    )
                )
                return
            lookup.inbox.put((identity, request))
            lookup.notify_engine()
        else:
            logger.warning("Connection listener got unexpected message %s", msg)

    def _prefill_backoff(self) -> int:
        """Trailing prompt tokens the prefiller must not compute; the decoder
        recomputes them locally.

        Mamba needs h(N-1) so the decoder can derive h(N) itself. DSV41's sliding
        window under bounded replay is not replayed after a P/D load, so it must
        be laid out for the token the decoder recomputes. Multi-module
        MTP needs to keep its whole lookahead window off of the prefiller, which
        would otherwise embed the unverified drafts in the MTP layer's KV cache. The
        decoder would never rebuild them, because the update is sized by the rejection
        count, which is zero for the first decode.
        """
        return max(
            1 if self._has_mamba or self._bounded_replay else 0,
            self.vllm_config.num_prefill_lookahead_tokens - 1,
        )

    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. The number of prompt tokens to load from the prefiller.
        Stops short of the trailing `_prefill_backoff()` tokens that the decoder
        will recompute locally."""
        backoff = self._prefill_backoff()
        if backoff and num_prompt_tokens > backoff:
            return num_prompt_tokens - backoff
        return num_prompt_tokens

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the trailing `_prefill_backoff()` prompt tokens
        so the prefiller stops short of what the decoder recomputes locally.
        For Mamba that is the single token needed to yield h(N-1); for
        multi-module MTP it is the drafter's whole lookahead window.

        Guarded by `_p_side_truncated` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        backoff = self._prefill_backoff()
        params = request.kv_transfer_params
        if (
            backoff
            and params is not None
            # Guard against repeated truncation after preemption/reschedule.
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > backoff
            # A request that skips reading external KV (prompt logprobs)
            # loads nothing on the decoder, which recomputes past its local hit.
            and not request.skip_reading_external_kv
        ):
            if request.prompt_token_ids is not None:
                del request.prompt_token_ids[-backoff:]
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-backoff]
            else:
                return

            del request._all_token_ids[-backoff:]
            request.num_prompt_tokens -= backoff
            request.max_tokens = 1
            hash_block_size = self.kv_cache_config.hash_block_size
            assert hash_block_size is not None
            request.truncate_block_hashes(
                hash_block_size, self.vllm_config.num_hash_lookahead_tokens
            )
            params["_p_side_truncated"] = True

    def _build_save_meta(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # only called when use_host_buffer is True to build the save metadata

        # NOTE: For the prefill side, there might be a chance that an early added
        # request is a chunked prefill, so we need to check if new blocks are added
        for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
            req_to_save = self._reqs_need_save.get(req_id)
            if req_to_save is None or new_block_id_groups is None:
                continue
            req = req_to_save

            assert req.kv_transfer_params is not None
            clipped_block_id_groups = self.get_exchange_clipped_blocks(
                new_block_id_groups, clip_ssm=False
            )
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=clipped_block_id_groups,
                kv_transfer_params=req.kv_transfer_params,
            )
            assert scheduler_output.num_scheduled_tokens is not None
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            is_partial = (
                req.num_computed_tokens + num_scheduled_tokens
            ) < req.num_prompt_tokens
            if not is_partial:
                # For non-partial prefills, once new req_meta is scheduled, it
                # can be removed from _reqs_need_save.
                # For partial prefill case, we will retain the request in
                # _reqs_need_save until all blocks are scheduled with req_meta.
                # Therefore, only pop if `not is_partial`.
                self._reqs_need_save.pop(req_id)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()
        # Serve before the step is counted: the allocations of the schedule
        # that is calling us already happened, and a fence of this step covers
        # the blocks it just cached.
        self._drain_peer_lookups()
        if self._peer_pins is not None:
            self._peer_pins.on_step_scheduled()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (
            req,
            block_ids,
            cached,
            awaiting_kvs,
        ) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                local_num_computed_blocks=cached,
                awaiting_kvs=awaiting_kvs,
            )

        self._release_orphaned_pins(meta)

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        if self._peer_pins is not None:
            # Ship pins together with the step that carries their reply, as a
            # finished prefill's send; the worker only tracks ids in both.
            new_sends = self._peer_pins.take_new_sends()
            self._reqs_need_send.update(new_sends)
            self._reqs_in_batch.update(new_sends)
        # After the ships above, so a reply leaves in the step that carries its
        # pin's id to the worker.
        self._flush_peer_lookup_replies()

        meta.reqs_to_send = self._reqs_need_send
        # Clock reference for reqs_to_send: deadlines above are in this
        # process's perf_counter domain; workers (possibly on other nodes,
        # where perf_counter has a different epoch) rebase against this.
        meta.scheduler_clock = time.perf_counter()
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        # Package heartbeats, throttled by heartbeat_interval.
        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta

    def _release_orphaned_pins(self, meta: NixlConnectorMetadata) -> None:
        """Free the pins of lookups that were cancelled before their hit arrived.

        Each is released with a notify-only recv, the same message a consumer
        sends when its local prefix hit already covers the remote blocks: the
        worker handshakes with the peer if needed and sends `"<pin_id>:<n>"`.
        It never reports the recv back, because it has neither local blocks nor
        `awaiting_kvs` and so no entry in the worker's recv metadata. A pin
        that cannot be released this way is left to its lease.
        """
        if self._lookup_client is None:
            return
        stats = self.peer_lookup_stats
        for host, port, reply in self._lookup_client.take_orphaned_pins():
            endpoint = self._peer_endpoints.get((host, port))
            if endpoint is None or endpoint.tp_size is None:
                # Without the peer's TP size a handshake could be built wrong.
                stats.orphans_unreleased += 1
                logger.warning(
                    "Peer pin %s on %s:%s is orphaned and cannot be released "
                    "(peer topology unknown); it expires with its lease.",
                    reply.remote_request_id,
                    host,
                    port,
                )
                continue
            stats.orphans_released += 1
            meta.add_new_req_to_recv(
                request_id=f"peer-release-{uuid.uuid4().hex}",
                local_block_ids=[],
                kv_transfer_params=dict(
                    remote_block_ids=reply.remote_block_ids,
                    remote_engine_id=endpoint.engine_id,
                    remote_request_id=reply.remote_request_id,
                    remote_host=host,
                    remote_port=port,
                    tp_size=endpoint.tp_size,
                    dcp_size=endpoint.dcp_size,
                    pp_size=endpoint.pp_size,
                ),
                awaiting_kvs=False,
            )

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Stop heartbeating for requests whose KV transfer completed."""
        if self._peer_pins is not None:
            # Pin ids are not requests: keep them from the scheduler's loop.
            if connector_output.finished_sending:
                connector_output.finished_sending = self._peer_pins.on_finished_sending(
                    set(connector_output.finished_sending)
                )
            self._peer_pins.on_invalid_blocks(connector_output.invalid_block_ids)
        for req_id in connector_output.finished_recving or ():
            self._stop_heartbeat(req_id)

    def has_pending_push_work(self) -> bool:
        # A queued lookup, a held pin or an unsent reply all need further steps
        # (and so an idle engine must not sleep on them).
        return (
            not self._lookup.inbox.empty()
            or not self._lookup.outbox.empty()
            or (self._peer_pins is not None and self._peer_pins.has_pending_work())
            or (self._lookup_client is not None and self._lookup_client.has_in_flight())
        )

    ############################################################
    # Abstract methods that subclasses must implement
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        raise NotImplementedError

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        raise NotImplementedError

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError
