# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECCPUScheduler — CPU offload scheduler delegate.

Owns the mmap region and the embedding cache, and handles the producer
(GPU->CPU offload) and consumer (CPU->GPU reload) scheduler-side logic
for the ECCPUConnector.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    _get_encoder_cache_hidden_dim,
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.step_tracker import (
    StepTracker,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECCPUScheduler:
    """Scheduler delegate for the ECCPUConnector."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._is_producer: bool = ec_config.is_ec_producer
        self._is_consumer: bool = ec_config.is_ec_consumer

        self._region = create_ec_shared_region(vllm_config)
        # Block allocator + LRU eviction policy for the shared region.
        self._cache = EmbeddingCache(self._region.num_blocks)

        max_batches = vllm_config.max_concurrent_batches
        # Delays mark_ready until the GPU→mmap DMA is guaranteed complete.
        self._ready_tracker = StepTracker(max_batches)
        # Delays unpin until the mmap→GPU DMA is guaranteed complete.
        self._unpin_tracker = StepTracker(max_batches)

        # mm_hash → block IDs allocated this step for GPU→mmap saves.
        self._pending_saves: dict[str, list[int]] = {}
        # mm_hash → block IDs to load from mmap→GPU this step.
        self._pending_loads: dict[str, list[int]] = {}

        self._ec_config = ec_config
        self._nixl_enabled = bool(getattr(ec_config, "ec_enable_nixl", False))
        # NIXL fields default to None/empty so the gate-off path is untouched.
        self._data: Any = None
        self._compat_hash: str | None = None
        self._first_in_batch = True
        self._transport: Any = None
        self._producer_session: Any = None
        self._sessions: dict = {}
        self._in_flight: set[str] = set()
        self._tombstones: set[str] = set()
        self._step_completed: set[str] = set()
        self._peer_host: str | None = None
        self._peer_port: int | None = None
        # Model shape for size checks + compat hash; only set by
        # _setup_nixl (the gate-off path never touches model_config).
        self._dtype: torch.dtype | None = None
        self._hidden_dim: int = 0
        self._element_size: int = 0
        if self._nixl_enabled:
            self._setup_nixl(vllm_config)

    def _setup_nixl(self, vllm_config: "VllmConfig") -> None:
        # Lazy imports keep nixl/zmq off the gate-off path.
        from vllm import envs
        from vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq import (
            ZmqClientTransport,
            ZmqServerTransport,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl import (
            NixlDataTransport,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.protocol import (
            compute_ec_compatibility_hash,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.session import (
            ProducerSession,
        )
        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
        from vllm.version import __version__ as VLLM_VERSION

        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError(
                "ec_enable_nixl=True requires NIXL; install the `nixl` package "
                "or set ec_enable_nixl=False."
            )
        self._dtype = vllm_config.model_config.dtype
        self._hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
        self._element_size = torch.empty(0, dtype=self._dtype).element_size()
        engine_id = self._ec_config.engine_id
        assert engine_id is not None
        self._data = NixlDataTransport(
            agent_name=engine_id,
            base_ptr=self._region.blocks.data_ptr(),
            num_blocks=self._region.num_blocks,
            block_size_bytes=self._region.block_size_bytes,
            total_size_bytes=self._region.num_blocks * self._region.block_size_bytes,
        )
        self._compat_hash = compute_ec_compatibility_hash(
            vllm_version=VLLM_VERSION,
            model=str(vllm_config.model_config.model),
            dtype=str(self._dtype),
            block_size_bytes=self._region.block_size_bytes,
        )
        if self._is_producer:
            self._peer_host = envs.VLLM_EC_SIDE_CHANNEL_HOST
            self._peer_port = envs.VLLM_EC_SIDE_CHANNEL_PORT
            self._producer_session = ProducerSession(
                transport=ZmqServerTransport(
                    host=self._peer_host, port=self._peer_port
                ),
                data=self._data,
                cache=self._cache,
                compat_hash=self._compat_hash,
            )
            self._producer_session.start()
        if self._is_consumer:
            self._transport = ZmqClientTransport()

    def has_cache_item(self, identifier: str) -> bool:
        if not self._is_consumer:
            return False
        entry = self._cache.get(identifier)
        return entry is not None and entry.ready

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        if not self._nixl_enabled:
            return True  # CPU offload never blocks.
        first = self._first_in_batch
        self._first_in_batch = False
        if not self._is_consumer:
            return True
        if first:
            self._poll_step()
        return self._nixl_consumer_admit(request, num_computed_tokens)

    def _nixl_consumer_admit(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """Admit a request only once all its remote encodings are cached.

        Returns True when every encoding the request needs is either already
        ready in the local cache or has no remote source. Any encoding whose
        NIXL READ is still in flight (or was just started this step) defers
        the request, which the scheduler re-presents on a later step.
        """
        params: dict[str, dict[str, Any]] = (
            getattr(request, "ec_transfer_params", None) or {}
        )
        if not params:
            return True
        pending = False
        for feature in request.mm_features:
            pos = feature.mm_position
            if pos.offset + pos.length <= num_computed_tokens:
                continue
            mm_hash = feature.identifier
            entry = self._cache.get(mm_hash)
            if entry is not None and entry.ready:
                # Local hit: upstream's update_state_after_alloc pins and
                # loads it through the same path as a natively cached entry.
                continue
            if mm_hash in self._in_flight:
                pending = True
                continue
            if mm_hash in self._step_completed:
                pending = True
                continue
            if mm_hash in self._tombstones:
                self._tombstones.discard(mm_hash)
                continue
            if entry is not None:
                # Present but not ready and not being fetched: its blocks are
                # held by a quarantined/settling DMA and cannot be reused.
                # Fall back to local compute this step; never re-alloc a
                # mm_hash already in the cache.
                continue
            info = params.get(mm_hash)
            if info is None:
                continue
            expected = pos.length * self._hidden_dim * self._element_size
            if int(info.get("size_bytes", -1)) != expected:
                logger.warning("EC: size mismatch mm_hash=%s; local encode", mm_hash)
                continue
            try:
                started = self._start_xfer(mm_hash, info, expected)
            except Exception:
                logger.exception("EC: start xfer failed mm_hash=%s", mm_hash)
                continue
            if not started:
                continue
            self._in_flight.add(mm_hash)
            pending = True
        return not pending

    def _start_xfer(
        self, mm_hash: str, info: "dict[str, Any]", size_bytes: int
    ) -> bool:
        """Allocate a not-ready cache entry and start a NIXL READ into it.

        Returns True if the transfer was started. Returns False when the
        cache cannot accommodate the encoding, in which case the request
        falls back to local recomputation.
        """
        import time
        from math import ceil

        from vllm.distributed.ec_transfer.ec_connector.cpu.session import (
            ConsumerSession,
        )

        n_blocks = max(1, ceil(size_bytes / self._region.block_size_bytes))
        entry = self._cache.alloc(mm_hash, n_blocks)
        if entry is None:
            logger.debug(
                "EC: cache full for mm_hash=%s (%d blocks); local encode",
                mm_hash,
                n_blocks,
            )
            return False
        indices = list(entry.block_ids)
        addr = (info["peer_host"], int(info["peer_port"]))
        if addr not in self._sessions:
            zmq_conn = self._transport.connect(addr)
            assert self._compat_hash is not None
            self._sessions[addr] = ConsumerSession(
                addr=addr,
                zmq_conn=zmq_conn,
                transport=self._transport,
                data=self._data,
                compat_hash=self._compat_hash,
            )
        deadline = time.monotonic() + 2.0  # CONSUMER_XFER_ACK_TIMEOUT_S
        try:
            self._sessions[addr].start_xfer(mm_hash, indices, deadline)
        except Exception:
            self._cache.discard(mm_hash)
            raise
        return True

    def _poll_step(self) -> None:
        import time

        now = time.monotonic()
        all_messages = self._transport.poll()
        for addr, session in list(self._sessions.items()):
            session.poll(all_messages.get(addr, []), now)
        for addr in self._transport.poll_dead():
            self._on_peer_down(addr)
        for session in self._sessions.values():
            self._process_session_results(session)

    def _process_session_results(self, session: Any) -> None:
        r = session.take_results()
        for mm_hash in r.completed:
            self._in_flight.discard(mm_hash)
            self._cache.mark_ready(mm_hash)
            self._step_completed.add(mm_hash)
        for mm_hash in r.tombstoned:
            self._in_flight.discard(mm_hash)
            self._cache.discard(mm_hash)
            self._tombstones.add(mm_hash)
        for mm_hash in r.quarantined:
            # DMA still running: keep the blocks reserved (entry stays
            # not-ready, hence non-evictable) until the xfer settles.
            self._in_flight.discard(mm_hash)
            self._tombstones.add(mm_hash)
        for mm_hash in r.cancelled:
            self._in_flight.discard(mm_hash)
            self._cache.discard(mm_hash)
        for mm_hash, _block_indices in r.settled:
            self._cache.discard(mm_hash)

    def _on_peer_down(self, addr: Any) -> None:
        session = self._sessions.pop(addr, None)
        if session is None:
            return
        session.on_peer_down()
        self._process_session_results(session)
        session.close()
        logger.info("EC: peer down addr=%s", addr)

    def _promote_completed_reads(self) -> None:
        """Drain the just-completed set built during ``_poll_step``.

        Reads are marked ready in ``_process_session_results``, so a deferred
        request is re-admitted by ``ensure_cache_available`` and loaded through
        the same local path as a natively cached entry
        (``update_state_after_alloc`` -> ``_pending_loads``). No explicit
        pin/load bookkeeping is needed here beyond clearing the set.
        """
        self._step_completed.clear()

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        feature = request.mm_features[index]
        mm_hash = feature.identifier

        if self._is_producer and self._cache.get(mm_hash) is None:
            entry = self._cache.alloc(mm_hash, feature.mm_position.length)
            if entry is not None:
                self._pending_saves[mm_hash] = list(entry.block_ids)
                self._ready_tracker.add(mm_hash, request.request_id)

        if self._is_consumer and mm_hash not in self._pending_loads:
            entry = self._cache.get(mm_hash)
            if entry is not None and entry.ready:
                self._cache.pin(mm_hash)
                self._pending_loads[mm_hash] = list(entry.block_ids)
                self._unpin_tracker.add(mm_hash, request.request_id)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        finished = (
            scheduler_output.finished_req_ids if scheduler_output is not None else set()
        )

        for key in self._ready_tracker.step(finished):
            entry = self._cache.get(key)
            if entry is not None and not entry.ready:
                self._cache.mark_ready(key)

        for key in self._unpin_tracker.step(finished):
            self._cache.unpin(key)

        meta = ECCPUConnectorMetadata()
        if self._is_producer:
            meta.saves = self._pending_saves
            self._pending_saves = {}
        if self._is_consumer:
            if self._nixl_enabled:
                self._promote_completed_reads()
            meta.loads = self._pending_loads
            self._pending_loads = {}
        if self._nixl_enabled:
            self._first_in_batch = True
        return meta

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, "dict[str, Any] | None"]:
        if not (self._nixl_enabled and self._is_producer):
            return False, None
        params: dict[str, dict[str, Any]] = {}
        for feature in request.mm_features:
            mm_hash = feature.identifier
            entry = self._cache.get(mm_hash)
            if entry is None or not entry.ready:
                continue
            size_bytes = (
                feature.mm_position.length * self._hidden_dim * self._element_size
            )
            params[mm_hash] = {
                "peer_host": self._peer_host,
                "peer_port": self._peer_port,
                "size_bytes": size_bytes,
            }
        logger.debug(
            "EC: request_finished req_id=%s params=%s", request.request_id, params
        )
        return False, (params or None)

    def shutdown(self) -> None:
        # drain_all() covers both entries still in _current (never
        # consumed by build_connector_meta) and entries in slots.
        self._pending_loads.clear()
        for mm_hash in self._unpin_tracker.drain_all():
            self._cache.unpin(mm_hash)
        self._ready_tracker.drain_all()

        self._is_producer = False
        self._is_consumer = False

        if self._nixl_enabled:
            if self._producer_session is not None:
                self._producer_session.stop()
            self._shutdown_nixl_consumer()
            if self._data is not None:
                try:
                    self._data.deregister()
                except Exception:
                    logger.debug("ec: deregister failed", exc_info=True)

        try:
            self._region.cleanup()
        except Exception:
            logger.debug("ec: region cleanup failed", exc_info=True)

    def _shutdown_nixl_consumer(self) -> None:
        for session in list(self._sessions.values()):
            session.close()
        self._sessions.clear()
        if self._transport is not None:
            self._transport.close()
