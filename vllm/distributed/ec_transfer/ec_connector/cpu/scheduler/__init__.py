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
    ECCPUWorkerMetadata,
    _get_encoder_cache_hidden_dim,
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)
from vllm.distributed.ec_transfer.ec_connector.utils import (
    PlaceholderMetadataResolver,
    collect_ec_item_metadata,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ECConnectorOutput
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
        self._metadata_resolver = PlaceholderMetadataResolver(vllm_config.model_config)

        # mm_hash → block IDs allocated this step for GPU→mmap saves.
        self._pending_saves: dict[str, list[int]] = {}
        # mm_hash → (transfer_id, block IDs) to load from mmap→GPU this step.
        self._pending_loads: dict[str, tuple[int, list[int]]] = {}

        # Dispatched loads awaiting completion reports, keyed by transfer id:
        # transfer_id → (mm_hash, reports still outstanding). The pin taken at
        # dispatch is released only once the count reaches zero.
        self._load_acks: dict[int, tuple[str, int]] = {}
        self._next_transfer_id = 0

        pc = vllm_config.parallel_config
        # Reports to expect per load. Only pipeline stage 0 runs the EC
        # connector, so the participants are the tp × pcp ranks of that stage.
        # Other executors deliver a single rank's output, and expecting reports
        # that cannot arrive would hold the pin forever.
        self._expected_load_reports = (
            pc.tensor_parallel_size * pc.prefill_context_parallel_size
            if pc.distributed_executor_backend == "mp"
            else 1
        )

        self._ec_config = ec_config
        # NIXL p2p is an option of this connector, not engine-wide behavior, so
        # it lives in extra config. Extra config is not type coerced, so a value
        # that arrived as a JSON/CLI string is parsed, not truth-tested.
        raw_nixl = ec_config.get_from_extra_config("ec_enable_nixl", False)
        self._nixl_enabled = (
            raw_nixl
            if isinstance(raw_nixl, bool)
            else str(raw_nixl).strip().lower() in ("true", "1", "yes")
        )
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
        self._ack_timeout_s: float = 0.0
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
                "ec_enable_nixl requires NIXL; install the `nixl` package or "
                "remove ec_enable_nixl from ec_connector_extra_config."
            )
        engine_id = self._ec_config.engine_id
        assert engine_id is not None
        self._dtype = vllm_config.model_config.dtype
        self._hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
        self._element_size = torch.empty(0, dtype=self._dtype).element_size()
        # How long a consumer waits for an XferAck. The producer answers from
        # its scheduler step, so its reply latency scales with the encoder's
        # batch size: a deployment whose steps run longer must raise this.
        self._ack_timeout_s = float(
            self._ec_config.get_from_extra_config("consumer_ack_timeout_s", 2.0)
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

        # Registering the region with NIXL and binding the control sockets are
        # the first side effects here. __init__ propagates a failure, so the
        # caller never receives a scheduler it could shut down: unwind through
        # the same teardown shutdown() uses.
        try:
            self._data = NixlDataTransport(
                agent_name=engine_id,
                base_ptr=self._region.blocks.data_ptr(),
                num_blocks=self._region.num_blocks,
                block_size_bytes=self._region.block_size_bytes,
                total_size_bytes=self._region.num_blocks
                * self._region.block_size_bytes,
            )
            if self._is_producer:
                assert self._peer_host is not None
                assert self._peer_port is not None
                self._producer_session = ProducerSession(
                    transport=ZmqServerTransport(
                        host=self._peer_host, port=self._peer_port
                    ),
                    data=self._data,
                    cache=self._cache,
                    compat_hash=self._compat_hash,
                )
            if self._is_consumer:
                self._transport = ZmqClientTransport()
        except Exception:
            self._teardown_nixl()
            raise

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
            # Absent entirely, or present with only placeholder metadata (no
            # cache entry existed on the producer side at request_finished
            # time): nothing to fetch, fall back to local compute.
            if not info or "peer_host" not in info:
                continue
            expected = pos.length * self._hidden_dim * self._element_size
            if int(info.get("size_bytes", -1)) != expected:
                logger.warning(
                    "EC consumer: size mismatch mm_hash=%s; local encode", mm_hash
                )
                continue
            try:
                started = self._start_xfer(mm_hash, info, expected)
            except Exception:
                logger.exception(
                    "EC consumer: failed to start NIXL xfer mm_hash=%s", mm_hash
                )
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
                "EC consumer: cache full for mm_hash=%s (%d blocks); local encode",
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
        deadline = time.monotonic() + self._ack_timeout_s
        try:
            self._sessions[addr].start_xfer(mm_hash, indices, deadline)
        except Exception:
            self._cache.discard(mm_hash)
            raise
        logger.debug(
            "EC consumer: starting NIXL xfer mm_hash=%s from %s:%d blocks=%d",
            mm_hash,
            addr[0],
            addr[1],
            n_blocks,
        )
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
            logger.debug("EC consumer: NIXL xfer complete mm_hash=%s", mm_hash)
        for mm_hash in r.tombstoned:
            self._in_flight.discard(mm_hash)
            self._cache.discard(mm_hash)
            self._tombstones.add(mm_hash)
            logger.debug("EC consumer: NIXL xfer failed mm_hash=%s", mm_hash)
        for mm_hash in r.quarantined:
            # DMA still running: keep the blocks reserved (entry stays
            # not-ready, hence non-evictable) until the xfer settles.
            self._in_flight.discard(mm_hash)
            self._tombstones.add(mm_hash)
            logger.debug(
                "EC consumer: NIXL xfer mm_hash=%s quarantined; DMA still running",
                mm_hash,
            )
        for mm_hash in r.retryable:
            # No tombstone: dropping every trace of the attempt is what lets
            # the next admit pass re-issue the read as if it were the first.
            self._in_flight.discard(mm_hash)
            self._cache.discard(mm_hash)
            logger.debug(
                "EC consumer: NIXL xfer mm_hash=%s retryable; will re-request",
                mm_hash,
            )
        for mm_hash, _block_indices in r.settled:
            self._cache.discard(mm_hash)
            logger.debug(
                "EC consumer: quarantined NIXL xfer mm_hash=%s settled", mm_hash
            )

    def _on_peer_down(self, addr: Any) -> None:
        session = self._sessions.pop(addr, None)
        if session is None:
            return
        session.on_peer_down()
        self._process_session_results(session)
        session.close()
        logger.info("EC consumer: producer peer down addr=%s", addr)

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

        if self._is_consumer and mm_hash not in self._pending_loads:
            entry = self._cache.get(mm_hash)
            if entry is not None and entry.ready:
                self._cache.pin(mm_hash)
                transfer_id = self._next_transfer_id
                self._next_transfer_id += 1
                self._pending_loads[mm_hash] = (transfer_id, list(entry.block_ids))
                self._load_acks[transfer_id] = (mm_hash, self._expected_load_reports)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        meta = ECCPUConnectorMetadata()
        if self._is_producer:
            if self._nixl_enabled and self._producer_session is not None:
                self._producer_session.poll_step()
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

    def update_connector_output(self, connector_output: "ECConnectorOutput") -> None:
        """Apply the worker's memcpy-completion report to the cache.

        Completed saves become safe to mark ready. A load is unpinned once
        every participating rank has reported its transfer id; reports for a
        transfer that has already been released, or for one this scheduler
        never dispatched, are ignored.
        """
        meta = connector_output.ec_connector_worker_meta
        if not isinstance(meta, ECCPUWorkerMetadata):
            return
        for mm_hash in meta.completed_saves:
            entry = self._cache.get(mm_hash)
            if entry is None:
                logger.debug(
                    "EC producer: worker reported completed save for unknown "
                    "mm_hash=%s (already discarded/evicted?)",
                    mm_hash,
                )
            elif not entry.ready:
                self._cache.mark_ready(mm_hash)
                logger.debug("EC producer: mm_hash=%s marked ready", mm_hash)
        for transfer_id in meta.completed_loads:
            pending = self._load_acks.get(transfer_id)
            if pending is None:
                continue
            mm_hash, outstanding = pending
            if outstanding > 1:
                self._load_acks[transfer_id] = (mm_hash, outstanding - 1)
                continue
            # Drop the entry before unpinning so a replayed report is treated
            # as stale rather than releasing the pin a second time.
            del self._load_acks[transfer_id]
            self._cache.unpin(mm_hash)
            logger.debug("EC consumer: mm_hash=%s unpinned after load", mm_hash)

    def has_pending_push_work(self) -> bool:
        """Keep the engine stepping so this connector's polls keep running.

        With NIXL enabled the engine tick is the only driver of
        ``ProducerSession.poll_step()``, and a producer's work arrives from a
        remote consumer, so nothing local would otherwise wake it. Without NIXL
        this only has to outlive dispatched saves and loads, so the engine can
        quiesce once the worker has confirmed them.
        """
        return self._nixl_enabled or self._cache.has_held_entries()

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, "dict[str, Any] | None"]:
        if not (self._nixl_enabled and self._is_producer):
            return False, None
        if not request.mm_features:
            return False, None

        items = collect_ec_item_metadata(request.mm_features, self._metadata_resolver)

        for feature in request.mm_features:
            mm_hash = feature.identifier
            entry = self._cache.get(mm_hash)
            if entry is None:
                # Never saved, or evicted since. Publishing placeholder
                # metadata without an address invites an orchestrator to
                # rewrite the media into a reference to an encoding no
                # consumer can fetch, leaving the decoder nothing to embed.
                # Publish neither, so the media stays on the request.
                items[mm_hash]["metadata"] = {}
                logger.debug(
                    "EC producer: mm_hash=%s absent at request_finished; "
                    "announcing no metadata so the media is not rewritten away",
                    mm_hash,
                )
                continue
            # Announce even if the save's GPU->mmap copy hasn't been
            # confirmed complete yet: a not-ready entry can't be evicted, so
            # it will still be here by the time a consumer's XferReq arrives.
            # A read arriving before the save lands is NACKed NACK_NOT_READY,
            # which the consumer retries on a later step rather than treating
            # as a miss.
            size_bytes = (
                feature.mm_position.length * self._hidden_dim * self._element_size
            )
            items[mm_hash].update(
                peer_host=self._peer_host,
                peer_port=self._peer_port,
                size_bytes=size_bytes,
            )
        logger.debug(
            "EC producer: announcing NIXL-readable encodings req_id=%s items=%s",
            request.request_id,
            items,
        )
        return False, items

    def shutdown(self) -> None:
        self._pending_saves.clear()
        self._pending_loads.clear()
        self._load_acks.clear()

        self._is_producer = False
        self._is_consumer = False

        if self._nixl_enabled:
            self._teardown_nixl()

        try:
            self._region.cleanup()
        except Exception:
            logger.debug("ec: region cleanup failed", exc_info=True)

    def _teardown_nixl(self) -> None:
        """Close the control sockets and release the NIXL agent.

        Guards each step and clears what it releases, so it is safe on a
        scheduler whose NIXL setup failed part-way and safe to call twice.
        The shared region is not touched — shutdown() owns that.
        """
        if self._producer_session is not None:
            try:
                self._producer_session.close()
            except Exception:
                logger.debug("ec: producer session close failed", exc_info=True)
            self._producer_session = None

        for session in list(self._sessions.values()):
            try:
                session.close()
            except Exception:
                logger.debug("ec: consumer session close failed", exc_info=True)
        self._sessions.clear()

        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:
                logger.debug("ec: client transport close failed", exc_info=True)
            self._transport = None

        if self._data is not None:
            try:
                self._data.deregister()
            except Exception:
                logger.debug("ec: deregister failed", exc_info=True)
            self._data = None
