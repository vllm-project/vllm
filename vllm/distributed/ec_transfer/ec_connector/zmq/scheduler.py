# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side of the ECZmqConnector.

Consumer: tracks which embeddings the workers have received (reported through
`ECZmqWorkerMetadata`) and holds back requests whose embeddings are still in
flight, so the engine never falls back to encoding an item locally that a
producer is about to deliver.

Producer: resolves where each freshly scheduled encoder input must be pushed,
and keeps the engine stepping until those pushes complete.
"""

import time
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.zmq.common import (
    ECZmqConnectorMetadata,
    ECZmqOptions,
    ECZmqWorkerMetadata,
    ZmqDst,
    parse_zmq_options,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ECConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECZmqScheduler:
    """Scheduler delegate for the ECZmqConnector."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._options: ECZmqOptions = parse_zmq_options(vllm_config)
        self._is_producer = ec_config.is_ec_producer
        self._is_consumer = ec_config.is_ec_consumer

        # Consumer: mm_hashes every rank has staged, ready to be loaded.
        self._ready: set[str] = set()
        # Consumer: mm_hash -> number of ranks that reported it so far.
        self._arrivals: dict[str, int] = {}
        # Consumer: mm_hash -> deadline after which we stop waiting for it.
        self._expected: dict[str, float] = {}
        # Consumer: loads to hand to the workers on the next step.
        self._pending_loads: list[str] = []

        # Producer: mm_hash -> destinations for the next step.
        self._pending_sends: dict[str, list[ZmqDst]] = {}
        # Producer: mm_hashes handed to the workers whose delivery has not
        # been confirmed yet.
        self._inflight_sends: set[str] = set()

    # ==============================
    # Consumer
    # ==============================

    def has_cache_item(self, identifier: str) -> bool:
        return self._is_consumer and identifier in self._ready

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """Hold back a request whose embeddings have not arrived yet.

        Waiting is bounded: once the deadline passes the request is released so
        it can fail (or encode locally) rather than block the queue forever.
        """
        if not self._is_consumer:
            return True

        self.expect_items(self._remote_items(request))

        now = time.monotonic()
        for feature in request.mm_features:
            mm_hash = feature.identifier
            if mm_hash in self._ready:
                continue
            deadline = self._expected.get(mm_hash)
            if deadline is None:
                continue
            if now < deadline:
                return False
            del self._expected[mm_hash]
            logger.error(
                "EC ZMQ: gave up waiting %.0fs for the embedding of mm_hash "
                "%s (request %s)",
                self._options.recv_timeout_s,
                mm_hash,
                request.request_id,
            )
        return True

    def expect_items(self, mm_hashes: list[str]) -> None:
        """Declare embeddings that a producer is going to push.

        Phase one learns this from the request's `ec_transfer_params`; an
        in-engine dispatcher can call this directly once it owns the encode
        request.
        """
        if not mm_hashes:
            return
        deadline = time.monotonic() + self._options.recv_timeout_s
        for mm_hash in mm_hashes:
            if mm_hash not in self._ready:
                self._expected.setdefault(mm_hash, deadline)

    def _remote_items(self, request: "Request") -> list[str]:
        """The request's mm_hashes whose embeddings come from a producer."""
        if self._options.wait_for_all_remote:
            return [feature.identifier for feature in request.mm_features]

        params = request.ec_transfer_params
        if not params:
            return []
        items = params.get("ec_items") or []
        return [
            mm_hash
            for item in items
            if isinstance(item, dict) and (mm_hash := item.get("mm_hash"))
        ]

    def update_connector_output(self, connector_output: "ECConnectorOutput") -> None:
        """Fold the worker reports of this step into the connector state."""
        meta = connector_output.ec_connector_worker_meta
        if isinstance(meta, ECZmqWorkerMetadata):
            for mm_hash, count in meta.staged.items():
                arrived = self._arrivals.get(mm_hash, 0) + count
                if arrived < self._options.num_recv_ranks:
                    self._arrivals[mm_hash] = arrived
                    continue
                # Every rank has the embedding in hand.
                self._arrivals.pop(mm_hash, None)
                self._expected.pop(mm_hash, None)
                self._ready.add(mm_hash)

        if connector_output.finished_sending:
            self._inflight_sends -= connector_output.finished_sending

    # ==============================
    # Producer
    # ==============================

    def _destinations(self, request: "Request") -> list[ZmqDst]:
        """Where this request's embeddings must be pushed.

        A request may name its own destination, which is what lets one encoder
        fleet serve several consumers; otherwise the statically configured
        consumers are used.
        """
        params = request.ec_transfer_params or {}
        raw_dst = params.get("ec_dst")
        if not raw_dst:
            return list(self._options.consumers)
        raw_dsts = raw_dst if isinstance(raw_dst, list) else [raw_dst]
        try:
            return [ZmqDst.from_dict(raw) for raw in raw_dsts]
        except ValueError:
            logger.exception(
                "EC ZMQ: ignoring the malformed ec_dst of request %s",
                request.request_id,
            )
            return list(self._options.consumers)

    # ==============================
    # Shared
    # ==============================

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash = request.mm_features[index].identifier

        if self._is_consumer and mm_hash in self._ready:
            # The workers hand the embedding over exactly once, so drop it from
            # the ready set as soon as the load is scheduled.
            self._ready.discard(mm_hash)
            if mm_hash not in self._pending_loads:
                self._pending_loads.append(mm_hash)
            return

        if self._is_producer and mm_hash not in self._pending_sends:
            dsts = self._destinations(request)
            if not dsts:
                return
            self._pending_sends[mm_hash] = dsts
            self._inflight_sends.add(mm_hash)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECZmqConnectorMetadata:
        meta = ECZmqConnectorMetadata(
            sends=self._pending_sends, loads=self._pending_loads
        )
        self._pending_sends = {}
        self._pending_loads = []
        return meta

    def has_pending_push_work(self) -> bool:
        return bool(self._inflight_sends) or bool(self._pending_sends)

    def shutdown(self) -> None:
        self._ready.clear()
        self._arrivals.clear()
        self._expected.clear()
        self._pending_loads.clear()
        self._pending_sends.clear()
        self._inflight_sends.clear()
