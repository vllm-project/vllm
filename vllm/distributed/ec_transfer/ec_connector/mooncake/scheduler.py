# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side planning and observation for Mooncake cache transfers.

The Scheduler prepares Producer reservations, consumes Consumer readiness
events, emits per-step Worker metadata, and converts Worker results back into
request availability without owning tensor memory or running data transfers.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    _get_encoder_cache_hidden_dim,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.config import (
    _RESERVATION_TTL_SECONDS,
    MooncakeECConfig,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.control import (
    ControlClient,
    EventInbox,
    make_cancel_request,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.state import (
    SchedulerTransferState,
    SchedulerTransferTable,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    ensure_mooncake_available,
)
from vllm.distributed.ec_transfer.ec_connector.utils import (
    PlaceholderMetadataResolver,
    collect_ec_item_metadata,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Unmatched `ec_items` entries are a protocol desync, not a per-request
# event: warn on the first few, then only once in a while.
_MAX_UNRESOLVED_TRANSFER_ID_WARNINGS = 5

_DRAIN_MIN_INTERVAL = 0.005
_MAX_PENDING_EVENTS = 4096
_MAX_TERMINAL_TRANSFER_RECORDS = 1 << 16
_CANCEL_ATTEMPTS = 2
_CONTROL_WORKERS = 8


class ECMooncakeScheduler:
    """Coordinate Mooncake transfers from the vLLM Scheduler process."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        ensure_mooncake_available()
        config = MooncakeECConfig.from_vllm_config(vllm_config)

        self._is_producer = config.is_producer
        self._is_consumer = config.is_consumer
        self._control_addr = config.control_addr
        self._push_wait_timeout = config.push_wait_timeout_s
        self._encoder_cache_hidden_dim = (
            _get_encoder_cache_hidden_dim(vllm_config) if config.is_producer else None
        )
        self._model_config = vllm_config.model_config
        self._control_client = ControlClient(config.control_timeout_ms)
        self._control_executor = ThreadPoolExecutor(
            max_workers=_CONTROL_WORKERS,
            thread_name_prefix="ec-mooncake-control",
        )
        self._event_inbox = EventInbox(self._control_client, self._control_executor)

        self._metadata_resolver = PlaceholderMetadataResolver(vllm_config.model_config)
        self._unresolved_transfer_ids = 0
        self._drain_pending = True
        self._drained_at = 0.0
        self._pending_cancels: dict[str, Future[Any]] = {}
        self._transfers = SchedulerTransferTable(
            config.pool_size, _RESERVATION_TTL_SECONDS
        )
        self._scheduler_pending_work = False
        self._pushes_to_prepare: dict[str, ECMooncakePushSpec] = {}
        self._prepared_push_transfer_ids: set[str] = set()
        self._event_ready_shards: OrderedDict[str, set[int]] = OrderedDict()
        # Mirror of the engine's encoder cache, maintained from the alloc and
        # free notifications the Scheduler already sends. Tracking it here
        # keeps `ensure_cache_available` on the upstream two-argument shape.
        self._local_cache: set[str] = set()
        self._failed_saves: set[str] = set()

    def _cancel_remote(self, consumer_zmq: str, transfer_id: str) -> bool:
        pending = None
        for _ in range(_CANCEL_ATTEMPTS):
            pending = self._control_client.discover_shards(consumer_zmq)
            if pending is not None:
                break
        if pending is None:
            raise RuntimeError(
                f"Could not discover every EC consumer shard at {consumer_zmq}"
            )

        cancelled = False
        error: BaseException | None = None
        for _ in range(_CANCEL_ATTEMPTS):
            failed = []
            for addr in pending:
                try:
                    result = self._control_client.request(
                        addr,
                        make_cancel_request(transfer_id),
                    )
                except Exception as exc:
                    if error is None:
                        error = exc
                    failed.append(addr)
                    continue
                cancelled |= isinstance(result, dict) and bool(result.get("cancelled"))
            if not failed:
                return cancelled
            pending = failed
        if error is not None:
            raise error
        return cancelled

    def _note_awaiting_push(
        self,
        mm_hash: str,
        transfer_id: str,
        request_id: str,
        now: float,
    ) -> None:
        record = self._transfers.wait_for_event(
            transfer_id,
            request_id,
            mm_hash,
            now + self._push_wait_timeout,
        )
        if record is None:
            # The id names another encoding; the request is already failed.
            return
        if record.state is not SchedulerTransferState.WAITING_EVENT:
            return
        assert record.deadline is not None
        if now < record.deadline:
            return
        elapsed = now - record.deadline + self._push_wait_timeout
        self._transfers.mark_unavailable(transfer_id, now)
        logger.warning(
            "EC Mooncake waited %.1fs for a push of mm_hash=%s "
            "(transfer_id=%s) that never arrived; "
            "requests needing it fail with a retryable error.",
            elapsed,
            mm_hash,
            transfer_id,
        )

    def take_unavailable_requests(self) -> set[str]:
        failed, self._failed_saves = self._failed_saves, set()
        return failed | self._transfers.take_unavailable_requests()

    def _poll_pending_cancels(self) -> None:
        pending = {}
        for transfer_id, future in self._pending_cancels.items():
            if not future.done():
                pending[transfer_id] = future
                continue
            try:
                future.result()
            except Exception:
                logger.warning(
                    "EC Mooncake reservation cancellation failed", exc_info=True
                )
        self._pending_cancels = pending

    def _note_shard_ready(self, data: dict[str, Any]) -> bool:
        shard_count = self._event_inbox.shard_count
        if shard_count <= 1:
            return True
        transfer_id = str(data["transfer_id"])
        record = self._transfers.get(transfer_id)
        if (
            record is not None
            and record.state is not SchedulerTransferState.WAITING_EVENT
        ):
            return False
        shard = data.get("shard")
        shards = self._event_ready_shards.setdefault(transfer_id, set())
        self._event_ready_shards.move_to_end(transfer_id)
        shards.add(int(shard) if shard is not None else len(shards))
        if len(shards) < shard_count:
            while len(self._event_ready_shards) > _MAX_PENDING_EVENTS:
                self._event_ready_shards.popitem(last=False)
            return False
        self._event_ready_shards.pop(transfer_id, None)
        return True

    def _forget_shard_readiness(self, transfer_id: str) -> None:
        self._event_ready_shards.pop(transfer_id, None)

    def _store_pushed_spec(self, data: dict[str, Any]) -> bool:
        transfer_id = str(data["transfer_id"])
        identifier = str(data["mm_hash"])
        _, accepted = self._transfers.observe_ready(
            ECMooncakeLoadSpec(
                mm_hash=identifier,
                nbytes=int(data["nbytes"]),
                shape=tuple(int(value) for value in data["shape"]),
                dtype=str(data["dtype"]),
                transfer_id=transfer_id,
            ),
            time.monotonic() + _RESERVATION_TTL_SECONDS,
        )
        return accepted

    def _queue_cancel(
        self,
        transfer_id: str,
        mm_hash: str = "",
        request_id: str = "",
    ) -> None:
        if not self._transfers.cancel(
            transfer_id,
            time.monotonic(),
            mm_hash=mm_hash,
            request_id=request_id,
        ):
            return
        self._forget_shard_readiness(transfer_id)
        self._pending_cancels[transfer_id] = self._control_executor.submit(
            self._cancel_remote,
            self._control_addr,
            transfer_id,
        )

    def _expire_transfers(self) -> None:
        now = time.monotonic()
        expired = self._transfers.expire(now, _MAX_TERMINAL_TRANSFER_RECORDS)
        for record in expired:
            self._queue_cancel(record.transfer_id)

    def _drain_push_notifications(self) -> None:
        now = time.monotonic()
        if not self._drain_pending and now - self._drained_at < _DRAIN_MIN_INTERVAL:
            return
        self._drain_pending = False
        self._drained_at = now
        self._poll_pending_cancels()
        self._expire_transfers()
        events = self._event_inbox.drain(self._control_addr)
        for data in events:
            try:
                if not isinstance(data, dict):
                    raise TypeError("EC readiness event must be an object")
                if not data.get("ready"):
                    continue
                self._accept_ready_event(data)
            except (KeyError, TypeError, ValueError):
                # Readiness events cross a plain PULL socket, so a malformed
                # one costs that event and not the engine.
                logger.warning(
                    "Discarding a malformed EC readiness event.", exc_info=True
                )

    def _accept_ready_event(self, data: dict[str, Any]) -> None:
        transfer_id = str(data["transfer_id"])
        record = self._transfers.get(transfer_id)
        if record is not None and record.state in {
            SchedulerTransferState.CANCELLED,
            SchedulerTransferState.UNAVAILABLE,
            SchedulerTransferState.EXPIRED,
            SchedulerTransferState.FAILED,
        }:
            return
        if not self._note_shard_ready(data):
            return
        self._store_pushed_spec(data)

    def has_cache_item(self, identifier: str) -> bool:
        if not self._is_consumer:
            return False
        self._drain_push_notifications()
        return self._transfers.has_state(
            identifier,
            (
                SchedulerTransferState.READY,
                SchedulerTransferState.RESIDENT,
                SchedulerTransferState.AVAILABLE,
            ),
        )

    def _warn_unresolved_transfer_id(
        self, request: Any, index: int, where: str
    ) -> None:
        """Report an `ec_items` entry that cannot be matched to a feature.

        A caller that cannot resolve a transfer id has to fall back to a
        locally invented one or skip its bookkeeping entirely, and either way
        the two sides of a transfer stop agreeing on its name. That desyncs
        silently -- reservations are never released and only surface minutes
        later as a full buffer pool -- so say it out loud the first time.

        `mm_hash` in `ec_items` must be a value some engine reported, never one
        the caller derived itself: `mm_features[i].identifier` folds in the
        engine's media_io_kwargs and mm_processor_kwargs, so a media uuid alone
        no longer equals it. Omit the field to match by position instead.
        """
        self._unresolved_transfer_ids += 1
        count = self._unresolved_transfer_ids
        if count > _MAX_UNRESOLVED_TRANSFER_ID_WARNINGS and count % 1000:
            return
        params = getattr(request, "ec_transfer_params", None)
        items = (params or {}).get("ec_items") or []
        logger.warning(
            "EC Mooncake could not resolve a transfer id at %s for req=%s "
            "index=%d: feature identifier=%s does not match any of the %d "
            "ec_items %s (ec_transfer_params present=%s). Occurrence %d. "
            "ec_items[].mm_hash must echo an engine-reported identifier, or "
            "be omitted so the entry is matched by position.",
            where,
            request.request_id,
            index,
            request.mm_features[index].identifier[:16],
            len(items),
            [str(item.get("mm_hash"))[:16] for item in items[:4]],
            params is not None,
            count,
        )

    @staticmethod
    def _request_transfer_id(request: Any, index: int) -> str | None:
        params = getattr(request, "ec_transfer_params", None) or {}
        items = params.get("ec_items") or []
        mm_hash = request.mm_features[index].identifier
        if index < len(items):
            item = items[index]
            if item.get("mm_hash") in (None, mm_hash) and item.get("transfer_id"):
                return str(item["transfer_id"])
        for item in items:
            if item.get("mm_hash") == mm_hash and item.get("transfer_id"):
                return str(item["transfer_id"])
        return None

    def ensure_cache_available(self, request: Any, num_computed_tokens: int) -> bool:
        if self._is_producer:
            for index, feature in enumerate(request.mm_features):
                if (
                    feature.mm_position.offset + feature.mm_position.length
                    > num_computed_tokens
                ):
                    self._prepare_push_spec(request, index)
        if not self._is_consumer:
            return True

        self._drain_push_notifications()
        # One timestamp for the whole decision, so the deadlines it hands out
        # cannot disagree between features.
        now = time.monotonic()
        all_ready = True
        for index, feature in enumerate(request.mm_features):
            if (
                feature.mm_position.offset + feature.mm_position.length
                <= num_computed_tokens
            ):
                continue
            mm_hash = feature.identifier
            transfer_id = self._request_transfer_id(request, index)
            if transfer_id is not None:
                self._transfers.touch_available(
                    transfer_id, now + _RESERVATION_TTL_SECONDS
                )
            if mm_hash in self._local_cache:
                continue
            if self._transfers.has_state(mm_hash, (SchedulerTransferState.READY,)):
                continue
            if self._transfers.has_state(mm_hash, (SchedulerTransferState.LOADING,)):
                all_ready = False
                continue
            record = self._transfers.begin_load(
                mm_hash,
                transfer_id,
                request.request_id,
                now + self._push_wait_timeout,
            )
            if record is not None:
                self._scheduler_pending_work = True
                all_ready = False
            else:
                waiting_id = transfer_id or f"{request.request_id}:{index}"
                self._note_awaiting_push(mm_hash, waiting_id, request.request_id, now)
                all_ready = False
        return all_ready

    def _prepare_push_spec(self, request: Any, index: int) -> None:
        params = getattr(request, "ec_transfer_params", None) or {}
        consumer_zmq = params.get("consumer_zmq")
        mm_hash = request.mm_features[index].identifier
        transfer_id = self._request_transfer_id(request, index)
        if transfer_id is None:
            # Still push: after the proxy has rewritten the item to embeds the
            # consumer has no media left to fall back on, so a nameless push
            # beats none. But the consumer knows this transfer by the id it
            # sent, not by the one invented here, so its cancel will never
            # reach the reservation this push is about to take.
            if consumer_zmq:
                self._warn_unresolved_transfer_id(request, index, "push prepare")
            transfer_id = f"{request.request_id}:{index}"
        if not consumer_zmq or transfer_id in self._prepared_push_transfer_ids:
            return
        num_tokens = request.get_num_encoder_embeds(index)
        dtype = self._model_config.dtype
        assert isinstance(dtype, torch.dtype)
        assert self._encoder_cache_hidden_dim is not None
        dtype_name = str(dtype).split(".")[-1]
        shape = (num_tokens, self._encoder_cache_hidden_dim)
        nbytes = math.prod(shape) * dtype.itemsize
        self._pushes_to_prepare[transfer_id] = ECMooncakePushSpec(
            mm_hash=mm_hash,
            nbytes=nbytes,
            shape=shape,
            dtype=dtype_name,
            consumer_zmq=str(consumer_zmq),
            transfer_id=transfer_id,
            request_id=request.request_id,
        )
        self._prepared_push_transfer_ids.add(transfer_id)

    def update_state_after_alloc(self, request: Any, index: int) -> None:
        self._local_cache.add(request.mm_features[index].identifier)
        if self._is_producer:
            self._prepare_push_spec(request, index)

    def update_state_after_free(self, request: Any, index: int) -> None:
        if not self._is_consumer:
            return
        transfer_id = self._request_transfer_id(request, index)
        if transfer_id is None:
            self._warn_unresolved_transfer_id(request, index, "encoder-cache free")
            return
        self._queue_cancel(
            transfer_id,
            mm_hash=request.mm_features[index].identifier,
            request_id=request.request_id,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self._local_cache.discard(mm_hash)
            self._transfers.release_ready(mm_hash, time.monotonic())
        for transfer_id in self._transfers.drain_orphaned():
            self._queue_cancel(transfer_id)
        meta = ECMooncakeConnectorMetadata(
            freed=scheduler_output.free_encoder_mm_hashes
        )
        for push_spec in self._pushes_to_prepare.values():
            meta.pushes.append(push_spec)
        self._pushes_to_prepare.clear()
        for record in self._transfers.take_loads_to_dispatch():
            assert record.spec is not None
            meta.loads.append(record.spec)
        self._poll_pending_cancels()
        self._drain_pending = True
        return meta

    def update_connector_output(self, connector_output: ECConnectorOutput) -> None:
        meta = connector_output.ec_connector_worker_meta
        if not isinstance(meta, ECMooncakeWorkerMetadata):
            return
        for mm_hash in meta.loaded:
            self._transfers.complete_load(mm_hash)
        for mm_hash in meta.failed_loads:
            self._transfers.fail_load(mm_hash, time.monotonic())
        for mm_hash in meta.reclaimed:
            self._transfers.reclaim(mm_hash, time.monotonic())
        self._scheduler_pending_work = meta.pending_saves
        self._failed_saves.update(meta.failed_saves)

    def has_pending_push_work(self) -> bool:
        return self._scheduler_pending_work

    def request_finished(self, request: Any) -> tuple[bool, dict[str, Any] | None]:
        if self._is_consumer:
            for index in range(len(request.mm_features)):
                transfer_id = self._request_transfer_id(request, index)
                if transfer_id is None:
                    self._warn_unresolved_transfer_id(request, index, "request finish")
                    continue
                self._queue_cancel(
                    transfer_id,
                    mm_hash=request.mm_features[index].identifier,
                    request_id=request.request_id,
                )
        if self._is_producer and self._prepared_push_transfer_ids:
            for index in range(len(request.mm_features)):
                transfer_id = self._request_transfer_id(request, index)
                if transfer_id is None:
                    transfer_id = f"{request.request_id}:{index}"
                self._prepared_push_transfer_ids.discard(transfer_id)
        if not self._is_producer:
            return False, None

        items = collect_ec_item_metadata(request.mm_features, self._metadata_resolver)
        for index, feature in enumerate(request.mm_features):
            transfer_id = self._request_transfer_id(request, index)
            if transfer_id is not None:
                items[feature.identifier]["transfer_id"] = transfer_id

        if not items:
            return False, None
        return False, items

    def close(self) -> None:
        self._control_executor.shutdown(wait=True, cancel_futures=True)
        self._control_client.close()
        self._event_inbox.close()
