# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side orchestration of Mooncake control, memory, and data planes.

Consumer Workers expose rank-local reservations and publish received tensors.
Producer Workers reserve every destination shard, bind computed sources, run
batched Mooncake writes, and report asynchronous completion to the Scheduler.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.config import (
    _RESERVATION_TTL_SECONDS,
    MooncakeECConfig,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.control import (
    ConsumerControlServer,
    ControlClient,
    make_cancel_request,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    ProducerMemoryPool,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.producer import (
    ProducerPushManager,
    ProducerPushRecord,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    ConsumerReservationManager,
    ConsumerReservationState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
    ensure_mooncake_available,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

logger = init_logger(__name__)

_T = TypeVar("_T")

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_RESERVATION_REFRESH_SECONDS = _RESERVATION_TTL_SECONDS / 2
_MAX_CANCELLED_TRANSFER_IDS = 1 << 16
_CANCEL_ATTEMPTS = 2
_TRANSFER_WORKERS = 4
_CONTROL_WORKERS = 8


class _FanoutError(RuntimeError):
    """Retain shard outcomes after every started task settles."""

    def __init__(self, error: BaseException, results: list[Any | None]) -> None:
        self.results = results
        super().__init__(str(error))


class ECMooncakeWorker:
    """Orchestrate consumer reservations and producer push batches.

    Consumers may use TP, PP, and DP. Only the first PP stage receives encoder
    outputs, and each TP rank exposes a consecutive control port and receives
    the same source concurrently. Producers remain unsharded and unreplicated.
    With DP, the caller must route both halves of a request to the same replica
    and pass that replica's control address to the producer.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        ensure_mooncake_available()
        config = MooncakeECConfig.from_vllm_config(vllm_config)
        self.is_producer = config.is_producer
        self.is_consumer = config.is_consumer
        self._buffer_device = config.buffer_device
        self._control_host = config.control_host
        self._control_port = config.control_port
        self._transfer = MooncakeTransfer(get_ip(), config.protocol)
        self._consumer_memory = ConsumerMemoryPool(
            config.pool_size,
            self._transfer,
        )
        self._reservations = ConsumerReservationManager(
            self._consumer_memory,
            _RESERVATION_TTL_SECONDS,
            _MAX_CANCELLED_TRANSFER_IDS,
        )
        self._consumer_rank_resolved = False
        self._is_receiving_rank = True
        self._tp_rank = 0
        self._tp_size = 1
        self._control_server: ConsumerControlServer | None = None
        # Worker producer
        self._producer_memory = ProducerMemoryPool(
            config.pool_size,
            self._transfer,
        )
        self._control_client = ControlClient(config.control_timeout_ms)
        self._shutdown_drain_timeout_s = config.control_timeout_ms / 1000
        self._io_executor = ThreadPoolExecutor(
            max_workers=_TRANSFER_WORKERS,
            thread_name_prefix="ec-mooncake-transfer",
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=_CONTROL_WORKERS,
            thread_name_prefix="ec-mooncake-control",
        )
        self._shard_pool: ThreadPoolExecutor | None = None
        self._shard_pool_lock = threading.Lock()
        self._push_ready = threading.Event()
        self._producer_pushes = ProducerPushManager(self._push_ready.set)
        self._dispatch_stop = threading.Event()
        self._dispatcher: threading.Thread | None = None
        self._failed_saves: set[str] = set()
        self._collecting_sources = False
        self._completed_loads: set[str] = set()
        self._failed_loads: set[str] = set()
        self._shutdown = False
        if self.is_producer:
            self._dispatcher = threading.Thread(
                target=self._dispatch_pushes, name="ec-mooncake-ready", daemon=True
            )
            self._dispatcher.start()

    def _dispatch_pushes(self) -> None:
        pending_event = False
        while not self._dispatch_stop.is_set():
            self._push_ready.wait(timeout=0.001 if pending_event else None)
            self._push_ready.clear()
            if self._dispatch_stop.is_set():
                break
            if self._collecting_sources:
                pending_event = False
                continue
            pending_event = self._producer_pushes.submit_batches(
                self._io_executor, self._push_batch
            )

    def _resolve_consumer_rank(self) -> None:
        """Place this worker in the consumer receive topology."""
        if self._consumer_rank_resolved:
            return
        self._consumer_rank_resolved = True
        try:
            from vllm.distributed.parallel_state import get_pp_group, get_tp_group

            tp_group = get_tp_group()
            self._tp_rank = tp_group.rank_in_group
            self._tp_size = tp_group.world_size
            self._is_receiving_rank = get_pp_group().is_first_rank
        except AssertionError:
            # Groups are only absent outside a distributed run, where this
            # worker is the whole consumer.
            self._tp_rank = 0
            self._tp_size = 1
            self._is_receiving_rank = True

    def start_services(self) -> None:
        if not self.is_consumer or self._control_server is not None:
            return
        self._resolve_consumer_rank()
        if not self._is_receiving_rank:
            # Later pipeline stages hold no encoder outputs, so they need
            # neither a receive pool nor a control channel.
            return
        self._consumer_memory.prepare(torch.device(self._buffer_device))
        consumer_pool = self._consumer_memory.tensor
        if consumer_pool is None:
            raise RuntimeError(
                "Mooncake push mode requires a registered consumer buffer pool."
            )
        base_port = self._control_port
        self._control_server = ConsumerControlServer(
            # The reservation channel hands out registered-memory addresses
            # and takes cancellations, so it listens where the Consumer
            # advertises itself (`ec_ip`), not on every interface.
            self._control_host,
            base_port + self._tp_rank,
            self._reserve_push_destination,
            self._push_status,
            self._reservations.complete,
            self._reservations.cancel,
            self._reservations.expire,
            peer_ports=[base_port + rank for rank in range(self._tp_size)],
            device=consumer_pool.device,
            drain_ready=self._reservations.drain_ready,
        )
        try:
            self._control_server.start()
        except Exception:
            self._control_server.close()
            self._control_server = None
            raise

    def _reserve_push_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        transfer_id = str(payload["transfer_id"])
        mm_hash = str(payload["mm_hash"])
        nbytes = int(payload["nbytes"])
        shape = tuple(int(value) for value in payload["shape"])
        dtype_name = str(payload["dtype"])
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported torch dtype string: {dtype_name!r}")
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if expected_nbytes != nbytes:
            raise ValueError("shape and dtype do not match nbytes")

        self._reservations.expire()
        reservation, should_write = self._reservations.reserve(
            transfer_id, mm_hash, nbytes, shape, dtype_name, dtype
        )
        if reservation is None:
            raise RuntimeError("EC consumer buffer pool is full")
        if reservation.state in {
            ConsumerReservationState.CANCEL_PENDING,
            ConsumerReservationState.CANCELLED,
        }:
            return {
                "reservation_id": "",
                "dst_session": "",
                "dst_ptr": 0,
                "nbytes": nbytes,
                "write": False,
                "ready": False,
                "cancelled": True,
            }
        assert reservation.allocation is not None

        return {
            "reservation_id": reservation.reservation_id,
            "dst_session": self._transfer.local_session(),
            "dst_ptr": reservation.allocation.tensor.data_ptr(),
            "nbytes": reservation.allocation.tensor.nbytes,
            "write": should_write,
            "ready": reservation.state is ConsumerReservationState.READY,
            "cached": reservation.lease is not None,
        }

    def _push_status(self, transfer_id: str) -> dict[str, Any] | None:
        reservation = self._reservations.status(transfer_id)
        if reservation is None:
            return None
        assert reservation.allocation is not None
        return {
            "mm_hash": reservation.mm_hash,
            "ready": reservation.state is ConsumerReservationState.READY,
            "reservation_id": reservation.reservation_id,
            "nbytes": reservation.allocation.tensor.nbytes,
            "shape": list(reservation.shape),
            "dtype": reservation.dtype,
        }

    def _shard_executor(self) -> ThreadPoolExecutor:
        """Use a separate pool so nested shard fan-out cannot deadlock."""
        with self._shard_pool_lock:
            if self._shard_pool is None:
                self._shard_pool = ThreadPoolExecutor(
                    max_workers=32, thread_name_prefix="ec-mooncake-shard"
                )
            return self._shard_pool

    def _reserve_one(self, addr: str, spec: ECMooncakePushSpec) -> dict[str, Any]:
        result = self._control_client.request(
            addr,
            {
                "op": "reserve",
                "transfer_id": spec.transfer_id,
                "mm_hash": spec.mm_hash,
                "nbytes": spec.nbytes,
                "shape": list(spec.shape),
                "dtype": spec.dtype,
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError("Invalid EC reservation response")
        result["_received_at"] = time.monotonic()
        result["addr"] = addr
        return result

    def _run_fanout(
        self,
        tasks: list[Callable[[], _T]],
        on_submit: Callable[[int, Future[_T]], None] | None = None,
    ) -> list[_T]:
        """Run every started shard task and retain partial results on failure.

        Waiting for all submitted tasks is what makes source-memory release and
        partial-reservation cleanup safe after one shard fails.
        """
        if not tasks:
            return []
        futures: list[tuple[int, Future[_T]]] = []
        results: list[_T | None] = [None] * len(tasks)
        error: BaseException | None = None
        for index, task in enumerate(tasks[1:], 1):
            try:
                future = self._shard_executor().submit(task)
            except Exception as exc:
                error = exc
                break
            futures.append((index, future))
            if on_submit is not None:
                on_submit(index, future)
        if error is None:
            try:
                results[0] = tasks[0]()
            except Exception as exc:
                error = exc
        for index, future in futures:
            try:
                results[index] = future.result()
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise _FanoutError(error, results)
        return cast(list[_T], results)

    def _cancel_reservations(
        self,
        spec: ECMooncakePushSpec,
        reservations: list[dict[str, Any]],
        *,
        refresh: bool = False,
        record: ProducerPushRecord | None = None,
    ) -> None:
        reservations = [
            shard for shard in reservations if not shard.get("cancelled", False)
        ]
        if not reservations:
            return

        def cancel(shard: dict[str, Any]) -> dict[str, Any]:
            addr = shard.get("addr")
            if not isinstance(addr, str) or not addr:
                raise RuntimeError(
                    "EC reservation is missing a confirmed shard address"
                )
            result = self._control_client.request(
                addr,
                make_cancel_request(
                    spec.transfer_id,
                    str(shard.get("reservation_id", "")),
                    abandon=True,
                    refresh=refresh,
                ),
            )
            if not isinstance(result, dict) or not result.get("cancelled"):
                raise RuntimeError(
                    f"Could not cancel EC reservation for mm_hash={spec.mm_hash}"
                )
            return shard

        def track(_index: int, future: Future[dict[str, Any]]) -> None:
            if record is not None:
                self._producer_pushes.track_shard_futures([record], [future])

        self._run_fanout([partial(cancel, shard) for shard in reservations], track)

    def _retry_cancel_reservations(
        self,
        spec: ECMooncakePushSpec,
        reservations: list[dict[str, Any]],
        *,
        record: ProducerPushRecord | None = None,
    ) -> None:
        pending = [shard for shard in reservations if not shard.get("cancelled", False)]
        error: _FanoutError | None = None
        for _ in range(_CANCEL_ATTEMPTS):
            try:
                self._cancel_reservations(spec, pending, record=record)
            except _FanoutError as exc:
                error = exc
                pending = [
                    shard
                    for index, shard in enumerate(pending)
                    if exc.results[index] is None
                ]
                continue
            return
        assert error is not None
        raise error

    def _reserve_remote(self, spec: ECMooncakePushSpec) -> list[dict[str, Any]]:
        """Reserve a destination on every shard of the consumer."""
        result = self._reserve_remote_many([spec])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def _refresh_remote_reservations(
        self,
        spec: ECMooncakePushSpec,
        reservations: list[dict[str, Any]],
        record: ProducerPushRecord | None = None,
    ) -> list[dict[str, Any]]:
        stale = [
            shard
            for shard in reservations
            if not shard.get("ready", False)
            and not shard.get("cached", False)
            and not shard.get("cancelled", False)
        ]
        try:
            self._cancel_reservations(spec, stale, refresh=True, record=record)
        except _FanoutError as exc:
            pending = [
                shard for index, shard in enumerate(stale) if exc.results[index] is None
            ]
            try:
                self._retry_cancel_reservations(spec, pending, record=record)
            except _FanoutError as cleanup_error:
                raise exc from cleanup_error
            raise
        return self._reserve_remote(spec)

    @staticmethod
    def _validate_push_source(push: ProducerPushRecord) -> None:
        tensor = push.source_tensor
        assert tensor is not None
        spec = push.spec
        if tuple(tensor.shape) != tuple(spec.shape):
            raise ValueError(f"EC source shape mismatch for mm_hash={spec.mm_hash}")
        if str(tensor.dtype).split(".")[-1] != spec.dtype:
            raise ValueError(f"EC source dtype mismatch for mm_hash={spec.mm_hash}")
        if not tensor.is_contiguous():
            raise ValueError(f"EC source must be contiguous for mm_hash={spec.mm_hash}")
        if tensor.nbytes != spec.nbytes:
            raise ValueError(f"EC source size mismatch for mm_hash={spec.mm_hash}")

    def start_save_caches(
        self,
        metadata: ECMooncakeConnectorMetadata,
        encoder_cache: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        self._collecting_sources = True
        new: dict[str, list[ProducerPushRecord]] = {}
        for spec in metadata.pushes:
            try:
                record, created = self._producer_pushes.reserve(spec, Future)
                if created:
                    new.setdefault(spec.consumer_zmq, []).append(record)
            except ValueError:
                logger.warning("Rejected conflicting EC push", exc_info=True)
                self._failed_saves.add(spec.request_id)
        for records in new.values():
            future = self._control_executor.submit(self._reserve_batch, records)
            future.add_done_callback(partial(self._reservation_batch_done, records))
        if not isinstance(encoder_cache, dict):
            return
        for mm_hash in dict.fromkeys(spec.mm_hash for spec in metadata.pushes):
            tensor = encoder_cache.get(mm_hash)
            if tensor is not None:
                self._bind_push_source(tensor, mm_hash)

    def _reserve_batch(self, records: list[ProducerPushRecord]) -> None:
        """Resolve independent item futures with one reserve RPC per shard."""
        if len(records) == 1:
            record = records[0]
            try:
                record.reservation_future.set_result(self._reserve_remote(record.spec))
            except Exception as exc:
                record.reservation_future.set_exception(exc)
            return
        outcomes = self._reserve_remote_many([record.spec for record in records])
        self._producer_pushes.finish_reservations(list(zip(records, outcomes)))

    def _reserve_remote_many(
        self, specs: list[ECMooncakePushSpec]
    ) -> list[list[dict[str, Any]] | Exception]:
        shards = None
        for _ in range(_CANCEL_ATTEMPTS):
            shards = self._control_client.discover_shards(specs[0].consumer_zmq)
            if shards is not None:
                break
        if shards is None:
            raise RuntimeError(
                f"Could not discover every EC consumer shard at {specs[0].consumer_zmq}"
            )

        def reserve(addr: str) -> list[dict[str, Any]]:
            if len(specs) == 1:
                return [{"ok": True, "result": self._reserve_one(addr, specs[0])}]
            response = self._control_client.request(
                addr,
                {
                    "op": "reserve_batch",
                    "items": [
                        {
                            "transfer_id": spec.transfer_id,
                            "mm_hash": spec.mm_hash,
                            "nbytes": spec.nbytes,
                            "shape": list(spec.shape),
                            "dtype": spec.dtype,
                        }
                        for spec in specs
                    ],
                },
            )
            items = response["items"]
            if len(items) != len(specs):
                raise RuntimeError("Malformed EC reservation batch response")
            for item in items:
                if item.get("ok"):
                    item["result"]["_received_at"] = time.monotonic()
            return items

        fanout_error = None
        results: list[Any]
        try:
            results = self._run_fanout([partial(reserve, addr) for addr in shards])
        except _FanoutError as exc:
            results = exc.results
            fanout_error = exc
        outcomes: list[list[dict[str, Any]] | Exception] = []
        for index, spec in enumerate(specs):
            reservations: list[dict[str, Any]] = []
            error = None
            for addr, items in zip(shards, results):
                item = items[index] if items is not None else {}
                if item.get("ok"):
                    reservations.append({**item["result"], "addr": addr})
                else:
                    error = fanout_error or RuntimeError(item["error"])
                    reservations.append({"addr": addr, "reservation_id": ""})
            if error is None:
                outcomes.append(reservations)
            else:
                failure = _FanoutError(error, list(reservations))
                try:
                    self._retry_cancel_reservations(spec, reservations)
                except Exception as cleanup_error:
                    failure.__cause__ = cleanup_error
                    logger.exception("Failed to release partial EC reservation batch")
                outcomes.append(failure)
        return outcomes

    @staticmethod
    def _reservation_batch_done(
        records: list[ProducerPushRecord], future: Future[None]
    ) -> None:
        error = future.exception()
        if error is not None:
            for record in records:
                if not record.reservation_future.done():
                    record.reservation_future.set_exception(error)

    def start_load_caches(
        self,
        metadata: ECMooncakeConnectorMetadata,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        self._resolve_consumer_rank()
        if not self._is_receiving_rank:
            # Later pipeline stages never gather multimodal embeddings.
            return
        self._transfer.ensure_ready()
        if self._buffer_device == "cuda" and not torch.accelerator.is_available():
            raise RuntimeError(
                "ECMooncakeConnector requires CUDA for ec_buffer_device=cuda"
            )
        self._reservations.retire_stale(encoder_cache, metadata.freed)

        for spec in metadata.loads:
            if spec.mm_hash in encoder_cache:
                if not spec.local:
                    # The spec's id is one shard's; cancel by transfer.
                    self._reservations.cancel(spec.transfer_id, "")
                self._completed_loads.add(spec.mm_hash)
                continue
            if spec.local:
                tensor = self._consumer_memory.take_resident(
                    spec.mm_hash, tuple(spec.shape), spec.dtype
                )
            else:
                try:
                    allocation = self._reservations.take(spec.transfer_id, spec.mm_hash)
                    tensor = allocation.tensor
                except RuntimeError as e:
                    logger.warning("EC Mooncake pushed load failed: %s", e)
                    tensor = None
            if tensor is None:
                self._failed_loads.add(spec.mm_hash)
            else:
                encoder_cache[spec.mm_hash] = tensor
                self._completed_loads.add(spec.mm_hash)

    def _push_batch(self, pushes: list[ProducerPushRecord]) -> None:
        started_at = time.monotonic()
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]] = []
        written_pushes: dict[str, ProducerPushRecord] = {}
        failure: Exception | None = None
        try:
            for push in pushes:
                self._validate_push_source(push)
                reservations = self._producer_pushes.resolve_reservations(push)
                stale = [
                    index
                    for index, shard in enumerate(reservations)
                    if not shard.get("ready", False)
                    and not shard.get("cancelled", False)
                    and time.monotonic() - float(shard.get("_received_at", started_at))
                    >= _RESERVATION_REFRESH_SECONDS
                ]
                if stale:
                    reservations = self._refresh_remote_reservations(
                        push.spec, reservations, push
                    )
                    self._producer_pushes.replace_reservations(push, reservations)
                self._producer_pushes.begin_writing(push)
                writable = [
                    shard
                    for shard in reservations
                    if not shard.get("cached", False)
                    and not shard.get("cancelled", False)
                    and shard.get("write", True)
                ]
                source = push.source_tensor
                assert source is not None
                for shard in writable:
                    if int(shard["nbytes"]) != source.nbytes:
                        raise RuntimeError(
                            "Reserved EC size does not match tensor for "
                            f"mm_hash={push.spec.mm_hash}"
                        )
                    ready.append((push, shard))
                    written_pushes.setdefault(push.spec.transfer_id, push)
            if ready:
                # Stage each source once, then write it to every destination.
                source_index = {
                    push.spec.transfer_id: index
                    for index, push in enumerate(written_pushes.values())
                }
                tensors = [
                    cast(torch.Tensor, push.source_tensor)
                    for push in written_pushes.values()
                    if push.source_tensor is not None
                ]
                lengths = [tensor.nbytes for tensor in tensors]
                staged = self._producer_memory.stage(tensors)
                registered_sources: list[int] = []
                if staged is not None:
                    sources = staged.tensors
                else:
                    sources = tensors
                    registered_sources = self._transfer.acquire_sources(tensors)
                addresses = [tensor.data_ptr() for tensor in sources]
                try:
                    by_session: dict[str, list[tuple[int, int]]] = {}
                    session_records: dict[str, dict[str, ProducerPushRecord]] = {}
                    for push, shard in ready:
                        session = str(shard["dst_session"])
                        by_session.setdefault(session, []).append(
                            (source_index[push.spec.transfer_id], int(shard["dst_ptr"]))
                        )
                        session_records.setdefault(session, {})[
                            push.spec.transfer_id
                        ] = push

                    def write(session: str, items: list[tuple[int, int]]) -> None:
                        self._transfer.write(
                            session,
                            [addresses[index] for index, _ in items],
                            [dst for _, dst in items],
                            [lengths[index] for index, _ in items],
                        )

                    sessions = list(by_session.items())

                    # Write shards concurrently to avoid serial TP latency.
                    def track_write(index: int, future: Future[None]) -> None:
                        session = sessions[index][0]
                        self._producer_pushes.track_shard_futures(
                            list(session_records[session].values()), [future]
                        )

                    writes: list[Callable[[], None]] = [
                        partial(write, *session) for session in sessions
                    ]
                    self._run_fanout(writes, track_write)
                finally:
                    if staged is not None:
                        self._producer_memory.release(staged)
                    self._transfer.release_sources(registered_sources)

            self._producer_pushes.begin_notifying(pushes)
            self._notify_completions(ready)
            self._producer_pushes.complete(pushes)
        except Exception as exc:
            # Report asynchronously; raising here would fail EngineCore.
            failure = exc
            logger.exception(
                "EC Mooncake push batch failed for mm_hashes=%s",
                [push.spec.mm_hash for push in pushes],
            )
            self._producer_pushes.settle_all(pushes)
            self._abandon_pushes(pushes)
        finally:
            if failure is not None:
                self._producer_pushes.fail(pushes, failure)

    def _notify_completions(
        self, notifications: list[tuple[ProducerPushRecord, dict[str, Any]]]
    ) -> None:
        """Tell the consumer, in one message per destination, what landed."""
        if not notifications:
            return
        by_destination: dict[str, list[tuple[ProducerPushRecord, dict[str, Any]]]] = {}
        for push, reservation in notifications:
            by_destination.setdefault(
                str(reservation.get("addr", push.spec.consumer_zmq)), []
            ).append((push, reservation))
        destinations = list(by_destination.items())

        def notify(
            consumer_zmq: str,
            items: list[tuple[ProducerPushRecord, dict[str, Any]]],
        ) -> None:
            result = self._control_client.request(
                consumer_zmq,
                {
                    "op": "complete_batch",
                    "items": [
                        {
                            "transfer_id": push.spec.transfer_id,
                            "reservation_id": reservation["reservation_id"],
                        }
                        for push, reservation in items
                    ],
                },
            )
            completions = result.get("items", []) if isinstance(result, dict) else []
            if len(completions) != len(items):
                raise RuntimeError("Malformed EC completion response")
            for (push, _), completion in zip(items, completions):
                if not completion.get("completed"):
                    raise RuntimeError(
                        f"Unknown EC reservation for mm_hash={push.spec.mm_hash}"
                    )

        def track(index: int, future: Future[None]) -> None:
            records = {
                push.spec.transfer_id: push for push, _ in destinations[index][1]
            }
            self._producer_pushes.track_shard_futures(list(records.values()), [future])

        self._run_fanout(
            [
                partial(notify, destination, items)
                for destination, items in destinations
            ],
            track,
        )

    @staticmethod
    def _known_reservations(record: ProducerPushRecord) -> list[dict[str, Any]]:
        if record.reservations:
            return list(record.reservations)
        try:
            return list(record.reservation_future.result())
        except _FanoutError as exc:
            return [result for result in exc.results if isinstance(result, dict)]
        except Exception:
            return []

    def _abandon_pushes(self, pushes: list[ProducerPushRecord]) -> None:
        """Release the consumer-side reservations of a batch that failed."""
        for push in pushes:
            shards = self._known_reservations(push)
            if not shards:
                # No confirmed topology means there is no safe address to
                # cancel.  The original push failure remains observable via
                # ProducerPushManager.fail below.
                continue
            try:
                self._retry_cancel_reservations(push.spec, shards, record=push)
            except _FanoutError:
                logger.exception(
                    "Failed to abandon EC reservations for transfer_id=%s",
                    push.spec.transfer_id,
                )

    def _flush_pending_pushes(self) -> None:
        self._producer_pushes.submit_batches(
            self._io_executor,
            self._push_batch,
        )

    def _bind_push_source(self, tensor: torch.Tensor, mm_hash: str) -> None:
        ready_event = None
        if tensor.device.type == "cuda":
            ready_event = torch.Event()
            ready_event.record(torch.accelerator.current_stream(tensor.device))
        self._producer_pushes.bind_source(mm_hash, tensor, ready_event)

    def _cancel_orphaned_reservation(self, record: ProducerPushRecord) -> None:
        resolution_error: Exception | None = None
        try:
            reservations = self._producer_pushes.resolve_reservations(record)
        except Exception as exc:
            resolution_error = exc
            reservations = self._known_reservations(record)
        reservations = [
            shard for shard in reservations if not shard.get("cancelled", False)
        ]
        if not reservations:
            if resolution_error is not None:
                self._producer_pushes.fail([record], resolution_error)
            else:
                self._producer_pushes.finish_cancel(record)
            return
        try:
            self._retry_cancel_reservations(record.spec, reservations, record=record)
        except Exception as cleanup_error:
            if resolution_error is not None:
                combined_error = RuntimeError(
                    f"EC reservation resolution failed ({resolution_error}); "
                    f"cleanup also failed ({cleanup_error})"
                )
                combined_error.__cause__ = resolution_error
                cleanup_error = combined_error
            self._producer_pushes.fail([record], cleanup_error)
            return
        if resolution_error is not None:
            self._producer_pushes.fail([record], resolution_error)
        else:
            self._producer_pushes.finish_cancel(record)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if not self.is_producer:
            return None, None

        for record in self._producer_pushes.cancel_requests(finished_req_ids):
            self._producer_pushes.submit_cancel(
                record,
                self._io_executor,
                self._cancel_orphaned_reservation,
            )
        return None, None

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any
    ) -> None:
        if not self.is_producer:
            return
        tensor = encoder_cache[mm_hash]
        self._bind_push_source(tensor, mm_hash)

    def build_connector_worker_meta(self) -> ECMooncakeWorkerMetadata | None:
        if self.is_consumer and not self._is_receiving_rank:
            # `loaded` is intersected across reporting ranks, so a stage that
            # never loads must not report at all rather than report nothing.
            return None

        self._collecting_sources = False
        self._push_ready.set()
        self._flush_pending_pushes()
        failures = self._producer_pushes.poll()
        for mm_hash, error in failures:
            logger.error(
                "EC Mooncake async save failed for mm_hash=%s: %s",
                mm_hash,
                error,
            )
        reclaimed = self._consumer_memory.drain_reclaimed()
        meta = ECMooncakeWorkerMetadata(
            loaded=self._completed_loads,
            failed_loads=self._failed_loads,
            reclaimed=reclaimed,
            pending_saves=self._producer_pushes.pending,
            failed_saves=self._failed_saves,
        )
        self._failed_saves = set()
        self._completed_loads = set()
        self._failed_loads = set()
        return meta

    def close(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is not None:
            self._dispatch_stop.set()
            self._push_ready.set()
            dispatcher.join()
        if self._control_server is not None:
            self._reservations.begin_shutdown()
        self._flush_pending_pushes()
        for record in self._producer_pushes.cancel_requests(None):
            self._producer_pushes.submit_cancel(
                record,
                self._io_executor,
                self._cancel_orphaned_reservation,
            )
        self._control_executor.shutdown(wait=True)
        self._producer_pushes.submit_batches(
            self._io_executor, self._push_batch, wait=True
        )
        self._io_executor.shutdown(wait=True)
        if self._shard_pool is not None:
            self._shard_pool.shutdown(wait=True)
        # Every producer-side thread that could hold a control socket is stopped.
        self._control_client.close()
        drained = True
        if self._control_server is not None:
            drained = self._reservations.wait_for_writers(
                self._shutdown_drain_timeout_s
            )
            self._control_server.close()
        if drained:
            self._consumer_memory.close()
        else:
            logger.error(
                "Timed out waiting for Mooncake EC writers; keeping the consumer "
                "receive pool registered"
            )
        self._producer_memory.close()
        self._transfer.close()
