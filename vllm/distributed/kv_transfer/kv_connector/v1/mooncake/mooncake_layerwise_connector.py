# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake layerwise: background address planning and device transfers.

One full-attention group, PP1, no CP; KV writes use a single compute stream.
Model hooks snapshot sends; the executor builds addresses, waits and performs WRITE.
The listener loop owns D sockets and ACKs. Requests retain KV through delay_free.
"""

import asyncio
import time
from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import msgspec
import torch
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    SendBlockMeta,
    TransferRegion,
    _align_transfer_regions,
    _validate_asymmetric_region_lengths,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    iter_layer_specs,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


logger = init_logger(__name__)


class MooncakeLayerwiseXferMetadata(MooncakeXferMetadata):
    logical_block_size: int = 0
    kernel_block_size: int = 0


@dataclass(frozen=True)
class LayerwisePushReqMeta:
    req_id: str
    transfer_id: str
    block_ids: tuple[tuple[int, ...], ...]
    num_computed_tokens: int
    num_scheduled_tokens: int
    is_final_chunk: bool
    num_prompt_tokens: int


@dataclass
class RequestState:
    """Mutable P state; only the model thread may access it."""

    transfer_id: str
    req_id: str | None = None
    block_ids: tuple[tuple[int, ...], ...] = ()
    num_computed_tokens: int = 0
    num_scheduled_tokens: int = 0
    num_prompt_tokens: int = 0
    is_final_chunk: bool = False
    inference_finished: bool = False
    preempted: bool = False
    sent_tokens: dict[tuple[int, int], int] = field(default_factory=dict)
    available_tokens: dict[int, int] = field(default_factory=dict)
    sealed: bool = False
    error: str | None = None
    completed_ranks: set[int] = field(default_factory=set)
    deadline: float = float("inf")
    destinations: dict[int, tuple[str, MooncakeLayerwiseXferMetadata]] = field(
        default_factory=dict
    )

    @property
    def scheduled_end_tokens(self) -> int:
        return self.num_computed_tokens + self.num_scheduled_tokens


@dataclass(frozen=True)
class LayerwiseSendTask:
    """One request/destination snapshot; contains no transfer regions."""

    transfer_id: str
    req_id: str
    remote_req_id: str
    meta: MooncakeLayerwiseXferMetadata
    block_ids: tuple[int, ...]
    remote_block_offset: int
    # (transformer layer index, first block, exclusive last block).
    layers: tuple[tuple[int, int, int], ...]


@dataclass
class LayerwiseSendResult:
    progress: list[tuple[str, int, int, int]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


class MooncakeLayerwiseMetadata(MooncakeConnectorMetadata):
    def __init__(self) -> None:
        super().__init__()
        self.requests: tuple[LayerwisePushReqMeta, ...] = ()
        self.finished_req_ids: set[str] = set()
        self.preempted_req_ids: set[str] = set()


class MooncakeLayerwiseScheduler(MooncakeConnectorScheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prefills: dict[str, Request] = {}

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        params = request.kv_transfer_params
        if (
            self.is_kv_producer
            and params
            and params.get("do_remote_decode")
            and params.get("transfer_id")
        ):
            self._prefills[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> MooncakeConnectorMetadata:
        base = super().build_connector_meta(scheduler_output)
        if not self.is_kv_producer:
            return base
        meta = MooncakeLayerwiseMetadata()
        meta.reqs_to_send = base.reqs_to_send
        meta.reqs_not_processed = base.reqs_not_processed
        meta.finished_req_ids = set(scheduler_output.finished_req_ids)
        meta.preempted_req_ids = set(scheduler_output.preempted_req_ids or ())
        for req_id in meta.finished_req_ids:
            self._prefills.pop(req_id, None)
        snapshots = []
        for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            request = self._prefills.get(req_id)
            if request is None:
                continue
            computed = request.num_computed_tokens
            if computed >= request.num_prompt_tokens:
                continue
            block_state = scheduler_output.kv_connector_block_state
            blocks = block_state.get_block_ids(req_id) if block_state else None
            if blocks is None:
                raise RuntimeError("Layerwise requires scheduler block state")
            assert request.kv_transfer_params is not None
            snapshots.append(
                LayerwisePushReqMeta(
                    req_id,
                    request.kv_transfer_params["transfer_id"],
                    tuple(tuple(group) for group in self.get_sw_clipped_blocks(blocks)),
                    computed,
                    scheduled,
                    computed + scheduled >= request.num_prompt_tokens,
                    request.num_prompt_tokens,
                )
            )
        meta.requests = tuple(snapshots)
        # P hooks require initialization before forward, including empty steps
        # that retire old state. D retains the original load scheduling policy.
        scheduler_output.has_sync_kv_loads = True
        return meta


@dataclass
class DecodeBatch:
    """Sender-loop ACK context, retaining immutable regions for executor readers."""

    identity: bytes
    sock: zmq.asyncio.Socket
    pending: dict[str, str]
    layout_id: int
    regions_by_layer: dict[int, tuple[list[TransferRegion], list[TransferRegion]]]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class LayerwiseControler:
    """Bridge model-thread transfer decisions and sender-loop control messages.

    Sender-loop owns sockets and the layout LRU. The executor borrows region
    references under a short lock; address planning and I/O run outside it.
    Model-thread ready notifications contain metadata only.
    """

    def __init__(self, worker: "MooncakeLayerwiseWorker") -> None:
        self._worker = worker
        self._ack_contexts: dict[str, dict[int, DecodeBatch]] = {}
        self._contexts_lock = Lock()
        self.send_ready_queue: SimpleQueue[MooncakeLayerwiseXferMetadata] = (
            SimpleQueue()
        )
        # Only the model thread submits ACKs and collects these Futures.
        self._ack_send_futures: list[Future[None]] = []

    def drain_decode_ready(self) -> Iterator[MooncakeLayerwiseXferMetadata]:
        """Model thread: consume ready metadata published by the listener."""
        while True:
            try:
                yield self.send_ready_queue.get_nowait()
            except Empty:
                return

    def get_transfer_regions(
        self, transfer_id: str, rank: int
    ) -> dict[int, tuple[list[TransferRegion], list[TransferRegion]]]:
        """Executor only: borrow regions pinned until this destination is ACKed."""
        with self._contexts_lock:
            batch = self._ack_contexts[transfer_id][rank]
            return batch.regions_by_layer

    def schedule_completion_ack(
        self, transfer_id: str, rank: int, error: str | None
    ) -> None:
        """Model thread: schedule an ACK without touching the ZMQ socket."""
        self._ack_send_futures.append(
            asyncio.run_coroutine_threadsafe(
                self._send_completion_ack(transfer_id, rank, error),
                self._worker.sender_loop,
            )
        )

    def collect_ack_results(self) -> None:
        """Model thread: report completed ACK failures without blocking."""
        pending = []
        for future in self._ack_send_futures:
            if not future.done():
                pending.append(future)
                continue
            try:
                future.result()
            except Exception:
                logger.exception("Failed to deliver layerwise completion ACK")
        self._ack_send_futures = pending

    async def receive_decode_ready(
        self,
        identity: bytes,
        sock: zmq.asyncio.Socket,
        meta: MooncakeLayerwiseXferMetadata,
    ) -> None:
        """Register D-ready and release the listener worker immediately."""
        worker = self._worker
        if (
            meta.remote_pp_size != 1
            or meta.logical_block_size != worker._logical_block_size
            or meta.kernel_block_size != worker.block_size
        ):
            raise ValueError(
                "Layerwise peers require PP1 and equal logical and kernel block sizes"
            )
        if meta.remote_tp_rank not in worker.transfer_topo.handshake_target_ranks(
            meta.remote_tp_size
        ):
            raise ValueError("D TP rank does not pair with this P worker")
        stored = worker._lookup_peer_layout(meta)
        if stored is None:
            if not meta.kv_caches_base_addr:
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.LAYOUT_MISS
                )
                await sock.send_multipart(
                    (identity, self._worker._encoder.encode(response))
                )
                return
            local, remote, error = self._prepare_transfer_regions(meta)
            if error is not None:
                raise ValueError(error)
        else:
            local, remote = stored
        pending: dict[str, str] = {}
        for req_id, (transfer_id, _) in meta.req_blocks.items():
            peers = self._ack_contexts.get(transfer_id, {})
            if meta.remote_tp_rank in peers:
                raise ValueError("Duplicate D-ready for this transfer and TP rank")
            pending[transfer_id] = req_id
        if not pending:
            response = MooncakeXferResponse(status=MooncakeXferResponseStatus.FINISH)
            await sock.send_multipart(
                (identity, self._worker._encoder.encode(response))
            )
            return
        regions_by_layer: dict[
            int, tuple[list[TransferRegion], list[TransferRegion]]
        ] = {}
        # P/D regions have already been aligned by their full cache names.
        for p, d in zip(local, remote):
            pair = regions_by_layer.setdefault(p.layer_index, ([], []))
            pair[0].append(p)
            pair[1].append(d)
        batch = DecodeBatch(identity, sock, pending, meta.layout_id, regions_by_layer)
        with self._contexts_lock:
            for transfer_id in pending:
                self._ack_contexts.setdefault(transfer_id, {})[meta.remote_tp_rank] = (
                    batch
                )
        self.send_ready_queue.put(meta)

    async def _send_completion_ack(
        self, transfer_id: str, rank: int, error: str | None
    ) -> None:
        with self._contexts_lock:
            peers = self._ack_contexts.get(transfer_id)
            if peers is None or rank not in peers:
                return
            batch = peers.pop(rank)
            if not peers:
                del self._ack_contexts[transfer_id]
        async with batch.lock:
            req_id = batch.pending.pop(transfer_id)
            response = MooncakeXferResponse(
                status=(
                    MooncakeXferResponseStatus.CONTINUE
                    if batch.pending
                    else MooncakeXferResponseStatus.FINISH
                ),
                ok_reqs=[req_id] if error is None else None,
                err_reqs=[req_id] if error is not None else None,
                err_msg=error,
                layout_id=batch.layout_id,
            )
            await batch.sock.send_multipart(
                (batch.identity, self._worker._encoder.encode(response))
            )

    def _prepare_transfer_regions(
        self, meta: MooncakeXferMetadata
    ) -> tuple[list[TransferRegion], list[TransferRegion], str | None]:
        worker = self._worker
        # The base implementation coalesces across layers, losing the layer
        # identities needed for readiness. Keep its alignment and TP validation.
        local = worker._get_transfer_regions(
            worker.kv_caches_base_addr,
            worker.block_len_per_layer,
            worker.kv_block_len_per_layer,
            worker.registered_layer_names,
            worker.registered_layer_indices,
            worker.registered_group_indices,
            worker.region_shared_groups,
            worker.region_row_offsets,
        )
        remote = worker._get_transfer_regions(
            meta.kv_caches_base_addr,
            meta.block_lens,
            meta.kv_block_lens,
            meta.registered_layer_names,
            meta.registered_layer_indices,
            meta.registered_group_indices,
            [tuple(groups) for groups in meta.registered_shared_group_ids] or None,
            meta.registered_row_offsets or None,
        )
        local, remote, error = _align_transfer_regions(local, remote)
        if error is None:
            error = worker._validate_head_resharding_layout(meta.remote_tp_size, local)
        if error is None:
            error = _validate_asymmetric_region_lengths(
                local_regions=local,
                remote_regions=remote,
                local_tp_size=worker.tp_size,
                remote_tp_size=meta.remote_tp_size,
                producer_cache_replicated=worker._producer_cache_is_replicated(),
                total_num_kv_heads=(
                    None if worker.use_mla else worker.transfer_topo.total_num_kv_heads
                ),
            )
        worker._remember_peer_layout(meta, (local, remote, error))
        return local, remote, error


class MooncakeLayerwiseWorker(MooncakeConnectorWorker):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.requests_to_send: dict[str, RequestState] = {}
        self._step_transfer_ids: tuple[str, ...] = ()
        self._logical_block_size = self.vllm_config.cache_config.block_size
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeLayerwiseXferMetadata)
        self._control_channel = LayerwiseControler(self)
        self._send_future: Future[LayerwiseSendResult] | None = None
        self._ready_event: torch.Event | None = None
        self._ready_layer_index: int | None = None
        self._retired: dict[str, float] = {}
        self._layer_index_by_name: dict[str, int] = {}
        self._layer_indices: tuple[int, ...] = ()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._kv_layout_id:
            raise RuntimeError("Layerwise does not support replacing KV storage")
        super().register_kv_caches(kv_caches)
        if self.opaque_packed_storages:
            raise ValueError("Layerwise requires individually addressable layers")
        self._layer_index_by_name = dict(
            zip(self.registered_layer_names, self.registered_layer_indices)
        )
        self._layer_indices = tuple(dict.fromkeys(self.registered_layer_indices))

    def shutdown(self) -> None:
        # Drain device readers before the base worker tears down the transport.
        executor = getattr(self, "_sender_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
        super().shutdown()

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ) -> None:
        """Keep the base listener entry point; control state lives in the channel."""
        await self._control_channel.receive_decode_ready(
            identity, sock, cast(MooncakeLayerwiseXferMetadata, meta)
        )

    def _drain_decode_ready(self) -> None:
        for meta in self._control_channel.drain_decode_ready():
            for req_id, (transfer_id, _) in meta.req_blocks.items():
                if transfer_id in self._retired:
                    self._control_channel.schedule_completion_ack(
                        transfer_id, meta.remote_tp_rank, "P transfer already retired"
                    )
                    continue
                state = self.requests_to_send.setdefault(
                    transfer_id, RequestState(transfer_id)
                )
                if state.req_id is None and not state.destinations:
                    state.deadline = (
                        time.monotonic() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                    )
                if meta.remote_tp_rank in state.destinations:
                    self._control_channel.schedule_completion_ack(
                        transfer_id, meta.remote_tp_rank, "Duplicate D-ready"
                    )
                    continue
                state.destinations[meta.remote_tp_rank] = req_id, meta

    def _build_pull_metadata(
        self, pull_metas: dict[str, PullReqMeta], *, include_layout: bool
    ) -> MooncakeLayerwiseXferMetadata:
        base = super()._build_pull_metadata(pull_metas, include_layout=include_layout)
        return MooncakeLayerwiseXferMetadata(
            **{name: getattr(base, name) for name in base.__struct_fields__},
            logical_block_size=self._logical_block_size,
            kernel_block_size=self.block_size,
        )

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        if not self.is_kv_producer:
            super().start_load_kv(metadata)
            return
        assert isinstance(metadata, MooncakeLayerwiseMetadata)
        self._collect_send_results(wait=True)
        self._drain_decode_ready()
        ids = tuple(request.transfer_id for request in metadata.requests)
        for state in self.requests_to_send.values():
            if state.req_id in metadata.preempted_req_ids:
                state.preempted = True
                state.sent_tokens.clear()
                state.available_tokens.clear()
                state.sealed = False
        for request in metadata.requests:
            state = self.requests_to_send.get(request.transfer_id)
            if state is None:
                state = RequestState(request.transfer_id)
                self.requests_to_send[request.transfer_id] = state
            state.req_id = request.req_id
            state.block_ids = request.block_ids
            state.num_computed_tokens = request.num_computed_tokens
            state.num_scheduled_tokens = request.num_scheduled_tokens
            state.num_prompt_tokens = request.num_prompt_tokens
            state.is_final_chunk = request.is_final_chunk
            state.preempted = False
            state.inference_finished = False
            # Scheduled requests are still computing. The send timeout starts
            # only when the scheduler publishes their finished prefill below.
            state.deadline = float("inf")
        for transfer_id, blocks in metadata.reqs_to_send.values():
            if not any(blocks):
                continue
            state = self.requests_to_send[transfer_id]
            state.inference_finished = True
            state.deadline = time.monotonic() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
        aborted = metadata.reqs_not_processed | {
            transfer_id
            for transfer_id, state in self.requests_to_send.items()
            if state.req_id in metadata.finished_req_ids
            and not state.inference_finished
        }
        for transfer_id in aborted:
            state = self.requests_to_send.get(transfer_id)
            if state is not None:
                state.error = "P request ended without a transferable prefill"
                self._finish_destinations(state)
            # The base scheduler already freed aborted requests' blocks.
            self._retire_request(transfer_id)
        self._step_transfer_ids = ids

    def _send_batch(
        self, tasks: tuple[LayerwiseSendTask, ...], event: torch.Event
    ) -> LayerwiseSendResult:
        """Executor: build and aggregate addresses, fence device writes, then WRITE."""
        result = LayerwiseSendResult()
        writes: dict[str, tuple[list[int], list[int], list[int]]] = {}
        try:
            for task in tasks:
                meta = task.meta
                rank = meta.remote_tp_rank
                regions_by_layer = self._control_channel.get_transfer_regions(
                    task.transfer_id, rank
                )
                remote_blocks = meta.req_blocks[task.remote_req_id][1][0]
                for layer_index, start, end in task.layers:
                    local, remote = regions_by_layer[layer_index]
                    send = SendBlockMeta(
                        task.req_id,
                        task.transfer_id,
                        [list(task.block_ids[start:end])],
                        asyncio.Event(),
                    )
                    destination_blocks = [
                        remote_blocks[
                            start - task.remote_block_offset : end
                            - task.remote_block_offset
                        ]
                    ]
                    sliced = msgspec.structs.replace(
                        meta,
                        req_blocks={
                            task.remote_req_id: (task.transfer_id, destination_blocks)
                        },
                    )
                    src, dst, sizes, errors, error = self._build_transfer_params_sync(
                        [(task.remote_req_id, send)],
                        sliced,
                        local,
                        remote,
                    )
                    if errors:
                        result.errors[task.transfer_id] = (
                            error or "Failed to plan layer transfer"
                        )
                        break
                    session = f"{meta.remote_hostname}:{meta.remote_port}"
                    batch = writes.setdefault(session, ([], [], []))
                    batch[0].extend(src)
                    batch[1].extend(dst)
                    batch[2].extend(sizes)
                    result.progress.append(
                        (
                            task.transfer_id,
                            rank,
                            layer_index,
                            end * self._logical_block_size,
                        )
                    )
            event.synchronize()
            for session, (src, dst, sizes) in writes.items():
                if src:
                    ret = self._send_blocks(session, src, dst, sizes)
                    if ret != 0:
                        raise RuntimeError(f"Mooncake WRITE returned {ret}")
        except Exception as exc:
            # Preserve the existing batch-wide failure policy. No partial
            # success is published after an event or transport failure.
            logger.exception("Layerwise send batch failed")
            result.errors.update((task.transfer_id, str(exc)) for task in tasks)
            result.progress.clear()
        return result

    def _collect_send_results(self, *, wait: bool = False) -> None:
        future = self._send_future
        if future is None or (not wait and not future.done()):
            return
        result = future.result()
        for transfer_id, rank, layer_index, end in result.progress:
            self.requests_to_send[transfer_id].sent_tokens[rank, layer_index] = end
        for transfer_id, error in result.errors.items():
            self.requests_to_send[transfer_id].error = error
        self._send_future = None

    def _submit_available_layers(self) -> None:
        """Model thread: snapshot block ranges without accessing regions."""
        if self._send_future is not None:
            return
        tasks = []
        block_size = self._logical_block_size
        now = time.monotonic()
        for state in self.requests_to_send.values():
            if now >= state.deadline:
                state.error = "Layerwise transfer expired before submission"
            if state.preempted or state.error or not state.block_ids:
                continue
            for rank, (req_id, meta) in state.destinations.items():
                if rank in state.completed_ranks:
                    continue
                remote_blocks = meta.req_blocks[req_id][1]
                if not any(remote_blocks):
                    continue
                target_end = (state.num_prompt_tokens + block_size - 1) // block_size
                # D requests the uncached suffix of the full prompt.
                offset = target_end - len(remote_blocks[0])
                if offset < 0:
                    state.error = "D requests more blocks than P's prompt contains"
                    break
                layers = []
                for layer_index in self._layer_indices:
                    if (
                        self._ready_layer_index is not None
                        and layer_index > self._ready_layer_index
                    ):
                        continue
                    available = state.available_tokens.get(layer_index, 0)
                    end = available // block_size
                    if state.sealed or (
                        state.is_final_chunk and available == state.scheduled_end_tokens
                    ):
                        end = (available + block_size - 1) // block_size
                    end = min(end, target_end)
                    start = (
                        state.sent_tokens.get((rank, layer_index), offset * block_size)
                        // block_size
                    )
                    if end > start:
                        layers.append((layer_index, start, end))
                if layers:
                    tasks.append(
                        LayerwiseSendTask(
                            state.transfer_id,
                            state.req_id or "",
                            req_id,
                            meta,
                            state.block_ids[0],
                            offset,
                            tuple(layers),
                        )
                    )
        if tasks and self._ready_event is not None:
            self._send_future = self._sender_executor.submit(
                self._send_batch, tuple(tasks), self._ready_event
            )

    def _progress_sends(self, *, wait: bool = False) -> None:
        self._drain_decode_ready()
        self._collect_send_results(wait=wait)
        self._submit_available_layers()
        if wait:
            # At most one executor batch is outstanding. Drain before the next
            # scheduler step can reuse blocks or write the next partial block.
            self._collect_send_results(wait=True)

    def _finish_destinations(self, state: RequestState) -> bool:
        """ACK each D only after every required region reached its target."""
        for rank, (req_id, meta) in state.destinations.items():
            if rank in state.completed_ranks:
                continue
            blocks = meta.req_blocks[req_id][1]
            if state.error is None and any(blocks):
                if not state.sealed:
                    continue
                block_size = self._logical_block_size
                target = (
                    (state.num_prompt_tokens + block_size - 1) // block_size
                ) * block_size
                if any(
                    state.sent_tokens.get((rank, layer_index), 0) < target
                    for layer_index in self._layer_indices
                ):
                    continue
            self._control_channel.schedule_completion_ack(
                state.transfer_id, rank, state.error
            )
            state.completed_ranks.add(rank)
        if not state.destinations:
            return False
        meta = next(iter(state.destinations.values()))[1]
        expected = set(self.transfer_topo.handshake_target_ranks(meta.remote_tp_size))
        return expected <= state.completed_ranks

    def _retire_request(self, transfer_id: str) -> None:
        self.requests_to_send.pop(transfer_id, None)
        self._retired[transfer_id] = (
            time.monotonic() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
        )

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        if not self.is_kv_producer:
            return super().get_finished()
        # Also runs on empty scheduler steps held alive by delay_free. Late
        # D-ready therefore flushes the sealed KV without another forward.
        self._progress_sends(wait=True)
        now = time.monotonic()
        finished: set[str] = set()
        for transfer_id, state in list(self.requests_to_send.items()):
            complete = self._finish_destinations(state)
            if not complete and state.error is None:
                continue
            if state.inference_finished:
                assert state.req_id is not None
                finished.add(state.req_id)
            elif state.req_id is not None or state.error is None:
                # Keep active P requests, and ready-only requests awaiting P.
                continue
            self._retire_request(transfer_id)
        self._retired = {
            tid: deadline for tid, deadline in self._retired.items() if deadline > now
        }
        self._control_channel.collect_ack_results()
        return finished or None, None

    @staticmethod
    def _record_event() -> torch.Event:
        stream = torch.accelerator.current_stream()
        event = torch.Event()
        event.record(stream)
        return event

    def save_layer(self, layer_name: str) -> None:
        if not self.is_kv_producer or not self._step_transfer_ids:
            return
        layer_index = self._layer_index_by_name.get(layer_name)
        if layer_index is None:
            return
        # The attention save hook must follow all KV writes for this transformer
        # layer, including an independent indexer. Its event fences the group.
        # All KV writes use one stream; this event fences earlier layers too.
        self._ready_event = self._record_event()
        self._ready_layer_index = layer_index
        for transfer_id in self._step_transfer_ids:
            state = self.requests_to_send[transfer_id]
            state.available_tokens[layer_index] = state.scheduled_end_tokens
        self._progress_sends()

    def wait_for_save(self) -> None:
        if not self.is_kv_producer:
            return
        if not self._step_transfer_ids:
            self._progress_sends(wait=True)
            return
        # Retain the final fence for D-ready arriving without another forward.
        self._ready_event = self._record_event()
        self._ready_layer_index = None
        for transfer_id in self._step_transfer_ids:
            state = self.requests_to_send[transfer_id]
            if state.is_final_chunk:
                state.sealed = True
                # End-of-forward fallback for regions without a per-layer hook.
                for layer_index in self._layer_indices:
                    state.available_tokens[layer_index] = state.scheduled_end_tokens
        self._progress_sends(wait=True)
        self._step_transfer_ids = ()


class MooncakeLayerwiseConnector(MooncakeConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        transfer = vllm_config.kv_transfer_config
        if transfer is None or transfer.kv_role not in ("kv_producer", "kv_consumer"):
            raise ValueError(
                "Layerwise requires a fixed kv_role: kv_producer (P) or "
                "kv_consumer (D). kv_both and role switching are not supported."
            )

        if (
            transfer.is_kv_producer
            and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            raise ValueError(
                "Layerwise P requires per-layer Python hooks and cannot use FULL "
                "CUDA graphs. Use PIECEWISE or eager mode on kv_producer."
            )
        assert transfer.engine_id is not None
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise ValueError("Layerwise requires PP1 without context parallelism")
        groups = kv_cache_config.transfer_groups
        if len(groups) != 1:
            raise ValueError("Layerwise requires one full-attention KV group")
        for spec in iter_layer_specs(groups[0].kv_cache_spec):
            if not isinstance(spec, FullAttentionSpec):
                raise ValueError("Layerwise requires full-attention KV specs")
            if spec.sliding_window is not None or spec.attention_chunk_size is not None:
                raise ValueError("Layerwise does not support windowed attention")
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        self.engine_id = transfer.engine_id
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeLayerwiseScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        else:
            self.connector_worker = MooncakeLayerwiseWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    @classmethod
    def supports_sleep_mode(cls, kv_transfer_config: Any) -> bool:
        return False

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        assert isinstance(self.connector_worker, MooncakeLayerwiseWorker)
        self.connector_worker.save_layer(layer_name)

    def wait_for_save(self) -> None:
        assert isinstance(self.connector_worker, MooncakeLayerwiseWorker)
        self.connector_worker.wait_for_save()
