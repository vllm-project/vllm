# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake layerwise: background address planning and device transfers.

Group-aware layerwise transfer with PP and explicitly replicated PCP caches.
KV writes use a single compute stream; DCP is not supported.
Model hooks snapshot sends; the executor builds addresses, waits and performs WRITE.
The listener loop owns D sockets and ACKs. Requests retain KV through delay_free.
"""

import asyncio
import time
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from queue import Empty, SimpleQueue
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
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.layerwise_mapping import (
    map_pending_blocks,
    map_state_blocks,
    map_window_blocks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    _MAX_PEER_LAYOUTS,
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
from vllm.distributed.parallel_state import get_pcp_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    SlidingWindowSpec,
    iter_layer_specs,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


logger = init_logger(__name__)

WorkerKey = tuple[str, int, int, int]


class MooncakeLayerwiseXferMetadata(MooncakeXferMetadata):
    remote_engine_id: str = ""
    remote_pp_rank: int = 0
    remote_pcp_rank: int = 0
    remote_pcp_size: int = 1
    pcp_cache_mode: str = "replicated"
    cache_specs: dict[str, str] = msgspec.field(default_factory=dict)
    logical_block_size: int = 0
    kernel_block_size: int = 0
    recv_timeout_s: float = 0.0

    @property
    def worker_key(self) -> WorkerKey:
        return (
            self.remote_engine_id,
            self.remote_tp_rank,
            self.remote_pp_rank,
            self.remote_pcp_rank,
        )


class LayerwiseXferResponse(MooncakeXferResponse):
    covered_layers: list[str] = msgspec.field(default_factory=list)


@dataclass(frozen=True)
class LayerwisePushReqMeta:
    req_id: str
    transfer_id: str
    block_ids: tuple[tuple[int, ...], ...]
    scheduled_end_tokens: int
    num_prompt_tokens: int


@dataclass(frozen=True)
class Destination:
    remote_req_id: str
    meta: MooncakeLayerwiseXferMetadata
    layout: "LayerwisePeerLayout"
    send_deadline: float
    required_layers: tuple[str, ...]


@dataclass
class RequestState:
    """Mutable P state; only the model thread may access it."""

    transfer_id: str
    req_id: str | None = None
    block_ids: tuple[tuple[int, ...], ...] = ()
    scheduled_end_tokens: int = 0
    num_prompt_tokens: int = 0
    inference_finished: bool = False
    preempted: bool = False
    sent_tokens: dict[tuple[WorkerKey, str], int] = field(default_factory=dict)
    available_tokens: dict[int, int] = field(default_factory=dict)
    sealed: bool = False
    error: str | None = None
    ack_scheduled_workers: set[WorkerKey] = field(default_factory=set)
    deadline: float = float("inf")
    destinations: dict[WorkerKey, Destination] = field(default_factory=dict)

    @property
    def is_final_chunk(self) -> bool:
        return self.scheduled_end_tokens >= self.num_prompt_tokens

    def mark_preempted(self) -> None:
        self.preempted = True
        self.sent_tokens.clear()
        self.available_tokens.clear()
        self.sealed = False

    def update_from_metadata(self, metadata: LayerwisePushReqMeta) -> None:
        """Apply the current prefill step's request metadata."""
        self.req_id = metadata.req_id
        self.block_ids = metadata.block_ids
        self.scheduled_end_tokens = metadata.scheduled_end_tokens
        self.num_prompt_tokens = metadata.num_prompt_tokens
        self.preempted = False
        self.inference_finished = False
        # The send timeout starts when the scheduler publishes finished prefill.
        self.deadline = float("inf")


@dataclass(frozen=True)
class LayerwiseSendTask:
    """Block snapshot retaining the destination's registered layout."""

    transfer_id: str
    req_id: str
    destination: Destination
    kv_cache_name: str
    block_ids: tuple[tuple[int, ...], ...]
    remote_block_ids: tuple[tuple[int, ...], ...]
    end_tokens: int


@dataclass
class LayerwiseSendResult:
    progress: list[tuple[str, WorkerKey, str, int]] = field(default_factory=list)
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
        self._prefill_requests: dict[str, Request] = {}
        for index, group in enumerate(self.kv_cache_config.transfer_groups):
            specs = list(iter_layer_specs(group.kv_cache_spec))
            windows = {getattr(spec, "sliding_window", None) for spec in specs}
            if len(windows) != 1:
                raise ValueError("Mixed window semantics within a transfer group")
            # A windowed FullAttentionSpec still uses the full-attention
            # manager. Its whole allocated suffix must be populated before D
            # can publish prefix-cache hashes for those blocks.
            if next(iter(windows)) is not None and all(
                isinstance(spec, SlidingWindowSpec) for spec in specs
            ):
                self._is_hma_required = True
                self.blocks_per_sw[index] = max(
                    (
                        getattr(
                            spec,
                            "num_retained_tokens",
                            spec.sliding_window - 1 + spec.extra_retained_tokens,
                        )
                        + spec.block_size
                        - 1
                    )
                    // spec.block_size
                    + 1
                    for spec in specs
                )

    def request_finished(self, request, block_ids):
        delayed, params = super().request_finished(request, block_ids)
        if delayed and self.is_kv_producer:
            # Final allocation state is authoritative, but retain logical null
            # positions for windows and align-mode state slots.
            self._reqs_need_send[request.request_id] = (
                request,
                [
                    list(g)
                    for g in self.kv_cache_config.select_transfer_block_ids(block_ids)
                ],
            )
        return delayed, params

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
            self._prefill_requests[request.request_id] = request

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
            self._prefill_requests.pop(req_id, None)
        snapshots = []
        for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            request = self._prefill_requests.get(req_id)
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
                    tuple(
                        tuple(group)
                        for group in self.kv_cache_config.select_transfer_block_ids(
                            blocks
                        )
                    ),
                    computed + scheduled,
                    request.num_prompt_tokens,
                )
            )
        meta.requests = tuple(snapshots)
        # P hooks require initialization before forward, including empty steps
        # that retire old state. D retains the original load scheduling policy.
        scheduler_output.has_sync_kv_loads = True
        return meta


@dataclass(frozen=True)
class LayerwisePeerLayout:
    regions_by_layer: dict[str, tuple[list[TransferRegion], list[TransferRegion]]]
    remote_group_by_layer: dict[str, int]
    remote_kv_cache_names: dict[int, list[str]]
    covered_layers: list[str]
    expected_workers: frozenset[WorkerKey]


@dataclass
class DecodeBatch:
    """Sender-loop-owned socket and pending ACKs for one ready message."""

    identity: bytes
    sock: zmq.asyncio.Socket
    pending: dict[str, str]
    layout_id: int
    layout: LayerwisePeerLayout
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class LayerwiseController:
    """Bridge model-thread transfer decisions and sender-loop control messages.

    Sender-loop owns sockets, ACK contexts and the layout LRU. The queue
    publishes destinations whose layouts remain alive through task references.
    """

    def __init__(self, worker: "MooncakeLayerwiseWorker") -> None:
        self._worker = worker
        self._ack_contexts: dict[tuple[str, WorkerKey], DecodeBatch] = {}
        self._layouts: OrderedDict[tuple, LayerwisePeerLayout] = OrderedDict()
        self.send_ready_queue: SimpleQueue[Destination] = SimpleQueue()

    def get_decode_ready_requests(self) -> Iterator[Destination]:
        """Yield and remove queued D-ready metadata without waiting for new messages."""
        while True:
            try:
                yield self.send_ready_queue.get_nowait()
            except Empty:
                return

    def schedule_completion_ack(
        self, transfer_id: str, worker_key: WorkerKey, error: str | None
    ) -> None:
        """Model thread: schedule an ACK without touching the ZMQ socket."""
        future = asyncio.run_coroutine_threadsafe(
            self._send_completion_ack(transfer_id, worker_key, error),
            self._worker.sender_loop,
        )
        future.add_done_callback(self._log_ack_result)

    @staticmethod
    def _log_ack_result(future: Future[None]) -> None:
        """Completion callback: log failures without accessing request state."""
        try:
            future.result()
        except Exception:
            logger.exception("Failed to deliver layerwise completion ACK")

    async def receive_decode_ready(
        self,
        identity: bytes,
        sock: zmq.asyncio.Socket,
        meta: MooncakeLayerwiseXferMetadata,
    ) -> None:
        """Register D-ready and release the listener worker immediately."""
        worker = self._worker
        # The send margin covers ready's network/listener delay. Start before
        # layout processing so it and the model-thread queue consume this budget.
        send_deadline = time.monotonic() + meta.recv_timeout_s
        if (
            meta.remote_pcp_size < 1
            or not 0 <= meta.remote_pcp_rank < meta.remote_pcp_size
            or meta.pcp_cache_mode != "replicated"
        ):
            raise ValueError("Layerwise PCP requires complete cache replicas")
        if (
            meta.logical_block_size != worker.logical_block_size
            or meta.kernel_block_size != worker.block_size
        ):
            raise ValueError(
                "Layerwise peers require equal logical and kernel block sizes"
            )
        if meta.remote_tp_rank not in worker.transfer_topo.handshake_target_ranks(
            meta.remote_tp_size
        ):
            raise ValueError("D TP rank does not pair with this P worker")
        key = (meta.worker_key, worker._peer_layout_key(meta), meta.remote_pcp_size)
        layout = self._layouts.get(key) if meta.layout_id else None
        if layout is None:
            if not meta.kv_caches_base_addr:
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.LAYOUT_MISS
                )
                await sock.send_multipart(
                    (identity, self._worker._encoder.encode(response))
                )
                return
            local, remote = self._prepare_transfer_regions(meta)
            regions_by_layer: dict[
                str, tuple[list[TransferRegion], list[TransferRegion]]
            ] = {}
            for p, d in zip(local, remote):
                pair = regions_by_layer.setdefault(p.layer_name, ([], []))
                pair[0].append(p)
                pair[1].append(d)
            remote_groups: dict[str, int] = {}
            for name, group in zip(
                meta.registered_layer_names, meta.registered_group_indices
            ):
                if group < 0 or remote_groups.setdefault(name, group) != group:
                    raise ValueError(f"Ambiguous remote group for {name}")
            remote_kv_cache_names: dict[int, list[str]] = {}
            for name in regions_by_layer:
                remote_kv_cache_names.setdefault(
                    worker._layer_index_by_name[name], []
                ).append(name)
            layout = LayerwisePeerLayout(
                regions_by_layer,
                remote_groups,
                remote_kv_cache_names,
                sorted(regions_by_layer),
                frozenset(
                    (meta.remote_engine_id, tp, pp, pcp)
                    for tp in worker.transfer_topo.handshake_target_ranks(
                        meta.remote_tp_size
                    )
                    for pp in range(meta.remote_pp_size)
                    for pcp in range(meta.remote_pcp_size)
                ),
            )
            if meta.layout_id:
                self._layouts[key] = layout
                while len(self._layouts) > _MAX_PEER_LAYOUTS:
                    self._layouts.popitem(last=False)
        else:
            self._layouts.move_to_end(key)
        pending: dict[str, str] = {}
        for req_id, (transfer_id, _) in meta.req_blocks.items():
            if (transfer_id, meta.worker_key) in self._ack_contexts:
                raise ValueError("Duplicate D-ready for this transfer and worker")
            pending[transfer_id] = req_id
        if not pending:
            response = MooncakeXferResponse(status=MooncakeXferResponseStatus.FINISH)
            await sock.send_multipart(
                (identity, self._worker._encoder.encode(response))
            )
            return
        batch = DecodeBatch(identity, sock, pending, meta.layout_id, layout)
        for transfer_id, req_id in pending.items():
            blocks = meta.req_blocks[req_id][1]
            required_layers = tuple(
                name
                for name in layout.regions_by_layer
                if blocks and blocks[layout.remote_group_by_layer[name]]
            )
            self._ack_contexts[transfer_id, meta.worker_key] = batch
            self.send_ready_queue.put(
                Destination(req_id, meta, layout, send_deadline, required_layers)
            )

    async def _send_completion_ack(
        self, transfer_id: str, worker_key: WorkerKey, error: str | None
    ) -> None:
        batch = self._ack_contexts.pop((transfer_id, worker_key), None)
        if batch is None:
            return
        async with batch.lock:
            req_id = batch.pending.pop(transfer_id)
            response = LayerwiseXferResponse(
                covered_layers=batch.layout.covered_layers,
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
        self, meta: MooncakeLayerwiseXferMetadata
    ) -> tuple[list[TransferRegion], list[TransferRegion]]:
        worker = self._worker
        # The base implementation coalesces across layers, losing the layer
        # identities needed for readiness. Keep its alignment and TP validation.
        local = worker._local_transfer_regions
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
        if any(r.group_index < 0 for r in (*local, *remote)):
            raise ValueError(
                "Layerwise requires individually attributable cache regions"
            )
        for index, region in enumerate(remote):
            if region.layer_name in worker._layer_specs:
                group = worker.get_group_from_layer_name(region.layer_name)
                remote[index] = replace(
                    region, group_index=group, shared_group_ids=(group,)
                )
        for kv_cache_name in set(worker._layer_specs) & set(
            meta.registered_layer_names
        ):
            if isinstance(
                worker._layer_specs[kv_cache_name], (MambaSpec, CircularBufferSpec)
            ) and (meta.remote_tp_size != worker.tp_size):
                raise ValueError("Recurrent states currently require equal TP sizes")
            if (
                meta.cache_specs.get(kv_cache_name)
                != worker._cache_specs[kv_cache_name]
            ):
                raise ValueError(f"Incompatible cache semantics for {kv_cache_name}")
        local, remote, error = _align_transfer_regions(
            local, remote, allow_partial_layers=True
        )
        if error:
            raise ValueError(error)
        if (worker.tp_size == meta.remote_tp_size or worker.use_mla) and any(
            p.kv_block_len != d.kv_block_len for p, d in zip(local, remote)
        ):
            raise ValueError("Peer cache component payload lengths differ")
        error = worker._validate_head_resharding_layout(
            meta.remote_tp_size, local
        ) or _validate_asymmetric_region_lengths(
            local_regions=local,
            remote_regions=remote,
            local_tp_size=worker.tp_size,
            remote_tp_size=meta.remote_tp_size,
            producer_cache_replicated=worker._producer_cache_is_replicated(),
            total_num_kv_heads=(
                None if worker.use_mla else worker.transfer_topo.total_num_kv_heads
            ),
        )
        if error:
            raise ValueError(error)
        return local, remote


class MooncakeLayerwiseWorker(MooncakeConnectorWorker):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        kv_config = self.vllm_config.kv_transfer_config
        assert kv_config is not None
        self.send_timeout_margin_s = float(
            kv_config.get_from_extra_config("layerwise_send_timeout_margin_s", 10)
        )
        if not 0 <= self.send_timeout_margin_s < float("inf"):
            raise ValueError("layerwise_send_timeout_margin_s must be finite and >= 0")
        self.requests_to_send: dict[str, RequestState] = {}
        # Transfer IDs of requests prefilling in the current scheduler step.
        self.prefilling_transfer_ids: tuple[str, ...] = ()
        self.logical_block_size = self.vllm_config.cache_config.block_size
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeLayerwiseXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(LayerwiseXferResponse)
        self._received_layers: dict[str, set[str]] = {}
        self._control_channel = LayerwiseController(self)
        self._pending_sends: list[
            tuple[tuple[LayerwiseSendTask, ...], Future[LayerwiseSendResult]]
        ] = []
        self._ready_event: torch.Event | None = None
        self.retired_transfer_expiry_times: dict[str, float] = {}
        self._layer_index_by_name: dict[str, int] = {}
        self._cache_specs: dict[str, str] = {}
        self._local_transfer_regions: list[TransferRegion] = []
        self._kernel_block_ratios: tuple[int, ...] = ()

    @property
    def pcp_rank(self) -> int:
        if self.vllm_config.parallel_config.prefill_context_parallel_size == 1:
            return 0
        return get_pcp_group().rank_in_group

    def _should_launch_bootstrap_server(self) -> bool:
        return self.pcp_rank == 0 and super()._should_launch_bootstrap_server()

    async def register_worker_with_bootstrap(self):
        # The base listener publishes this worker before register_kv_caches returns.
        self._init_layerwise_layout()
        # Only replica zero is an advertised source. The remaining ranks still
        # take part in model execution and worker completion aggregation.
        if self.pcp_rank == 0:
            await super().register_worker_with_bootstrap()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._kv_layout_id:
            raise RuntimeError("Layerwise does not support replacing KV storage")
        super().register_kv_caches(kv_caches)
        if self.is_kv_consumer:
            self._init_layerwise_layout()

    def _init_layerwise_layout(self) -> None:
        if self.opaque_packed_storages:
            raise ValueError("Layerwise requires individually addressable layers")
        self._layer_index_by_name = dict(
            zip(self.registered_layer_names, self.registered_layer_indices)
        )
        if self._layer_index_by_name.keys() != self._layer_specs.keys():
            raise ValueError("Registration collapsed cache writer identities")
        self._cache_specs = {
            name: self._spec_signature(name) for name in self._layer_specs
        }
        self._kernel_block_ratios = tuple(
            self._physical_blocks_for_spec(
                next(iter(iter_layer_specs(group.kv_cache_spec)))
            )
            for group in self.kv_cache_config.transfer_groups
        )
        self._local_transfer_regions = [
            replace(region, shared_group_ids=(region.group_index,))
            for region in self._get_transfer_regions(
                self.kv_caches_base_addr,
                self.block_len_per_layer,
                self.kv_block_len_per_layer,
                self.registered_layer_names,
                self.registered_layer_indices,
                self.registered_group_indices,
                self.region_shared_groups,
                self.region_row_offsets,
            )
        ]

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

    def process_decode_ready_requests(self) -> None:
        for destination in self._control_channel.get_decode_ready_requests():
            meta = destination.meta
            transfer_id, _ = meta.req_blocks[destination.remote_req_id]
            if transfer_id in self.retired_transfer_expiry_times:
                self._control_channel.schedule_completion_ack(
                    transfer_id, meta.worker_key, "P transfer already retired"
                )
                continue
            state = self.requests_to_send.get(transfer_id)
            if state is None:
                state = RequestState(transfer_id)
                self.requests_to_send[transfer_id] = state
            if state.req_id is None and not state.destinations:
                state.deadline = (
                    time.monotonic() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
            if meta.worker_key in state.destinations:
                self._control_channel.schedule_completion_ack(
                    transfer_id, meta.worker_key, "Duplicate D-ready"
                )
                continue
            state.destinations[meta.worker_key] = destination

    def get_group_from_layer_name(self, kv_cache_name: str) -> int:
        return self.kv_cache_config.transfer_group_index_by_layer[kv_cache_name]

    def _spec_signature(self, kv_cache_name: str) -> str:
        spec = self._layer_specs[kv_cache_name]
        if isinstance(spec, CircularBufferSpec):
            return repr(
                (
                    type(spec).__name__,
                    spec.block_size,
                    spec.page_size_bytes,
                    spec.head_size,
                    str(spec.dtype),
                    spec.tokens_per_state,
                )
            )
        if isinstance(spec, MambaSpec):
            return repr(
                (
                    type(spec).__name__,
                    spec.block_size,
                    spec.shapes,
                    tuple(str(dtype) for dtype in spec.dtypes),
                    spec.mamba_type,
                    spec.mamba_cache_mode,
                    spec.num_speculative_blocks,
                    spec.num_prefill_checkpoint_blocks,
                    spec.tokens_per_state,
                    spec.tp_replicated,
                )
            )
        return repr(
            (
                type(spec).__name__,
                spec.block_size,
                str(spec.dtype),
                spec.head_size,
                getattr(spec, "head_size_v", None),
                str(spec.tokens_per_state),
                getattr(spec, "cache_dtype_str", None),
                str(spec.kv_quant_mode),
                getattr(spec, "model_version", None),
                getattr(spec, "sliding_window", None),
                getattr(spec, "extra_retained_tokens", 0),
                getattr(spec, "bounded_replay", False),
            )
        )

    def _logical_to_kernel_block_ids(self, block_ids):
        return [
            [block * ratio + offset for block in ids for offset in range(ratio)]
            for ratio, ids in zip(self._kernel_block_ratios, block_ids)
        ]

    def _preserve_cache_region_identity(self) -> bool:
        return True

    def _physical_blocks_for_spec(self, spec):
        if isinstance(spec, (MambaSpec, CircularBufferSpec)):
            return 1
        if spec.block_size % self.block_size:
            raise ValueError("Group block size must divide into kernel blocks")
        return spec.block_size // self.block_size

    def receive_kv(self, remote_engine_id, pull_metas):
        # Query all P stages even for equal PP sizes: layer partitions can differ.
        ranks = self.transfer_topo.handshake_target_ranks(
            self._tp_size[remote_engine_id]
        )
        addresses = {
            address
            for rank in ranks
            for address in self._remote_agents[remote_engine_id][rank].values()
        }
        if not addresses:
            self._handle_failed_recv(pull_metas, list(pull_metas), "No source stages")
            return
        for pull in pull_metas.values():
            pull.pull_tasks_count = len(addresses)
            self._received_layers[pull.d_req_id] = set()
        for address in addresses:
            asyncio.create_task(self.receive_kv_from_single_worker(address, pull_metas))

    def process_pulling_result(self, response, pull_metas):
        for req_id in response.ok_reqs or ():
            pull = pull_metas.get(req_id)
            if pull is None:
                continue
            covered = self._received_layers[pull.d_req_id]
            covered.update(response.covered_layers)
            if (
                pull.pull_tasks_count == 1
                and not pull.failed
                and not self._layer_index_by_name.keys() <= covered
            ):
                raise RuntimeError("Producer stages omitted required cache layers")
        super().process_pulling_result(response, pull_metas)

    def _complete_pull_task(self, pull_meta):
        super()._complete_pull_task(pull_meta)
        if pull_meta.pull_tasks_count == 0:
            self._received_layers.pop(pull_meta.d_req_id, None)

    def _build_pull_metadata(
        self, pull_metas: dict[str, PullReqMeta], *, include_layout: bool
    ) -> MooncakeLayerwiseXferMetadata:
        base = super()._build_pull_metadata(pull_metas, include_layout=include_layout)
        return MooncakeLayerwiseXferMetadata(
            **{name: getattr(base, name) for name in base.__struct_fields__},
            cache_specs=self._cache_specs if include_layout else {},
            remote_engine_id=self.engine_id,
            remote_pp_rank=self.pp_rank,
            remote_pcp_rank=self.pcp_rank,
            remote_pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
            logical_block_size=self.logical_block_size,
            kernel_block_size=self.block_size,
            # Match D's base receiver RCVTIMEO without exchanging clock values.
            recv_timeout_s=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60,
        )

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        if not self.is_kv_producer:
            super().start_load_kv(metadata)
            return
        assert isinstance(metadata, MooncakeLayerwiseMetadata)
        # wait_for_save must drain device readers before replacing block tables.
        assert not self._pending_sends
        self.process_decode_ready_requests()
        ids = tuple(request.transfer_id for request in metadata.requests)
        # Reset preempted progress before applying this step's prefill metadata.
        for state in self.requests_to_send.values():
            if state.req_id in metadata.preempted_req_ids:
                state.mark_preempted()
        for request in metadata.requests:
            state = self.requests_to_send.get(request.transfer_id)
            if state is None:
                state = RequestState(request.transfer_id)
                self.requests_to_send[request.transfer_id] = state
            state.update_from_metadata(request)
        # Mooncake uses empty blocks for initialization, final blocks for handoff.
        for transfer_id, blocks in metadata.reqs_to_send.values():
            if not any(blocks):
                continue
            state = self.requests_to_send[transfer_id]
            state.block_ids = tuple(tuple(group) for group in blocks)
            state.inference_finished = True
            state.deadline = time.monotonic() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
        # Finished requests without a normal handoff are treated as aborted.
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
            # Retire worker state; KV block release belongs to the scheduler.
            self._retire_request(transfer_id)
        self.prefilling_transfer_ids = ids
        # Steps without prefill work drain transfers here.
        if not ids:
            self._process_pending_transfers(wait=True)

    def _send_batch(
        self, session: str, tasks: tuple[LayerwiseSendTask, ...], event: torch.Event
    ) -> LayerwiseSendResult:
        """Executor: prepare and send one destination's batch."""
        result = LayerwiseSendResult()
        planned_transfers = []
        for layer_transfer_task in tasks:
            destination = layer_transfer_task.destination
            meta = destination.meta
            regions_by_layer = destination.layout.regions_by_layer
            local, remote = regions_by_layer[layer_transfer_task.kv_cache_name]
            send_meta = SendBlockMeta(
                layer_transfer_task.req_id,
                layer_transfer_task.transfer_id,
                [list(g) for g in layer_transfer_task.block_ids],
                asyncio.Event(),
            )
            task_meta = msgspec.structs.replace(
                meta,
                req_blocks={
                    destination.remote_req_id: (
                        layer_transfer_task.transfer_id,
                        [list(g) for g in layer_transfer_task.remote_block_ids],
                    )
                },
            )
            src, dst, sizes, errors, error = self._build_transfer_params_sync(
                [(destination.remote_req_id, send_meta)],
                task_meta,
                local,
                remote,
            )
            if errors:
                result.errors[layer_transfer_task.transfer_id] = (
                    error or "Address planning failed"
                )
                continue
            planned_transfers.append((layer_transfer_task, src, dst, sizes))
        event.synchronize()
        now = time.monotonic()
        for layer_transfer_task in tasks:
            if (
                now + self.send_timeout_margin_s
                >= layer_transfer_task.destination.send_deadline
            ):
                result.errors[layer_transfer_task.transfer_id] = (
                    "D receive deadline too close for WRITE"
                )
        batch_src: list[int] = []
        batch_dst: list[int] = []
        batch_sizes: list[int] = []
        for layer_transfer_task, src, dst, sizes in planned_transfers:
            # Drop every layer of a failed request, retaining healthy requests
            # in the same WRITE batch.
            if layer_transfer_task.transfer_id in result.errors:
                continue
            batch_src.extend(src)
            batch_dst.extend(dst)
            batch_sizes.extend(sizes)
            result.progress.append(
                (
                    layer_transfer_task.transfer_id,
                    layer_transfer_task.destination.meta.worker_key,
                    layer_transfer_task.kv_cache_name,
                    layer_transfer_task.end_tokens,
                )
            )
        if batch_src:
            ret = self._send_blocks(session, batch_src, batch_dst, batch_sizes)
            if ret != 0:
                raise RuntimeError(f"Mooncake WRITE returned {ret}")
        return result

    def _finish_pending_send(self, *, wait: bool = False) -> None:
        if not self._pending_sends:
            return
        if not wait and any(not future.done() for _, future in self._pending_sends):
            return
        # Drain every batch before allowing completion or block release.
        for tasks, future in self._pending_sends:
            try:
                result = future.result()
            except Exception as exc:
                logger.exception("Layerwise send batch failed")
                error = str(exc)
                for layer_transfer_task in tasks:
                    self.requests_to_send[layer_transfer_task.transfer_id].error = error
            else:
                for transfer_id, worker_key, layer_name, end in result.progress:
                    self.requests_to_send[transfer_id].sent_tokens[
                        worker_key, layer_name
                    ] = end
                for transfer_id, error in result.errors.items():
                    self.requests_to_send[transfer_id].error = error
        self._pending_sends.clear()

    def _build_layer_transfer_task(self, state, destination, kv_cache_name):
        meta = destination.meta
        local_group_index = self.get_group_from_layer_name(kv_cache_name)
        remote_group_index = destination.layout.remote_group_by_layer[kv_cache_name]
        _, remote_block_ids = meta.req_blocks[destination.remote_req_id]
        remote_group_block_ids = remote_block_ids[remote_group_index]
        if not remote_group_block_ids:
            return None
        spec = self._layer_specs[kv_cache_name]
        block_size = spec.block_size
        layer_index = self._layer_index_by_name[kv_cache_name]
        available_tokens = state.available_tokens[layer_index]
        final_layer_ready = state.sealed or (
            state.is_final_chunk and available_tokens == state.scheduled_end_tokens
        )
        sent_tokens = state.sent_tokens.get((meta.worker_key, kv_cache_name))
        is_state = isinstance(spec, (MambaSpec, CircularBufferSpec))
        is_window = getattr(spec, "sliding_window", None) is not None
        # State and window caches are sent once, when this layer's final KV is ready.
        if (is_state or is_window) and (
            not final_layer_ready or sent_tokens is not None
        ):
            return None
        if is_state:
            block_mapping = map_state_blocks(
                state.block_ids[local_group_index],
                remote_group_block_ids,
                token_end=state.num_prompt_tokens,
                null_block=NULL_BLOCK_ID,
            )
        elif is_window:
            block_mapping = map_window_blocks(
                state.block_ids[local_group_index],
                remote_group_block_ids,
                token_end=state.num_prompt_tokens,
                block_size=block_size,
                null_block=NULL_BLOCK_ID,
            )
        else:
            block_mapping = map_pending_blocks(
                state.block_ids[local_group_index],
                remote_group_block_ids,
                token_end=state.num_prompt_tokens,
                available=available_tokens,
                sent=sent_tokens,
                block_size=block_size,
                final=final_layer_ready,
            )
        if block_mapping is None:
            return None
        local_groups = [()] * len(state.block_ids)
        remote_groups = [()] * len(state.block_ids)
        local_groups[local_group_index] = block_mapping.source
        remote_groups[local_group_index] = block_mapping.destination
        return LayerwiseSendTask(
            transfer_id=state.transfer_id,
            req_id=cast(str, state.req_id),
            destination=destination,
            kv_cache_name=kv_cache_name,
            block_ids=tuple(local_groups),
            remote_block_ids=tuple(remote_groups),
            end_tokens=block_mapping.end_tokens,
        )

    def _submit_available_layers(self) -> None:
        """Model thread: snapshot block ranges and retain their destination layouts."""
        if self._pending_sends:
            return
        session_tasks: dict[str, list[LayerwiseSendTask]] = {}
        now = time.monotonic()
        for state in self.requests_to_send.values():
            if now >= state.deadline:
                state.error = "Layerwise transfer expired before submission"
            if any(
                worker_key not in state.ack_scheduled_workers
                and destination.required_layers
                and now + self.send_timeout_margin_s >= destination.send_deadline
                for worker_key, destination in state.destinations.items()
            ):
                state.error = "D receive deadline too close for WRITE"
            if state.preempted or state.error or not state.block_ids:
                continue
            for worker_key, destination in state.destinations.items():
                meta = destination.meta
                if (
                    worker_key in state.ack_scheduled_workers
                    or not destination.required_layers
                ):
                    continue
                remote_kv_cache_names = destination.layout.remote_kv_cache_names
                session = f"{meta.remote_hostname}:{meta.remote_port}"
                try:
                    for layer_index in state.available_tokens:
                        for kv_cache_name in remote_kv_cache_names.get(layer_index, ()):
                            layer_transfer_task = self._build_layer_transfer_task(
                                state, destination, kv_cache_name
                            )
                            if layer_transfer_task is not None:
                                session_tasks.setdefault(session, []).append(
                                    layer_transfer_task
                                )
                except ValueError as exc:
                    state.error = str(exc)
                    break
        if session_tasks and self._ready_event is not None:
            for session, tasks in session_tasks.items():
                task_snapshot = tuple(
                    layer_transfer_task
                    for layer_transfer_task in tasks
                    if self.requests_to_send[layer_transfer_task.transfer_id].error
                    is None
                )
                if not task_snapshot:
                    continue
                future = self._sender_executor.submit(
                    self._send_batch, session, task_snapshot, self._ready_event
                )
                self._pending_sends.append((task_snapshot, future))

    def _process_pending_transfers(self, wait: bool = False) -> None:
        self.process_decode_ready_requests()
        self._finish_pending_send(wait=wait)
        self._submit_available_layers()
        if wait:
            # Finish all destination batches before advancing transfer state.
            self._finish_pending_send(wait=True)

    def _finish_destinations(self, state: RequestState) -> bool:
        """ACK each D only after every required region reached its target."""
        for worker_key, destination in state.destinations.items():
            if worker_key in state.ack_scheduled_workers:
                continue
            required_layers = destination.required_layers
            if (
                state.error is None
                and required_layers
                and (
                    not state.sealed
                    or any(
                        state.sent_tokens.get((worker_key, name), -1)
                        < state.num_prompt_tokens
                        for name in required_layers
                    )
                )
            ):
                continue
            self._control_channel.schedule_completion_ack(
                state.transfer_id, worker_key, state.error
            )
            state.ack_scheduled_workers.add(worker_key)
        if not state.destinations:
            return False
        layout = next(iter(state.destinations.values())).layout
        return layout.expected_workers <= state.ack_scheduled_workers

    def _retire_request(self, transfer_id: str) -> None:
        self.requests_to_send.pop(transfer_id, None)
        if self.pcp_rank == 0:
            self.retired_transfer_expiry_times[transfer_id] = (
                time.monotonic() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
            )

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        if not self.is_kv_producer:
            return super().get_finished()
        if self.pcp_rank != 0:
            finished = set()
            for transfer_id, state in list(self.requests_to_send.items()):
                if state.inference_finished and state.req_id is not None:
                    finished.add(state.req_id)
                    self.requests_to_send.pop(transfer_id)
            return finished or None, None
        # Speculative runners may collect output before wait_for_save. Completion
        # must not release blocks until that step's send barrier has run.
        if self._pending_sends:
            return None, None
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
        self.retired_transfer_expiry_times = {
            transfer_id: deadline
            for transfer_id, deadline in self.retired_transfer_expiry_times.items()
            if deadline > now
        }
        return finished or None, None

    @staticmethod
    def _record_event() -> torch.Event:
        stream = torch.accelerator.current_stream()
        event = torch.Event()
        event.record(stream)
        return event

    def save_kv_layer(self, layer_name: str) -> None:
        if not self.is_kv_producer or not self.prefilling_transfer_ids:
            return
        layer_index = self._layer_index_by_name.get(layer_name)
        if layer_index is None:
            return
        # The attention save hook must follow all KV writes for this transformer
        # layer across groups, including an independent indexer.
        # All KV writes use one stream; this event fences earlier layers too.
        self._ready_event = self._record_event()
        for transfer_id in self.prefilling_transfer_ids:
            state = self.requests_to_send[transfer_id]
            state.available_tokens[layer_index] = state.scheduled_end_tokens
        self._process_pending_transfers()

    def wait_for_save(self) -> None:
        if not self.is_kv_producer or not self.prefilling_transfer_ids:
            return
        # Retain the final fence for D-ready arriving without another forward.
        self._ready_event = self._record_event()
        layer_indices = set(self._layer_index_by_name.values())
        for transfer_id in self.prefilling_transfer_ids:
            state = self.requests_to_send[transfer_id]
            if state.is_final_chunk:
                state.sealed = True
                # End-of-forward fallback for regions without a per-layer hook.
                for layer_index in layer_indices:
                    state.available_tokens[layer_index] = state.scheduled_end_tokens
        self._process_pending_transfers(wait=True)
        self.prefilling_transfer_ids = ()


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
        if parallel.decode_context_parallel_size != 1:
            raise ValueError("Layerwise requires DCP1")
        if parallel.prefill_context_parallel_size > 1:
            if transfer.get_from_extra_config("pcp_cache_mode", None) != "replicated":
                raise ValueError("PCP requires explicit pcp_cache_mode=replicated")
            if any(
                spec.dcp_sharded
                for group in kv_cache_config.transfer_groups
                for spec in iter_layer_specs(group.kv_cache_spec)
            ):
                raise ValueError("PCP-sharded cache cannot be sent as a replica")
            if role == KVConnectorRole.WORKER:
                transfer.engine_id = get_pcp_group().broadcast_object(
                    transfer.engine_id, src=0
                )
        groups = kv_cache_config.transfer_groups
        if not groups:
            raise ValueError("Layerwise requires transfer groups")
        for spec in (
            spec for group in groups for spec in iter_layer_specs(group.kv_cache_spec)
        ):
            if not isinstance(
                spec,
                (FullAttentionSpec, SlidingWindowSpec, MambaSpec, CircularBufferSpec),
            ):
                raise ValueError("Layerwise requires full-attention KV specs")
            if getattr(spec, "attention_chunk_size", None) is not None:
                raise ValueError("Chunked-local attention is not yet supported")
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
        self.connector_worker.save_kv_layer(layer_name)

    def wait_for_save(self) -> None:
        assert isinstance(self.connector_worker, MooncakeLayerwiseWorker)
        self.connector_worker.wait_for_save()
