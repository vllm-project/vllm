# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side auxiliary-output data plane."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.aux_output_connector.artifacts import (
    AuxOutputBuffer,
    aux_output_keys,
    materialize_aux_output,
    publish_aux_output,
)
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxRequestOutput,
    retained_objects,
)
from vllm.distributed.aux_output_connector.object_store import (
    MooncakeAuxObjectStore,
)
from vllm.distributed.aux_output_connector.session_objects import (
    SessionObjectWriter,
)
from vllm.distributed.aux_output_connector.store import (
    BackgroundBlockObjectStore,
    BlockObjectStore,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    _get_routed_experts_shape,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.indexer_capture import IndexerCapturer
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.sample.prompt_logprob import CHUNK_SIZE, TOPK_MAX_ELEMENTS

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.input_batch import InputBatch


logger = init_logger(__name__)


PP_ROUTED_EXPERTS_KEY = "routed_experts"


def add_pp_routes_buffer(
    intermediate_tensors: IntermediateTensors,
    vllm_config: VllmConfig,
    max_num_tokens: int,
    device: torch.device,
) -> None:
    if not vllm_config.aux_output_config.enable_return_routed_experts:
        return
    num_layers, _, num_experts_per_tok = _get_routed_experts_shape(vllm_config)
    intermediate_tensors[PP_ROUTED_EXPERTS_KEY] = torch.zeros(
        (max_num_tokens, num_layers, num_experts_per_tok),
        dtype=torch.int32,
        device=device,
    )


@dataclass(slots=True)
class _WorkerRequestState:
    aux_output_keys: list[str] = field(default_factory=list)
    session: str | None = None
    block_hashes: list[bytes] = field(default_factory=list)
    hash_start: int = 0
    staged: list[tuple[int, np.ndarray]] = field(default_factory=list)
    fields: dict[str, tuple[int, int, list[np.ndarray]]] = field(default_factory=dict)
    tail: tuple[str, np.ndarray, dict[str, np.ndarray]] | None = None
    pending_blocks: list[tuple[int, np.ndarray]] = field(default_factory=list)
    capture_cursor: int | None = None
    buffer_start: int = 0
    scheduled_cursor: int = 0
    emit_cursor: int = 0
    pending_outputs: int = 0
    # Terminal event received; teardown waits for in-flight step outputs.
    finished: bool = False


@dataclass(slots=True)
class PendingAuxOutput:
    """Own one step's capture snapshots until the asynchronous copy completes."""

    connector: AuxOutputWorkerConnector
    request_ids: list[str]
    token_starts: np.ndarray
    query_start_loc: np.ndarray
    # GPU snapshots taken on the main stream.
    routed_experts_snapshot: torch.Tensor | None
    # Filled by enqueue_cpu_copy on the output copy stream.
    routed_experts: np.ndarray | None = None
    num_sampled: np.ndarray | None = None
    num_rejected: np.ndarray | None = None
    indexer_snapshot: torch.Tensor | None = None
    indexer: np.ndarray | None = None
    fields_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)
    fields: dict[str, np.ndarray] = field(default_factory=dict)

    def enqueue_cpu_copy(
        self, num_sampled: np.ndarray, num_rejected: np.ndarray
    ) -> None:
        """Enqueue asynchronous D2H copies; call on the output copy stream."""
        self.num_sampled = num_sampled
        self.num_rejected = num_rejected

        def copy_rows(tensor: torch.Tensor) -> np.ndarray:
            host = torch.empty_like(tensor, device="cpu", pin_memory=tensor.is_cuda)
            host.copy_(tensor, non_blocking=True)
            return host.numpy()

        if self.routed_experts_snapshot is not None:
            self.routed_experts = copy_rows(self.routed_experts_snapshot)
        if self.indexer_snapshot is not None:
            self.indexer = copy_rows(self.indexer_snapshot)
        self.fields = {
            name: copy_rows(rows) for name, rows in self.fields_snapshot.items()
        }

    def process_output(self) -> dict[str, AuxRequestOutput]:
        return self.connector.process_output(self)


class AuxOutputWorkerConnector:
    """Own capture, request tails, and backend resources on the output worker."""

    _return_hidden_states = False
    _topk_logprobs = 0

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
    ) -> None:
        self._vllm_config = vllm_config
        self._routed_experts_capturer: RoutedExpertsCapturer | None = None
        self._indexer: IndexerCapturer | None = None
        try:
            if vllm_config.aux_output_config.enable_return_routed_experts:
                self._routed_experts_capturer = RoutedExpertsCapturer(
                    max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                    vllm_config=vllm_config,
                )
                bind_routed_experts_capturer(model, self._routed_experts_capturer)
            if vllm_config.aux_output_config.enable_return_indexer_topk:
                self._indexer = IndexerCapturer(
                    model, vllm_config, output_rank=get_tp_group().is_first_rank
                )
        except BaseException:
            self._close_capture()
            raise
        self._return_hidden_states = (
            vllm_config.aux_output_config.enable_return_hidden_states
        )
        self._topk_logprobs = vllm_config.aux_output_config.return_topk_logprobs
        self._session_request_ids: set[str] = set()
        self._session_generation = 0
        self._r3_shape: tuple[int, ...] = ()
        self._r3_dtype: np.dtype[Any] = np.dtype("uint8")
        self._r3_nbytes = 0
        self._store: BackgroundBlockObjectStore | None = None
        self._buffer: AuxOutputBuffer | None = None
        self._object_writer: SessionObjectWriter | None = None
        self._object_size = 0
        self._requests: dict[str, _WorkerRequestState] = {}
        self._generation = 0
        self._step_metadata: AuxOutputConnectorMetadata | None = None
        # Steps whose asynchronous copy has not been consumed yet. The engine
        # consumes step outputs in order, so at most max_concurrent_batches
        # are outstanding; guarded by _lock against the output thread.
        self._pending_outputs: list[PendingAuxOutput] = []
        self._lock = Lock()
        self._max_concurrent_batches = vllm_config.max_concurrent_batches

    def _close_capture(self) -> None:
        """Release both capturers; safe to call on a partially built connector."""
        capturers = self._routed_experts_capturer, self._indexer
        self._routed_experts_capturer = None
        self._indexer = None
        for capturer in capturers:
            if capturer is None:
                continue
            try:
                capturer.close()
            except Exception:
                logger.warning(
                    "Failed to close auxiliary-output capture", exc_info=True
                )

    def initialize_output_storage(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize the SHM data plane after KV capacity is resolved."""
        vllm_config = self._vllm_config
        if vllm_config.aux_output_config.enable_return_routed_experts:
            assert self._routed_experts_capturer is not None, (
                "capture must be initialized before storage"
            )
        if vllm_config.aux_output_config.enable_return_indexer_topk:
            assert self._indexer is not None, (
                "capture must be initialized before storage"
            )
        if self._store is not None or self._buffer is not None:
            raise RuntimeError("auxiliary-output storage is already initialized")
        if not get_tp_group().is_first_rank or not get_pp_group().is_last_rank:
            return
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        shape_per_token, dtype = self._get_storage_layout(max_num_batched_tokens)
        num_objects, hash_block_size = retained_objects(vllm_config, kv_cache_config)
        block_nbytes = hash_block_size * int(np.prod(shape_per_token)) * dtype.itemsize
        max_bytes = num_objects * block_nbytes
        store = BackgroundBlockObjectStore(
            BlockObjectStore(max_bytes=max_bytes, object_nbytes=block_nbytes),
            max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
        )
        try:
            buffer = AuxOutputBuffer(
                dtype,
                shape_per_token,
                hash_block_size,
                vllm_config.scheduler_config.max_num_seqs,
                max_num_batched_tokens,
                vllm_config.max_concurrent_batches,
            )
        except BaseException:
            try:
                store.close()
            except Exception:
                logger.warning(
                    "Failed to close the auxiliary-output store after "
                    "initialization failed",
                    exc_info=True,
                )
            raise
        self._store = store
        self._buffer = buffer
        key_prefix = vllm_config.aux_output_config.object_key_prefix
        if key_prefix is not None:
            self._object_writer = SessionObjectWriter(
                MooncakeAuxObjectStore.create(
                    vllm_config.aux_output_config.mooncake_config_path,
                    vllm_config.aux_output_config.object_hard_pin,
                ),
                key_prefix=key_prefix,
                max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
                fields=self._object_fields,
                put_stall_timeout_s=min(
                    vllm_config.aux_output_config.object_put_stall_timeout_s,
                    envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS / 2,
                ),
            )

    def _get_storage_layout(
        self, max_num_batched_tokens: int
    ) -> tuple[tuple[int, ...], np.dtype[Any]]:
        if self._routed_experts_capturer is not None:
            self._r3_shape = self._routed_experts_capturer.shape_per_token
            self._r3_dtype = np.dtype(self._routed_experts_capturer.output_dtype_name)
            self._r3_nbytes = int(np.prod(self._r3_shape)) * self._r3_dtype.itemsize
        if self._indexer is None:
            if self._routed_experts_capturer is None:
                return (1,), np.dtype("uint8")
            return self._r3_shape, self._r3_dtype
        assert len(self._indexer.device_buffer) >= max_num_batched_tokens, (
            "auxiliary-output capture capacity is smaller than the token budget"
        )
        return (
            (self._r3_nbytes + self._indexer.fixed_bytes_per_token,),
            np.dtype("uint8"),
        )

    @cached_property
    def _pp_start_layer(self) -> int:
        pp_group = get_pp_group()
        assert self._routed_experts_capturer is not None
        num_layers = self._routed_experts_capturer.shape_per_token[0]
        start_layer, _ = get_pp_indices(
            num_layers, pp_group.rank_in_group, pp_group.world_size
        )
        return start_layer

    def receive_pp_routes(
        self, intermediate_tensors: IntermediateTensors, num_tokens: int
    ) -> None:
        if not self._vllm_config.aux_output_config.enable_return_routed_experts:
            return
        start = self._pp_start_layer
        received = intermediate_tensors[PP_ROUTED_EXPERTS_KEY]
        assert self._routed_experts_capturer is not None
        buffer = self._routed_experts_capturer.device_buffer
        buffer[:num_tokens, :start] = received[:num_tokens, :start]

    def send_pp_routes(
        self, output_intermediate_tensors: IntermediateTensors
    ) -> IntermediateTensors:
        if not self._vllm_config.aux_output_config.enable_return_routed_experts:
            return output_intermediate_tensors
        assert self._routed_experts_capturer is not None
        return IntermediateTensors(
            output_intermediate_tensors.tensors
            | {PP_ROUTED_EXPERTS_KEY: self._routed_experts_capturer.device_buffer}
        )

    def _object_fields(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        output = self._make_output(0, rows)
        fields = {
            "routed_experts": output.routed_rows,
            "indexer_topk": output.indexer_rows,
        }
        return {name: rows for name, rows in fields.items() if rows is not None}

    def prepare_output(
        self,
        input_batch: InputBatch,
        hidden_states: torch.Tensor | None = None,
        logits_fn: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
    ) -> PendingAuxOutput | None:
        """Snapshot one step's auxiliary tensors for asynchronous CPU transfer.

        `hidden_states` are what the model's forward pass returned, one row per
        query token, on the last pipeline stage, and `logits_fn` computes logits
        from them. Call on every rank: the top-k logprobs need the logits
        collective.
        """
        if self._step_metadata is None or (
            self._buffer is None and not self._topk_logprobs
        ):
            return None
        request_ids = list(input_batch.req_ids)
        query_start_loc = input_batch.query_start_loc_np[: len(request_ids) + 1]
        num_rows = int(query_start_loc[-1])
        rows = session_rows(request_ids, query_start_loc, self._session_request_ids)
        topk = None
        if (
            self._topk_logprobs
            and len(rows)
            and hidden_states is not None
            and logits_fn is not None
        ):
            index = async_tensor_h2d(rows, hidden_states.device)
            topk = compute_topk_logprobs(
                hidden_states.index_select(0, index),
                logits_fn,
                self._topk_logprobs,
                store=self._buffer is not None,
            )
        if self._buffer is None or not (
            len(rows) or self._routed_experts_capturer or self._indexer
        ):
            return None

        snapshot = self._indexer.snapshot(num_rows) if self._indexer else None
        fields: dict[str, torch.Tensor] = {}
        if len(rows) and self._return_hidden_states and hidden_states is not None:
            index = async_tensor_h2d(rows, hidden_states.device)
            hidden = hidden_states.index_select(0, index)
            if hidden.dtype == torch.bfloat16:
                hidden = hidden.view(torch.int16)
            fields["hidden_states"] = hidden
        if topk is not None:
            fields["topk_token_ids"], fields["topk_logprobs"] = topk
        pending_output = PendingAuxOutput(
            connector=self,
            request_ids=request_ids,
            token_starts=input_batch.num_computed_tokens_np[: len(request_ids)],
            query_start_loc=query_start_loc,
            routed_experts_snapshot=(
                self._routed_experts_capturer.snapshot_routing_data(num_rows)
                if self._routed_experts_capturer
                else None
            ),
            indexer_snapshot=snapshot,
            fields_snapshot=fields,
        )
        with self._lock:
            assert len(self._pending_outputs) < self._max_concurrent_batches, (
                "auxiliary output step outputs are not consumed in order"
            )
            self._pending_outputs.append(pending_output)
            for request_id in pending_output.request_ids:
                if request_id in self._requests:
                    self._requests[request_id].pending_outputs += 1
        return pending_output

    def process_output(self, pending: PendingAuxOutput) -> dict[str, AuxRequestOutput]:
        """Commit one consumed R3 snapshot and build request outputs.

        Request teardown deferred by begin_step is completed here once no
        in-flight step covers the request.
        """
        with self._lock:
            teardown = []
            release_keys: list[str] = []
            try:
                outputs = self._commit_output(pending)
            finally:
                self._pending_outputs.remove(pending)
                for request_id in pending.request_ids:
                    state = self._requests.get(request_id)
                    if state is None:
                        continue
                    state.pending_outputs -= 1
                    if state.pending_outputs == 0 and state.finished:
                        teardown.append(request_id)
                        release_keys.extend(reversed(state.aux_output_keys))
            if teardown:
                self._publish_blocks([], release_keys=release_keys)
                self._teardown(teardown)
            return outputs

    def _commit_output(self, pending: PendingAuxOutput) -> dict[str, AuxRequestOutput]:
        buffer = self._buffer
        store = self._store
        assert buffer is not None and store is not None
        block_size = buffer.block_size

        packed_rows = pending.routed_experts
        if self._indexer is not None:
            assert pending.indexer is not None
            indexer_rows = (
                np.ascontiguousarray(pending.indexer)
                .view(np.uint8)
                .reshape(len(pending.indexer), -1)
            )
            if packed_rows is None:
                packed_rows = indexer_rows
            else:
                packed_rows = np.concatenate(
                    (
                        np.ascontiguousarray(packed_rows)
                        .view(np.uint8)
                        .reshape(len(packed_rows), -1),
                        indexer_rows,
                    ),
                    axis=1,
                )
        captured = packed_rows is not None
        if not captured:
            packed_rows = np.zeros((pending.query_start_loc[-1], 1), dtype=np.uint8)
        num_sampled = pending.num_sampled
        num_rejected = pending.num_rejected
        assert (
            packed_rows is not None
            and num_sampled is not None
            and num_rejected is not None
        ), "auxiliary output CPU copy was not enqueued"

        # Publish the whole batch before materializing any consumer output.
        materialize_outputs: list[tuple[str, int, int]] = []
        block_batches = []
        outputs: dict[str, AuxRequestOutput] = {}
        field_start = 0

        # Use the ModelRunner's actual batch boundaries rather than rebuilding them.
        for request_id, token_start, start, end, sampled, rejected in zip(
            pending.request_ids,
            pending.token_starts,
            pending.query_start_loc[:-1],
            pending.query_start_loc[1:],
            num_sampled,
            num_rejected,
            strict=True,
        ):
            request_num_tokens = end - start
            assert request_num_tokens > 0, (
                "auxiliary output request token count must be positive"
            )
            state = self._requests.get(request_id)
            if state is None:
                assert not captured, f"auxiliary output request {request_id} is unknown"
                continue

            # Capture precedes speculative acceptance, so discard the rejected
            # suffix. Batch boundaries still span the full executed range.
            rejected = int(rejected)
            assert 0 <= rejected <= request_num_tokens, (
                "auxiliary output rejected-token count is invalid"
            )
            rows = packed_rows[start : end - rejected]

            capture_start = token_start
            capture_cursor = state.capture_cursor
            if capture_cursor is None:
                capture_cursor = capture_start
                # KV loaded up to mid-block has no rows here for that block.
                state.buffer_start = round_up(capture_start, block_size)

            assert capture_start >= capture_cursor, (
                "auxiliary output capture moved backwards"
            )
            if capture_start > capture_cursor:
                # Reattach after an optimistically scheduled suffix was rejected.
                assert capture_cursor < state.scheduled_cursor, (
                    "auxiliary output capture has an unbacked token gap"
                )
                capture_start = capture_cursor
            if state.session is not None:
                for name, values in pending.fields.items():
                    first, _, chunks = state.fields.get(name, (capture_start, 0, []))
                    chunk = values[field_start : field_start + len(rows)]
                    chunks.append(chunk if len(chunk) == len(values) else chunk.copy())
                    state.fields[name] = first, capture_start + len(rows), chunks
                field_start += request_num_tokens

            skip = max(state.buffer_start - capture_start, 0)

            emit_start = state.emit_cursor
            # Complete blocks without keys remain pending until a hash update.
            completed = buffer.capture(request_id, capture_start + skip, rows[skip:])
            token_end = capture_start + len(rows)
            state.capture_cursor = token_end
            state.scheduled_cursor = token_start + request_num_tokens
            block_batches.append((state, completed))

            if state.session is None and sampled > 0 and emit_start <= token_end:
                if emit_start >= capture_start:
                    output_slice = slice(
                        start + emit_start - capture_start, end - rejected
                    )
                    outputs[request_id] = AuxRequestOutput(
                        token_start=emit_start,
                        routed_rows=(
                            pending.routed_experts[output_slice]
                            if pending.routed_experts is not None
                            else None
                        ),
                        indexer_rows=(
                            pending.indexer[output_slice]
                            if pending.indexer is not None
                            else None
                        ),
                    )
                    state.emit_cursor = token_end
                else:
                    materialize_outputs.append((request_id, emit_start, token_end))

        # A consumer may reuse a block produced earlier in the same batch.
        self._publish_blocks(block_batches)

        for request_id, emit_start, token_end in materialize_outputs:
            state = self._requests[request_id]
            stored_end = (
                min(token_end // block_size, len(state.aux_output_keys)) * block_size
            )
            if emit_start < stored_end:
                first_block = emit_start // block_size
                stored = materialize_aux_output(
                    store,
                    state.aux_output_keys[first_block : stored_end // block_size],
                    shape_per_token=buffer.shape_per_token,
                    dtype=buffer.dtype,
                )
                local_start = emit_start % block_size
                rows = stored[local_start : local_start + stored_end - emit_start]
                if stored_end < token_end:
                    suffix = buffer.read(
                        request_id, stored_end, token_end, state.pending_blocks
                    )
                    rows = np.concatenate((rows, suffix))
            else:
                rows = buffer.read(
                    request_id, emit_start, token_end, state.pending_blocks
                )
            outputs[request_id] = self._make_output(emit_start, rows)
            state.emit_cursor = token_end
        return outputs

    def _make_output(self, token_start: int, rows: np.ndarray) -> AuxRequestOutput:
        if self._indexer is None:
            return AuxRequestOutput(
                token_start=token_start,
                routed_rows=(
                    rows if self._routed_experts_capturer is not None else None
                ),
            )
        routed_rows = None
        combined = self._routed_experts_capturer is not None
        if combined:
            routed_rows = self._unpack_rows(
                rows[:, : self._r3_nbytes],
                self._r3_dtype,
                self._r3_shape,
                copy=True,
            )
        indexer = self._unpack_rows(
            rows[:, self._r3_nbytes :],
            np.dtype("int32"),
            self._indexer.shape_per_token,
            copy=combined,
        )
        return AuxRequestOutput(
            token_start=token_start,
            routed_rows=routed_rows,
            indexer_rows=indexer,
        )

    @staticmethod
    def _unpack_rows(
        rows: np.ndarray,
        dtype: np.dtype[Any],
        shape_per_token: tuple[int, ...],
        *,
        copy: bool,
    ) -> np.ndarray:
        # Detach packed artifacts into contiguous arrays and keep store-backed
        # outputs writable. A writable single-artifact array can remain a view.
        if copy or not rows.flags.writeable:
            rows = rows.copy()
        return rows.view(dtype).reshape(len(rows), *shape_per_token)

    def _publish_blocks(
        self,
        batches: list[tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]],
        retain_keys: Sequence[str] = (),
        release_keys: Sequence[str] = (),
    ) -> None:
        store = self._store
        buffer = self._buffer
        assert store is not None and buffer is not None
        ready_batches = []
        for state, completed in batches:
            blocks = state.pending_blocks + completed
            keys = (
                state.aux_output_keys if state.session is None else state.block_hashes
            )
            keyed_end = (state.hash_start + len(keys)) * buffer.block_size
            ready = [(start, rows) for start, rows in blocks if start < keyed_end]
            state.pending_blocks = [
                (start, buffer.retain_block(rows))
                for start, rows in blocks
                if start >= keyed_end
            ]
            if not ready:
                continue
            if state.session is None:
                ready_batches.append((state.aux_output_keys, ready))
                continue
            assert self._object_writer is not None
            self._object_writer.add_blocks(state, ready, self._object_size)
            for _, rows in ready:
                buffer.release_block(rows)
        if ready_batches or retain_keys or release_keys:
            publish_aux_output(
                store,
                batches=ready_batches,
                block_size=buffer.block_size,
                retain_keys=retain_keys,
                release_keys=release_keys,
            )
        for _, blocks in ready_batches:
            for _, rows in blocks:
                buffer.release_block(rows)

    def _teardown(self, request_ids: Iterable[str]) -> None:
        """Drop finished requests' temporary state after publishing key releases."""
        buffer = self._buffer
        assert buffer is not None
        for request_id in request_ids:
            state = self._requests.pop(request_id)
            if state.tail is not None:
                assert self._object_writer is not None
                self._object_writer.add_tail(*state.tail)
            for _, rows in state.pending_blocks:
                buffer.release_block(rows)
            buffer.discard(request_id)

    def _get_namespace(self, generation: int) -> str:
        namespace = str(generation)
        if self._indexer is not None:
            namespace += "/" + self._indexer.namespace
            if self._routed_experts_capturer is not None:
                namespace += f"/r3-{self._r3_shape}-{self._r3_dtype.name}"
        return namespace

    def _take_tail(
        self, request_id: str, state: _WorkerRequestState, tail: tuple[str, int, int]
    ) -> None:
        assert self._buffer is not None
        key, start, end = tail
        blocks = state.staged + state.pending_blocks
        state.tail = (
            key,
            self._buffer.read(request_id, start, end, blocks),
            SessionObjectWriter.field_rows(state, start, end),
        )

    def begin_step(self, metadata: AuxOutputConnectorMetadata | None) -> None:
        """Apply one scheduler step's request and block-hash updates.

        While an unconsumed step covers a finished request, its teardown is
        deferred to that step's process_output.
        """
        self._step_metadata = metadata
        if (self._topk_logprobs or self._return_hidden_states) and metadata is not None:
            if metadata.generation != self._session_generation:
                self._session_request_ids.clear()
                self._session_generation = metadata.generation
            self._session_request_ids.update(metadata.sessions)
            self._session_request_ids.difference_update(metadata.finished_requests)
        if self._object_writer is not None:
            self._object_writer.check()
        if self._buffer is None or metadata is None:
            return
        self._object_size = metadata.session_object_size
        assert not metadata.requests.keys() & metadata.finished_requests, (
            "auxiliary output request cannot run and finish in one step"
        )
        assert metadata.generation >= self._generation, (
            "auxiliary output metadata generation moved backwards"
        )
        with self._lock:
            namespace = self._get_namespace(self._generation)
            hashes = metadata.retained_block_hashes or ()
            retained_keys = aux_output_keys(hashes, namespace)
            hashes = metadata.released_block_hashes or ()
            release_keys = aux_output_keys(hashes, namespace)
            carried: list[str] = []
            if metadata.generation > self._generation:
                # A generation change follows a prefix-cache reset, which the
                # scheduler only performs once all model output is consumed.
                assert not self._pending_outputs, (
                    "auxiliary output generation changed with output in flight"
                )
                carried = [
                    request_id
                    for request_id in metadata.finished_requests
                    if request_id in self._requests
                ]
                for request_id in carried:
                    state = self._requests[request_id]
                    if request_id in metadata.session_tails:
                        tail = metadata.session_tails[request_id]
                        self._take_tail(request_id, state, tail)
                    state.block_hashes.extend(metadata.block_hashes.get(request_id, ()))
                flushed = [
                    request_id
                    for request_id in metadata.flushed_block_hashes
                    if request_id in self._requests
                ]
                for request_id in flushed:
                    self._requests[request_id].block_hashes.extend(
                        metadata.flushed_block_hashes[request_id]
                    )
                self._publish_blocks(
                    [(self._requests[r], []) for r in carried + flushed]
                )
                self._teardown(carried)
                release_keys.extend(
                    key
                    for state in self._requests.values()
                    for key in reversed(state.aux_output_keys)
                )
                self._buffer.reset()
                self._requests.clear()
                self._generation = metadata.generation
            for request_id, emit_start in metadata.requests.items():
                state = self._requests.setdefault(
                    request_id, _WorkerRequestState(emit_cursor=emit_start)
                )
                state.session = metadata.sessions.get(request_id, state.session)
                state.hash_start = metadata.session_hash_starts.get(
                    request_id, state.hash_start
                )
                if state.pending_outputs == 0:
                    # In-flight steps leave the worker cursor behind the
                    # scheduler's optimistic view; only check settled requests.
                    assert emit_start <= state.emit_cursor, (
                        "auxiliary output Scheduler emit cursor moved ahead"
                    )
            block_batches: list[
                tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]
            ] = []
            for request_id, block_hashes in metadata.block_hashes.items():
                if request_id in carried:
                    continue
                state = self._requests[request_id]
                if state.session is not None:
                    state.block_hashes.extend(block_hashes)
                else:
                    namespace = self._get_namespace(self._generation)
                    keys = aux_output_keys(block_hashes, namespace)
                    state.aux_output_keys.extend(keys)
                    retained_keys.extend(keys)
                block_batches.append((state, []))
            finished_now: list[str] = []
            for request_id in metadata.finished_requests:
                if request_id in carried:
                    continue
                state = self._requests[request_id]
                state.finished = True
                if request_id in metadata.session_tails:
                    tail = metadata.session_tails[request_id]
                    self._take_tail(request_id, state, tail)
                if state.pending_outputs:
                    continue
                finished_now.append(request_id)
                release_keys.extend(reversed(state.aux_output_keys))
            self._publish_blocks(block_batches, retained_keys, release_keys)
            self._teardown(finished_now)

    def drain(self, metadata: AuxOutputConnectorMetadata) -> None:
        self.begin_step(metadata)
        self._step_metadata = None
        if self._object_writer is not None:
            self._object_writer.drain()

    def close(self) -> None:
        with self._lock:
            self._pending_outputs.clear()
            writer, self._object_writer = self._object_writer, None
            store, self._store = self._store, None
            try:
                if writer is not None:
                    writer.close()
            finally:
                try:
                    if store is not None:
                        store.close()
                finally:
                    self._buffer = None
                    self._requests.clear()
                    self._step_metadata = None
                    self._close_capture()


def session_rows(
    request_ids: Sequence[str],
    query_start_loc: np.ndarray,
    session_request_ids: set[str],
) -> np.ndarray:
    """Batch rows of the requests in `session_request_ids`."""
    ranges = [
        np.arange(query_start_loc[i], query_start_loc[i + 1])
        for i, request_id in enumerate(request_ids)
        if request_id in session_request_ids
    ]
    return np.concatenate(ranges) if ranges else np.empty(0, dtype=np.int64)


def compute_topk_logprobs(
    hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor | None],
    k: int,
    chunk_size: int = CHUNK_SIZE,
    store: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Top-k next-token ids (int32) and log-probabilities (float32) of every
    row, or None on ranks that do not receive the logits or do not `store`
    them; every rank computes the logits, a collective. Chunks hold at most
    `TOPK_MAX_ELEMENTS // (k + 1)` rows, as for prompt logprobs."""
    chunk_size = min(chunk_size, max(1, TOPK_MAX_ELEMENTS // (k + 1)))
    token_ids, logprobs = [], []
    for start in range(0, len(hidden_states), chunk_size):
        logits = logits_fn(hidden_states[start : start + chunk_size])
        if logits is None or not store:
            continue
        values, indices = torch.log_softmax(logits.float(), dim=-1).topk(k, dim=-1)
        token_ids.append(indices.to(torch.int32))
        logprobs.append(values)
    if not token_ids:
        return None
    return torch.cat(token_ids), torch.cat(logprobs)


def get_aux_output_connector(
    model: torch.nn.Module, vllm_config: VllmConfig
) -> AuxOutputWorkerConnector:
    return AuxOutputWorkerConnector(model=model, vllm_config=vllm_config)
