# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side auxiliary-output data plane."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

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
)
from vllm.distributed.aux_output_connector.logprobs import (
    LogprobRows,
    concat_rows,
    decode_rows,
    encode_rows,
    rows_from_lists,
    rows_to_lists,
)
from vllm.distributed.aux_output_connector.store import (
    BackgroundBlockObjectStore,
    BlockObject,
    BlockObjectStore,
    VariableBlockObjectStore,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.indexer_capture import IndexerCapturer
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu.input_batch import InputBatch


logger = init_logger(__name__)


@dataclass
class _WorkerRequestState:
    aux_output_keys: list[str] = field(default_factory=list)
    pending_blocks: list[tuple[int, np.ndarray]] = field(default_factory=list)
    capture_cursor: int | None = None
    scheduled_cursor: int = 0
    emit_cursor: int = 0
    pending_outputs: int = 0
    # Terminal event received; teardown waits for in-flight step outputs.
    finished: bool = False
    logprob_keys: list[str] = field(default_factory=list)
    pending_logprob_blocks: list[tuple[bytes, str]] = field(default_factory=list)
    logprob_rows: dict[int, LogprobRows] = field(default_factory=dict)
    pending_publish_logprobs: set[int] = field(default_factory=set)
    prompt_len: int = 0


@dataclass
class PendingAuxOutput:
    """Own one step's capture snapshots until the asynchronous copy completes."""

    connector: AuxOutputWorkerConnector
    request_ids: list[str]
    batch_indices: np.ndarray
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
    logprobs_tensors: LogprobsTensors | None = None
    logprobs: dict[str, Any] = field(default_factory=dict)
    logprob_replays: set[str] = field(default_factory=set)
    existing_logprob_artifacts: dict[str, bytes | None] = field(default_factory=dict)
    aux_logprobs: frozenset[str] = frozenset()
    logprob_widths: dict[str, int] = field(default_factory=dict)

    def enqueue_cpu_copy(
        self,
        num_sampled: np.ndarray,
        num_rejected: np.ndarray,
        logprobs: LogprobsTensors | None = None,
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
        self.logprobs_tensors = logprobs if self.aux_logprobs else None

    def process_output(self) -> dict[str, AuxRequestOutput]:
        return self.connector.process_output(self)


class AuxOutputWorkerConnector:
    """Own capture, request tails, and backend resources on the output worker."""

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
        self._r3_shape: tuple[int, ...] = ()
        self._r3_dtype: np.dtype[Any] = np.dtype("uint8")
        self._r3_nbytes = 0
        self._store: BackgroundBlockObjectStore | None = None
        self._logprob_store: BackgroundBlockObjectStore | None = None
        self._buffer: AuxOutputBuffer | None = None
        self._requests: dict[str, _WorkerRequestState] = {}
        self._generation = 0
        self._step_metadata: AuxOutputConnectorMetadata | None = None
        # Steps whose asynchronous copy has not been consumed yet. The engine
        # consumes step outputs in order, so at most max_concurrent_batches
        # are outstanding; guarded by _lock against the output thread.
        self._pending_outputs: list[PendingAuxOutput] = []
        self._lock = Lock()
        self._max_concurrent_batches = vllm_config.max_concurrent_batches
        self._logprob_block_size = 1
        self._enable_logprobs = vllm_config.aux_output_config.enable_return_logprobs

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
        if not get_tp_group().is_first_rank:
            return
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self._logprob_block_size = hash_block_size
        if self._routed_experts_capturer is not None or self._indexer is not None:
            shape_per_token, dtype = self._get_storage_layout(max_num_batched_tokens)
            bytes_per_token = int(np.prod(shape_per_token)) * dtype.itemsize
            block_nbytes = hash_block_size * bytes_per_token
            max_bytes = vllm_config.aux_output_config.max_bytes
            if max_bytes is None:
                max_bytes = (
                    kv_cache_config.num_blocks * scheduler_block_size * bytes_per_token
                )
            store = BackgroundBlockObjectStore(
                BlockObjectStore(max_bytes=max_bytes, object_nbytes=block_nbytes),
                max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
            )
            try:
                self._buffer = AuxOutputBuffer(
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
        if self._enable_logprobs:
            logprob_max_bytes = vllm_config.aux_output_config.max_bytes
            if logprob_max_bytes is None:
                max_logprobs = max(vllm_config.model_config.max_logprobs, 0)
                bytes_per_row = max(16, (max_logprobs + 1) * 8 + 4)
                num_hash_blocks = max(
                    kv_cache_config.num_blocks
                    * scheduler_block_size
                    // self._logprob_block_size,
                    1,
                )
                logprob_max_bytes = num_hash_blocks * (
                    self._logprob_block_size * bytes_per_row + 4096
                )
            self._logprob_store = BackgroundBlockObjectStore(
                VariableBlockObjectStore(max_bytes=logprob_max_bytes),
                max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
            )

    def _get_storage_layout(
        self, max_num_batched_tokens: int
    ) -> tuple[tuple[int, ...], np.dtype[Any]]:
        if self._routed_experts_capturer is not None:
            self._r3_shape = self._routed_experts_capturer.shape_per_token
            self._r3_dtype = np.dtype(self._routed_experts_capturer.output_dtype_name)
            self._r3_nbytes = int(np.prod(self._r3_shape)) * self._r3_dtype.itemsize
        if self._indexer is None:
            assert self._routed_experts_capturer is not None
            return self._r3_shape, self._r3_dtype
        assert len(self._indexer.device_buffer) >= max_num_batched_tokens, (
            "auxiliary-output capture capacity is smaller than the token budget"
        )
        return (
            (self._r3_nbytes + self._indexer.fixed_bytes_per_token,),
            np.dtype("uint8"),
        )

    def prepare_output(self, input_batch: InputBatch) -> PendingAuxOutput | None:
        """Snapshot one step's capture tensors for asynchronous CPU transfer."""
        if (
            self._buffer is None and self._logprob_store is None
        ) or self._step_metadata is None:
            return None
        if self._buffer is None and not self._step_metadata.logprobs:
            return None
        if self._buffer is None:
            request_ids = [
                req_id
                for req_id in input_batch.req_ids
                if req_id in self._step_metadata.logprobs
            ]
            if not request_ids:
                return None
            indices = np.asarray(
                [input_batch.req_ids.index(req_id) for req_id in request_ids],
                dtype=np.int32,
            )
            query_start_loc = np.zeros(len(request_ids) + 1, dtype=np.int32)
            np.cumsum(
                input_batch.num_scheduled_tokens[indices], out=query_start_loc[1:]
            )
            token_starts = input_batch.num_computed_tokens_np[indices]
        else:
            request_ids = list(input_batch.req_ids)
            indices = np.arange(len(request_ids), dtype=np.int32)
            query_start_loc = input_batch.query_start_loc_np[: len(request_ids) + 1]
            token_starts = input_batch.num_computed_tokens_np[: len(request_ids)]
        num_rows = int(query_start_loc[-1])
        snapshot = self._indexer.snapshot(num_rows) if self._indexer else None
        pending_output = PendingAuxOutput(
            connector=self,
            request_ids=request_ids,
            batch_indices=indices,
            token_starts=token_starts,
            query_start_loc=query_start_loc,
            routed_experts_snapshot=(
                self._routed_experts_capturer.snapshot_routing_data(num_rows)
                if self._routed_experts_capturer
                else None
            ),
            indexer_snapshot=snapshot,
            aux_logprobs=frozenset(self._step_metadata.logprobs),
            logprob_widths={
                request_id: self._step_metadata.logprob_widths[request_id]
                for request_id in request_ids
                if request_id in self._step_metadata.logprob_widths
            },
        )
        with self._lock:
            assert len(self._pending_outputs) < self._max_concurrent_batches, (
                "auxiliary output step outputs are not consumed in order"
            )
            self._pending_outputs.append(pending_output)
            for request_id in pending_output.request_ids:
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
            release_logprob_keys: list[str] = []
            try:
                outputs = self._commit_output(pending)
            finally:
                self._pending_outputs.remove(pending)
                for request_id in pending.request_ids:
                    state = self._requests[request_id]
                    state.pending_outputs -= 1
                    if state.pending_outputs == 0 and state.finished:
                        teardown.append(request_id)
                        release_keys.extend(reversed(state.aux_output_keys))
                        release_logprob_keys.extend(reversed(state.logprob_keys))
            if teardown:
                if self._buffer is not None:
                    self._publish_blocks([], release_keys=release_keys)
                logprob_store = self._logprob_store
                if logprob_store is not None:
                    logprob_store.put([], release_keys=release_logprob_keys)
                self._teardown(teardown)
            return outputs

    def _commit_output(self, pending: PendingAuxOutput) -> dict[str, AuxRequestOutput]:
        buffer = self._buffer
        store = self._store
        assert store is not None or self._logprob_store is not None
        block_size = (
            buffer.block_size if buffer is not None else self._logprob_block_size
        )

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
        num_sampled = pending.num_sampled
        num_rejected = pending.num_rejected
        assert num_sampled is not None and num_rejected is not None, (
            "auxiliary output CPU copy was not enqueued"
        )
        num_sampled = num_sampled[pending.batch_indices]
        num_rejected = num_rejected[pending.batch_indices]

        sampler_logprobs = pending.logprobs_tensors
        sampler_cu = None
        if sampler_logprobs is not None:
            sampler_cu = sampler_logprobs.cu_num_generated_tokens
            if (
                sampler_cu is None
                and sampler_logprobs.cu_num_generated_tokens_tensor is not None
            ):
                sampler_cu = sampler_logprobs.cu_num_generated_tokens_tensor.tolist()

        for index, request_id in enumerate(pending.request_ids):
            count = int(num_sampled[index])
            if not count or request_id not in pending.aux_logprobs:
                continue
            state = self._requests[request_id]
            sample_start = max(
                state.prompt_len,
                int(pending.token_starts[index]) + 1,
            )
            positions = np.arange(sample_start, sample_start + count, dtype=np.int64)
            sampled_token_ids = None
            if sampler_logprobs is not None:
                batch_index = int(pending.batch_indices[index])
                offset = (
                    sampler_cu[batch_index] if sampler_cu is not None else batch_index
                )
                sampled_token_ids = (
                    sampler_logprobs.logprob_token_ids[offset : offset + count, 0]
                    .numpy()
                    .astype(np.int32, copy=False)
                )
                self._bind_pending_logprob_blocks(state, positions, sampled_token_ids)
            cached = self._get_cached_logprob_rows(
                state,
                positions,
                sampled_token_ids,
                pending.logprob_widths[request_id],
                pending.existing_logprob_artifacts,
            )
            if cached is not None:
                pending.logprobs[request_id] = rows_to_lists(cached)
                pending.logprob_replays.add(request_id)

        sampler_values = None
        for index, request_id in enumerate(pending.request_ids):
            count = int(num_sampled[index])
            if (
                not count
                or request_id not in pending.aux_logprobs
                or request_id in pending.logprob_replays
                or sampler_logprobs is None
            ):
                continue
            if sampler_values is None:
                sampler_values = sampler_logprobs.tolists()
            # logprobs_tensors covers the whole sampler batch, while
            # request_ids may be a filtered subset of it.
            batch_index = int(pending.batch_indices[index])
            cu = sampler_values.cu_num_generated_tokens
            offset = cu[batch_index] if cu is not None else batch_index
            width = pending.logprob_widths[request_id]
            assert width <= sampler_values.logprob_token_ids.shape[1], (
                "sampler logprob width is smaller than the request width"
            )
            pending.logprobs[request_id] = type(sampler_values)(
                sampler_values.logprob_token_ids[offset : offset + count, :width],
                sampler_values.logprobs[offset : offset + count, :width],
                sampler_values.sampled_token_ranks[offset : offset + count],
                None,
            )

        # Publish the whole batch before materializing any consumer output.
        materialize_outputs: list[tuple[str, int, int]] = []
        block_batches = []
        outputs: dict[str, AuxRequestOutput] = {}

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
            state = self._requests[request_id]

            # Capture precedes speculative acceptance, so discard the rejected
            # suffix. Batch boundaries still span the full executed range.
            rejected = int(rejected)
            assert 0 <= rejected <= request_num_tokens, (
                "auxiliary output rejected-token count is invalid"
            )
            rows = (
                packed_rows[start : end - rejected] if packed_rows is not None else None
            )

            if (
                request_id in pending.logprobs
                and request_id not in pending.logprob_replays
            ):
                proposed_start = max(state.prompt_len, int(token_start) + 1)
                # The batch token position is authoritative across streaming
                # prompt extensions and speculative stop rollbacks.
                sample_start = proposed_start
                rows_logprobs = rows_from_lists(
                    pending.logprobs[request_id], sample_start
                )
                state.logprob_rows[sample_start] = rows_logprobs
                state.pending_publish_logprobs.update(
                    self._block_starts(
                        rows_logprobs.positions, self._logprob_block_size
                    )
                )

            if buffer is None:
                state.scheduled_cursor = int(token_start) + request_num_tokens
                if request_id in pending.logprobs:
                    outputs[request_id] = AuxRequestOutput(
                        token_start=int(token_start) + 1,
                        logprobs=pending.logprobs[request_id],
                    )
                continue
            assert rows is not None

            capture_start = token_start
            capture_cursor = state.capture_cursor
            if capture_cursor is None:
                capture_cursor = capture_start

            assert capture_start >= capture_cursor, (
                "auxiliary output capture moved backwards"
            )
            if capture_start > capture_cursor:
                # Reattach after an optimistically scheduled suffix was rejected.
                assert capture_cursor < state.scheduled_cursor, (
                    "auxiliary output capture has an unbacked token gap"
                )
                capture_start = capture_cursor

            emit_start = state.emit_cursor
            # Complete blocks without keys remain pending until a hash update.
            completed = buffer.capture(request_id, capture_start, rows)
            token_end = capture_start + len(rows)
            state.capture_cursor = token_end
            state.scheduled_cursor = token_start + request_num_tokens
            block_batches.append((state, completed))

            if sampled > 0 and emit_start <= token_end:
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
                        logprobs=pending.logprobs.get(request_id),
                    )
                    state.emit_cursor = token_end
                else:
                    materialize_outputs.append((request_id, emit_start, token_end))

        # A consumer may reuse a block produced earlier in the same batch.
        if buffer is not None:
            self._publish_blocks(block_batches)

        for request_id, emit_start, token_end in materialize_outputs:
            state = self._requests[request_id]
            assert buffer is not None and store is not None
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
                    rows = np.concatenate(
                        (rows, buffer.read(request_id, stored_end, token_end))
                    )
            else:
                rows = buffer.read(request_id, emit_start, token_end)
            outputs[request_id] = self._make_output(emit_start, rows)
            if request_id in pending.logprobs:
                outputs[request_id].logprobs = pending.logprobs[request_id]
            state.emit_cursor = token_end
        self._publish_logprob_blocks(
            pending.request_ids, pending.existing_logprob_artifacts
        )
        for index, request_id in enumerate(pending.request_ids):
            state = self._requests[request_id]
            if request_id not in outputs and request_id in pending.aux_logprobs:
                outputs[request_id] = AuxRequestOutput(0)
            output = outputs.get(request_id)
            if output is None or request_id not in pending.aux_logprobs:
                continue
            if request_id in pending.logprobs:
                output.logprobs = pending.logprobs[request_id]
            elif state.logprob_keys and int(num_sampled[index]):
                position = max(
                    state.prompt_len,
                    int(pending.token_starts[index]) + 1,
                )
                block_index = max((position - 1) // self._logprob_block_size, 0)
                if block_index >= len(state.logprob_keys):
                    raise RuntimeError(
                        "auxiliary logprobs artifact has no key for sampled "
                        f"position {position}: request={request_id}"
                    )
                rows = self._read_logprob_rows([state.logprob_keys[block_index]])
                mask = rows.positions == position
                if int(mask.sum()) != 1:
                    raise RuntimeError(
                        "auxiliary logprobs artifact is missing sampled position "
                        f"{position}: request={request_id}"
                    )
                output.logprobs = rows_to_lists(
                    LogprobRows(
                        rows.positions[mask],
                        rows.token_ids[mask],
                        rows.values[mask],
                        rows.ranks[mask],
                    )
                )
        return outputs

    def _read_logprob_rows(self, keys: Sequence[str]) -> LogprobRows:
        store = self._logprob_store
        if store is None:
            raise RuntimeError("logprob replay store is disabled")
        chunks = [decode_rows(store.get_concatenated([key])) for key in keys]
        rows = concat_rows(chunks)
        if rows is None:
            raise RuntimeError("auxiliary logprobs artifact contains no rows")
        return rows

    def _make_output(self, token_start: int, rows: np.ndarray) -> AuxRequestOutput:
        if self._indexer is None:
            return AuxRequestOutput(token_start=token_start, routed_rows=rows)
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

    @staticmethod
    def _block_starts(positions: np.ndarray, block_size: int) -> set[int]:
        return {
            (int(position) - 1) // block_size * block_size for position in positions
        }

    def _publish_logprob_blocks(
        self,
        request_ids: Iterable[str],
        existing_payloads: dict[str, bytes | None] | None = None,
    ) -> None:
        store = self._logprob_store
        if store is None:
            return
        block_size = self._logprob_block_size
        for request_id in request_ids:
            state = self._requests[request_id]
            starts = sorted(
                start
                for start in state.pending_publish_logprobs
                if start // block_size < len(state.logprob_keys)
            )
            if not starts:
                continue
            # One queue barrier per step instead of one per artifact.
            keys = [state.logprob_keys[start // block_size] for start in starts]
            if existing_payloads is None:
                existing = store.get_optional_many(keys)
            else:
                missing = [key for key in keys if key not in existing_payloads]
                if missing:
                    existing_payloads.update(store.get_optional_many(missing))
                existing = {key: existing_payloads[key] for key in keys}
            completed = []
            for start in starts:
                key = state.logprob_keys[start // block_size]
                end = start + block_size
                rows = concat_rows(
                    [
                        value
                        for value in state.logprob_rows.values()
                        if np.any((value.positions > start) & (value.positions <= end))
                    ]
                )
                if rows is None:
                    continue
                mask = (rows.positions > start) & (rows.positions <= end)
                artifact = LogprobRows(
                    rows.positions[mask],
                    rows.token_ids[mask],
                    rows.values[mask],
                    rows.ranks[mask],
                )
                payload = existing.get(key)
                if payload is not None:
                    merged = concat_rows([decode_rows(payload), artifact])
                    assert merged is not None
                    artifact = merged
                store.put([BlockObject(key, encode_rows(artifact))])
                state.pending_publish_logprobs.discard(start)
                if max(artifact.positions) >= end:
                    completed.append((start, end))
            for start, end in completed:
                self._discard_logprob_rows(state.logprob_rows, start, end)

    @staticmethod
    def _discard_logprob_rows(
        rows_by_start: dict[int, LogprobRows], start: int, end: int
    ) -> None:
        remaining = []
        for rows in rows_by_start.values():
            keep = (rows.positions <= start) | (rows.positions > end)
            if np.any(keep):
                remaining.append(
                    LogprobRows(
                        rows.positions[keep],
                        rows.token_ids[keep],
                        rows.values[keep],
                        rows.ranks[keep],
                    )
                )
        rows_by_start.clear()
        merged = concat_rows(remaining)
        if merged is not None:
            rows_by_start[int(merged.positions[0])] = merged

    def _bind_pending_logprob_blocks(
        self,
        state: _WorkerRequestState,
        positions: np.ndarray,
        sampled_token_ids: np.ndarray,
    ) -> None:
        store = self._logprob_store
        if store is None or not state.pending_logprob_blocks:
            return
        retained = []
        for position, token_id in zip(positions, sampled_token_ids, strict=True):
            if (
                int(position) % self._logprob_block_size
                or not state.pending_logprob_blocks
            ):
                continue
            block_hash, fingerprint = state.pending_logprob_blocks.pop(0)
            key = self._logprob_key(block_hash, int(token_id), fingerprint)
            state.logprob_keys.append(key)
            retained.append(key)
        if retained:
            store.put([], retain_keys=retained)

    def _get_cached_logprob_rows(
        self,
        state: _WorkerRequestState,
        positions: np.ndarray,
        sampled_token_ids: np.ndarray | None,
        width: int,
        existing_payloads: dict[str, bytes | None],
    ) -> LogprobRows | None:
        store = self._logprob_store
        if store is None or sampled_token_ids is None:
            return None
        block_size = self._logprob_block_size
        block_indices = [
            max((int(position) - 1) // block_size, 0) for position in positions
        ]
        if any(index >= len(state.logprob_keys) for index in block_indices):
            return None
        keys = list(dict.fromkeys(state.logprob_keys[index] for index in block_indices))
        missing = [key for key in keys if key not in existing_payloads]
        if missing:
            existing_payloads.update(store.get_optional_many(missing))
        decoded = {
            key: decode_rows(payload)
            for key, payload in existing_payloads.items()
            if key in keys
            if payload is not None
        }
        chunks = []
        for row_index, (position, block_index) in enumerate(
            zip(positions, block_indices, strict=True)
        ):
            key = state.logprob_keys[block_index]
            rows = decoded.get(key)
            if rows is None or rows.token_ids.shape[1] != width:
                return None
            matches = np.flatnonzero(rows.positions == position)
            if len(matches) != 1:
                return None
            match = int(matches[0])
            if (
                sampled_token_ids is not None
                and rows.token_ids[match, 0] != sampled_token_ids[row_index]
            ):
                return None
            chunks.append(
                LogprobRows(
                    rows.positions[match : match + 1],
                    rows.token_ids[match : match + 1],
                    rows.values[match : match + 1],
                    rows.ranks[match : match + 1],
                )
            )
        return concat_rows(chunks)

    def _logprob_key(
        self, block_hash: bytes, boundary_token_id: int, fingerprint: str
    ) -> str:
        return (
            f"vllm-logprobs/v1/{self._generation}/{fingerprint}/"
            f"{block_hash.hex()}/{boundary_token_id}"
        )

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
            keyed_end = len(state.aux_output_keys) * buffer.block_size
            ready = [(start, rows) for start, rows in blocks if start < keyed_end]
            state.pending_blocks = [
                (start, buffer.retain_block(rows))
                for start, rows in blocks
                if start >= keyed_end
            ]
            if ready:
                ready_batches.append((state.aux_output_keys, ready))
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
        for request_id in request_ids:
            state = self._requests.pop(request_id)
            if buffer is not None:
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

    def begin_step(self, metadata: AuxOutputConnectorMetadata | None) -> None:
        """Apply one scheduler step's request and block-hash updates.

        While an unconsumed step covers a finished request, its teardown is
        deferred to that step's process_output.
        """
        self._step_metadata = metadata
        if (self._buffer is None and self._logprob_store is None) or metadata is None:
            return
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
            if metadata.generation > self._generation:
                # A generation change follows a prefix-cache reset, which the
                # scheduler only performs once all model output is consumed.
                assert not self._pending_outputs, (
                    "auxiliary output generation changed with output in flight"
                )
                release_keys.extend(
                    key
                    for state in self._requests.values()
                    for key in reversed(state.aux_output_keys)
                )
                release_logprob_keys = [
                    key
                    for state in self._requests.values()
                    for key in reversed(state.logprob_keys)
                ]
                if self._buffer is not None:
                    self._buffer.reset()
                logprob_store = self._logprob_store
                if logprob_store is not None and release_logprob_keys:
                    logprob_store.put([], release_keys=release_logprob_keys)
                self._requests.clear()
                self._generation = metadata.generation
            for request_id, emit_start in metadata.requests.items():
                state = self._requests.setdefault(
                    request_id, _WorkerRequestState(emit_cursor=emit_start)
                )
                state.prompt_len = metadata.prompt_lens.get(request_id, 0)
                if state.pending_outputs == 0 and self._buffer is not None:
                    # In-flight steps leave the worker cursor behind the
                    # scheduler's optimistic view; only check settled requests.
                    # Logprobs-only steps have no emit cursor to compare against.
                    assert emit_start <= state.emit_cursor, (
                        "auxiliary output Scheduler emit cursor moved ahead"
                    )
            block_batches: list[
                tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]
            ] = []
            for request_id, block_hashes in metadata.block_hashes.items():
                state = self._requests[request_id]
                namespace = self._get_namespace(self._generation)
                keys = aux_output_keys(block_hashes, namespace)
                state.aux_output_keys.extend(keys)
                retained_keys.extend(keys)
                block_batches.append((state, []))
            retained_logprob_keys: list[str] = []
            for request_id, fingerprint in metadata.logprobs.items():
                state = self._requests[request_id]
                logprob_hashes: list[bytes] = list(
                    metadata.logprob_block_hashes.get(request_id, ())
                )
                boundaries = metadata.logprob_boundary_token_ids.get(request_id, ())
                if len(logprob_hashes) != len(boundaries):
                    raise ValueError("auxiliary logprobs block metadata is misaligned")
                for block_hash, boundary in zip(
                    logprob_hashes, boundaries, strict=True
                ):
                    if boundary is None:
                        state.pending_logprob_blocks.append((block_hash, fingerprint))
                    else:
                        key = self._logprob_key(block_hash, boundary, fingerprint)
                        state.logprob_keys.append(key)
                        retained_logprob_keys.append(key)
            self._publish_logprob_blocks(metadata.logprob_block_hashes)
            finished_now: list[str] = []
            for request_id in metadata.finished_requests:
                state = self._requests[request_id]
                state.finished = True
                if state.pending_outputs:
                    continue
                finished_now.append(request_id)
                release_keys.extend(reversed(state.aux_output_keys))
                logprob_store = self._logprob_store
                if logprob_store is not None:
                    logprob_store.put(
                        [], release_keys=list(reversed(state.logprob_keys))
                    )
            if self._buffer is not None:
                self._publish_blocks(block_batches, retained_keys, release_keys)
            logprob_store = self._logprob_store
            if logprob_store is not None:
                logprob_store.put([], retain_keys=retained_logprob_keys)
            self._teardown(finished_now)

    def close(self) -> None:
        with self._lock:
            self._pending_outputs.clear()
            store, self._store = self._store, None
            try:
                if store is not None:
                    store.close()
            finally:
                self._buffer = None
                logprob_store = self._logprob_store
                self._logprob_store = None
                if logprob_store is not None:
                    logprob_store.close()
                self._requests.clear()
                self._step_metadata = None
                self._close_capture()


def get_aux_output_connector(
    model: torch.nn.Module, vllm_config: VllmConfig
) -> AuxOutputWorkerConnector:
    return AuxOutputWorkerConnector(model=model, vllm_config=vllm_config)
