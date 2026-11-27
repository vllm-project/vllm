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
from vllm.distributed.aux_output_connector.connector import (
    AuxObject,
    AuxOutputConnectorMetadata,
    AuxRequestOutput,
    object_put_timeout_s,
    retained_objects,
)
from vllm.distributed.aux_output_connector.object_store import (
    AuxObjects,
    AuxObjectWriter,
    create_aux_object_store,
)
from vllm.distributed.aux_output_connector.routed_experts import (
    RoutedExpertsBuffer,
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_keys,
)
from vllm.distributed.aux_output_connector.store import (
    BackgroundBlockObjectStore,
    BlockObjectStore,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.utils.math_utils import cdiv, round_up

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass(slots=True)
class _ObjectRun:
    """Rows a namespace request computed since it was last scheduled, by KV
    block, and the objects still to write from them. A block a step computes
    whole is a view of that step's rows; a block computed over several steps
    is filled in place in its own buffer. Positions computed again, after a
    failed KV load, go to copies of their blocks: the writer may still read
    objects taken from them. Runs start on a block boundary and every object
    starts at one; a tail object may span blocks."""

    block_size: int
    end: int = -1
    blocks: dict[int, np.ndarray] = field(default_factory=dict)
    writes: list[AuxObject] = field(default_factory=list)

    def add(self, token_start: int, rows: np.ndarray) -> None:
        if 0 <= self.end < token_start:
            # A step scheduled while drafts were in flight starts after the
            # ones rejected: its rows follow those kept.
            token_start = self.end
        size = self.block_size
        if token_start < self.end:
            for block in range(token_start // size, cdiv(self.end, size)):
                if (buffer := self.blocks.get(block)) is not None:
                    self.blocks[block] = buffer.copy()
        position = token_start
        self.end = end = token_start + len(rows)
        while position < end:
            block, offset = divmod(position, size)
            stop = min(end, position - offset + size)
            part = rows[position - token_start : stop - token_start]
            position = stop
            if len(part) == size:
                self.blocks[block] = part
                continue
            if (buffer := self.blocks.get(block)) is None:
                buffer = self.blocks[block] = np.empty(
                    (size, *part.shape[1:]), part.dtype
                )
            buffer[offset : offset + len(part)] = part

    def take_ready(self) -> AuxObjects:
        """Objects whose rows are all here. Later writes start at or after
        the start of the last one, so earlier blocks are dropped."""
        if not self.writes:
            return []
        ready = [write for write in self.writes if write[2] <= self.end]
        if not ready:
            return []
        self.writes = [write for write in self.writes if write[2] > self.end]
        size = self.block_size
        objects = [
            (key, {"routed_experts": self._rows(start, end)})
            for key, start, end in ready
        ]
        keep = min(
            (start for _, start, _ in self.writes),
            default=max(start for _, start, _ in ready),
        )
        for block in [block for block in self.blocks if block < keep // size]:
            del self.blocks[block]
        return objects

    def _rows(self, start: int, end: int) -> np.ndarray:
        size = self.block_size
        first = start // size
        if end - start <= size:
            return self.blocks[first][: end - start]
        blocks = [self.blocks[block] for block in range(first, cdiv(end, size))]
        return np.concatenate(blocks)[: end - start]


@dataclass(slots=True)
class _WorkerRequestState:
    aux_output_keys: list[str] = field(default_factory=list)
    pending_blocks: list[tuple[int, np.ndarray]] = field(default_factory=list)
    capture_cursor: int | None = None
    buffer_start: int = 0
    scheduled_cursor: int = 0
    emit_cursor: int = 0
    pending_outputs: int = 0
    # Terminal event received; teardown waits for in-flight step outputs.
    finished: bool = False
    run: _ObjectRun | None = None


@dataclass(slots=True)
class PendingAuxOutput:
    """Own one step's R3 snapshot until its asynchronous copy is consumed."""

    connector: AuxOutputWorkerConnector
    request_ids: list[str]
    token_starts: np.ndarray
    query_start_loc: np.ndarray
    # GPU snapshot taken on the main stream.
    routed_experts_gpu: torch.Tensor
    # Filled by enqueue_cpu_copy on the output copy stream.
    routed_experts: np.ndarray | None = None
    num_sampled: np.ndarray | None = None
    num_rejected: np.ndarray | None = None
    runs: dict[str, _ObjectRun] = field(default_factory=dict)

    def enqueue_cpu_copy(
        self, num_sampled: np.ndarray, num_rejected: np.ndarray
    ) -> None:
        """Enqueue asynchronous D2H copies; call on the output copy stream."""
        self.num_sampled = num_sampled
        self.num_rejected = num_rejected
        self.routed_experts = self.routed_experts_gpu.to(
            "cpu", non_blocking=True
        ).numpy()

    def process_output(self) -> dict[str, AuxRequestOutput]:
        return self.connector.process_output(self)

    def take_dropped_objects(self) -> list[str] | None:
        return self.connector.take_dropped_objects()


class AuxOutputWorkerConnector:
    """Own capture, request tails, and backend resources on the output worker."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
    ) -> None:
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=max_num_batched_tokens, vllm_config=vllm_config
        )
        bind_routed_experts_capturer(model, capturer)
        self._capturer = capturer
        self._vllm_config = vllm_config
        self._store: BackgroundBlockObjectStore | None = None
        self._buffer: RoutedExpertsBuffer | None = None
        self._writer: AuxObjectWriter | None = None
        self._requests: dict[str, _WorkerRequestState] = {}
        self._generation = 0
        self._step_metadata: AuxOutputConnectorMetadata | None = None
        # Steps whose asynchronous copy has not been consumed yet. The engine
        # consumes step outputs in order, so at most max_concurrent_batches
        # are outstanding; guarded by _lock against the output thread.
        self._pending_outputs: list[PendingAuxOutput] = []
        self._lock = Lock()
        self._max_concurrent_batches = vllm_config.max_concurrent_batches

    def initialize_output_storage(self, kv_cache_config: KVCacheConfig) -> None:
        """Create the output data plane once the KV cache layout is resolved."""
        # Every TP rank participates in capture collectives, but only the
        # executor output rank owns the auxiliary output data plane.
        if not get_tp_group().is_first_rank:
            return

        vllm_config = self._vllm_config
        shape_per_token = self._capturer.shape_per_token
        dtype: np.dtype[Any] = np.dtype(self._capturer.output_dtype_name)
        num_objects, hash_block_size = retained_objects(vllm_config, kv_cache_config)
        block_nbytes = hash_block_size * int(np.prod(shape_per_token)) * dtype.itemsize
        max_bytes = num_objects * block_nbytes
        self._store = BackgroundBlockObjectStore(
            BlockObjectStore(max_bytes=max_bytes, object_nbytes=block_nbytes),
            max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
        )
        self._buffer = RoutedExpertsBuffer(
            dtype,
            shape_per_token,
            hash_block_size,
            vllm_config.scheduler_config.max_num_seqs,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.max_concurrent_batches,
        )
        config = vllm_config.aux_output_config
        if config.object_key_prefix is not None:
            self._writer = AuxObjectWriter(
                create_aux_object_store(
                    config.object_store, config.object_store_extra_config
                ),
                timeout_s=object_put_timeout_s(),
            )

    def prepare_output(self, input_batch: InputBatch) -> PendingAuxOutput | None:
        """Snapshot one step's R3 tensor for asynchronous CPU transfer."""
        if self._buffer is None or self._step_metadata is None:
            return None

        request_ids = list(input_batch.req_ids)
        query_start_loc = input_batch.query_start_loc_np[: len(request_ids) + 1]
        num_rows = int(query_start_loc[-1])
        pending_output = PendingAuxOutput(
            connector=self,
            request_ids=request_ids,
            token_starts=input_batch.num_computed_tokens_np,
            query_start_loc=query_start_loc,
            routed_experts_gpu=self._capturer.snapshot_routing_data(num_rows),
        )
        with self._lock:
            assert len(self._pending_outputs) < self._max_concurrent_batches, (
                "auxiliary output step outputs are not consumed in order"
            )
            self._pending_outputs.append(pending_output)
            for request_id in pending_output.request_ids:
                state = self._requests[request_id]
                state.pending_outputs += 1
                if state.run is not None:
                    pending_output.runs[request_id] = state.run
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
                    state = self._requests[request_id]
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

        routed_experts = pending.routed_experts
        num_sampled = pending.num_sampled
        num_rejected = pending.num_rejected
        assert (
            routed_experts is not None
            and num_sampled is not None
            and num_rejected is not None
        ), "auxiliary output CPU copy was not enqueued"

        # Publish the whole batch before materializing any consumer output.
        materialize_outputs: list[tuple[str, int, int]] = []
        block_batches = []
        outputs: dict[str, AuxRequestOutput] = {}
        objects: AuxObjects = []

        # Use the ModelRunner's actual batch boundaries rather than rebuilding them.
        query_start_loc = pending.query_start_loc.tolist()
        for request_id, token_start, start, end, sampled, rejected in zip(
            pending.request_ids,
            pending.token_starts.tolist(),
            query_start_loc[:-1],
            query_start_loc[1:],
            num_sampled.tolist(),
            num_rejected.tolist(),
            strict=True,
        ):
            request_num_tokens = end - start
            assert request_num_tokens > 0, (
                "auxiliary output request token count must be positive"
            )
            state = self._requests[request_id]

            # Capture precedes speculative acceptance, so discard the rejected
            # suffix. Batch boundaries still span the full executed range.
            assert 0 <= rejected <= request_num_tokens, (
                "auxiliary output rejected-token count is invalid"
            )
            rows = routed_experts[start : end - rejected]
            if (run := pending.runs.get(request_id)) is not None:
                run.add(token_start, rows)
                objects.extend(run.take_ready())
                continue

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

            skip = max(state.buffer_start - capture_start, 0)

            emit_start = state.emit_cursor
            # Complete blocks without keys remain pending until a hash update.
            completed = buffer.capture(request_id, capture_start + skip, rows[skip:])
            token_end = capture_start + len(rows)
            state.capture_cursor = token_end
            state.scheduled_cursor = token_start + request_num_tokens
            block_batches.append((state, completed))

            if sampled > 0 and emit_start <= token_end:
                if emit_start >= capture_start:
                    outputs[request_id] = AuxRequestOutput(
                        emit_start, rows[emit_start - capture_start :]
                    )
                    state.emit_cursor = token_end
                else:
                    materialize_outputs.append((request_id, emit_start, token_end))

        self._put_objects(objects)
        # A consumer may reuse a block produced earlier in the same batch.
        self._publish_blocks(block_batches)

        for request_id, emit_start, token_end in materialize_outputs:
            state = self._requests[request_id]
            stored_end = (
                min(token_end // block_size, len(state.aux_output_keys)) * block_size
            )
            if emit_start < stored_end:
                first_block = emit_start // block_size
                stored = materialize_routed_experts(
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
            outputs[request_id] = AuxRequestOutput(emit_start, rows)
            state.emit_cursor = token_end
        return outputs

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
            publish_routed_experts(
                store,
                batches=ready_batches,
                block_size=buffer.block_size,
                retain_keys=retain_keys,
                release_keys=release_keys,
            )
        for _, blocks in ready_batches:
            for _, rows in blocks:
                buffer.release_block(rows)

    def take_dropped_objects(self) -> list[str] | None:
        """Keys of the objects the writer dropped since the last call."""
        writer = self._writer
        return (writer.take_dropped() or None) if writer is not None else None

    def _put_objects(self, objects: AuxObjects) -> None:
        """One put per step, so a step waits for the writer at most once."""
        if objects:
            assert self._writer is not None
            self._writer.put(objects)

    def _teardown(self, request_ids: Iterable[str]) -> None:
        """Drop finished requests' temporary state after publishing key releases."""
        buffer = self._buffer
        assert buffer is not None
        for request_id in request_ids:
            state = self._requests.pop(request_id)
            for _, rows in state.pending_blocks:
                buffer.release_block(rows)
            buffer.discard(request_id)

    def begin_step(self, metadata: AuxOutputConnectorMetadata | None) -> None:
        """Apply one scheduler step's request and block-hash updates.

        While an unconsumed step covers a finished request, its teardown is
        deferred to that step's process_output.
        """
        self._step_metadata = metadata
        if self._buffer is None or metadata is None:
            return
        assert not metadata.requests.keys() & metadata.finished_requests, (
            "auxiliary output request cannot run and finish in one step"
        )
        assert metadata.generation >= self._generation, (
            "auxiliary output metadata generation moved backwards"
        )
        with self._lock:
            objects: AuxObjects = []
            for request_id, writes in (metadata.object_writes or {}).items():
                run = self._requests[request_id].run
                assert run is not None
                run.writes.extend(writes)
                objects.extend(run.take_ready())
            self._put_objects(objects)
            namespace = str(self._generation)
            hashes = metadata.retained_block_hashes or ()
            retained_keys = routed_experts_keys(hashes, namespace)
            hashes = metadata.released_block_hashes or ()
            release_keys = routed_experts_keys(hashes, namespace)
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
                self._buffer.reset()
                self._requests.clear()
                self._generation = metadata.generation
            for request_id, emit_start in metadata.requests.items():
                state = self._requests.setdefault(
                    request_id, _WorkerRequestState(emit_cursor=emit_start)
                )
                if state.pending_outputs == 0:
                    # In-flight steps leave the worker cursor behind the
                    # scheduler's optimistic view; only check settled requests.
                    assert emit_start <= state.emit_cursor, (
                        "auxiliary output Scheduler emit cursor moved ahead"
                    )
            for request_id in metadata.namespace_runs or ():
                state = self._requests.setdefault(request_id, _WorkerRequestState())
                state.run = _ObjectRun(metadata.object_block_size)
            block_batches: list[
                tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]
            ] = []
            for request_id, block_hashes in metadata.block_hashes.items():
                state = self._requests[request_id]
                keys = routed_experts_keys(block_hashes, str(self._generation))
                state.aux_output_keys.extend(keys)
                retained_keys.extend(keys)
                block_batches.append((state, []))
            finished_now: list[str] = []
            for request_id in metadata.finished_requests:
                state = self._requests[request_id]
                state.finished = True
                if state.pending_outputs:
                    continue
                finished_now.append(request_id)
                release_keys.extend(reversed(state.aux_output_keys))
            self._publish_blocks(block_batches, retained_keys, release_keys)
            self._teardown(finished_now)

    def close(self) -> None:
        with self._lock:
            self._pending_outputs.clear()
            writer, self._writer = self._writer, None
            try:
                if writer is not None:
                    writer.close()
            finally:
                if self._store is not None:
                    try:
                        self._store.close()
                    finally:
                        self._store = None


def get_aux_output_connector(
    model: torch.nn.Module, vllm_config: VllmConfig
) -> AuxOutputWorkerConnector:
    return AuxOutputWorkerConnector(model=model, vllm_config=vllm_config)
