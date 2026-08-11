# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side execution-artifact data plane."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.artifact_connector.connector import (
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactRequestMetadata,
    ArtifactRequestOutput,
)
from vllm.distributed.artifact_connector.routed_experts import (
    RoutedExpertsArtifactBuffer,
    get_routing_shape_and_dtype,
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_keys,
)
from vllm.distributed.artifact_connector.shm import (
    BackgroundArtifactStore,
    LocalSharedMemoryArtifactStore,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

_RequestKey = tuple[str, int]


@dataclass
class _WorkerRequestState:
    block_hashes: list[bytes] = field(default_factory=list)
    artifact_keys: list[str] = field(default_factory=list)
    pending_blocks: list[tuple[int, np.ndarray]] = field(default_factory=list)
    capture_cursor: int | None = None
    emit_cursor: int | None = None
    pending_outputs: int = 0
    finished_block_hashes: Sequence[bytes] | None = None
    finished: bool = False


class PendingArtifactOutput:
    """Own one step's Artifact tensors until its async D2H copy completes."""

    def __init__(
        self,
        connector: ArtifactWorkerConnector,
        metadata: ArtifactConnectorMetadata,
        routed_experts: torch.Tensor,
        request_ids: list[str],
        num_rejected: torch.Tensor,
    ) -> None:
        self._connector = connector
        self._metadata = metadata
        self._routed_experts = routed_experts
        self._request_ids = request_ids
        self._num_rejected = num_rejected
        self._routed_experts_cpu: np.ndarray | None = None
        self._num_rejected_cpu: np.ndarray | None = None

    def to_cpu_nonblocking(self) -> None:
        self._routed_experts_cpu = self._routed_experts.to(
            "cpu", non_blocking=True
        ).numpy()
        self._num_rejected_cpu = self._num_rejected.to("cpu", non_blocking=True).numpy()

    def finish(self) -> ArtifactConnectorOutput:
        assert self._routed_experts_cpu is not None
        assert self._num_rejected_cpu is not None
        try:
            return self._connector.process_output(
                self._metadata,
                self._routed_experts_cpu,
                self._request_ids,
                self._num_rejected_cpu,
            )
        finally:
            self._connector.output_finished(self._metadata)


class ArtifactWorkerConnector:
    """Own capture, request tails, and backend resources on the output worker."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        kv_cache_config: KVCacheConfig,
        max_num_batched_tokens: int,
    ) -> None:
        capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=max_num_batched_tokens,
            vllm_config=vllm_config,
        )
        bind_routed_experts_capturer(model, capturer)
        self._capturer = capturer
        self._store: BackgroundArtifactStore | None = None
        self._buffer: RoutedExpertsArtifactBuffer | None = None
        self._requests: dict[_RequestKey, _WorkerRequestState] = {}
        self._generation = -1
        self._lock = Lock()
        self._step_metadata: ArtifactConnectorMetadata | None = None
        self._pending_capture: tuple[ArtifactConnectorMetadata, torch.Tensor] | None = (
            None
        )

        # Every TP rank participates in capture collectives, but only the
        # executor output rank owns the artifact data plane.
        if not get_tp_group().is_first_rank:
            return

        shape_per_token, dtype = get_routing_shape_and_dtype(vllm_config)
        self._shape_per_token = shape_per_token
        self._dtype: np.dtype[Any] = np.dtype(dtype)
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        hashes_per_kv_block = scheduler_block_size // hash_block_size
        block_nbytes = (
            hash_block_size * int(np.prod(shape_per_token)) * self._dtype.itemsize
        )
        max_bytes = vllm_config.artifact_config.max_shm_bytes
        if max_bytes is None:
            max_bytes = kv_cache_config.num_blocks * hashes_per_kv_block * block_nbytes
        self._store = BackgroundArtifactStore(
            LocalSharedMemoryArtifactStore(
                vllm_config.artifact_config.shm_dir,
                vllm_config.instance_id,
                vllm_config.parallel_config.data_parallel_rank,
                max_bytes=max_bytes,
                object_nbytes=block_nbytes,
                ttl_seconds=vllm_config.artifact_config.shm_ttl_seconds,
            ),
            max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
        )
        self._buffer = RoutedExpertsArtifactBuffer(
            self._dtype,
            shape_per_token,
            hash_block_size,
            vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens,
        )

    def capture_step(self, num_tokens: int) -> None:
        """Capture one step without exposing Artifact internals to the runner."""
        metadata = self._step_metadata
        if metadata is None or self._store is None:
            return
        self._pending_capture = (
            metadata,
            self._capturer.get_routing_data(num_tokens),
        )

    def prepare_output(
        self,
        request_ids: list[str],
        num_rejected: torch.Tensor,
    ) -> PendingArtifactOutput | None:
        capture = self._pending_capture
        self._pending_capture = None
        if capture is None:
            return None
        metadata, routed_experts = capture
        with self._lock:
            for request in metadata.requests:
                key = (request.request_id, request.epoch)
                self._requests[key].pending_outputs += 1
        return PendingArtifactOutput(
            self,
            metadata,
            routed_experts,
            request_ids,
            num_rejected,
        )

    def process_output(
        self,
        metadata: ArtifactConnectorMetadata,
        routed_experts: np.ndarray,
        request_ids: list[str],
        num_rejected: np.ndarray,
    ) -> ArtifactConnectorOutput:
        with self._lock:
            if metadata.generation != self._generation:
                return ArtifactConnectorOutput({})
            outputs = self._process_output(
                metadata,
                routed_experts,
                request_ids,
                num_rejected,
            )
        return ArtifactConnectorOutput(outputs)

    def _process_output(
        self,
        metadata: ArtifactConnectorMetadata,
        routed_experts: np.ndarray,
        request_ids: list[str],
        num_rejected: np.ndarray,
    ) -> dict[str, ArtifactRequestOutput]:
        assert self._store is not None and self._buffer is not None
        by_request = {request.request_id: request for request in metadata.requests}
        captured: list[
            tuple[
                ArtifactRequestMetadata,
                int,
                np.ndarray,
                list[tuple[int, np.ndarray]],
                int,
            ]
        ] = []
        commit_batches: list[tuple[Sequence[str], list[tuple[int, np.ndarray]]]] = []
        committed_rows: list[np.ndarray] = []
        outputs: dict[str, ArtifactRequestOutput] = {}
        offset = 0
        for request_index, request_id in enumerate(request_ids):
            request = by_request.get(request_id)
            if request is None:
                raise RuntimeError(f"artifact metadata is missing {request_id}")
            end = offset + request.num_tokens
            key = (request_id, request.epoch)
            state = self._requests[key]
            artifact_keys = state.artifact_keys
            valid_end = end - int(num_rejected[request_index])
            rows = routed_experts[offset:valid_end]
            capture_start = request.token_start
            capture_cursor = state.capture_cursor
            if capture_cursor is None:
                block_start = capture_start // metadata.block_size * metadata.block_size
                if block_start < capture_start:
                    if capture_start > len(artifact_keys) * metadata.block_size:
                        raise RuntimeError("artifact capture has an unbacked token gap")
                    cached_rows = self._materialize(
                        block_start,
                        capture_start,
                        artifact_keys,
                        metadata.block_size,
                    )
                    rows = np.concatenate((cached_rows, rows))
                    capture_start = block_start
            elif capture_start < capture_cursor:
                rows = rows[min(capture_cursor - capture_start, len(rows)) :]
                capture_start = capture_cursor
            elif capture_start > capture_cursor:
                if capture_start > len(artifact_keys) * metadata.block_size:
                    raise RuntimeError("artifact capture has an unbacked token gap")
                cached_rows = self._materialize(
                    capture_cursor,
                    capture_start,
                    artifact_keys,
                    metadata.block_size,
                )
                rows = np.concatenate((cached_rows, rows))
                capture_start = capture_cursor
            emit_start = max(
                request.emit_start,
                state.emit_cursor
                if state.emit_cursor is not None
                else request.emit_start,
            )
            completed = self._buffer.capture(key, capture_start, rows)
            state.capture_cursor = capture_start + len(rows)
            pending, ready = self._take_available_blocks(
                state,
                completed,
                metadata.block_size,
            )
            if ready:
                commit_batches.append((artifact_keys, ready))
                committed_rows.extend(rows for _, rows in ready)
            token_end = capture_start + len(rows)
            if request.emit_output and emit_start < token_end:
                if emit_start >= capture_start:
                    outputs[request_id] = ArtifactRequestOutput(
                        emit_start,
                        rows[emit_start - capture_start :],
                    )
                    state.emit_cursor = token_end
                else:
                    captured.append(
                        (
                            request,
                            capture_start,
                            rows,
                            [*ready, *pending],
                            emit_start,
                        )
                    )
            offset = end
        if offset != len(routed_experts):
            raise RuntimeError("artifact capture output has an invalid row count")
        publish_routed_experts(
            self._store,
            batches=commit_batches,
            block_size=metadata.block_size,
        )
        for request, capture_start, rows, local_segments, emit_start in captured:
            key = (request.request_id, request.epoch)
            state = self._requests[key]
            block_hashes = state.block_hashes
            artifact_keys = state.artifact_keys
            token_end = capture_start + len(rows)
            stored_end = min(
                capture_start // metadata.block_size * metadata.block_size,
                len(block_hashes) * metadata.block_size,
            )
            segments: list[tuple[int, np.ndarray]] = []
            if emit_start < stored_end:
                segments.append(
                    (
                        emit_start,
                        self._materialize(
                            emit_start,
                            stored_end,
                            artifact_keys,
                            metadata.block_size,
                        ),
                    )
                )
            segments.append((capture_start, rows))
            segments.extend(local_segments)
            segments.sort(key=lambda item: item[0])
            complete_segments: list[tuple[int, np.ndarray]] = []
            cursor = emit_start
            for start, segment in segments:
                if cursor < capture_start and start > cursor:
                    gap_end = min(start, capture_start)
                    complete_segments.append(
                        (cursor, self._buffer.read(key, cursor, gap_end))
                    )
                    cursor = gap_end
                complete_segments.append((start, segment))
                if start <= cursor:
                    cursor = max(cursor, start + len(segment))
            if cursor < capture_start:
                complete_segments.append(
                    (cursor, self._buffer.read(key, cursor, capture_start))
                )
            outputs[request.request_id] = ArtifactRequestOutput(
                emit_start,
                self._assemble_segments(emit_start, token_end, complete_segments),
            )
            state.emit_cursor = token_end
        for rows in committed_rows:
            self._buffer.release_block(rows)
        return outputs

    def _take_available_blocks(
        self,
        state: _WorkerRequestState,
        completed: list[tuple[int, np.ndarray]],
        block_size: int,
    ) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]]]:
        buffer = self._buffer
        assert buffer is not None
        blocks = state.pending_blocks
        keyed_end = len(state.block_hashes) * block_size
        ready = [(start, rows) for start, rows in blocks if start < keyed_end]
        pending = [(start, rows) for start, rows in blocks if start >= keyed_end]
        for start, rows in completed:
            if start < keyed_end:
                ready.append((start, rows))
            else:
                # Completed tail slots and model-runner output buffers are
                # reusable after this call, so retain only unkeyed blocks.
                pending.append((start, buffer.retain_block(rows)))
        state.pending_blocks = pending
        return pending, ready

    @staticmethod
    def _assemble_segments(
        token_start: int,
        token_end: int,
        segments: list[tuple[int, np.ndarray]],
    ) -> np.ndarray:
        if len(segments) == 1:
            segment_start, rows = segments[0]
            local_start = token_start - segment_start
            local_end = token_end - segment_start
            if 0 <= local_start < local_end <= len(rows):
                return rows[local_start:local_end].copy()
        cursor = token_start
        chunks = []
        for segment_start, rows in segments:
            segment_end = segment_start + len(rows)
            if segment_end <= cursor:
                continue
            if segment_start > cursor:
                raise RuntimeError("artifact output has a missing token range")
            end = min(segment_end, token_end)
            chunks.append(rows[cursor - segment_start : end - segment_start])
            cursor = end
            if cursor == token_end:
                break
        if cursor != token_end:
            raise RuntimeError("artifact output has a missing token range")
        return chunks[0].copy() if len(chunks) == 1 else np.concatenate(chunks)

    def begin_step(self, metadata: ArtifactConnectorMetadata | None) -> None:
        self._step_metadata = metadata
        if self._buffer is None or metadata is None:
            return
        with self._lock:
            if metadata.generation != self._generation:
                self._buffer.reset()
                self._requests.clear()
                self._generation = metadata.generation
            for request in metadata.requests:
                self._merge_block_hashes(request)
            for key, block_hashes in metadata.finished_requests.items():
                state = self._requests.get(key)
                if state is None:
                    continue
                if state.pending_outputs:
                    state.finished = True
                    state.finished_block_hashes = block_hashes
                else:
                    self._finish_request(key, state, block_hashes, metadata.block_size)

    def output_finished(self, metadata: ArtifactConnectorMetadata) -> None:
        if self._buffer is None:
            return
        with self._lock:
            if metadata.generation != self._generation:
                return
            for request in metadata.requests:
                key = (request.request_id, request.epoch)
                state = self._requests[key]
                state.pending_outputs -= 1
                if not state.pending_outputs and state.finished:
                    self._finish_request(
                        key, state, state.finished_block_hashes, metadata.block_size
                    )

    def _merge_block_hashes(self, request: ArtifactRequestMetadata) -> None:
        key = (request.request_id, request.epoch)
        state = self._requests.setdefault(key, _WorkerRequestState())
        block_hashes = state.block_hashes
        start = request.block_hash_start
        if start > len(block_hashes):
            raise RuntimeError(
                f"artifact block-hash delta is missing for {request.request_id}"
            )
        if start < len(block_hashes):
            overlap = min(len(request.block_hashes), len(block_hashes) - start)
            if (
                list(request.block_hashes[:overlap])
                != block_hashes[start : start + overlap]
            ):
                raise RuntimeError(
                    f"artifact block-hash history changed for {request.request_id}"
                )
            new_hashes = request.block_hashes[overlap:]
        else:
            new_hashes = request.block_hashes
        new_hashes = list(new_hashes)
        block_hashes.extend(new_hashes)
        state.artifact_keys.extend(
            routed_experts_keys(new_hashes, str(self._generation))
        )

    def _finish_request(
        self,
        key: _RequestKey,
        state: _WorkerRequestState,
        block_hashes: Sequence[bytes] | None,
        block_size: int,
    ) -> None:
        assert self._store is not None and self._buffer is not None
        if block_hashes is not None:
            num_existing_hashes = len(state.block_hashes)
            if (
                num_existing_hashes > len(block_hashes)
                or list(block_hashes[:num_existing_hashes]) != state.block_hashes
            ):
                raise RuntimeError("artifact block-hash history changed")
            new_hashes = list(block_hashes[num_existing_hashes:])
            state.block_hashes.extend(new_hashes)
            state.artifact_keys.extend(
                routed_experts_keys(new_hashes, str(self._generation))
            )
            pending, ready = self._take_available_blocks(state, [], block_size)
            publish_routed_experts(
                self._store,
                batches=[(state.artifact_keys, ready)],
                block_size=block_size,
            )
            for _, rows in [*ready, *pending]:
                self._buffer.release_block(rows)
        else:
            for _, rows in state.pending_blocks:
                self._buffer.release_block(rows)
        self._buffer.discard(key)
        del self._requests[key]

    def _materialize(
        self,
        token_start: int,
        token_end: int,
        artifact_keys: Sequence[str],
        block_size: int,
    ) -> np.ndarray:
        assert self._store is not None
        first_block = token_start // block_size
        last_block = (token_end + block_size - 1) // block_size
        stored = materialize_routed_experts(
            self._store,
            list(artifact_keys[first_block:last_block]),
            shape_per_token=self._shape_per_token,
            dtype=self._dtype,
            rows_per_object=block_size,
        )
        local_start = token_start - first_block * block_size
        return stored[local_start : local_start + token_end - token_start]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
