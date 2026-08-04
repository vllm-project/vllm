# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side execution-artifact data plane."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.artifact_connector.buffer import RoutedExpertsArtifactBuffer
from vllm.distributed.artifact_connector.connector import (
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactRequestMetadata,
    ArtifactRequestOutput,
)
from vllm.distributed.artifact_connector.routed_experts import (
    get_routing_shape_and_dtype,
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_key,
)
from vllm.distributed.artifact_connector.shm import LocalSharedMemoryArtifactStore
from vllm.distributed.artifact_connector.store import BackgroundArtifactStore
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


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
        self._pending_blocks: dict[str, list[tuple[int, np.ndarray]]] = {}
        self._capture_cursors: dict[str, int] = {}
        self._emit_cursors: dict[str, int] = {}
        self._generation = -1
        self._pending_requests: dict[str, int] = {}
        self._finished_requests: dict[str, Sequence[bytes] | None] = {}
        self._pending_lock = Lock()

        # Every TP rank participates in capture collectives, but only the
        # executor output rank owns the artifact data plane.
        if not get_tp_group().is_first_rank:
            return

        shape_per_token, dtype = get_routing_shape_and_dtype(vllm_config)
        self._shape_per_token = shape_per_token
        self._dtype: np.dtype[Any] = np.dtype(dtype)
        _, block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
        capacity_block_size = vllm_config.cache_config.block_size
        assert capacity_block_size is not None
        block_nbytes = (
            capacity_block_size * int(np.prod(shape_per_token)) * self._dtype.itemsize
        )
        max_bytes = vllm_config.artifact_config.max_shm_bytes
        if max_bytes is None:
            max_bytes = kv_cache_config.num_blocks * block_nbytes
        self._store = BackgroundArtifactStore(
            LocalSharedMemoryArtifactStore(
                vllm_config.artifact_config.shm_dir,
                vllm_config.instance_id,
                vllm_config.parallel_config.data_parallel_rank,
                max_bytes=max_bytes,
                ttl_seconds=vllm_config.artifact_config.shm_ttl_seconds,
            ),
            max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
        )
        self._buffer = RoutedExpertsArtifactBuffer(
            self._dtype,
            shape_per_token,
            block_size,
            vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens,
        )

    def capture_routed_experts(self, num_tokens: int) -> torch.Tensor:
        """Return a stable routed-experts snapshot for the current step."""
        return self._capturer.get_routing_data(num_tokens)

    def prepare_output(
        self,
        metadata: ArtifactConnectorMetadata | None,
        routed_experts: torch.Tensor | None,
        request_ids: list[str],
        num_rejected: torch.Tensor,
    ) -> PendingArtifactOutput | None:
        # Warmup runs capture without scheduler metadata.
        if self._store is None or metadata is None or routed_experts is None:
            return None
        with self._pending_lock:
            for request in metadata.requests:
                request_id = request.request_id
                self._pending_requests[request_id] = (
                    self._pending_requests.get(request_id, 0) + 1
                )
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
        assert self._store is not None and self._buffer is not None
        by_request = {request.request_id: request for request in metadata.requests}
        captured: list[
            tuple[
                ArtifactRequestMetadata,
                np.ndarray,
                list[tuple[int, np.ndarray]],
                int,
            ]
        ] = []
        commit_batches: list[tuple[Sequence[bytes], list[tuple[int, np.ndarray]]]] = []
        committed_rows: list[np.ndarray] = []
        offset = 0
        for request_index, request_id in enumerate(request_ids):
            request = by_request.get(request_id)
            if request is None:
                raise RuntimeError(f"artifact metadata is missing {request_id}")
            end = offset + request.num_tokens
            valid_end = end - int(num_rejected[request_index])
            rows: np.ndarray = routed_experts[offset:valid_end].astype(
                self._dtype, copy=False
            )
            capture_start = min(
                request.token_start,
                self._capture_cursors.get(request_id, request.token_start),
            )
            emit_start = max(
                request.emit_start,
                self._emit_cursors.get(request_id, request.emit_start),
            )
            completed = self._buffer.capture(request_id, capture_start, rows)
            self._capture_cursors[request_id] = capture_start + len(rows)
            pending, ready = self._take_available_blocks(
                request_id,
                request.block_hashes,
                completed,
                metadata.block_size,
            )
            if ready:
                commit_batches.append((request.block_hashes, ready))
                committed_rows.extend(rows for _, rows in ready)
            captured.append(
                (request, capture_start, rows, [*ready, *pending], emit_start)
            )
            offset = end
        if offset != len(routed_experts):
            raise RuntimeError("artifact capture output has an invalid row count")
        publish_routed_experts(
            self._store,
            artifact_namespace=str(self._generation),
            batches=commit_batches,
            block_size=metadata.block_size,
        )
        outputs: dict[str, ArtifactRequestOutput] = {}
        for request, capture_start, rows, local_segments, emit_start in captured:
            token_end = capture_start + len(rows)
            if not request.emit_output or emit_start >= token_end:
                continue
            stored_end = min(
                capture_start // metadata.block_size * metadata.block_size,
                len(request.block_hashes) * metadata.block_size,
            )
            segments: list[tuple[int, np.ndarray]] = []
            if emit_start < stored_end:
                segments.append(
                    (
                        emit_start,
                        self._materialize(
                            emit_start,
                            stored_end,
                            request.block_hashes,
                            metadata.block_size,
                        ),
                    )
                )
            segments.extend(local_segments)
            segments.append((capture_start, rows))
            outputs[request.request_id] = ArtifactRequestOutput(
                emit_start,
                self._assemble_segments(emit_start, token_end, segments),
            )
            self._emit_cursors[request.request_id] = token_end
        for rows in committed_rows:
            self._buffer.release_block(rows)
        return ArtifactConnectorOutput(outputs)

    def _take_available_blocks(
        self,
        request_id: str,
        block_hashes: Sequence[bytes],
        completed: list[tuple[int, np.ndarray]],
        block_size: int,
    ) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]]]:
        buffer = self._buffer
        assert buffer is not None
        blocks = self._pending_blocks.pop(request_id, [])
        keyed_end = len(block_hashes) * block_size
        ready = [(start, rows) for start, rows in blocks if start < keyed_end]
        pending = [(start, rows) for start, rows in blocks if start >= keyed_end]
        for start, rows in completed:
            if start < keyed_end:
                ready.append((start, rows))
            else:
                # Completed tail slots and model-runner output buffers are
                # reusable after this call, so retain only unkeyed blocks.
                pending.append((start, buffer.retain_block(rows)))
        if pending:
            self._pending_blocks[request_id] = pending
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
                return rows[local_start:local_end]
        cursor = token_start
        chunks = []
        for segment_start, rows in sorted(segments, key=lambda item: item[0]):
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
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def begin_step(self, metadata: ArtifactConnectorMetadata | None) -> None:
        if self._buffer is None or metadata is None:
            return
        if metadata.generation != self._generation:
            self._buffer.reset()
            self._pending_blocks.clear()
            self._capture_cursors.clear()
            self._emit_cursors.clear()
            self._finished_requests.clear()
            self._generation = metadata.generation
        finished = []
        with self._pending_lock:
            for request_id, block_hashes in metadata.finished_requests.items():
                if self._pending_requests.get(request_id, 0):
                    self._finished_requests[request_id] = block_hashes
                else:
                    finished.append((request_id, block_hashes))
        for request_id, block_hashes in finished:
            self._finish_request(request_id, block_hashes, metadata.block_size)

    def output_finished(self, metadata: ArtifactConnectorMetadata) -> None:
        if self._buffer is None:
            return
        finished = []
        with self._pending_lock:
            for request in metadata.requests:
                request_id = request.request_id
                pending = self._pending_requests[request_id] - 1
                if pending:
                    self._pending_requests[request_id] = pending
                else:
                    del self._pending_requests[request_id]
                    if request_id in self._finished_requests:
                        finished.append(
                            (request_id, self._finished_requests.pop(request_id))
                        )
        for request_id, block_hashes in finished:
            self._finish_request(request_id, block_hashes, metadata.block_size)

    def _finish_request(
        self,
        request_id: str,
        block_hashes: Sequence[bytes] | None,
        block_size: int,
    ) -> None:
        if block_hashes is not None:
            _, ready = self._take_available_blocks(
                request_id, block_hashes, [], block_size
            )
            assert self._store is not None and self._buffer is not None
            publish_routed_experts(
                self._store,
                artifact_namespace=str(self._generation),
                batches=[(block_hashes, ready)],
                block_size=block_size,
            )
            for _, rows in ready:
                self._buffer.release_block(rows)
        else:
            for _, rows in self._pending_blocks.pop(request_id, []):
                assert self._buffer is not None
                self._buffer.release_block(rows)
        self._discard_request(request_id)

    def _discard_request(self, request_id: str) -> None:
        assert self._buffer is not None
        if self._pending_blocks.get(request_id):
            raise RuntimeError(
                f"finished request has uncommitted artifact blocks: {request_id}"
            )
        self._pending_blocks.pop(request_id, None)
        self._capture_cursors.pop(request_id, None)
        self._emit_cursors.pop(request_id, None)
        self._buffer.discard(request_id)

    def _materialize(
        self,
        token_start: int,
        token_end: int,
        block_hashes: Sequence[bytes],
        block_size: int,
    ) -> np.ndarray:
        assert self._store is not None
        first_block = token_start // block_size
        last_block = token_end // block_size
        stored = materialize_routed_experts(
            self._store,
            [
                routed_experts_key(block_hash, str(self._generation))
                for block_hash in block_hashes[first_block:last_block]
            ],
            shape_per_token=self._shape_per_token,
            dtype=self._dtype,
            rows_per_object=block_size,
        )
        return stored[token_start - first_block * block_size :]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
