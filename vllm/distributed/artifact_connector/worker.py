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
import torch.distributed

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

_RequestKey = tuple[str, int]


@dataclass
class _WorkerRequestState:
    block_hashes: list[bytes] = field(default_factory=list)
    artifact_keys: list[str] = field(default_factory=list)
    kv_block_ids: set[int] = field(default_factory=set)
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
        metadata: ArtifactConnectorMetadata | None,
        routed_experts: torch.Tensor | None,
        request_ids: list[str],
        num_rejected: torch.Tensor | None,
    ) -> None:
        self._connector = connector
        self._metadata = metadata
        self._routed_experts = routed_experts
        self._request_ids = request_ids
        self._num_rejected = num_rejected
        self._routed_experts_cpu: np.ndarray | None = None
        self._num_rejected_cpu: np.ndarray | None = None

    def to_cpu_nonblocking(self) -> None:
        if self._routed_experts is None:
            return
        assert self._num_rejected is not None
        self._routed_experts_cpu = self._routed_experts.to(
            "cpu", non_blocking=True
        ).numpy()
        self._num_rejected_cpu = self._num_rejected.to("cpu", non_blocking=True).numpy()

    def finish(self, invalid_block_ids: set[int]) -> ArtifactConnectorOutput:
        invalid_block_ids = self._connector.sync_invalid_block_ids(invalid_block_ids)
        if self._metadata is None:
            return ArtifactConnectorOutput({})
        assert self._routed_experts_cpu is not None
        assert self._num_rejected_cpu is not None
        try:
            return self._connector.process_output(
                self._metadata,
                self._routed_experts_cpu,
                self._request_ids,
                self._num_rejected_cpu,
                invalid_block_ids,
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
        external_capacity_blocks: int | None,
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
        self._sync_invalid_blocks = (
            vllm_config.kv_transfer_config is not None and get_tp_group().world_size > 1
        )

        # Every TP rank participates in capture collectives, but only the
        # executor output rank owns the artifact data plane.
        if not get_tp_group().is_first_rank:
            return

        shape_per_token, dtype = get_routing_shape_and_dtype(vllm_config)
        self._shape_per_token = shape_per_token
        self._dtype: np.dtype[Any] = np.dtype(dtype)
        scheduler_block_size, block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        hashes_per_kv_block = scheduler_block_size // block_size
        block_nbytes = block_size * int(np.prod(shape_per_token)) * self._dtype.itemsize
        max_bytes = vllm_config.artifact_config.max_shm_bytes
        if max_bytes is None:
            if external_capacity_blocks is None:
                raise ValueError(
                    "artifact_config.max_shm_bytes is required when the KV "
                    "connector does not report its storage capacity"
                )
            max_bytes = (
                max(
                    kv_cache_config.num_blocks * hashes_per_kv_block,
                    (vllm_config.cache_config.num_cpu_blocks or 0)
                    * hashes_per_kv_block,
                    external_capacity_blocks,
                )
                * block_nbytes
            )
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

    def sync_invalid_block_ids(self, block_ids: set[int]) -> set[int]:
        if not self._sync_invalid_blocks:
            return block_ids
        tp_group = get_tp_group()
        gathered: list[set[int] | None] = [None] * tp_group.world_size
        torch.distributed.all_gather_object(
            gathered,
            block_ids,
            group=tp_group.cpu_group,
        )
        return set().union(*(ids or () for ids in gathered))

    def prepare_output(
        self,
        metadata: ArtifactConnectorMetadata | None,
        routed_experts: torch.Tensor | None,
        request_ids: list[str],
        num_rejected: torch.Tensor,
    ) -> PendingArtifactOutput | None:
        # Warmup runs capture without scheduler metadata.
        if metadata is None or routed_experts is None:
            return None
        if self._store is None:
            if not self._sync_invalid_blocks:
                return None
            return PendingArtifactOutput(self, None, None, [], None)
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
        invalid_block_ids: set[int] | None = None,
    ) -> ArtifactConnectorOutput:
        with self._lock:
            if metadata.generation != self._generation:
                return ArtifactConnectorOutput({})
            return self._process_output(
                metadata,
                routed_experts,
                request_ids,
                num_rejected,
                invalid_block_ids or set(),
            )

    def _process_output(
        self,
        metadata: ArtifactConnectorMetadata,
        routed_experts: np.ndarray,
        request_ids: list[str],
        num_rejected: np.ndarray,
        invalid_block_ids: set[int],
    ) -> ArtifactConnectorOutput:
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
        invalid_requests: set[tuple[str, int]] = set()
        if invalid_block_ids:
            for state in self._requests.values():
                if state.kv_block_ids.intersection(invalid_block_ids):
                    state.finished = True
                    state.finished_block_hashes = None
        offset = 0
        for request_index, request_id in enumerate(request_ids):
            request = by_request.get(request_id)
            if request is None:
                raise RuntimeError(f"artifact metadata is missing {request_id}")
            end = offset + request.num_tokens
            key = (request_id, request.epoch)
            state = self._requests[key]
            if state.finished and state.finished_block_hashes is None:
                invalid_requests.add(key)
                offset = end
                continue
            block_hashes = state.block_hashes
            artifact_keys = state.artifact_keys
            valid_end = end - int(num_rejected[request_index])
            rows: np.ndarray = routed_experts[offset:valid_end].astype(
                self._dtype, copy=False
            )
            capture_start = min(
                request.token_start,
                state.capture_cursor
                if state.capture_cursor is not None
                else request.token_start,
            )
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
            captured.append(
                (request, capture_start, rows, [*ready, *pending], emit_start)
            )
            offset = end
        if offset != len(routed_experts):
            raise RuntimeError("artifact capture output has an invalid row count")
        publish_routed_experts(
            self._store,
            batches=commit_batches,
            block_size=metadata.block_size,
        )
        outputs: dict[str, ArtifactRequestOutput] = {}
        for request, capture_start, rows, local_segments, emit_start in captured:
            key = (request.request_id, request.epoch)
            state = self._requests[key]
            block_hashes = state.block_hashes
            artifact_keys = state.artifact_keys
            token_end = capture_start + len(rows)
            if not request.emit_output or emit_start >= token_end:
                continue
            stored_end = min(
                capture_start // metadata.block_size * metadata.block_size,
                len(block_hashes) * metadata.block_size,
            )
            if emit_start >= capture_start:
                # The D2H allocation stays owned by this output. Only history
                # read from reusable tail slots needs the copying path below.
                output_rows = rows[emit_start - capture_start :]
            else:
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
                segments.extend(local_segments)
                tail_cursor = emit_start
                for start, segment in sorted(segments, key=lambda item: item[0]):
                    if tail_cursor >= capture_start:
                        break
                    if start > tail_cursor:
                        gap_end = min(start, capture_start)
                        segments.append(
                            (
                                tail_cursor,
                                self._buffer.read(key, tail_cursor, gap_end),
                            )
                        )
                        tail_cursor = gap_end
                    if start <= tail_cursor:
                        tail_cursor = max(tail_cursor, start + len(segment))
                if tail_cursor < capture_start:
                    segments.append(
                        (
                            tail_cursor,
                            self._buffer.read(key, tail_cursor, capture_start),
                        )
                    )
                segments.append((capture_start, rows))
                output_rows = self._assemble_segments(emit_start, token_end, segments)
            outputs[request.request_id] = ArtifactRequestOutput(
                emit_start,
                output_rows,
            )
            state.emit_cursor = token_end
        for rows in committed_rows:
            self._buffer.release_block(rows)
        return ArtifactConnectorOutput(outputs, invalid_requests)

    def _take_available_blocks(
        self,
        state: _WorkerRequestState,
        completed: list[tuple[int, np.ndarray]],
        block_size: int,
    ) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]]]:
        buffer = self._buffer
        assert buffer is not None
        blocks = state.pending_blocks
        state.pending_blocks = []
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
        if pending:
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
        return chunks[0].copy() if len(chunks) == 1 else np.concatenate(chunks)

    def begin_step(self, metadata: ArtifactConnectorMetadata | None) -> None:
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
        state.kv_block_ids.update(request.kv_block_ids)
        block_hashes = state.block_hashes
        start = request.block_hash_start
        if start > len(block_hashes):
            raise RuntimeError(
                f"artifact block-hash delta is missing for {request.request_id}"
            )
        overlap = min(len(request.block_hashes), len(block_hashes) - start)
        if (
            list(request.block_hashes[:overlap])
            != block_hashes[start : start + overlap]
        ):
            raise RuntimeError(
                f"artifact block-hash history changed for {request.request_id}"
            )
        new_hashes = request.block_hashes[overlap:]
        block_hashes.extend(new_hashes)
        state.artifact_keys.extend(
            routed_experts_key(block_hash, str(self._generation))
            for block_hash in new_hashes
        )

    def _finish_request(
        self,
        key: _RequestKey,
        state: _WorkerRequestState,
        block_hashes: Sequence[bytes] | None,
        block_size: int,
    ) -> None:
        if block_hashes is not None:
            state.block_hashes = list(block_hashes)
            state.artifact_keys = [
                routed_experts_key(block_hash, str(self._generation))
                for block_hash in block_hashes
            ]
            pending, ready = self._take_available_blocks(state, [], block_size)
            assert self._store is not None and self._buffer is not None
            publish_routed_experts(
                self._store,
                batches=[(state.artifact_keys, ready)],
                block_size=block_size,
            )
            for _, rows in [*ready, *pending]:
                self._buffer.release_block(rows)
        else:
            for _, rows in state.pending_blocks:
                assert self._buffer is not None
                self._buffer.release_block(rows)
        assert self._buffer is not None
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
        last_block = token_end // block_size
        stored = materialize_routed_experts(
            self._store,
            list(artifact_keys[first_block:last_block]),
            shape_per_token=self._shape_per_token,
            dtype=self._dtype,
            rows_per_object=block_size,
        )
        return stored[token_start - first_block * block_size :]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
