# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side execution-artifact data plane."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.artifact_connector.connector import (
    ArtifactConnectorMetadata,
    ArtifactRequestOutput,
)
from vllm.distributed.artifact_connector.routed_experts import (
    RoutedExpertsArtifactBuffer,
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_keys,
)
from vllm.distributed.artifact_connector.store import (
    BackgroundArtifactStore,
    InProcessArtifactStore,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass
class _WorkerRequestState:
    artifact_keys: list[str] = field(default_factory=list)
    pending_blocks: list[tuple[int, np.ndarray]] = field(default_factory=list)
    capture_cursor: int | None = None
    scheduled_cursor: int = 0
    emit_cursor: int = 0


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
        self._requests: dict[str, _WorkerRequestState] = {}
        self._generation = 0
        self._step_metadata: ArtifactConnectorMetadata | None = None
        # Every TP rank participates in capture collectives, but only the
        # executor output rank owns the artifact data plane.
        if not get_tp_group().is_first_rank:
            return

        shape_per_token = self._capturer.shape_per_token
        dtype: np.dtype[Any] = np.dtype(self._capturer.output_dtype_name)
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        hashes_per_kv_block = scheduler_block_size // hash_block_size
        block_nbytes = hash_block_size * int(np.prod(shape_per_token)) * dtype.itemsize
        max_bytes = vllm_config.artifact_config.max_bytes
        if max_bytes is None:
            max_bytes = kv_cache_config.num_blocks * hashes_per_kv_block * block_nbytes
        self._store = BackgroundArtifactStore(
            InProcessArtifactStore(
                max_bytes=max_bytes,
                object_nbytes=block_nbytes,
            ),
            max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
        )
        self._buffer = RoutedExpertsArtifactBuffer(
            dtype,
            shape_per_token,
            hash_block_size,
            vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens,
            vllm_config.max_concurrent_batches,
        )

    def prepare_output(
        self,
        request_ids: list[str],
        token_starts: np.ndarray,
        query_start_loc: np.ndarray,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> dict[str, ArtifactRequestOutput] | None:
        buffer = self._buffer
        if buffer is None or self._step_metadata is None:
            return None
        store = self._store
        assert store is not None

        query_start_loc = query_start_loc[: len(request_ids) + 1]
        # Freeze the packed batch before splitting it into request ranges.
        num_rows = int(query_start_loc[-1])
        routed_experts = self._capturer.snapshot_routing_data(num_rows).cpu().numpy()
        block_size = buffer.block_size

        # Publish the whole batch before materializing any consumer output.
        materialize_outputs: list[tuple[str, int, int]] = []
        block_batches = []
        outputs: dict[str, ArtifactRequestOutput] = {}

        # Use the ModelRunner's actual batch boundaries rather than rebuilding them.
        for request_id, token_start, start, end, sampled, rejected in zip(
            request_ids,
            token_starts,
            query_start_loc[:-1],
            query_start_loc[1:],
            num_sampled.cpu().numpy(),
            num_rejected.cpu().numpy(),
            strict=True,
        ):
            request_num_tokens = end - start
            if request_num_tokens <= 0:
                raise RuntimeError("artifact request token count must be positive")
            state = self._requests[request_id]

            # Capture precedes speculative acceptance, so discard the rejected
            # suffix. Batch boundaries still span the full executed range.
            rejected = int(rejected)
            if not 0 <= rejected <= request_num_tokens:
                raise RuntimeError("artifact rejected-token count is invalid")
            rows = routed_experts[start : end - rejected]

            capture_start = token_start
            capture_cursor = state.capture_cursor
            if capture_cursor is None:
                capture_cursor = capture_start

            if capture_start < capture_cursor:
                raise RuntimeError("artifact capture moved backwards")
            if capture_start > capture_cursor:
                # Reattach after an optimistically scheduled suffix was rejected.
                if capture_cursor >= state.scheduled_cursor:
                    raise RuntimeError("artifact capture has an unbacked token gap")
                capture_start = capture_cursor

            emit_start = state.emit_cursor
            # Complete blocks without keys remain pending until a hash update.
            completed = buffer.capture(request_id, capture_start, rows)
            state.capture_cursor = capture_start + len(rows)
            state.scheduled_cursor = token_start + request_num_tokens
            block_batches.append((state, completed))

            token_end = capture_start + len(rows)
            if sampled > 0 and emit_start < token_end:
                if emit_start >= capture_start:
                    outputs[request_id] = ArtifactRequestOutput(
                        emit_start,
                        rows[emit_start - capture_start :],
                    )
                    state.emit_cursor = token_end
                else:
                    materialize_outputs.append((request_id, emit_start, token_end))

        # A consumer may reuse a block produced earlier in the same batch.
        self._publish_blocks(block_batches)

        for request_id, emit_start, token_end in materialize_outputs:
            state = self._requests[request_id]
            stored_end = (
                min(token_end // block_size, len(state.artifact_keys)) * block_size
            )
            if emit_start < stored_end:
                first_block = emit_start // block_size
                stored = materialize_routed_experts(
                    store,
                    state.artifact_keys[first_block : stored_end // block_size],
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
            outputs[request_id] = ArtifactRequestOutput(emit_start, rows)
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
            keyed_end = len(state.artifact_keys) * buffer.block_size
            ready = [(start, rows) for start, rows in blocks if start < keyed_end]
            state.pending_blocks = [
                (start, buffer.retain_block(rows))
                for start, rows in blocks
                if start >= keyed_end
            ]
            if ready:
                ready_batches.append((state.artifact_keys, ready))
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

    def begin_step(self, metadata: ArtifactConnectorMetadata | None) -> None:
        self._step_metadata = metadata
        if self._buffer is None or metadata is None:
            return
        if metadata.requests.keys() & metadata.finished_requests:
            raise RuntimeError("artifact request cannot run and finish in one step")
        if metadata.generation < self._generation:
            raise RuntimeError("artifact metadata generation moved backwards")
        release_keys: list[str] = []
        if metadata.generation > self._generation:
            release_keys.extend(
                key
                for state in self._requests.values()
                for key in reversed(state.artifact_keys)
            )
            self._buffer.reset()
            self._requests.clear()
            self._generation = metadata.generation
        for request_id, emit_start in metadata.requests.items():
            state = self._requests.setdefault(
                request_id, _WorkerRequestState(emit_cursor=emit_start)
            )
            if emit_start > state.emit_cursor:
                raise RuntimeError("artifact Scheduler emit cursor moved ahead")
        block_batches: list[
            tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]
        ] = []
        retained_keys: list[str] = []
        for request_id, block_hashes in metadata.block_hashes.items():
            state = self._requests[request_id]
            keys = routed_experts_keys(block_hashes, str(self._generation))
            state.artifact_keys.extend(keys)
            retained_keys.extend(keys)
            block_batches.append((state, []))
        release_keys.extend(
            key
            for request_id in metadata.finished_requests
            for key in reversed(self._requests[request_id].artifact_keys)
        )
        self._publish_blocks(block_batches, retained_keys, release_keys)
        for request_id in metadata.finished_requests:
            state = self._requests.pop(request_id)
            for _, rows in state.pending_blocks:
                self._buffer.release_block(rows)
            self._buffer.discard(request_id)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
