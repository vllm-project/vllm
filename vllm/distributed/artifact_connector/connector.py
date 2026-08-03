# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side connector for routed-experts execution artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.artifact_connector.buffer import RoutedExpertsArtifactBuffer
from vllm.distributed.artifact_connector.request_core import (
    RoutedExpertsRequestCore,
    get_routing_shape_and_dtype,
    materialize_routed_experts,
    routed_experts_key,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactStore,
)
from vllm.distributed.artifact_connector.store import BackgroundArtifactStore

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class _RequestState:
    next_full_end: int
    emit_cursor: int


class ArtifactSchedulerConnector:
    """Persist worker-captured R3 under logical KV-cache block hashes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        *,
        kv_connector: KVConnectorBase_V1 | None,
        block_size: int,
    ) -> None:
        config = vllm_config.artifact_config
        parallel_config = vllm_config.parallel_config
        shape_per_token, dtype = get_routing_shape_and_dtype(vllm_config)
        self._shape_per_token = shape_per_token
        self._dtype: np.dtype[Any] = np.dtype(dtype)

        num_cpu_blocks = getattr(
            getattr(kv_connector, "scheduler_manager", None),
            "num_cpu_blocks",
            0,
        )
        num_kv_blocks = max(kv_cache_config.num_blocks, num_cpu_blocks)
        block_nbytes = block_size * int(np.prod(shape_per_token)) * self._dtype.itemsize
        minimum_shm_bytes = num_kv_blocks * block_nbytes
        max_shm_bytes = config.max_shm_bytes
        if max_shm_bytes is None:
            max_shm_bytes = minimum_shm_bytes
        elif max_shm_bytes < minimum_shm_bytes:
            raise ValueError(
                "artifact_config.max_shm_bytes is smaller than required by the "
                "KV cache capacity: "
                f"configured={max_shm_bytes}, minimum={minimum_shm_bytes}, "
                f"gpu_blocks={kv_cache_config.num_blocks}, "
                f"cpu_blocks={num_cpu_blocks}, "
                f"r3_bytes_per_block={block_nbytes}"
            )
        self._store = BackgroundArtifactStore(
            LocalSharedMemoryArtifactStore(
                config.shm_dir,
                vllm_config.instance_id,
                parallel_config.data_parallel_rank,
                max_bytes=max_shm_bytes,
                ttl_seconds=config.shm_ttl_seconds,
            ),
            max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
        )
        self._buffer = RoutedExpertsArtifactBuffer(
            self._dtype, shape_per_token, block_size
        )
        self._request_core = RoutedExpertsRequestCore(self._store, self._buffer)
        self._states: dict[str, _RequestState] = {}
        self._resume_emit_cursors: dict[str, int] = {}
        self._step_rows: dict[str, tuple[int, np.ndarray]] = {}
        self._kv_cache_generation = 0
        self._reuse_kv_hashes = vllm_config.cache_config.enable_prefix_caching

    def _state(self, request_id: str) -> _RequestState:
        try:
            return self._states[request_id]
        except KeyError as error:
            raise RuntimeError(
                f"artifact request has not started: {request_id}"
            ) from error

    def request_started(
        self,
        *,
        request: Request,
        cached_token_end: int,
        hash_block_size: int,
    ) -> None:
        if request.request_id in self._states:
            return
        if cached_token_end % hash_block_size:
            raise RuntimeError(
                "cached KV boundary is not aligned with the artifact hash block: "
                f"request={request.request_id}, cached={cached_token_end}, "
                f"block_size={hash_block_size}"
            )
        sampling_params = request.sampling_params
        prompt_start = (
            sampling_params.routed_experts_prompt_start
            if sampling_params is not None
            else 0
        )
        if prompt_start < 0 or prompt_start >= request.num_prompt_tokens:
            raise ValueError(
                "routed_experts_prompt_start "
                f"({prompt_start}) must be >= 0 and < num_prompt_tokens "
                f"({request.num_prompt_tokens})"
            )
        emit_cursor = self._resume_emit_cursors.pop(
            request.request_id,
            prompt_start,
        )
        if not prompt_start <= emit_cursor <= request.num_tokens - 1:
            raise RuntimeError(
                "invalid artifact resume cursor: "
                f"request={request.request_id}, cursor={emit_cursor}, "
                f"range=[{prompt_start}, {request.num_tokens - 1}]"
            )
        self._states[request.request_id] = _RequestState(
            next_full_end=cached_token_end,
            emit_cursor=emit_cursor,
        )

    def _block_hashes(
        self,
        request: Request,
        block_end: int,
    ) -> list[bytes]:
        if self._reuse_kv_hashes:
            block_hashes = request.block_hashes[:block_end]
            if len(block_hashes) != block_end:
                raise RuntimeError(
                    "missing KV-compatible hashes for completed artifact blocks"
                )
            return list(block_hashes)

        return [
            f"{request.request_id}:{block_index}".encode()
            for block_index in range(block_end)
        ]

    def capture_step(
        self,
        scheduler_output: SchedulerOutput,
        routed_experts: np.ndarray | None,
        request_ids: list[str],
        stale_request_ids: set[str] | None = None,
    ) -> None:
        """Split #50721's stable per-step R3 snapshot by logical request."""
        if routed_experts is None:
            if scheduler_output.total_num_scheduled_tokens:
                raise RuntimeError("artifact capture output is missing")
            return

        token_starts = {
            request.req_id: request.num_computed_tokens
            for request in scheduler_output.scheduled_new_reqs
        }
        cached = scheduler_output.scheduled_cached_reqs
        token_starts.update(
            zip(cached.req_ids, cached.num_computed_tokens, strict=True)
        )

        stale_request_ids = stale_request_ids or set()
        self._step_rows.clear()
        offset = 0
        for request_id in request_ids:
            num_tokens = scheduler_output.num_scheduled_tokens[request_id]
            end = offset + num_tokens
            try:
                token_start = token_starts[request_id]
            except KeyError as error:
                raise RuntimeError(
                    f"artifact token start is missing for request {request_id}"
                ) from error
            if request_id in self._states and request_id not in stale_request_ids:
                request_rows = routed_experts[offset:end]
                self._buffer.capture(
                    request_id,
                    token_start,
                    request_rows,
                )
                self._step_rows[request_id] = (token_start, request_rows)
            offset = end
        if offset != len(routed_experts):
            raise RuntimeError("artifact capture output has an invalid row count")

    def request_progress(
        self,
        *,
        request: Request,
        accepted_token_end: int,
        hash_block_size: int,
    ) -> None:
        state = self._state(request.request_id)
        full_end = accepted_token_end // hash_block_size * hash_block_size
        if full_end <= state.next_full_end:
            return
        first_block = state.next_full_end // hash_block_size
        last_block = full_end // hash_block_size
        block_hashes = self._block_hashes(request, last_block)[first_block:]
        self._request_core.commit(
            request_id=request.request_id,
            artifact_namespace=self._artifact_namespace,
            block_hashes=block_hashes,
            block_start=state.next_full_end,
            block_size=hash_block_size,
        )
        state.next_full_end = full_end

    def take_output(
        self,
        *,
        request: Request,
        token_end: int,
        hash_block_size: int,
    ) -> np.ndarray | None:
        """Return the next inline R3 delta for the SHM backend."""
        state = self._state(request.request_id)
        token_start = state.emit_cursor
        if token_end <= token_start:
            return None

        step = self._step_rows.get(request.request_id)
        if step is not None:
            step_start, step_rows = step
            if step_start <= token_start and token_end <= step_start + len(step_rows):
                state.emit_cursor = token_end
                return step_rows[token_start - step_start : token_end - step_start]

        chunks: list[np.ndarray] = []
        stored_end = min(state.next_full_end, token_end)
        if token_start < stored_end:
            first_block = token_start // hash_block_size
            last_block = stored_end // hash_block_size
            block_hashes = self._block_hashes(request, last_block)[first_block:]
            object_start = first_block * hash_block_size
            stored = materialize_routed_experts(
                self._store,
                [
                    routed_experts_key(block_hash, self._artifact_namespace)
                    for block_hash in block_hashes
                ],
                shape_per_token=self._shape_per_token,
                dtype=self._dtype,
                rows_per_object=hash_block_size,
            )
            chunks.append(stored[token_start - object_start :])

        buffer_start = max(token_start, stored_end)
        if buffer_start < token_end:
            chunks.append(
                self._buffer.read(request.request_id, buffer_start, token_end)
            )

        output = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if len(output) != token_end - token_start:
            raise RuntimeError("inline artifact output has an invalid row count")
        state.emit_cursor = token_end
        return output

    def request_finished(
        self,
        request_id: str,
    ) -> None:
        self._states.pop(request_id)
        self._buffer.discard(request_id)

    def request_aborted(self, request_id: str) -> None:
        self._states.pop(request_id, None)
        self._resume_emit_cursors.pop(request_id, None)
        self._buffer.discard(request_id)

    @property
    def _artifact_namespace(self) -> str:
        return str(self._kv_cache_generation)

    def reset(self) -> None:
        """Reset artifact state after a successful KV cache reset."""
        for request_id, state in self._states.items():
            self._resume_emit_cursors[request_id] = state.emit_cursor
            self._buffer.discard(request_id)
        self._states.clear()
        self._kv_cache_generation += 1

    def shutdown(self) -> None:
        self._store.close()
