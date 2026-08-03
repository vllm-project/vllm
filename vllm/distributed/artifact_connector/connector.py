# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side connector for routed-experts execution artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.artifact_connector.buffer import RoutedExpertsArtifactBuffer
from vllm.distributed.artifact_connector.request_core import (
    ArtifactCommit,
    ArtifactFinalize,
    RoutedExpertsRequestCore,
    get_routing_shape_and_dtype,
    materialize_routed_experts,
    routed_experts_key,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactStore,
)
from vllm.distributed.artifact_connector.store import BackgroundArtifactStore
from vllm.utils import random_uuid
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    hash_block_tokens,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class _RequestState:
    next_full_end: int
    emit_cursor: int
    artifact_namespace: str


class ArtifactSchedulerConnector:
    """Persist worker-captured R3 under logical KV-cache block hashes."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        config = vllm_config.artifact_config
        parallel_config = vllm_config.parallel_config
        self._store = BackgroundArtifactStore(
            LocalSharedMemoryArtifactStore(
                config.shm_dir,
                vllm_config.instance_id,
                parallel_config.data_parallel_rank,
                max_bytes=config.max_shm_bytes,
                ttl_seconds=config.shm_ttl_seconds,
            ),
            max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
        )
        shape_per_token, dtype = get_routing_shape_and_dtype(vllm_config)
        self._shape_per_token = shape_per_token
        self._dtype: np.dtype[Any] = np.dtype(dtype)
        self._buffer = RoutedExpertsArtifactBuffer(self._dtype, shape_per_token)
        self._request_core = RoutedExpertsRequestCore(self._store, self._buffer)
        self._states: dict[str, _RequestState] = {}
        self._resume_emit_cursors: dict[str, int] = {}
        self._session_id = random_uuid()
        self._kv_cache_generation = 0
        self._hash_fn = get_hash_fn_by_name(
            vllm_config.cache_config.prefix_caching_hash_algo
        )

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
        if prompt_start is None:
            prompt_start = 0
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
            artifact_namespace=self._artifact_namespace,
        )

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
                self._buffer.capture(
                    request_id,
                    token_start,
                    routed_experts[offset:end],
                )
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
        block_hashes = list(request.block_hashes[first_block:last_block])
        if len(block_hashes) != last_block - first_block:
            raise RuntimeError(
                "missing KV-compatible hashes for completed artifact blocks"
            )
        self._request_core.commit(
            [
                ArtifactCommit(
                    request_id=request.request_id,
                    artifact_namespace=state.artifact_namespace,
                    block_hashes=block_hashes,
                    block_start=state.next_full_end,
                    hash_block_size=hash_block_size,
                )
            ]
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

        chunks: list[np.ndarray] = []
        stored_end = min(state.next_full_end, token_end)
        if token_start < stored_end:
            first_block = token_start // hash_block_size
            last_block = stored_end // hash_block_size
            block_hashes = list(request.block_hashes[first_block:last_block])
            if len(block_hashes) != last_block - first_block:
                raise RuntimeError(
                    "missing KV-compatible hashes for inline artifact output"
                )
            object_start = first_block * hash_block_size
            stored = materialize_routed_experts(
                self._store,
                [
                    routed_experts_key(block_hash, state.artifact_namespace)
                    for block_hash in block_hashes
                ],
                expected_shape_per_token=self._shape_per_token,
                expected_dtype=self._dtype,
                expected_token_start=object_start,
                expected_token_end=stored_end,
                hash_block_size=hash_block_size,
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
        *,
        request: Request,
        token_end: int,
        hash_block_size: int,
    ) -> list[str]:
        state = self._state(request.request_id)
        full_end = token_end // hash_block_size * hash_block_size
        tail_block_hash = None
        if full_end < token_end:
            if token_end > request.num_tokens:
                raise RuntimeError(
                    "artifact token boundary exceeds the request token count"
                )
            parent_hash = (
                request.block_hashes[full_end // hash_block_size - 1]
                if full_end
                else None
            )
            extra_keys, _ = generate_block_hash_extra_keys(
                request,
                full_end,
                token_end,
                -1 if full_end else 0,
            )
            tail_block_hash = hash_block_tokens(
                self._hash_fn,
                parent_hash,
                request.all_token_ids[full_end:token_end],
                extra_keys,
            )

        finalize = ArtifactFinalize(
            request_id=request.request_id,
            artifact_namespace=state.artifact_namespace,
            block_hashes=list(request.block_hashes),
            tail_block_hash=tail_block_hash,
            token_end=token_end,
            hash_block_size=hash_block_size,
        )
        keys = self._request_core.finalize(finalize)
        self._states.pop(request.request_id)
        return keys

    def request_aborted(self, request_id: str) -> None:
        self._states.pop(request_id, None)
        self._resume_emit_cursors.pop(request_id, None)
        self._buffer.discard(request_id)

    @property
    def _artifact_namespace(self) -> str:
        return f"{self._session_id}:{self._kv_cache_generation}"

    def advance_kv_cache_generation(self) -> None:
        """Start a new namespace after a successful KV cache reset."""
        for request_id, state in self._states.items():
            self._resume_emit_cursors[request_id] = state.emit_cursor
            self._buffer.discard(request_id)
        self._states.clear()
        self._kv_cache_generation += 1

    def shutdown(self) -> None:
        self._store.close()
