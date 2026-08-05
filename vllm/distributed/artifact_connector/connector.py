# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class PackedBlockHashes(Sequence[bytes]):
    """Contiguous, self-contained block hashes for scheduler-worker IPC."""

    data: bytes
    item_size: int

    def __len__(self) -> int:
        return len(self.data) // self.item_size

    @overload
    def __getitem__(self, index: int) -> bytes: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[bytes]: ...

    def __getitem__(self, index: int | slice) -> bytes | Sequence[bytes]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return PackedBlockHashes(
                    self.data[start * self.item_size : stop * self.item_size],
                    self.item_size,
                )
            return [self[i] for i in range(start, stop, step)]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.item_size
        return self.data[start : start + self.item_size]

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self.data), self.item_size):
            yield self.data[start : start + self.item_size]


@dataclass
class ArtifactRequestMetadata:
    request_id: str
    token_start: int
    num_tokens: int
    emit_start: int
    emit_output: bool
    block_hashes: Sequence[bytes]
    block_hash_start: int = 0
    epoch: int = 0
    kv_block_ids: Sequence[int] = ()


@dataclass
class ArtifactConnectorMetadata:
    generation: int
    block_size: int
    requests: list[ArtifactRequestMetadata]
    finished_requests: dict[tuple[str, int], Sequence[bytes] | None]


@dataclass
class ArtifactRequestOutput:
    token_start: int
    rows: np.ndarray


@dataclass
class ArtifactConnectorOutput:
    requests: dict[str, ArtifactRequestOutput]


@dataclass
class _RequestState:
    emit_cursor: int
    epoch: int = 0
    packed_hashes: bytearray = field(default_factory=bytearray)
    num_hashes: int = 0
    hash_size: int = 0


class ArtifactSchedulerConnector:
    """Build worker metadata without owning artifact payloads or stores."""

    def __init__(self, vllm_config: VllmConfig, *, block_size: int) -> None:
        self._block_size = block_size
        self._reuse_kv_hashes = vllm_config.cache_config.enable_prefix_caching
        self._states: dict[str, _RequestState] = {}
        self._resume_emit_cursors: dict[str, int] = {}
        self._finished_requests: dict[tuple[str, int], Sequence[bytes] | None] = {}
        self._generation = 0

    def request_started(self, request: Request) -> None:
        if request.request_id in self._states:
            return
        sampling_params = request.sampling_params
        prompt_start = (
            sampling_params.routed_experts_prompt_start
            if sampling_params is not None
            else 0
        )
        if prompt_start < 0 or prompt_start >= request.num_prompt_tokens:
            raise ValueError("routed_experts_prompt_start is outside the prompt")
        self._states[request.request_id] = _RequestState(
            self._resume_emit_cursors.pop(request.request_id, prompt_start),
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
        kv_block_ids: Mapping[str, Sequence[int]] | None = None,
    ) -> ArtifactConnectorMetadata:
        if kv_block_ids is None:
            kv_block_ids = {}
        for request_id in scheduler_output.preempted_req_ids or ():
            self.request_restarted(request_id)
        token_starts = {
            request.req_id: request.num_computed_tokens
            for request in scheduler_output.scheduled_new_reqs
        }
        cached = scheduler_output.scheduled_cached_reqs
        token_starts.update(
            zip(cached.req_ids, cached.num_computed_tokens, strict=True)
        )
        metadata = []
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            state = self._states.get(request_id)
            request = requests.get(request_id)
            if state is None or request is None:
                continue
            if self._reuse_kv_hashes:
                if state.num_hashes > len(request.block_hashes):
                    raise RuntimeError("KV block-hash history shrank")
                new_hashes: Sequence[bytes] = request.block_hashes[state.num_hashes :]
            else:
                num_hashes = (token_starts[request_id] + num_tokens) // self._block_size
                new_hashes = [
                    hashlib.sha256(f"{request_id}:{i}".encode()).digest()
                    for i in range(state.num_hashes, num_hashes)
                ]
            block_hash_start = state.num_hashes
            block_hashes = self._pack_block_hashes(state, new_hashes)
            metadata.append(
                ArtifactRequestMetadata(
                    request_id=request_id,
                    token_start=token_starts[request_id],
                    num_tokens=num_tokens,
                    emit_start=state.emit_cursor,
                    emit_output=(
                        token_starts[request_id] + num_tokens
                        >= request.num_prompt_tokens
                    ),
                    block_hashes=block_hashes,
                    block_hash_start=block_hash_start,
                    epoch=state.epoch,
                    kv_block_ids=kv_block_ids.get(request_id, ()),
                )
            )
        finished_requests = self._finished_requests
        self._finished_requests = {}
        return ArtifactConnectorMetadata(
            self._generation,
            self._block_size,
            metadata,
            finished_requests,
        )

    def take_output(
        self,
        request: Request,
        emit_output: bool,
        output: ArtifactConnectorOutput | None,
    ) -> np.ndarray | None:
        if not emit_output:
            return None
        token_end = min(
            request.num_tokens - 1,
            request.num_computed_tokens - request.num_in_flight_tokens,
        )
        if token_end <= 0:
            return None
        request_id = request.request_id
        state = self._states[request_id]
        if token_end <= state.emit_cursor:
            return None
        if output is None or request_id not in output.requests:
            raise RuntimeError(f"artifact worker output is missing {request_id}")
        request_output = output.requests[request_id]
        local_start = state.emit_cursor - request_output.token_start
        local_end = token_end - request_output.token_start
        if local_start < 0 or local_end > len(request_output.rows):
            raise RuntimeError("artifact worker output has an invalid token range")
        state.emit_cursor = token_end
        return request_output.rows[local_start:local_end]

    def request_finished(self, request: Request) -> None:
        request_id = request.request_id
        state = self._states[request_id]
        if self._reuse_kv_hashes:
            if state.num_hashes > len(request.block_hashes):
                raise RuntimeError("KV block-hash history shrank")
            self._pack_block_hashes(state, request.block_hashes[state.num_hashes :])
        self._finished_requests[(request_id, state.epoch)] = PackedBlockHashes(
            bytes(state.packed_hashes), state.hash_size or 1
        )
        self._states.pop(request_id, None)
        self._resume_emit_cursors.pop(request_id, None)

    @staticmethod
    def _pack_block_hashes(
        state: _RequestState, new_hashes: Sequence[bytes]
    ) -> PackedBlockHashes:
        packed_hashes = b"".join(new_hashes)
        if new_hashes:
            if state.hash_size == 0:
                state.hash_size = len(new_hashes[0])
            if any(len(block_hash) != state.hash_size for block_hash in new_hashes):
                raise RuntimeError("KV block hashes have inconsistent sizes")
            state.packed_hashes.extend(packed_hashes)
            state.num_hashes += len(new_hashes)
        return PackedBlockHashes(packed_hashes, state.hash_size or 1)

    def request_aborted(self, request_id: str) -> None:
        state = self._states.pop(request_id, None)
        if state is not None:
            self._finished_requests[(request_id, state.epoch)] = None
        self._resume_emit_cursors.pop(request_id, None)

    def request_restarted(self, request_id: str) -> None:
        """Start a fresh worker epoch while preserving delivered output."""
        state = self._states.get(request_id)
        if state is None:
            return
        self._finished_requests[(request_id, state.epoch)] = None
        state.epoch += 1
        state.packed_hashes.clear()
        state.num_hashes = 0
        state.hash_size = 0

    def reset(self) -> None:
        for request_id, state in self._states.items():
            self._resume_emit_cursors[request_id] = state.emit_cursor
        self._states.clear()
        self._finished_requests.clear()
        self._generation += 1
