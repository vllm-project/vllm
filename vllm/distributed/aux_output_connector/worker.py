# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side auxiliary-output data plane."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil
import torch

from vllm.config import VllmConfig
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxRequestOutput,
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
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.indexer_capture import IndexerCapturer
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.input_batch import InputBatch


logger = init_logger(__name__)


def validate_shm_budget(
    *,
    num_blocks: int,
    scheduler_block_size: int,
    bytes_per_token: int,
    max_bytes: int | None,
) -> int:
    required = num_blocks * scheduler_block_size * bytes_per_token
    if max_bytes is not None and max_bytes < required:
        supported = max_bytes // (scheduler_block_size * bytes_per_token)
        gib = 1024**3
        raise ValueError(
            f"Required AuxOutput SHM: {required / gib:.3f} GiB; "
            f"Allowed AuxOutput SHM: {max_bytes / gib:.3f} GiB. "
            f"Current KV Blocks: {num_blocks}; Maximum Supported Blocks: {supported}. "
            "Increase aux_output_config.max_bytes "
            "or reduce KV blocks using gpu_memory_utilization or "
            "kv_cache_memory_bytes. Host staging and runtime memory are additional."
        )
    return required if max_bytes is None else max_bytes


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


@dataclass
class PendingAuxOutput:
    """Own one step's R3 snapshot until its asynchronous copy is consumed."""

    connector: AuxOutputWorkerConnector
    request_ids: list[str]
    token_starts: np.ndarray
    query_start_loc: np.ndarray
    # GPU snapshot taken on the main stream.
    routed_experts_gpu: torch.Tensor | None
    # Filled by enqueue_cpu_copy on the output copy stream.
    routed_experts: np.ndarray | None = None
    num_sampled: np.ndarray | None = None
    num_rejected: np.ndarray | None = None
    indexer_gpu: torch.Tensor | None = None
    indexer_valid_gpu: torch.Tensor | None = None
    indexer: np.ndarray | None = None
    indexer_valid: np.ndarray | None = None
    indexer_snapshot_slot: int | None = None

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

        if self.routed_experts_gpu is not None:
            if self.indexer_gpu is None:
                self.routed_experts = self.routed_experts_gpu.to(
                    "cpu", non_blocking=True
                ).numpy()
            else:
                self.routed_experts = copy_rows(self.routed_experts_gpu)
        if self.indexer_gpu is not None:
            self.indexer = copy_rows(self.indexer_gpu)
            assert self.indexer_valid_gpu is not None
            self.indexer_valid = copy_rows(self.indexer_valid_gpu)

    def process_output(self) -> dict[str, AuxRequestOutput]:
        return self.connector.process_output(self)


class AuxOutputWorkerConnector:
    """Own capture, request tails, and backend resources on the output worker."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        kv_cache_config: KVCacheConfig,
        indexer_capturer: IndexerCapturer | None = None,
    ) -> None:
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._capturer: RoutedExpertsCapturer | None = None
        if vllm_config.aux_output_config.enable_return_routed_experts:
            self._capturer = RoutedExpertsCapturer(
                max_num_batched_tokens=max_num_batched_tokens, vllm_config=vllm_config
            )
            bind_routed_experts_capturer(model, self._capturer)
        self._indexer = indexer_capturer
        if (
            self._indexer is None
            and vllm_config.aux_output_config.enable_return_indexer_topk
        ):
            self._indexer = IndexerCapturer(
                model,
                vllm_config,
                output_rank=get_tp_group().is_first_rank,
            )
        self._r3_shape = self._capturer.shape_per_token if self._capturer else ()
        self._r3_dtype = (
            np.dtype(self._capturer.output_dtype_name)
            if self._capturer
            else np.dtype("uint8")
        )
        self._r3_nbytes = (
            int(np.prod(self._r3_shape)) * self._r3_dtype.itemsize
            if self._capturer
            else 0
        )
        self._store: BackgroundBlockObjectStore | None = None
        self._buffer: RoutedExpertsBuffer | None = None
        self._requests: dict[str, _WorkerRequestState] = {}
        self._generation = 0
        self._step_metadata: AuxOutputConnectorMetadata | None = None
        # Steps whose asynchronous copy has not been consumed yet. The engine
        # consumes step outputs in order, so at most max_concurrent_batches
        # are outstanding; guarded by _lock against the output thread.
        self._pending_outputs: list[PendingAuxOutput] = []
        self._lock = Lock()
        self._max_concurrent_batches = vllm_config.max_concurrent_batches
        # Every TP rank participates in capture collectives, but only the
        # executor output rank owns the auxiliary output data plane.
        if not get_tp_group().is_first_rank:
            return

        shape_per_token: tuple[int, ...]
        if self._indexer is None:
            assert self._capturer is not None
            shape_per_token = self._capturer.shape_per_token
            dtype: np.dtype[Any] = self._r3_dtype
        else:
            shape_per_token = (
                self._r3_nbytes + self._indexer.profile.fixed_bytes_per_token,
            )
            dtype = np.dtype("uint8")
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        hashes_per_kv_block = scheduler_block_size // hash_block_size
        block_nbytes = hash_block_size * int(np.prod(shape_per_token)) * dtype.itemsize
        max_bytes = vllm_config.aux_output_config.max_bytes
        if self._indexer is not None:
            max_bytes = validate_shm_budget(
                num_blocks=kv_cache_config.num_blocks,
                scheduler_block_size=scheduler_block_size,
                bytes_per_token=shape_per_token[0],
                max_bytes=max_bytes,
            )
        elif max_bytes is None:
            max_bytes = kv_cache_config.num_blocks * hashes_per_kv_block * block_nbytes
        if self._indexer is not None:
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            max_step_blocks = (
                max_num_batched_tokens + hash_block_size - 1
            ) // hash_block_size + max_seqs
            staging_blocks = (
                vllm_config.max_concurrent_batches * max_step_blocks + max_seqs
            )
            staging_bytes = (
                staging_blocks * block_nbytes
                + vllm_config.max_concurrent_batches
                * max_num_batched_tokens
                * shape_per_token[0]
            )
            metadata_bytes = (max_bytes // block_nbytes) * 512
            available = psutil.virtual_memory().available
            reserve = max(512 << 20, available // 20)
            local_output_workers = max(
                1, vllm_config.parallel_config.data_parallel_size_local
            )
            host_required = (
                local_output_workers * (max_bytes + staging_bytes + metadata_bytes)
                + reserve
            )
            logger.info(
                "AuxOutput per output worker: R3=%d B/token, indexer=%d B/token, "
                "block=%d B, SHM=%d B, host staging=%d B, metadata estimate=%d B, "
                "runtime reserve=%d B; local output workers=%d",
                self._r3_nbytes,
                self._indexer.profile.fixed_bytes_per_token,
                block_nbytes,
                max_bytes,
                staging_bytes,
                metadata_bytes,
                reserve,
                local_output_workers,
            )
            if host_required > available:
                raise ValueError(
                    f"AuxOutput needs approximately {host_required / 1024**3:.3f} GiB "
                    f"of host memory including staging and reserve, but only "
                    f"{available / 1024**3:.3f} GiB is available. Increase host memory "
                    "or reduce KV cache capacity/aux_output_config.max_bytes "
                    "while retaining enough SHM for every KV block."
                )
        self._store = BackgroundBlockObjectStore(
            BlockObjectStore(max_bytes=max_bytes, object_nbytes=block_nbytes),
            max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
        )
        self._buffer = RoutedExpertsBuffer(
            dtype,
            shape_per_token,
            hash_block_size,
            vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens,
            vllm_config.max_concurrent_batches,
        )

    def prepare_output(self, input_batch: InputBatch) -> PendingAuxOutput | None:
        """Snapshot one step's R3 tensor for asynchronous CPU transfer."""
        if self._buffer is None or self._step_metadata is None:
            return None

        request_ids = list(input_batch.req_ids)
        query_start_loc = input_batch.query_start_loc_np[: len(request_ids) + 1]
        num_rows = int(query_start_loc[-1])
        snapshot = self._indexer.snapshot(num_rows) if self._indexer else None
        pending_output = PendingAuxOutput(
            connector=self,
            request_ids=request_ids,
            token_starts=input_batch.num_computed_tokens_np[: len(request_ids)].copy(),
            query_start_loc=query_start_loc.copy(),
            routed_experts_gpu=(
                self._capturer.snapshot_routing_data(num_rows)
                if self._capturer
                else None
            ),
            indexer_gpu=snapshot.rows if snapshot else None,
            indexer_valid_gpu=snapshot.valid if snapshot else None,
            indexer_snapshot_slot=snapshot.slot if snapshot else None,
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
            try:
                outputs = self._commit_output(pending)
            finally:
                if (
                    self._indexer is not None
                    and pending.indexer_snapshot_slot is not None
                ):
                    self._indexer.release_snapshot(pending.indexer_snapshot_slot)
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
        if self._indexer is not None:
            assert pending.indexer is not None and pending.indexer_valid is not None
            if not pending.indexer_valid.all():
                raise RuntimeError("Indexer capture missed source layers or token rows")
            parts = []
            if routed_experts is not None:
                parts.append(
                    np.ascontiguousarray(routed_experts)
                    .view(np.uint8)
                    .reshape(len(routed_experts), -1)
                )
            parts.append(np.ascontiguousarray(pending.indexer).view(np.uint8))
            routed_experts = np.concatenate(parts, axis=1)
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
            rows = routed_experts[start : end - rejected]

            capture_start = token_start
            capture_cursor = state.capture_cursor
            if capture_cursor is None:
                capture_cursor = capture_start

            replay_prefix = None
            if capture_start < capture_cursor:
                # Preemption/bounded replay recomputes an earlier logical range.
                buffer.rewind(request_id, capture_start)
                boundary = capture_start // block_size * block_size
                retained = []
                for block_start, block_rows in state.pending_blocks:
                    if block_start < boundary:
                        retained.append((block_start, block_rows))
                    else:
                        if block_start == boundary and capture_start % block_size:
                            replay_prefix = block_rows[
                                : capture_start % block_size
                            ].copy()
                        buffer.release_block(block_rows)
                state.pending_blocks = retained
                capture_cursor = capture_start
            if capture_start > capture_cursor:
                # Reattach after an optimistically scheduled suffix was rejected.
                assert capture_cursor < state.scheduled_cursor, (
                    "auxiliary output capture has an unbacked token gap"
                )
                capture_start = capture_cursor

            emit_start = state.emit_cursor
            # Complete blocks without keys remain pending until a hash update.
            if capture_start % block_size and not buffer.has_tail(request_id):
                block_index = capture_start // block_size
                prefix = replay_prefix
                if prefix is None:
                    if block_index >= len(state.aux_output_keys):
                        raise RuntimeError(
                            "AuxOutput replay boundary has no KV block hash"
                        )
                    prefix = materialize_routed_experts(
                        store,
                        [state.aux_output_keys[block_index]],
                        shape_per_token=buffer.shape_per_token,
                        dtype=buffer.dtype,
                    )[: capture_start % block_size]
                buffer.capture(request_id, block_index * block_size, prefix)
            completed = buffer.capture(request_id, capture_start, rows)
            token_end = capture_start + len(rows)
            state.capture_cursor = token_end
            state.scheduled_cursor = token_start + request_num_tokens
            block_batches.append((state, completed))

            if sampled > 0 and emit_start <= token_end:
                if emit_start >= capture_start:
                    outputs[request_id] = self._make_output(
                        emit_start, rows[emit_start - capture_start :]
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
                stored = materialize_routed_experts(
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
            state.emit_cursor = token_end
        return outputs

    def _make_output(self, token_start: int, rows: np.ndarray) -> AuxRequestOutput:
        if self._indexer is None:
            return AuxRequestOutput(token_start=token_start, routed_rows=rows)
        n = len(rows)
        routed_rows = rows[:, : self._r3_nbytes].copy()
        if self._capturer is not None:
            routed_rows = routed_rows.view(self._r3_dtype).reshape(n, *self._r3_shape)
        indexer = rows[:, self._r3_nbytes :].copy().view(np.int32)
        return AuxRequestOutput(
            token_start=token_start,
            routed_rows=routed_rows,
            indexer_rows=indexer,
            indexer_profile=self._indexer.profile_metadata,
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
        if self._indexer is not None:
            self._indexer.begin_step()
        if self._buffer is None or metadata is None:
            return
        assert not metadata.requests.keys() & metadata.finished_requests, (
            "auxiliary output request cannot run and finish in one step"
        )
        assert metadata.generation >= self._generation, (
            "auxiliary output metadata generation moved backwards"
        )
        with self._lock:
            release_keys: list[str] = []
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
            block_batches: list[
                tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]
            ] = []
            retained_keys: list[str] = []
            for request_id, block_hashes in metadata.block_hashes.items():
                state = self._requests[request_id]
                namespace = str(self._generation)
                if self._indexer is not None:
                    namespace += "/" + self._indexer.profile.profile_id
                    if self._capturer is not None:
                        namespace += f"/r3-{self._r3_shape}-{self._r3_dtype.name}"
                keys = routed_experts_keys(block_hashes, namespace)
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
            try:
                if self._store is not None:
                    self._store.close()
            finally:
                self._store = None
                if self._indexer is not None:
                    self._indexer.close()
                    self._indexer = None


def get_aux_output_connector(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    *,
    indexer_capturer: IndexerCapturer | None = None,
) -> AuxOutputWorkerConnector:
    return AuxOutputWorkerConnector(
        model=model,
        kv_cache_config=kv_cache_config,
        vllm_config=vllm_config,
        indexer_capturer=indexer_capturer,
    )
