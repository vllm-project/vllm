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
    AuxOutputConnectorMetadata,
    AuxRequestOutput,
)
from vllm.distributed.aux_output_connector.logprobs import (
    LogprobRows,
    concat_rows,
    decode_rows,
    encode_rows,
    rows_from_lists,
    rows_from_tensors,
    rows_to_lists,
    rows_to_tensors,
)
from vllm.distributed.aux_output_connector.routed_experts import (
    RoutedExpertsBuffer,
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_keys,
)
from vllm.distributed.aux_output_connector.store import (
    BackgroundBlockObjectStore,
    BlockObject,
    BlockObjectStore,
    VariableBlockObjectStore,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import LogprobsLists, LogprobsTensors
    from vllm.v1.worker.gpu.input_batch import InputBatch


def _format_position_ranges(positions: np.ndarray) -> str:
    """Format sorted positions as compact inclusive ranges."""
    if not len(positions):
        return "[]"
    starts = np.r_[True, np.diff(positions) > 1]
    ends = np.r_[np.diff(positions) > 1, True]
    ranges = [
        f"{start}"
        if start == end
        else f"{start}-{end}"
        for start, end in zip(positions[starts], positions[ends], strict=True)
    ]
    return ",".join(ranges)


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
    prompt_logprob_keys: list[str] = field(default_factory=list)
    pending_logprob_blocks: list[tuple[bytes, str]] = field(default_factory=list)
    logprob_rows: dict[int, LogprobRows] = field(default_factory=dict)
    logprob_cursor: int | None = None
    prompt_logprob_rows: dict[int, LogprobRows] = field(default_factory=dict)
    published_artifact_positions: dict[str, frozenset[int]] = field(
        default_factory=dict
    )
    prompt_len: int = 0
    prompt_replay_prefix: int | None = None
    prompt_logprobs_artifact: LogprobsTensors | None = None


@dataclass
class PendingAuxOutput:
    """Own one step's R3 snapshot until its asynchronous copy is consumed."""

    connector: AuxOutputWorkerConnector
    request_ids: list[str]
    batch_indices: np.ndarray
    token_starts: np.ndarray
    query_start_loc: np.ndarray
    # GPU snapshot taken on the main stream.
    routed_experts_gpu: torch.Tensor | None
    # Filled by enqueue_cpu_copy on the output copy stream.
    routed_experts: np.ndarray | None = None
    num_sampled: np.ndarray | None = None
    num_rejected: np.ndarray | None = None
    logprobs_tensors: LogprobsTensors | None = None
    logprobs: dict[str, LogprobsLists] = field(default_factory=dict)
    prompt_logprobs: dict[str, LogprobsTensors] = field(default_factory=dict)
    replay_logprobs: frozenset[str] = frozenset()
    replay_prompt_logprobs: frozenset[str] = frozenset()

    def enqueue_cpu_copy(
        self,
        num_sampled: np.ndarray,
        num_rejected: np.ndarray,
        logprobs: LogprobsTensors | None = None,
        prompt_logprobs: dict[str, LogprobsTensors | None] | None = None,
    ) -> None:
        """Enqueue asynchronous D2H copies; call on the output copy stream."""
        self.num_sampled = num_sampled[self.batch_indices]
        self.num_rejected = num_rejected[self.batch_indices]
        if self.routed_experts_gpu is not None:
            self.routed_experts = self.routed_experts_gpu.to(
                "cpu", non_blocking=True
            ).numpy()
        self.logprobs_tensors = logprobs if self.replay_logprobs else None
        if prompt_logprobs:
            self.prompt_logprobs = {
                request_id: value
                for request_id, value in prompt_logprobs.items()
                if request_id in self.replay_prompt_logprobs and value is not None
            }

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
    ) -> None:
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        capturer = None
        if vllm_config.aux_output_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer(
                max_num_batched_tokens=max_num_batched_tokens, vllm_config=vllm_config
            )
            bind_routed_experts_capturer(model, capturer)
        self._capturer = capturer
        self._store: BackgroundBlockObjectStore | None = None
        self._logprob_store: BackgroundBlockObjectStore | None = None
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
        self._logprob_block_size = 1
        self._enable_logprobs = vllm_config.aux_output_config.enable_logprobs_replay
        self._enable_prompt_logprobs = (
            vllm_config.aux_output_config.enable_prompt_logprobs_replay
        )
        # Every TP rank participates in capture collectives, but only the
        # executor output rank owns the auxiliary output data plane.
        if not get_tp_group().is_first_rank:
            return

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        hashes_per_kv_block = scheduler_block_size // hash_block_size
        self._logprob_block_size = hash_block_size
        max_bytes = vllm_config.aux_output_config.max_bytes
        if capturer is not None:
            shape_per_token = capturer.shape_per_token
            dtype: np.dtype[Any] = np.dtype(capturer.output_dtype_name)
            block_nbytes = (
                hash_block_size * int(np.prod(shape_per_token)) * dtype.itemsize
            )
            if max_bytes is None:
                max_bytes = (
                    kv_cache_config.num_blocks * hashes_per_kv_block * block_nbytes
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
        if max_bytes is None:
            max_logprobs = max(vllm_config.model_config.max_logprobs, 0)
            bytes_per_row = max(16, (max_logprobs + 1) * 8 + 4)
            # NPZ stores array headers and metadata in addition to row data.
            # Reserve this per artifact so the derived capacity does not fail
            # closed solely because a block has a larger serialization header.
            artifact_overhead = 4096
            bytes_per_block = hash_block_size * bytes_per_row + artifact_overhead
            max_bytes = (
                kv_cache_config.num_blocks * hashes_per_kv_block * bytes_per_block
            )
        if self._enable_logprobs or self._enable_prompt_logprobs:
            self._logprob_store = BackgroundBlockObjectStore(
                VariableBlockObjectStore(max_bytes=max_bytes),
                max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
            )

    def prepare_output(self, input_batch: InputBatch) -> PendingAuxOutput | None:
        """Snapshot one step's R3 tensor for asynchronous CPU transfer."""
        if (
            self._buffer is None and getattr(self, "_logprob_store", None) is None
        ) or self._step_metadata is None:
            return None
        if (
            self._buffer is None
            and not self._step_metadata.logprobs
            and not self._step_metadata.prompt_logprobs
        ):
            return None

        if self._buffer is None:
            request_ids = [
                request_id
                for request_id in input_batch.req_ids
                if request_id in self._step_metadata.logprobs
                or request_id in self._step_metadata.prompt_logprobs
            ]
            if not request_ids:
                return None
            batch_indices = np.fromiter(
                (input_batch.req_ids.index(request_id) for request_id in request_ids),
                dtype=np.int32,
            )
            query_start_loc = np.empty(len(request_ids) + 1, dtype=np.int32)
            query_start_loc[0] = 0
            np.cumsum(
                input_batch.num_scheduled_tokens[batch_indices],
                out=query_start_loc[1:],
            )
        else:
            request_ids = list(input_batch.req_ids)
            batch_indices = np.arange(len(request_ids), dtype=np.int32)
            query_start_loc = input_batch.query_start_loc_np[: len(request_ids) + 1]
        num_rows = int(query_start_loc[-1])
        pending_output = PendingAuxOutput(
            connector=self,
            request_ids=request_ids,
            batch_indices=batch_indices,
            token_starts=input_batch.num_computed_tokens_np[batch_indices],
            query_start_loc=query_start_loc,
            routed_experts_gpu=(
                self._capturer.snapshot_routing_data(num_rows)
                if self._capturer is not None
                else None
            ),
            replay_logprobs=frozenset(self._step_metadata.logprobs),
            replay_prompt_logprobs=frozenset(self._step_metadata.prompt_logprobs),
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
                        release_logprob_keys.extend(
                            reversed(
                                list(
                                    dict.fromkeys(
                                        state.logprob_keys + state.prompt_logprob_keys
                                    )
                                )
                            )
                        )
            if teardown:
                if self._buffer is not None:
                    self._publish_blocks([], release_keys=release_keys)
                logprob_store = getattr(self, "_logprob_store", None)
                if logprob_store is not None and release_logprob_keys:
                    logprob_store.put([], release_keys=release_logprob_keys)
                self._teardown(teardown)
            return outputs

    def _commit_output(self, pending: PendingAuxOutput) -> dict[str, AuxRequestOutput]:
        buffer = self._buffer
        store = self._store
        assert store is not None or getattr(self, "_logprob_store", None) is not None
        block_size = (
            buffer.block_size if buffer is not None else self._logprob_block_size
        )

        routed_experts = pending.routed_experts
        num_sampled = pending.num_sampled
        num_rejected = pending.num_rejected
        assert num_sampled is not None and num_rejected is not None, (
            "auxiliary output CPU copy was not enqueued"
        )
        if pending.logprobs_tensors is not None:
            values = pending.logprobs_tensors.tolists()
            for index, request_id in enumerate(pending.request_ids):
                count = int(num_sampled[index])
                if not count or request_id not in pending.replay_logprobs:
                    continue
                cu = values.cu_num_generated_tokens
                batch_index = int(pending.batch_indices[index])
                row_index = batch_index if cu is None else cu[batch_index]
                end = row_index + count
                pending.logprobs[request_id] = type(values)(
                    values.logprob_token_ids[row_index:end],
                    values.logprobs[row_index:end],
                    values.sampled_token_ranks[row_index:end],
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
            if request_id in pending.prompt_logprobs:
                # A streaming session can reuse the worker request state. The
                # next completed prompt must replace the prior turn's
                # one-shot artifact and establish a fresh cache prefix.
                if state.prompt_logprobs_artifact is not None:
                    state.prompt_logprobs_artifact = None
                    state.prompt_replay_prefix = None
            if (
                request_id in pending.replay_prompt_logprobs
                and state.prompt_replay_prefix is None
            ):
                state.prompt_replay_prefix = int(token_start)

            if request_id in pending.logprobs:
                proposed_start = max(state.prompt_len, int(token_start) + 1)
                sample_start = state.logprob_cursor
                if sample_start is None:
                    sample_start = proposed_start
                assert proposed_start >= sample_start, (
                    "auxiliary logprobs capture moved backwards"
                )
                if proposed_start > sample_start:
                    assert sample_start < state.scheduled_cursor, (
                        "auxiliary logprobs capture has an unbacked token gap"
                    )
                rows = rows_from_lists(pending.logprobs[request_id], sample_start)
                self._bind_pending_logprob_blocks(state, rows)
                state.logprob_rows[sample_start] = rows
                state.logprob_cursor = sample_start + len(rows.positions)
            if request_id in pending.prompt_logprobs:
                value = pending.prompt_logprobs[request_id]
                num_rows = value.logprobs.shape[0]
                full_prompt_rows = max(state.prompt_len - 1, 0)
                if num_rows == full_prompt_rows:
                    # The V1 GPU runner returns its accumulated full-prompt
                    # tensor on the final prefill step. With a cached prefix,
                    # the prefix portion is allocated but not populated.
                    row_start = state.prompt_replay_prefix or 0
                    prompt_start = row_start + 1
                    value = type(value)(
                        value.logprob_token_ids[row_start:],
                        value.logprobs[row_start:],
                        value.selected_token_ranks[row_start:],
                    )
                else:
                    # The sampler's chunk output is the suffix ending at the
                    # prompt boundary, so align it from the prompt length.
                    prompt_start = state.prompt_len - num_rows
                rows = rows_from_tensors(value, prompt_start)
                state.prompt_logprob_rows[prompt_start] = rows

            # Capture precedes speculative acceptance, so discard the rejected
            # suffix. Batch boundaries still span the full executed range.
            rejected = int(rejected)
            assert 0 <= rejected <= request_num_tokens, (
                "auxiliary output rejected-token count is invalid"
            )
            if buffer is None:
                state.scheduled_cursor = token_start + request_num_tokens
                continue
            routed_rows = (
                routed_experts[start : end - rejected]
                if routed_experts is not None
                else None
            )

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
            completed = (
                buffer.capture(request_id, capture_start, routed_rows)
                if routed_rows is not None
                else []
            )
            token_end = capture_start + (
                len(routed_rows) if routed_rows is not None else 0
            )
            state.capture_cursor = token_end
            state.scheduled_cursor = token_start + request_num_tokens
            block_batches.append((state, completed))

            if routed_experts is not None and sampled > 0 and emit_start <= token_end:
                assert routed_rows is not None
                if emit_start >= capture_start:
                    outputs[request_id] = AuxRequestOutput(
                        emit_start, rows=routed_rows[emit_start - capture_start :]
                    )
                    state.emit_cursor = token_end
                else:
                    materialize_outputs.append((request_id, emit_start, token_end))

        # A consumer may reuse a block produced earlier in the same batch.
        if buffer is not None:
            self._publish_blocks(block_batches)

        # Publish logprob blocks before materializing outputs. A later request
        # in this batch may share a prefix block produced by an earlier one.
        # Keep the in-memory rows until materialization has consumed them.
        self._publish_logprob_blocks(pending.request_ids, discard_rows=False)

        for request_id, emit_start, token_end in materialize_outputs:
            state = self._requests[request_id]
            assert store is not None and buffer is not None
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
            outputs[request_id] = AuxRequestOutput(emit_start, rows=rows)
            state.emit_cursor = token_end
        for index, request_id in enumerate(pending.request_ids):
            state = self._requests[request_id]
            if request_id not in outputs:
                if (
                    request_id not in pending.replay_logprobs
                    and request_id not in pending.replay_prompt_logprobs
                ):
                    continue
                outputs[request_id] = AuxRequestOutput(0)
            output = outputs[request_id]
            if request_id in pending.logprobs:
                output.logprobs = pending.logprobs[request_id]
            elif (
                request_id in pending.replay_logprobs
                and state.logprob_keys
                and int(num_sampled[index])
            ):
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
            if request_id not in pending.replay_prompt_logprobs:
                continue
            visible_output = (
                int(num_sampled[index]) > 0
                or int(pending.token_starts[index]) >= state.prompt_len
            )
            if state.prompt_logprobs_artifact is not None:
                # A final prefill chunk can complete the artifact without
                # producing an EngineCoreOutput. Keep it pending until the
                # first visible output step consumes it.
                if visible_output:
                    output.prompt_logprobs = state.prompt_logprobs_artifact
                continue
            if state.prompt_replay_prefix is None:
                raise RuntimeError(
                    "auxiliary prompt logprobs replay prefix is missing: "
                    f"request={request_id}"
                )
            num_hit_blocks = state.prompt_replay_prefix // self._logprob_block_size
            prompt_chunks = [*state.prompt_logprob_rows.values()]
            if state.prompt_logprob_keys and num_hit_blocks:
                prompt_chunks.append(
                    self._read_logprob_rows(state.prompt_logprob_keys[:num_hit_blocks])
                )
            merged_prompt = concat_rows(prompt_chunks)
            step_tokens = int(pending.query_start_loc[index + 1]) - int(
                pending.query_start_loc[index]
            )
            if (
                state.prompt_len > 0
                and int(pending.token_starts[index]) + step_tokens >= state.prompt_len
            ):
                expected = np.arange(1, state.prompt_len, dtype=np.int64)
                if merged_prompt is not None:
                    mask = merged_prompt.positions < state.prompt_len
                    merged_prompt = LogprobRows(
                        merged_prompt.positions[mask],
                        merged_prompt.token_ids[mask],
                        merged_prompt.values[mask],
                        merged_prompt.ranks[mask],
                    )
                if merged_prompt is None and not len(expected):
                    generated = pending.logprobs.get(request_id)
                    if generated is None:
                        raise RuntimeError(
                            "auxiliary prompt logprobs width is unavailable for "
                            f"empty prompt artifact: request={request_id}"
                        )
                    width = generated.logprobs.shape[1]
                    merged_prompt = LogprobRows(
                        expected,
                        np.empty((0, width), dtype=np.int32),
                        np.empty((0, width), dtype=np.float32),
                        np.empty(0, dtype=np.int32),
                    )
                actual = (
                    np.empty(0, dtype=np.int64)
                    if merged_prompt is None
                    else merged_prompt.positions
                )
                if not np.array_equal(actual, expected):
                    missing = np.setdiff1d(expected, actual)
                    extra = np.setdiff1d(actual, expected)
                    raise RuntimeError(
                        "auxiliary prompt logprobs are incomplete: "
                        f"request={request_id}, "
                        f"missing_positions={_format_position_ranges(missing)}, "
                        f"unexpected_positions={_format_position_ranges(extra)}"
                    )
                tensors = rows_to_tensors(merged_prompt)
                state.prompt_logprobs_artifact = tensors
                if visible_output:
                    output.prompt_logprobs = tensors

        # Reclaim complete in-memory rows after all outputs in this batch have
        # consumed them. The first publication above makes shared blocks
        # visible to later requests in the same batch.
        self._publish_logprob_blocks(pending.request_ids, discard_rows=True)
        return outputs

    def _read_logprob_rows(self, keys: Sequence[str]) -> LogprobRows:
        store = getattr(self, "_logprob_store", None)
        if store is None:
            raise RuntimeError("logprob replay store is disabled")
        chunks = [decode_rows(store.get_concatenated([key])) for key in keys]
        result = concat_rows(chunks)
        if result is None:
            raise RuntimeError(
                "auxiliary logprobs artifact contains no rows for requested keys"
            )
        return result

    def _publish_logprob_blocks(
        self, request_ids: Iterable[str], *, discard_rows: bool = True
    ) -> None:
        store = getattr(self, "_logprob_store", None)
        if store is None:
            return
        block_size = self._logprob_block_size
        for request_id in request_ids:
            state = self._requests[request_id]
            generated_chunks = [*state.logprob_rows.values()]
            prompt_chunks = [*state.prompt_logprob_rows.values()]
            key_lists = [(state.logprob_keys, generated_chunks, [state.logprob_rows])]
            if state.prompt_logprob_keys is not state.logprob_keys:
                key_lists.append(
                    (
                        state.prompt_logprob_keys,
                        prompt_chunks,
                        [state.prompt_logprob_rows],
                    )
                )
            else:
                key_lists[0][1].extend(prompt_chunks)
                key_lists[0][2].append(state.prompt_logprob_rows)
            for keys, row_chunks, row_maps in key_lists:
                if not keys:
                    continue
                max_position = max(
                    (int(chunk.positions[-1]) for chunk in row_chunks), default=0
                )
                for block_index, key in enumerate(keys):
                    start = block_index * block_size
                    end = start + block_size
                    rows = concat_rows(
                        [
                            value
                            for value in row_chunks
                            if np.any(
                                (value.positions > start) & (value.positions <= end)
                            )
                        ]
                    )
                    if rows is None:
                        continue
                    mask = (rows.positions > start) & (rows.positions <= end)
                    positions = frozenset(int(pos) for pos in rows.positions[mask])
                    if not positions:
                        continue
                    already_published = (
                        positions == state.published_artifact_positions.get(key)
                    )
                    if already_published:
                        if discard_rows and max_position >= end:
                            for rows_by_start in row_maps:
                                self._discard_logprob_rows(rows_by_start, start, end)
                        continue
                    artifact = LogprobRows(
                        rows.positions[mask],
                        rows.token_ids[mask],
                        rows.values[mask],
                        rows.ranks[mask],
                    )
                    existing = store.get_optional(key)
                    if existing is not None:
                        # Keep the newly computed row when a preemption or
                        # resumed execution recomputes the same position.
                        merged = concat_rows([decode_rows(existing), artifact])
                        if merged is None:
                            raise RuntimeError(
                                "auxiliary logprobs block merge produced no rows: "
                                f"key={key}"
                            )
                        artifact = merged
                    store.put(
                        [
                            BlockObject(
                                key,
                                encode_rows(artifact),
                            )
                        ],
                    )
                    state.published_artifact_positions[key] = positions
                    if discard_rows and max_position >= end:
                        for rows_by_start in row_maps:
                            self._discard_logprob_rows(rows_by_start, start, end)

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
        for request_id in request_ids:
            state = self._requests.pop(request_id)
            if buffer is not None:
                for _, rows in state.pending_blocks:
                    buffer.release_block(rows)
                buffer.discard(request_id)

    def begin_step(self, metadata: AuxOutputConnectorMetadata | None) -> None:
        """Apply one scheduler step's request and block-hash updates.

        While an unconsumed step covers a finished request, its teardown is
        deferred to that step's process_output.
        """
        self._step_metadata = metadata
        if (
            self._buffer is None and getattr(self, "_logprob_store", None) is None
        ) or metadata is None:
            return
        assert not metadata.requests.keys() & metadata.finished_requests, (
            "auxiliary output request cannot run and finish in one step"
        )
        assert metadata.generation >= self._generation, (
            "auxiliary output metadata generation moved backwards"
        )
        with self._lock:
            release_keys: list[str] = []
            release_logprob_keys: list[str] = []
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
                release_logprob_keys.extend(
                    key
                    for state in self._requests.values()
                    for key in reversed(
                        list(
                            dict.fromkeys(
                                state.logprob_keys + state.prompt_logprob_keys
                            )
                        )
                    )
                )
                if self._buffer is not None:
                    self._buffer.reset()
                self._requests.clear()
                self._generation = metadata.generation
            for request_id, emit_start in metadata.requests.items():
                state = self._requests.setdefault(
                    request_id, _WorkerRequestState(emit_cursor=emit_start)
                )
                prompt_len = metadata.prompt_lens.get(request_id, 0)
                if state.prompt_len != prompt_len:
                    # Streaming input reuses the request ID for a new prompt
                    # turn. The completed artifact is one-shot per turn, and
                    # the prefix must be recalculated from the new schedule.
                    state.prompt_logprobs_artifact = None
                    state.prompt_replay_prefix = None
                state.prompt_len = prompt_len
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
            retained_logprob_keys: list[str] = []
            for request_id, block_hashes in metadata.block_hashes.items():
                state = self._requests[request_id]
                if self._buffer is not None:
                    keys = routed_experts_keys(block_hashes, str(self._generation))
                    state.aux_output_keys.extend(keys)
                    retained_keys.extend(keys)
                    block_batches.append((state, []))
            for request_id, fingerprint in metadata.logprobs.items():
                state = self._requests[request_id]
                keys, pending_blocks = self._resolve_logprob_keys(
                    metadata.logprob_block_hashes.get(request_id, ()),
                    metadata.logprob_boundary_token_ids.get(request_id, ()),
                    fingerprint,
                )
                state.logprob_keys.extend(keys)
                state.pending_logprob_blocks.extend(pending_blocks)
                retained_logprob_keys.extend(keys)
            for request_id, fingerprint in metadata.prompt_logprobs.items():
                state = self._requests[request_id]
                if request_id in metadata.logprobs:
                    state.prompt_logprob_keys = state.logprob_keys
                else:
                    keys = self._logprob_keys(
                        metadata.logprob_block_hashes.get(request_id, ()),
                        metadata.logprob_boundary_token_ids.get(request_id, ()),
                        fingerprint,
                    )
                    state.prompt_logprob_keys.extend(keys)
                    retained_logprob_keys.extend(keys)
            self._publish_logprob_blocks(metadata.logprob_block_hashes)
            finished_now: list[str] = []
            for request_id in metadata.finished_requests:
                state = self._requests[request_id]
                state.finished = True
                if state.pending_outputs:
                    continue
                finished_now.append(request_id)
                release_keys.extend(reversed(state.aux_output_keys))
                release_logprob_keys.extend(reversed(state.logprob_keys))
                if state.prompt_logprob_keys is not state.logprob_keys:
                    release_logprob_keys.extend(reversed(state.prompt_logprob_keys))
            if self._buffer is not None:
                self._publish_blocks(block_batches, retained_keys, release_keys)
            logprob_store = getattr(self, "_logprob_store", None)
            if logprob_store is not None:
                logprob_store.put(
                    [],
                    retain_keys=retained_logprob_keys,
                    release_keys=release_logprob_keys,
                )
            self._teardown(finished_now)

    def _logprob_keys(
        self,
        block_hashes: Iterable[bytes],
        boundary_token_ids: Sequence[int | None],
        fingerprint: str,
    ) -> list[str]:
        keys, pending = self._resolve_logprob_keys(
            block_hashes, boundary_token_ids, fingerprint
        )
        if pending:
            raise ValueError(
                "auxiliary logprobs boundary token is missing for a completed "
                "request"
            )
        return keys

    def _resolve_logprob_keys(
        self,
        block_hashes: Iterable[bytes],
        boundary_token_ids: Sequence[int | None],
        fingerprint: str,
    ) -> tuple[list[str], list[tuple[bytes, str]]]:
        prefix = f"vllm-logprobs/v1/{self._generation}/{fingerprint}/"
        hashes = list(block_hashes)
        if len(hashes) != len(boundary_token_ids):
            raise ValueError(
                "auxiliary logprobs block metadata is misaligned: "
                f"hashes={len(hashes)}, boundaries={len(boundary_token_ids)}"
            )
        keys = []
        pending = []
        for block_hash, token_id in zip(hashes, boundary_token_ids, strict=True):
            if token_id is None:
                pending.append((block_hash, fingerprint))
            else:
                if pending:
                    raise ValueError(
                        "only trailing auxiliary logprobs blocks may lack "
                        "boundary tokens"
                    )
                keys.append(f"{prefix}{block_hash.hex()}/{token_id}")
        if len(pending) > 1:
            raise ValueError(
                "only one trailing auxiliary logprobs block may lack a boundary "
                "token"
            )
        return keys, pending

    def _bind_pending_logprob_blocks(
        self, state: _WorkerRequestState, rows: LogprobRows
    ) -> None:
        store = self._logprob_store
        if store is None or not state.pending_logprob_blocks:
            return
        retained = []
        for position, token_ids in zip(rows.positions, rows.token_ids, strict=True):
            if position % self._logprob_block_size or not state.pending_logprob_blocks:
                continue
            block_hash, fingerprint = state.pending_logprob_blocks.pop(0)
            key = self._logprob_keys([block_hash], [int(token_ids[0])], fingerprint)[0]
            state.logprob_keys.append(key)
            if state.prompt_logprob_keys is not state.logprob_keys:
                state.prompt_logprob_keys.append(key)
            retained.append(key)
        if retained:
            store.put([], retain_keys=retained)

    def close(self) -> None:
        with self._lock:
            self._pending_outputs.clear()
            if self._store is not None:
                try:
                    self._store.close()
                finally:
                    self._store = None
            logprob_store = getattr(self, "_logprob_store", None)
            if logprob_store is not None:
                try:
                    logprob_store.close()
                finally:
                    self._logprob_store = None


def get_aux_output_connector(
    model: torch.nn.Module, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> AuxOutputWorkerConnector:
    return AuxOutputWorkerConnector(
        model=model,
        kv_cache_config=kv_cache_config,
        vllm_config=vllm_config,
    )
