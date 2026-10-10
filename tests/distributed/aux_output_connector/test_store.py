# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import io
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxOutputSchedulerConnector,
    AuxRequestOutput,
    PackedBlockHashes,
)
from vllm.distributed.aux_output_connector.logprobs import (
    LogprobRows,
    decode_rows,
    encode_rows,
    rows_from_lists,
    rows_from_tensors,
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
    BlockObjectStoreError,
    VariableBlockObjectStore,
)
from vllm.distributed.aux_output_connector.worker import (
    AuxOutputWorkerConnector,
    PendingAuxOutput,
    _WorkerRequestState,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker.gpu import async_utils
from vllm.v1.worker.gpu.sample.output import SamplerOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

_SHAPE = (3, 2)
_DTYPE = np.dtype("uint8")
_BLOCK_SIZE = 4


def test_background_store_publishes_without_blocking_caller():
    started = threading.Event()
    release = threading.Event()
    underlying = Mock()

    def put(*_args, **_kwargs):
        started.set()
        release.wait()

    underlying.put.side_effect = put
    store = BackgroundBlockObjectStore(underlying, max_pending_batches=1)

    store.put([BlockObject("key", b"value")])
    assert started.wait(timeout=1)
    release.set()
    store.close()

    underlying.close.assert_called_once_with()


def test_variable_store_round_trips_logprob_rows():
    rows = LogprobRows(
        positions=np.array([1, 2], dtype=np.int64),
        token_ids=np.array([[4, 5], [6, 7]], dtype=np.int32),
        values=np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32),
        ranks=np.array([0, 1], dtype=np.int32),
    )
    payload = encode_rows(rows)
    store = VariableBlockObjectStore(max_bytes=len(payload) * 2)
    store.put([BlockObject("logprobs", payload)])

    result = decode_rows(store.get_concatenated(["logprobs"]))

    np.testing.assert_array_equal(result.positions, rows.positions)
    np.testing.assert_array_equal(result.token_ids, rows.token_ids)
    np.testing.assert_array_equal(result.values, rows.values)
    np.testing.assert_array_equal(result.ranks, rows.ranks)
    store.close()


def test_logprob_rows_preserve_top_k_width_and_ranks():
    """Auxiliary rows retain the request-level top-k payload shape."""
    lists = LogprobsLists(
        np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64),
        np.array([[-0.1, -0.2, -0.3], [-1.1, -1.2, -1.3]], dtype=np.float64),
        np.array([0, 2], dtype=np.int64),
    )
    rows = rows_from_lists(lists, start=7)

    np.testing.assert_array_equal(rows.positions, [7, 8])
    np.testing.assert_array_equal(rows.token_ids, lists.logprob_token_ids)
    np.testing.assert_allclose(rows.values, lists.logprobs)
    np.testing.assert_array_equal(rows.ranks, lists.sampled_token_ranks)
    assert rows.token_ids.dtype == np.dtype("int32")
    assert rows.values.dtype == np.dtype("float32")
    assert rows.ranks.dtype == np.dtype("int32")

    tensors = LogprobsTensors(
        torch.from_numpy(lists.logprob_token_ids),
        torch.from_numpy(lists.logprobs),
        torch.from_numpy(lists.sampled_token_ranks),
    )
    tensor_rows = rows_from_tensors(tensors, start=7)
    np.testing.assert_array_equal(tensor_rows.token_ids, rows.token_ids)
    np.testing.assert_allclose(tensor_rows.values, rows.values)
    np.testing.assert_array_equal(tensor_rows.ranks, rows.ranks)


def test_variable_store_rolls_back_references_after_failed_put():
    store = VariableBlockObjectStore(max_bytes=4)
    store.put([BlockObject("first", b"1111")], retain_keys=["first"])

    with pytest.raises(BlockObjectStoreError, match="cannot retain"):
        store.put(
            [BlockObject("second", b"2222")],
            retain_keys=["second", "first"],
        )

    assert store.get_concatenated(["first"]) == b"1111"
    store.put([], release_keys=["first"])
    store.put([BlockObject("third", b"3333")])
    store.close()


def test_decode_rows_rejects_missing_and_non_scalar_schema():
    missing = io.BytesIO()
    np.savez(missing, positions=np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="malformed auxiliary"):
        decode_rows(missing.getvalue())

    non_scalar = io.BytesIO()
    np.savez(
        non_scalar,
        schema_version=np.array([1], dtype=np.int32),
        positions=np.array([], dtype=np.int64),
        token_ids=np.empty((0, 1), dtype=np.int32),
        values=np.empty((0, 1), dtype=np.float32),
        ranks=np.empty(0, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="unsupported auxiliary"):
        decode_rows(non_scalar.getvalue())


def _logprob_rows(start: int, count: int) -> LogprobRows:
    positions = np.arange(start, start + count, dtype=np.int64)
    token_ids = np.stack((positions, positions + 100), axis=1).astype(np.int32)
    values = -token_ids.astype(np.float32)
    return LogprobRows(positions, token_ids, values, positions.astype(np.int32))


def _make_logprob_worker() -> AuxOutputWorkerConnector:
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._store = None
    worker._buffer = None
    worker._logprob_store = BackgroundBlockObjectStore(
        VariableBlockObjectStore(max_bytes=1 << 20),
        max_pending_batches=4,
    )
    worker._logprob_block_size = _BLOCK_SIZE
    worker._requests = {}
    worker._generation = 0
    worker._step_metadata = None
    worker._pending_outputs = []
    worker._lock = threading.Lock()
    worker._max_concurrent_batches = 2
    return worker


def test_worker_publishes_causally_aligned_logprob_block():
    worker = _make_logprob_worker()
    state = _WorkerRequestState(prompt_len=_BLOCK_SIZE)
    state.logprob_keys = ["block"]
    state.prompt_logprob_keys = state.logprob_keys
    state.prompt_logprob_rows[1] = _logprob_rows(1, _BLOCK_SIZE - 1)
    state.logprob_rows[_BLOCK_SIZE] = _logprob_rows(_BLOCK_SIZE, 1)
    worker._requests["request"] = state

    worker._publish_logprob_blocks(["request"])

    stored = worker._read_logprob_rows(["block"])
    np.testing.assert_array_equal(stored.positions, np.arange(1, _BLOCK_SIZE + 1))
    worker.close()


def test_worker_incrementally_updates_sparse_generated_logprob_block():
    worker = _make_logprob_worker()
    state = _WorkerRequestState(prompt_len=_BLOCK_SIZE)
    state.logprob_keys = ["block"]
    state.logprob_rows[_BLOCK_SIZE] = _logprob_rows(_BLOCK_SIZE, 1)
    worker._requests["request"] = state

    worker._publish_logprob_blocks(["request"])
    np.testing.assert_array_equal(
        worker._read_logprob_rows(["block"]).positions, [_BLOCK_SIZE]
    )
    assert not state.logprob_rows

    second = _WorkerRequestState(prompt_len=_BLOCK_SIZE)
    second.logprob_keys = ["block"]
    second.logprob_rows[_BLOCK_SIZE - 1] = _logprob_rows(_BLOCK_SIZE - 1, 1)
    worker._requests["second"] = second
    worker._publish_logprob_blocks(["second"])
    np.testing.assert_array_equal(
        worker._read_logprob_rows(["block"]).positions,
        [_BLOCK_SIZE - 1, _BLOCK_SIZE],
    )
    worker.close()


def test_worker_recomputed_logprob_row_replaces_cached_value():
    worker = _make_logprob_worker()
    state = _WorkerRequestState(prompt_len=_BLOCK_SIZE)
    state.logprob_keys = ["block"]
    state.logprob_rows[_BLOCK_SIZE] = _logprob_rows(_BLOCK_SIZE, 1)
    worker._requests["request"] = state
    worker._publish_logprob_blocks(["request"])

    replacement = _logprob_rows(_BLOCK_SIZE, 1)
    replacement.token_ids[:] = 999
    replacement.values[:] = 999
    replacement.ranks[:] = 999
    second = _WorkerRequestState(prompt_len=_BLOCK_SIZE)
    second.logprob_keys = ["block"]
    second.logprob_rows[_BLOCK_SIZE] = replacement
    worker._requests["second"] = second
    worker._publish_logprob_blocks(["second"])

    stored = worker._read_logprob_rows(["block"])
    np.testing.assert_array_equal(stored.token_ids, replacement.token_ids)
    np.testing.assert_array_equal(stored.values, replacement.values)
    worker.close()


def test_worker_replays_generated_logprobs_after_speculative_rejection():
    worker = _make_logprob_worker()
    worker._requests["request"] = _WorkerRequestState(prompt_len=_BLOCK_SIZE)

    first_rows = _logprob_rows(_BLOCK_SIZE, 1)
    first = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([_BLOCK_SIZE - 1], dtype=np.int32),
        query_start_loc=np.array([0, 5], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1], dtype=np.int32),
        num_rejected=np.array([4], dtype=np.int32),
        logprobs={
            "request": LogprobsLists(
                first_rows.token_ids,
                first_rows.values,
                first_rows.ranks,
            )
        },
        replay_logprobs=frozenset({"request"}),
    )
    worker._commit_output(first)

    second_rows = _logprob_rows(_BLOCK_SIZE + 1, 1)
    second = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([2 * _BLOCK_SIZE], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1], dtype=np.int32),
        num_rejected=np.array([0], dtype=np.int32),
        logprobs={
            "request": LogprobsLists(
                second_rows.token_ids,
                second_rows.values,
                second_rows.ranks,
            )
        },
        replay_logprobs=frozenset({"request"}),
    )
    worker._commit_output(second)

    state = worker._requests["request"]
    assert state.logprob_cursor == _BLOCK_SIZE + 2
    assert sorted(state.logprob_rows) == [_BLOCK_SIZE, _BLOCK_SIZE + 1]
    worker.close()


def test_worker_merges_cached_prompt_rows_with_chunked_live_suffix():
    worker = _make_logprob_worker()
    cached = _logprob_rows(1, _BLOCK_SIZE)
    worker._logprob_store.put([BlockObject("block", encode_rows(cached))])
    state = _WorkerRequestState(
        prompt_len=2 * _BLOCK_SIZE,
        prompt_replay_prefix=_BLOCK_SIZE,
    )
    state.logprob_keys = ["block"]
    state.prompt_logprob_keys = state.logprob_keys
    worker._requests["request"] = state

    live = _logprob_rows(_BLOCK_SIZE + 1, _BLOCK_SIZE - 1)
    prompt_logprobs = LogprobsTensors(
        torch.from_numpy(live.token_ids),
        torch.from_numpy(live.values),
        torch.from_numpy(live.ranks),
    )
    generated = _logprob_rows(2 * _BLOCK_SIZE, 1)
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([2 * _BLOCK_SIZE - 1], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1], dtype=np.int32),
        num_rejected=np.array([0], dtype=np.int32),
        logprobs={
            "request": LogprobsLists(
                generated.token_ids,
                generated.values,
                generated.ranks,
            )
        },
        prompt_logprobs={"request": prompt_logprobs},
        replay_logprobs=frozenset({"request"}),
        replay_prompt_logprobs=frozenset({"request"}),
    )

    output = worker._commit_output(pending)["request"]

    assert output.prompt_logprobs is not None
    np.testing.assert_array_equal(
        output.prompt_logprobs.logprob_token_ids.numpy(),
        np.concatenate((cached.token_ids, live.token_ids)),
    )
    np.testing.assert_array_equal(
        output.prompt_logprobs.selected_token_ranks.numpy(),
        np.arange(1, 2 * _BLOCK_SIZE, dtype=np.int32),
    )
    worker.close()


def test_worker_publishes_shared_prompt_block_before_same_batch_consumer_reads():
    worker = _make_logprob_worker()
    producer = _WorkerRequestState(prompt_len=5, prompt_replay_prefix=0)
    producer.prompt_logprob_keys = ["shared"]
    consumer = _WorkerRequestState(prompt_len=5, prompt_replay_prefix=4)
    consumer.prompt_logprob_keys = ["shared"]
    worker._requests = {"producer": producer, "consumer": consumer}

    live = _logprob_rows(1, 4)
    prompt_logprobs = LogprobsTensors(
        torch.from_numpy(live.token_ids),
        torch.from_numpy(live.values),
        torch.from_numpy(live.ranks),
    )
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["producer", "consumer"],
        batch_indices=np.array([0, 1], dtype=np.int32),
        token_starts=np.array([0, 4], dtype=np.int32),
        query_start_loc=np.array([0, 5, 6], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1, 1], dtype=np.int32),
        num_rejected=np.array([0, 0], dtype=np.int32),
        prompt_logprobs={"producer": prompt_logprobs},
        replay_prompt_logprobs=frozenset({"producer", "consumer"}),
    )

    outputs = worker._commit_output(pending)

    assert outputs["producer"].prompt_logprobs is not None
    assert outputs["consumer"].prompt_logprobs is not None
    np.testing.assert_array_equal(
        outputs["consumer"].prompt_logprobs.logprob_token_ids.numpy(),
        live.token_ids,
    )
    worker.close()


def test_worker_discards_unpopulated_full_prompt_cache_prefix():
    worker = _make_logprob_worker()
    cached = _logprob_rows(1, _BLOCK_SIZE)
    worker._logprob_store.put([BlockObject("block", encode_rows(cached))])
    state = _WorkerRequestState(
        prompt_len=2 * _BLOCK_SIZE,
        prompt_replay_prefix=_BLOCK_SIZE,
    )
    state.logprob_keys = ["block"]
    state.prompt_logprob_keys = state.logprob_keys
    worker._requests["request"] = state

    full = _logprob_rows(1, 2 * _BLOCK_SIZE - 1)
    full.token_ids[:_BLOCK_SIZE] = -999
    full.values[:_BLOCK_SIZE] = -999
    full.ranks[:_BLOCK_SIZE] = -999
    prompt_logprobs = LogprobsTensors(
        torch.from_numpy(full.token_ids),
        torch.from_numpy(full.values),
        torch.from_numpy(full.ranks),
    )
    generated = _logprob_rows(2 * _BLOCK_SIZE, 1)
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([2 * _BLOCK_SIZE - 1], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1], dtype=np.int32),
        num_rejected=np.array([0], dtype=np.int32),
        logprobs={
            "request": LogprobsLists(
                generated.token_ids,
                generated.values,
                generated.ranks,
            )
        },
        prompt_logprobs={"request": prompt_logprobs},
        replay_logprobs=frozenset({"request"}),
        replay_prompt_logprobs=frozenset({"request"}),
    )

    output = worker._commit_output(pending)["request"]

    assert output.prompt_logprobs is not None
    np.testing.assert_array_equal(
        output.prompt_logprobs.logprob_token_ids.numpy(),
        np.concatenate((cached.token_ids, full.token_ids[_BLOCK_SIZE:])),
    )
    worker.close()


def test_worker_fails_when_generated_logprob_artifact_is_missing():
    worker = _make_logprob_worker()
    state = _WorkerRequestState(prompt_len=_BLOCK_SIZE)
    state.logprob_keys = ["missing"]
    worker._requests["request"] = state
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([_BLOCK_SIZE - 1], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1], dtype=np.int32),
        num_rejected=np.array([0], dtype=np.int32),
        replay_logprobs=frozenset({"request"}),
    )

    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        worker._commit_output(pending)
    worker.close()


def test_worker_binds_full_prompt_block_to_first_generated_token():
    worker = _make_logprob_worker()
    block_hash = b"a" * 32
    fingerprint = "fingerprint"
    generated = _logprob_rows(_BLOCK_SIZE, 1)
    key = worker._logprob_keys(
        [block_hash], [int(generated.token_ids[0, 0])], fingerprint
    )[0]
    cached = _logprob_rows(1, _BLOCK_SIZE)
    worker._logprob_store.put([BlockObject(key, encode_rows(cached))])
    state = _WorkerRequestState(
        prompt_len=_BLOCK_SIZE,
        prompt_replay_prefix=_BLOCK_SIZE,
        pending_logprob_blocks=[(block_hash, fingerprint)],
    )
    state.prompt_logprob_keys = state.logprob_keys
    worker._requests["request"] = state
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([_BLOCK_SIZE - 1], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1], dtype=np.int32),
        num_rejected=np.array([0], dtype=np.int32),
        logprobs={
            "request": LogprobsLists(
                generated.token_ids,
                generated.values,
                generated.ranks,
            )
        },
        replay_logprobs=frozenset({"request"}),
        replay_prompt_logprobs=frozenset({"request"}),
    )

    output = worker._commit_output(pending)["request"]

    assert state.logprob_keys == [key]
    assert not state.pending_logprob_blocks
    assert output.prompt_logprobs is not None
    np.testing.assert_array_equal(
        output.prompt_logprobs.logprob_token_ids.numpy(), cached.token_ids[:-1]
    )
    worker.close()


def test_worker_emits_empty_prompt_logprobs_for_single_token_prompt():
    worker = _make_logprob_worker()
    worker._requests["request"] = _WorkerRequestState(prompt_len=1)
    generated = _logprob_rows(1, 1)
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([0], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1], dtype=np.int32),
        num_rejected=np.array([0], dtype=np.int32),
        logprobs={
            "request": LogprobsLists(
                generated.token_ids,
                generated.values,
                generated.ranks,
            )
        },
        replay_logprobs=frozenset({"request"}),
        replay_prompt_logprobs=frozenset({"request"}),
    )

    output = worker._commit_output(pending)["request"]

    assert output.prompt_logprobs is not None
    assert output.prompt_logprobs.logprobs.shape == (0, 2)
    worker.close()


def test_worker_attaches_completed_prompt_logprobs_only_once():
    worker = _make_logprob_worker()
    worker._requests["request"] = _WorkerRequestState(prompt_len=1)
    generated = _logprob_rows(1, 1)

    def pending(*, sampled: int):
        return PendingAuxOutput(
            connector=worker,
            request_ids=["request"],
            batch_indices=np.array([0], dtype=np.int32),
            token_starts=np.array([0], dtype=np.int32),
            query_start_loc=np.array([0, 1], dtype=np.int32),
            routed_experts_gpu=None,
            num_sampled=np.array([sampled], dtype=np.int32),
            num_rejected=np.array([0], dtype=np.int32),
            logprobs=(
                {
                    "request": LogprobsLists(
                        generated.token_ids,
                        generated.values,
                        generated.ranks,
                    )
                }
                if sampled
                else {}
            ),
            replay_logprobs=frozenset({"request"}) if sampled else frozenset(),
            replay_prompt_logprobs=frozenset({"request"}),
        )

    first = worker._commit_output(pending(sampled=1))["request"]
    second = worker._commit_output(pending(sampled=0))["request"]
    assert first.prompt_logprobs is not None
    assert second.prompt_logprobs is None
    worker.close()


def test_worker_defers_prompt_artifact_until_visible_output_step():
    worker = _make_logprob_worker()
    worker._requests["request"] = _WorkerRequestState(prompt_len=5)
    prompt = _logprob_rows(1, 4)
    prompt_tensors = LogprobsTensors(
        torch.from_numpy(prompt.token_ids),
        torch.from_numpy(prompt.values),
        torch.from_numpy(prompt.ranks),
    )

    def pending(*, sampled: int, prompt_rows=None):
        return PendingAuxOutput(
            connector=worker,
            request_ids=["request"],
            batch_indices=np.array([0], dtype=np.int32),
            token_starts=np.array(
                [0 if prompt_rows is not None else 5], dtype=np.int32
            ),
            query_start_loc=np.array(
                [0, 5 if prompt_rows is not None else 1], dtype=np.int32
            ),
            routed_experts_gpu=None,
            num_sampled=np.array([sampled], dtype=np.int32),
            num_rejected=np.array([0], dtype=np.int32),
            prompt_logprobs=(
                {"request": prompt_rows} if prompt_rows is not None else {}
            ),
            replay_prompt_logprobs=frozenset({"request"}),
        )

    prefill = worker._commit_output(
        pending(sampled=0, prompt_rows=prompt_tensors)
    )["request"]
    assert prefill.prompt_logprobs is None

    # The request can finish on the first decode step without returning a
    # sampled token; the prompt artifact must still be delivered there.
    decode = worker._commit_output(pending(sampled=0))["request"]
    assert decode.prompt_logprobs is not None
    np.testing.assert_array_equal(
        decode.prompt_logprobs.logprob_token_ids.numpy(), prompt.token_ids
    )
    worker.close()


def test_worker_reports_missing_prompt_positions_as_ranges():
    worker = _make_logprob_worker()
    worker._requests["request"] = _WorkerRequestState(prompt_len=8)
    partial = _logprob_rows(1, 2)
    prompt_tensors = LogprobsTensors(
        torch.from_numpy(partial.token_ids),
        torch.from_numpy(partial.values),
        torch.from_numpy(partial.ranks),
    )
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([0], dtype=np.int32),
        query_start_loc=np.array([0, 8], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1], dtype=np.int32),
        num_rejected=np.array([0], dtype=np.int32),
        prompt_logprobs={"request": prompt_tensors},
        replay_prompt_logprobs=frozenset({"request"}),
    )

    with pytest.raises(
        RuntimeError,
        match=r"missing_positions=1-5",
    ):
        worker._commit_output(pending)
    worker.close()


def test_worker_reports_missing_generated_logprob_row():
    worker = _make_logprob_worker()
    state = _WorkerRequestState(prompt_len=_BLOCK_SIZE)
    state.logprob_keys = ["block"]
    worker._requests["request"] = state
    worker._logprob_store.put(
        [BlockObject("block", encode_rows(_logprob_rows(_BLOCK_SIZE + 1, 1)))]
    )
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([_BLOCK_SIZE - 1], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1], dtype=np.int32),
        num_rejected=np.array([0], dtype=np.int32),
        replay_logprobs=frozenset({"request"}),
    )

    with pytest.raises(RuntimeError, match="missing sampled position"):
        worker._commit_output(pending)
    worker.close()


def test_worker_preserves_all_generated_logprob_rows_for_multiple_samples():
    worker = _make_logprob_worker()
    worker._requests["request"] = _WorkerRequestState(prompt_len=_BLOCK_SIZE)
    sampled = _logprob_rows(_BLOCK_SIZE + 1, 2)
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["request"],
        batch_indices=np.array([0], dtype=np.int32),
        token_starts=np.array([_BLOCK_SIZE], dtype=np.int32),
        query_start_loc=np.array([0, 2], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([2], dtype=np.int32),
        num_rejected=np.array([0], dtype=np.int32),
        logprobs={
            "request": LogprobsLists(
                sampled.token_ids,
                sampled.values,
                sampled.ranks,
            )
        },
        replay_logprobs=frozenset({"request"}),
    )

    output = worker._commit_output(pending)["request"]

    assert output.logprobs is not None
    np.testing.assert_array_equal(output.logprobs.logprob_token_ids, sampled.token_ids)
    worker.close()


def test_worker_handles_fanned_out_n_requests_independently():
    worker = _make_logprob_worker()
    worker._requests = {
        "request-0": _WorkerRequestState(prompt_len=_BLOCK_SIZE),
        "request-1": _WorkerRequestState(prompt_len=_BLOCK_SIZE),
    }
    rows = {
        request_id: _logprob_rows(_BLOCK_SIZE + 1 + index, 1)
        for index, request_id in enumerate(("request-0", "request-1"))
    }
    pending = PendingAuxOutput(
        connector=worker,
        request_ids=["request-0", "request-1"],
        batch_indices=np.array([0, 1], dtype=np.int32),
        token_starts=np.array([_BLOCK_SIZE, _BLOCK_SIZE], dtype=np.int32),
        query_start_loc=np.array([0, 1, 2], dtype=np.int32),
        routed_experts_gpu=None,
        num_sampled=np.array([1, 1], dtype=np.int32),
        num_rejected=np.array([0, 0], dtype=np.int32),
        logprobs={
            request_id: LogprobsLists(value.token_ids, value.values, value.ranks)
            for request_id, value in rows.items()
        },
        replay_logprobs=frozenset(rows),
    )

    outputs = worker._commit_output(pending)

    for request_id, expected in rows.items():
        assert outputs[request_id].logprobs is not None
        np.testing.assert_array_equal(
            outputs[request_id].logprobs.logprob_token_ids, expected.token_ids
        )
    worker.close()


def test_worker_resets_prompt_artifact_when_streaming_prompt_grows():
    worker = _make_logprob_worker()
    state = _WorkerRequestState(
        prompt_len=4,
        prompt_replay_prefix=0,
        prompt_logprobs_artifact=LogprobsTensors(
            torch.empty((1, 2), dtype=torch.int32),
            torch.empty((1, 2), dtype=torch.float32),
            torch.empty(1, dtype=torch.int32),
        ),
    )
    worker._requests["session"] = state
    worker.begin_step(
        AuxOutputConnectorMetadata(
            generation=0,
            requests={"session": 0},
            block_hashes={},
            finished_requests=(),
            prompt_lens={"session": 8},
        )
    )

    assert state.prompt_logprobs_artifact is None
    assert state.prompt_replay_prefix is None
    assert state.prompt_len == 8
    worker.close()


def _make_connector():
    return AuxOutputSchedulerConnector()


@dataclass
class _Execution:
    request_id: str
    token_start: int
    num_tokens: int
    emit_start: int
    block_hashes: list[bytes] | PackedBlockHashes


@dataclass
class _WorkerStep:
    metadata: AuxOutputConnectorMetadata
    executions: list[_Execution]


@dataclass
class _SchedulerRequest:
    request_id: str
    block_hashes: list[bytes]
    num_tokens: int
    num_output_tokens: int = 1
    num_computed_tokens: int = 0
    num_in_flight_tokens: int = 0
    finished: bool = False
    token_ids: list[int] | None = None
    sampling_params: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(routed_experts_prompt_start=0)
    )

    @property
    def num_prompt_tokens(self) -> int:
        return self.num_tokens - self.num_output_tokens

    @property
    def all_token_ids(self) -> list[int]:
        return self.token_ids or list(range(self.num_tokens))

    def is_finished(self) -> bool:
        return self.finished


def _request_metadata(
    request_id: str,
    token_start: int,
    num_tokens: int,
    emit_start: int,
    block_hashes,
) -> _Execution:
    return _Execution(request_id, token_start, num_tokens, emit_start, block_hashes)


def _metadata(
    generation: int,
    requests: list[_Execution],
    finished_requests: dict[str, list[bytes] | PackedBlockHashes],
    hash_updates: dict[str, list[bytes] | PackedBlockHashes] | None = None,
) -> _WorkerStep:
    block_hashes = {
        request.request_id: request.block_hashes
        for request in requests
        if request.block_hashes
    }
    if hash_updates:
        block_hashes.update(
            (request_id, hashes)
            for request_id, hashes in hash_updates.items()
            if hashes
        )
    block_hashes.update(
        (request_id, hashes)
        for request_id, hashes in finished_requests.items()
        if hashes
    )
    return _WorkerStep(
        AuxOutputConnectorMetadata(
            generation,
            {request.request_id: request.emit_start for request in requests},
            block_hashes,
            tuple(finished_requests),
        ),
        requests,
    )


def _execution_ranges(metadata, request_ids):
    by_request = {request.request_id: request for request in metadata.executions}
    token_starts = tuple(
        by_request[request_id].token_start for request_id in request_ids
    )
    num_tokens = tuple(by_request[request_id].num_tokens for request_id in request_ids)
    return token_starts, num_tokens


def _begin_step(worker: AuxOutputWorkerConnector, step: _WorkerStep) -> None:
    worker.begin_step(step.metadata)


def _make_worker(
    max_num_seqs: int,
    max_num_batched_tokens: int | None = None,
    max_concurrent_batches: int = 2,
    max_store_blocks: int | None = None,
) -> AuxOutputWorkerConnector:
    object_nbytes = _BLOCK_SIZE * int(np.prod(_SHAPE))
    store = _make_store(
        max_bytes=(
            1 << 20 if max_store_blocks is None else max_store_blocks * object_nbytes
        ),
        object_nbytes=object_nbytes,
    )
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._store = store
    worker._buffer = RoutedExpertsBuffer(
        _DTYPE,
        _SHAPE,
        _BLOCK_SIZE,
        max_num_seqs,
        max_num_batched_tokens or max_num_seqs * _BLOCK_SIZE,
        max_concurrent_batches,
    )
    worker._requests = {}
    worker._generation = 0
    worker._step_metadata = None
    worker._pending_outputs = []
    worker._lock = threading.Lock()
    worker._max_concurrent_batches = max_concurrent_batches
    return worker


def test_worker_rejects_metadata_generation_rollback():
    worker = _make_worker(1)
    _begin_step(worker, _metadata(1, [], {}))

    with pytest.raises(AssertionError, match="generation moved backwards"):
        _begin_step(worker, _metadata(0, [], {}))

    worker.close()


def test_worker_rejects_run_and_finish_in_one_step():
    worker = _make_worker(1)
    request = _request_metadata("request", 0, 1, 0, [])
    metadata = _metadata(
        0,
        [request],
        {"request": []},
    )

    with pytest.raises(AssertionError, match="cannot run and finish"):
        _begin_step(worker, metadata)

    worker.close()


def _input_batch(request_ids, token_starts, query_start_loc):
    return SimpleNamespace(
        req_ids=request_ids,
        num_computed_tokens_np=token_starts,
        query_start_loc_np=query_start_loc,
    )


def test_non_output_rank_skips_capture_snapshot():
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._store = None
    worker._buffer = None
    worker._capturer = Mock()
    worker._step_metadata = Mock()
    assert worker.prepare_output(_input_batch([], np.array([]), np.array([]))) is None
    worker._capturer.snapshot_routing_data.assert_not_called()


def test_worker_skips_aux_outputs_for_internal_warmup_step():
    worker = _make_worker(1)
    worker._capturer = Mock()

    worker.begin_step(None)

    assert worker.prepare_output(_input_batch([], np.array([]), np.array([]))) is None
    worker._capturer.snapshot_routing_data.assert_not_called()
    worker.close()


def test_next_step_does_not_consume_pending_output(monkeypatch):
    """begin_step must not wait for or consume an unconsumed step output."""
    event = Mock()
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(async_utils, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(
        async_utils, "async_copy_to_np", lambda tensor: tensor.numpy().copy()
    )
    worker = _make_worker(1)
    worker.begin_step(
        _metadata(0, [_request_metadata("request", 0, 1, 0, [])], {}).metadata
    )
    rows = torch.ones((1, *_SHAPE), dtype=torch.uint8)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.return_value = rows
    process_output = Mock(wraps=worker.process_output)
    monkeypatch.setattr(worker, "process_output", process_output)
    pending = worker.prepare_output(
        _input_batch(["request"], np.array([0]), np.array([0, 1]))
    )
    assert pending is not None
    output = async_utils.AsyncOutput(
        ModelRunnerOutput(["request"], {"request": 0}),
        SamplerOutput(
            torch.tensor([[7]]), None, None, None, num_rejected=torch.tensor([0])
        ),
        torch.tensor([1]),
        Mock(),
        Mock(),
        check_ep_fault=False,
        pending_aux_output=pending,
    )
    event.record.assert_called_once()
    assert worker._pending_outputs == [pending]

    worker.begin_step(_metadata(0, [], {}).metadata)
    process_output.assert_not_called()
    assert worker._pending_outputs == [pending]

    result = output.get_output()

    assert result.sampled_token_ids == [[7]]
    np.testing.assert_array_equal(
        result.aux_output_connector_output["request"].rows, rows.numpy()
    )
    process_output.assert_called_once()
    assert worker._pending_outputs == []
    worker.close()


@pytest.mark.parametrize("num_pending", [1, 2])
def test_finished_request_teardown_waits_for_pending_output(num_pending):
    """Terminal cleanup waits for every outstanding output, not just the first."""
    worker = _make_worker(1)
    rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, *_SHAPE)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.return_value = torch.from_numpy(rows)
    pending_outputs = []
    for i in range(num_pending):
        request = _request_metadata("request", 4 * i, 4, 4 * i, [bytes([i]) * 4])
        worker.begin_step(_metadata(0, [request], {}).metadata)
        pending = worker.prepare_output(
            _input_batch(["request"], np.array([4 * i]), np.array([0, 4]))
        )
        assert pending is not None
        pending.enqueue_cpu_copy(np.array([1]), np.array([0]))
        pending_outputs.append(pending)

    # The next step finishes the request while its output is unconsumed.
    worker.begin_step(_metadata(0, [], {"request": []}).metadata)
    assert "request" in worker._requests

    for i, pending in enumerate(pending_outputs):
        output = worker.process_output(pending)
        np.testing.assert_array_equal(output["request"].rows, rows)
        if i + 1 < num_pending:
            assert "request" in worker._requests
            assert worker._store._references
    assert "request" not in worker._requests
    # The finished request's store keys must be released after commit.
    assert not worker._store._references
    worker.close()


def test_pending_outputs_bounded_by_max_concurrent_batches():
    worker = _make_worker(1, max_concurrent_batches=2)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.side_effect = lambda num_rows: torch.zeros(
        (num_rows, *_SHAPE), dtype=torch.uint8
    )
    metadata = _metadata(0, [_request_metadata("request", 0, 1, 0, [])], {}).metadata
    batch = _input_batch(["request"], np.array([0]), np.array([0, 1]))
    for _ in range(2):
        worker.begin_step(metadata)
        assert worker.prepare_output(batch) is not None

    worker.begin_step(metadata)
    with pytest.raises(AssertionError, match="not consumed in order"):
        worker.prepare_output(batch)

    worker.close()


def test_worker_rejects_invalid_rejected_token_count():
    worker = _make_worker(1)
    metadata = _metadata(
        0,
        [_request_metadata("request", 0, 1, 0, [])],
        {},
    )
    with pytest.raises(AssertionError, match="rejected-token count is invalid"):
        _process_output(
            worker,
            metadata,
            np.zeros((1, *_SHAPE), dtype=_DTYPE),
            ["request"],
            np.array([-1]),
            token_starts=(0,),
            num_tokens=(1,),
        )
    assert worker._pending_outputs == []

    worker.close()


def _process_output(
    worker,
    step,
    rows,
    request_ids,
    num_rejected,
    num_sampled=None,
    *,
    token_starts=None,
    num_tokens=None,
    query_start_loc=None,
):
    if num_sampled is None:
        num_sampled = np.ones(len(request_ids), dtype=np.int32)
    if token_starts is None or num_tokens is None:
        assert isinstance(step, _WorkerStep)
        token_starts, num_tokens = _execution_ranges(step, request_ids)
    worker._capturer = Mock()
    worker._capturer.snapshot_routing_data.return_value = torch.from_numpy(rows)
    metadata = step.metadata if isinstance(step, _WorkerStep) else step
    worker.begin_step(metadata)
    if query_start_loc is None:
        query_start_loc = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(num_tokens, dtype=np.int32))
        )
    pending = worker.prepare_output(
        _input_batch(request_ids, np.asarray(token_starts), query_start_loc)
    )
    assert pending is not None
    pending.enqueue_cpu_copy(num_sampled, num_rejected)
    return worker.process_output(pending)


def test_worker_ignores_cudagraph_query_padding():
    worker = _make_worker(2)
    logical = np.arange(2 * 3 * 2, dtype=np.uint8).reshape(2, 3, 2)
    metadata = _metadata(
        0,
        [
            _request_metadata("first", 0, 1, 0, []),
            _request_metadata("second", 0, 1, 0, []),
        ],
        {},
    )

    output = _process_output(
        worker,
        metadata,
        logical,
        ["first", "second"],
        np.zeros(2, dtype=np.int32),
        query_start_loc=np.array([0, 1, 2, 2], dtype=np.int32),
    )

    np.testing.assert_array_equal(output["first"].rows, logical[:1])
    np.testing.assert_array_equal(output["second"].rows, logical[1:])
    worker.close()


def test_worker_rejects_scheduler_emit_cursor_ahead():
    worker = _make_worker(1)
    _begin_step(
        worker,
        _metadata(
            0,
            [_request_metadata("request", 0, 1, 0, [])],
            {},
        ),
    )

    with pytest.raises(AssertionError, match="Scheduler emit cursor moved ahead"):
        _begin_step(
            worker,
            _metadata(
                0,
                [_request_metadata("request", 1, 1, 1, [])],
                {},
            ),
        )

    worker.close()


@pytest.mark.parametrize(
    "routing",
    [
        np.zeros((1, _SHAPE[0], _SHAPE[1] + 1), dtype=_DTYPE),
        np.zeros((1, *_SHAPE), dtype=np.int32),
    ],
    ids=["shape", "dtype"],
)
def test_worker_rejects_mismatched_capture_profile(routing):
    worker = _make_worker(1)
    metadata = _metadata(
        0,
        [_request_metadata("request", 0, 1, 0, [])],
        {},
    )

    with pytest.raises(AssertionError, match="capture profile changed"):
        _process_output(
            worker,
            metadata,
            routing,
            ["request"],
            np.array([0]),
        )
    worker.close()


@pytest.mark.parametrize(
    ("initial_start", "initial_rows", "invalid_start"),
    [
        (4, [4, 5, 6], 6),
        (0, [0], 2),
        (None, [], 3),
    ],
    ids=["overlap", "gap", "unaligned-initial-capture"],
)
def test_logical_buffer_rejects_noncontiguous_capture(
    initial_start, initial_rows, invalid_start
):
    buffer = RoutedExpertsBuffer(np.dtype("uint8"), (1,), 4, 1, 4, 1)
    if initial_start is not None:
        rows = np.asarray(initial_rows, dtype=np.uint8).reshape(-1, 1)
        assert buffer.capture("request", initial_start, rows) == []

    with pytest.raises(AssertionError, match="not contiguous"):
        buffer.capture(
            "request", invalid_start, np.array([[invalid_start]], dtype=np.uint8)
        )


def test_logical_buffer_captures_one_row_per_decode_step():
    buffer = RoutedExpertsBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8, 2)
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=_DTYPE).reshape(_BLOCK_SIZE, 3, 2)

    completed = []
    for step in range(_BLOCK_SIZE):
        completed += buffer.capture("request", step, logical[step : step + 1])

    assert len(completed) == 1
    np.testing.assert_array_equal(completed[0][1], logical)


def test_logical_buffer_copies_borrowed_block_when_retained():
    buffer = RoutedExpertsBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 4, 2)
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=_DTYPE).reshape(_BLOCK_SIZE, *_SHAPE)

    expected = logical.copy()
    completed = buffer.capture("request", 0, logical)
    retained = buffer.retain_block(completed[0][1])
    logical[:] = 0

    assert not np.shares_memory(retained, logical)
    np.testing.assert_array_equal(retained, expected)
    buffer.release_block(retained)


def _make_store(
    *,
    max_bytes: int = 1 << 20,
    object_nbytes: int = 4,
):
    return BlockObjectStore(
        max_bytes=max_bytes,
        object_nbytes=object_nbytes,
    )


def test_publish_routed_experts_publishes_full_blocks():
    store = _make_store(object_nbytes=_BLOCK_SIZE * int(np.prod(_SHAPE)))
    buffer = RoutedExpertsBuffer(_DTYPE, _SHAPE, _BLOCK_SIZE, 1, 8, 2)
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    hashes = [b"a" * 32, b"b" * 32]
    keys = routed_experts_keys(hashes, "0")
    blocks = buffer.capture("request", 0, logical)
    publish_routed_experts(
        store,
        batches=[(keys, blocks)],
        block_size=_BLOCK_SIZE,
    )
    expected = logical.copy()
    logical[:] = 0

    np.testing.assert_array_equal(
        materialize_routed_experts(
            store,
            keys,
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        expected,
    )
    store.close()


def test_worker_data_plane_publishes_blocks_and_reuses_prefix():
    worker = _make_worker(2)

    hashes = [b"a" * 32, b"b" * 32, b"c" * 32]
    logical = np.arange(10 * 3 * 2, dtype=np.uint8).reshape(10, 3, 2)
    first = _metadata(
        generation=0,
        requests=[_request_metadata("first", 0, 8, 0, hashes)],
        finished_requests={},
    )
    output = _process_output(worker, first, logical[:8], ["first"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output["first"].rows, logical[:8])

    second = _metadata(
        generation=0,
        requests=[_request_metadata("second", 8, 2, 0, hashes)],
        finished_requests={"first": []},
    )
    output = _process_output(worker, second, logical[8:], ["second"], np.array([0]))
    assert output is not None
    np.testing.assert_array_equal(output["second"].rows, logical)
    worker.close()


def test_worker_matches_kv_eviction_order_after_requests_finish():
    worker = _make_worker(2, max_store_blocks=3)
    hashes = [bytes([value]) * 32 for value in range(5)]
    blocks = [
        np.full((_BLOCK_SIZE, *_SHAPE), value, dtype=_DTYPE) for value in range(5)
    ]

    def publish_and_finish(request_id, block_indexes):
        rows = np.concatenate([blocks[index] for index in block_indexes])
        step = _metadata(
            0,
            [
                _request_metadata(
                    request_id,
                    0,
                    len(rows),
                    0,
                    [hashes[index] for index in block_indexes],
                )
            ],
            {},
        )
        _process_output(worker, step, rows, [request_id], np.array([0]))
        _begin_step(worker, _metadata(0, [], {request_id: []}))

    publish_and_finish("s", [0])
    publish_and_finish("ab", [1, 2])
    publish_and_finish("c", [3])
    publish_and_finish("d", [4])

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([hashes[1], hashes[3], hashes[4]], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        np.concatenate([blocks[1], blocks[3], blocks[4]]),
    )
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([hashes[2]], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        )
    worker.close()


def test_worker_rejects_unbacked_capture_gap():
    worker = _make_worker(1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = _metadata(
        0,
        [_request_metadata("request", 0, 1, 0, [])],
        {},
    )
    _process_output(
        worker,
        first,
        logical[:1],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    second = _metadata(
        0,
        [_request_metadata("request", 2, 1, 0, [])],
        {},
    )

    with pytest.raises(AssertionError, match="unbacked token gap"):
        _process_output(
            worker,
            second,
            logical[2:3],
            ["request"],
            np.array([0]),
            np.zeros(1, dtype=np.int32),
        )
    worker.close()


def test_worker_replays_rejected_speculative_gap():
    worker = _make_worker(1)
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    first = _metadata(
        0,
        [_request_metadata("request", 0, 5, 0, [])],
        {},
    )
    first_output = _process_output(
        worker, first, logical[:5], ["request"], np.array([4])
    )

    queued = _metadata(
        0,
        [_request_metadata("request", 5, 4, 0, [])],
        {},
    )
    queued_output = _process_output(
        worker,
        queued,
        logical[1:],
        ["request"],
        np.array([3]),
    )

    rolled_back = _metadata(
        0,
        [_request_metadata("request", 6, 3, 0, [b"a" * 32])],
        {},
    )

    rolled_back_output = _process_output(
        worker,
        rolled_back,
        logical[2:],
        ["request"],
        np.array([0]),
    )

    np.testing.assert_array_equal(first_output["request"].rows, logical[:1])
    assert queued_output["request"].token_start == 1
    np.testing.assert_array_equal(
        queued_output["request"].rows,
        logical[1:2],
    )
    np.testing.assert_array_equal(
        rolled_back_output["request"].rows,
        logical[2:],
    )
    assert worker._requests["request"].capture_cursor == len(logical)
    worker.close()


def test_worker_rejects_overlapping_capture_rows():
    worker = _make_worker(1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = _metadata(
        0,
        [_request_metadata("request", 0, 3, 0, [])],
        {},
    )
    _process_output(
        worker,
        first,
        logical[:3],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    stale_overlap = logical[2:].copy()
    stale_overlap[0] += 100
    second = _metadata(
        0,
        [_request_metadata("request", 2, 2, 0, [b"a" * 32])],
        {},
    )
    with pytest.raises(AssertionError, match="capture moved backwards"):
        _process_output(
            worker,
            second,
            stale_overlap,
            ["request"],
            np.array([0]),
        )
    worker.close()


def test_worker_publishes_entire_batch_before_materializing_prefix():
    worker = _make_worker(2)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)
    metadata = _metadata(
        generation=0,
        requests=[
            _request_metadata("consumer", 4, 4, 0, hashes),
            _request_metadata("producer", 0, 4, 0, hashes),
        ],
        finished_requests={},
    )

    output = _process_output(
        worker,
        metadata,
        np.concatenate((logical[4:], logical[:4])),
        ["consumer", "producer"],
        np.array([0, 0]),
    )

    assert output is not None
    np.testing.assert_array_equal(output["consumer"].rows, logical)
    np.testing.assert_array_equal(output["producer"].rows, logical[:4])
    worker.close()


def test_worker_defers_full_block_until_kv_hash_arrives():
    worker = _make_worker(1)
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    first = _metadata(
        0,
        [_request_metadata("request", 0, 4, 0, [])],
        {},
    )

    output = _process_output(
        worker,
        first,
        logical[:4],
        ["request"],
        np.array([0]),
    )

    assert output is not None
    np.testing.assert_array_equal(output["request"].rows, logical[:4])
    assert worker._requests["request"].pending_blocks[0][0] == 0

    block_hash = b"a" * 32
    second = _metadata(
        0,
        [_request_metadata("request", 4, 1, 4, [block_hash])],
        {},
    )
    output = _process_output(worker, second, logical[4:], ["request"], np.array([0]))

    assert output is not None
    np.testing.assert_array_equal(output["request"].rows, logical[4:])
    assert not worker._requests["request"].pending_blocks
    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[:4],
    )
    worker.close()


def test_worker_releases_newly_keyed_blocks_before_capture():
    worker = _make_worker(1, max_num_batched_tokens=16)
    connector = _make_connector()
    block_hashes = [b"a" * 32]
    request = _scheduler_request("request", block_hashes, num_tokens=100)
    logical = np.arange(52 * 3 * 2, dtype=np.uint8).reshape(52, 3, 2)

    for token_start, num_tokens in [(0, 4), (4, 16), (20, 16)]:
        request.num_computed_tokens = token_start
        metadata = connector.build_connector_meta(
            _step_output([request.request_id], [token_start], [num_tokens]),
            {request.request_id: request},
        )
        _process_output(
            worker,
            metadata,
            logical[token_start : token_start + num_tokens],
            [request.request_id],
            np.array([0]),
            token_starts=(token_start,),
            num_tokens=(num_tokens,),
        )

    block_hashes.extend(bytes([value]) * 32 for value in range(1, 5))
    request.num_computed_tokens = 36
    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [36], [16]),
        {request.request_id: request},
    )
    _process_output(
        worker,
        metadata,
        logical[36:52],
        [request.request_id],
        np.array([0]),
        token_starts=(36,),
        num_tokens=(16,),
    )

    state = worker._requests[request.request_id]
    assert [start for start, _ in state.pending_blocks] == [
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
    ]
    worker.close()


def test_worker_publishes_pending_block_without_forward():
    worker = _make_worker(1)
    _begin_step(
        worker,
        _metadata(
            0,
            [_request_metadata("request", 0, 1, 0, [])],
            {},
        ),
    )
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(_BLOCK_SIZE, 3, 2)
    assert worker._buffer is not None
    worker._requests["request"].pending_blocks = [
        (0, worker._buffer.retain_block(logical))
    ]
    block_hash = b"a" * 32

    _begin_step(worker, _metadata(0, [], {}, {"request": [block_hash]}))

    assert not worker._requests["request"].pending_blocks
    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical,
    )
    worker.close()


def test_worker_releases_unscheduled_request_blocks_before_capture():
    worker = _make_worker(2, 20, max_concurrent_batches=1)
    logical = np.arange(48 * 3 * 2, dtype=np.uint8).reshape(48, 3, 2)
    prompt_hashes = [b"a" * 32, b"b" * 32]
    prompt = _metadata(
        0,
        [
            _request_metadata("first", 0, 4, 0, prompt_hashes[:1]),
            _request_metadata("second", 0, 4, 0, prompt_hashes[1:]),
        ],
        {},
    )
    _process_output(
        worker,
        prompt,
        logical[:8],
        ["first", "second"],
        np.zeros(2),
        np.zeros(2, dtype=np.int32),
    )

    first_decode = _metadata(
        0,
        [_request_metadata("first", 4, 20, 0, [])],
        {},
    )
    _process_output(
        worker,
        first_decode,
        logical[8:28],
        ["first"],
        np.zeros(1),
        np.zeros(1, dtype=np.int32),
    )
    assert len(worker._requests["first"].pending_blocks) == 5

    first_hashes = [bytes([value]) * 32 for value in range(2, 7)]
    second_decode = _metadata(
        0,
        [_request_metadata("second", 4, 20, 0, [])],
        {},
        {"first": first_hashes},
    )
    _process_output(
        worker,
        second_decode,
        logical[28:48],
        ["second"],
        np.zeros(1),
        np.zeros(1, dtype=np.int32),
    )

    assert not worker._requests["first"].pending_blocks
    assert len(worker._requests["second"].pending_blocks) == 5
    worker.close()


def test_worker_does_not_rematerialize_emitted_rows():
    worker = _make_worker(1)
    hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(8 * 3 * 2, dtype=np.uint8).reshape(8, 3, 2)

    first = _metadata(
        0,
        [_request_metadata("request", 0, 4, 0, hashes)],
        {},
    )
    output = _process_output(worker, first, logical[:4], ["request"], np.array([0]))
    assert output is not None

    assert worker._store is not None
    worker._store.get_concatenated = Mock(wraps=worker._store.get_concatenated)
    second = _metadata(
        0,
        # Async scheduling may build this before the scheduler consumes first.
        [_request_metadata("request", 4, 4, 0, [])],
        {},
    )
    output = _process_output(worker, second, logical[4:], ["request"], np.array([0]))

    assert output is not None
    np.testing.assert_array_equal(output["request"].rows, logical[4:])
    worker._store.get_concatenated.assert_not_called()
    worker.close()


def test_worker_resumes_from_scheduler_emit_boundary():
    worker = _make_worker(1)
    logical = np.arange(5 * 3 * 2, dtype=np.uint8).reshape(5, 3, 2)
    resumed = _metadata(
        0,
        [_request_metadata("request", 4, 1, 4, [b"a" * 32])],
        {},
    )

    output = _process_output(worker, resumed, logical[4:], ["request"], np.array([0]))

    assert output["request"].token_start == 4
    np.testing.assert_array_equal(output["request"].rows, logical[4:])
    assert worker._requests["request"].emit_cursor == 5
    worker.close()


def test_worker_emits_mid_block_chunked_prefill():
    worker = _make_worker(1)
    logical = np.arange(3 * 3 * 2, dtype=np.uint8).reshape(3, 3, 2)

    first = _metadata(
        0,
        [_request_metadata("request", 0, 2, 0, [])],
        {},
    )
    output = _process_output(
        worker,
        first,
        logical[:2],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    assert output is not None and not output

    second = _metadata(
        0,
        [_request_metadata("request", 2, 1, 0, [])],
        {},
    )
    output = _process_output(worker, second, logical[2:], ["request"], np.array([0]))

    np.testing.assert_array_equal(output["request"].rows, logical)
    worker.close()


def test_worker_emits_published_block_and_mid_block_tail():
    worker = _make_worker(1)
    logical = np.arange(7 * 3 * 2, dtype=np.uint8).reshape(7, 3, 2)

    first = _metadata(
        0,
        [_request_metadata("request", 0, 6, 0, [b"a" * 32])],
        {},
    )
    output = _process_output(
        worker,
        first,
        logical[:6],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    assert output is not None and not output

    second = _metadata(
        0,
        [_request_metadata("request", 6, 1, 0, [])],
        {},
    )
    output = _process_output(worker, second, logical[6:], ["request"], np.array([0]))

    np.testing.assert_array_equal(output["request"].rows, logical)
    worker.close()


def test_worker_output_does_not_alias_released_tail_buffer():
    worker = _make_worker(1)
    logical = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    first = _metadata(
        0,
        [_request_metadata("request", 0, 2, 0, [b"a" * 32])],
        {},
    )
    _process_output(
        worker,
        first,
        logical[:2],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )
    second = _metadata(
        0,
        [_request_metadata("request", 2, 2, 0, [])],
        {},
    )

    output = _process_output(worker, second, logical[2:], ["request"], np.array([0]))
    emitted = output["request"].rows
    expected = logical.copy()

    replacement = np.full_like(logical, 99)
    reuse = _metadata(
        0,
        [_request_metadata("reuse", 0, 4, 0, [b"b" * 32])],
        {},
    )
    _process_output(
        worker,
        reuse,
        replacement,
        ["reuse"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
    )

    np.testing.assert_array_equal(emitted, expected)
    worker.close()


def test_worker_merges_block_hash_deltas():
    worker = _make_worker(1)
    _begin_step(
        worker,
        _metadata(
            0,
            [_request_metadata("request", 0, 1, 0, [b"a" * 32, b"b" * 32])],
            {},
        ),
    )
    _begin_step(
        worker,
        _metadata(
            0,
            [
                _request_metadata(
                    "request",
                    1,
                    1,
                    0,
                    [b"c" * 32],
                )
            ],
            {},
        ),
    )

    assert worker._requests["request"].aux_output_keys == [
        "vllm-artifact/0/" + (b"a" * 32).hex(),
        "vllm-artifact/0/" + (b"b" * 32).hex(),
        "vllm-artifact/0/" + (b"c" * 32).hex(),
    ]
    worker.close()


def test_worker_finish_publishes_keyed_block_and_discards_request_state():
    worker = _make_worker(1)
    logical = np.arange(2 * _BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(
        2 * _BLOCK_SIZE, 3, 2
    )
    running = _metadata(
        0,
        [_request_metadata("request", 0, len(logical), 0, [])],
        {},
    )
    _process_output(
        worker,
        running,
        logical,
        ["request"],
        np.array([0]),
        np.array([0]),
    )
    block_hash = b"a" * 32

    _begin_step(
        worker,
        _metadata(
            0,
            [],
            {"request": [block_hash]},
        ),
    )

    assert "request" not in worker._requests
    assert worker._buffer is not None
    assert "request" not in worker._buffer._requests
    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[:_BLOCK_SIZE],
    )
    worker.close()


def test_worker_fails_when_cached_aux_output_is_missing():
    worker = _make_worker(1)
    worker._generation = 0
    metadata = _metadata(
        0,
        [
            _request_metadata(
                "request",
                8,
                1,
                0,
                [b"a" * 32, b"b" * 32],
            )
        ],
        {},
    )

    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        _process_output(
            worker,
            metadata,
            np.zeros((1, *_SHAPE), dtype=_DTYPE),
            ["request"],
            np.array([0]),
        )
    worker.close()


def test_store_rejects_oversized_batch_without_partial_write():
    store = _make_store(max_bytes=6, object_nbytes=3)
    store.put([BlockObject("retained", b"rrr")])

    with pytest.raises(BlockObjectStoreError, match="cannot retain"):
        store.put(
            [
                BlockObject("first", b"111"),
                BlockObject("second", b"222"),
                BlockObject("third", b"333"),
            ]
        )

    assert store.get_concatenated(["retained"]) == b"rrr"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["first"])
    store.close()


def test_store_lru_and_immutable_put():
    store = _make_store(max_bytes=8)
    store.put([BlockObject("first", b"1111"), BlockObject("second", b"2222")])
    assert store.get_concatenated(["first"]) == b"1111"

    store.put([BlockObject("first", b"1111"), BlockObject("third", b"3333")])

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(BlockObjectStoreError, match="Increase aux_output_config"):
        store.get_concatenated(["second"])
    store.close()


def test_store_does_not_evict_referenced_aux_outputs():
    store = _make_store(max_bytes=8)
    store.put([BlockObject("first", b"1111"), BlockObject("second", b"2222")])
    store.put([], retain_keys=["first"])
    store.put([], retain_keys=["first"], release_keys=["first"])

    store.put([BlockObject("third", b"3333")])

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["second"])
    store.put([], release_keys=["first"])
    store.close()


def test_store_terminal_put_preserves_tail_first_eviction_order():
    store = _make_store(max_bytes=8)
    keys = ["first", "second", "third"]
    store.put(
        [BlockObject("first", b"1111"), BlockObject("second", b"2222")],
        retain_keys=keys,
    )

    store.put(
        [BlockObject("third", b"3333")],
        release_keys=reversed(keys),
    )

    assert store.get_concatenated(keys[:2]) == b"11112222"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["third"])
    store.close()


def test_store_terminal_put_orders_survivors_for_later_eviction():
    store = _make_store(max_bytes=12)
    keys = ["first", "second", "third"]
    store.put(
        [BlockObject("first", b"1111"), BlockObject("second", b"2222")],
        retain_keys=keys,
    )

    store.put(
        [BlockObject("third", b"3333")],
        release_keys=reversed(keys),
    )
    store.put([BlockObject("fourth", b"4444")])

    assert store.get_concatenated(["first", "second", "fourth"]) == b"111122224444"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["third"])
    store.close()


def test_store_terminal_put_keeps_a_shared_incoming_key():
    store = _make_store(max_bytes=8)
    keys = ["first", "second", "third"]
    store.put(
        [BlockObject("first", b"1111"), BlockObject("second", b"2222")],
        retain_keys=(*keys, "third"),
    )

    store.put(
        [BlockObject("third", b"3333")],
        release_keys=reversed(keys),
    )

    assert store.get_concatenated(["first", "third"]) == b"11113333"
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        store.get_concatenated(["second"])
    store.put([], release_keys=["third"])
    store.close()


def test_store_rejects_access_after_close():
    store = _make_store(max_bytes=8)
    store.close()

    with pytest.raises(RuntimeError, match="closed"):
        store.put([BlockObject("first", b"1111")])
    with pytest.raises(RuntimeError, match="closed"):
        store.get_concatenated(["first"])


def test_store_reuses_evicted_slot_without_moving_live_objects():
    store = _make_store(max_bytes=12)
    store.put(
        [
            BlockObject("first", b"1111"),
            BlockObject("second", b"2222"),
            BlockObject("third", b"3333"),
        ]
    )
    assert store.get_concatenated(["second"]) == b"2222"
    store.put([BlockObject("fourth", b"4444")])

    assert store.get_concatenated(["second", "third", "fourth"]) == b"222233334444"
    store.close()


def _scheduler_request(
    request_id: str,
    block_hashes: list[bytes],
    *,
    num_tokens: int = 10,
    num_output_tokens: int = 1,
    prompt_start: int = 0,
):
    return _SchedulerRequest(
        request_id=request_id,
        block_hashes=block_hashes,
        num_tokens=num_tokens,
        num_output_tokens=num_output_tokens,
        num_computed_tokens=num_tokens,
        sampling_params=SimpleNamespace(routed_experts_prompt_start=prompt_start),
    )


def _step_output(
    request_ids: list[str], token_starts: list[int], token_counts: list[int]
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=request_ids,
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in request_ids],
        all_token_ids={},
        new_block_ids=[None for _ in request_ids],
        num_computed_tokens=token_starts,
        num_output_tokens=[0 for _ in request_ids],
    )
    output.num_scheduled_tokens = dict(zip(request_ids, token_counts, strict=True))
    output.total_num_scheduled_tokens = sum(token_counts)
    return output


def test_scheduler_connector_builds_worker_metadata_and_forwards_output():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    request.num_computed_tokens = 0
    scheduler_output = _step_output([request.request_id], [0], [4])

    metadata = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )

    assert metadata.generation == 0
    assert metadata.requests == {request.request_id: 4}
    assert list(metadata.block_hashes[request.request_id]) == [b"a" * 32]

    routing = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    output = {"request": AuxRequestOutput(0, routing)}
    request.num_computed_tokens = 4
    np.testing.assert_array_equal(connector.take_output(request, output), routing)


def test_scheduler_starts_worker_output_at_requested_prompt_token():
    connector = _make_connector()
    request = _scheduler_request(
        "request",
        [b"a" * 32],
        num_tokens=4,
        num_output_tokens=0,
        prompt_start=2,
    )

    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )

    assert metadata.requests == {request.request_id: 2}

    # A P/D prefiller cuts its prompt short of the last token, which can pass
    # the requested start; the rows then start at the end of the cut prompt.
    connector = _make_connector()
    request = _scheduler_request(
        "cut", [b"a" * 32], num_tokens=3, num_output_tokens=0, prompt_start=4
    )
    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [0], [3]),
        {request.request_id: request},
    )
    assert metadata.requests == {request.request_id: 3}


def test_scheduler_marks_only_opted_in_logprob_requests():
    connector = AuxOutputSchedulerConnector(
        enable_routed_experts=False,
        enable_logprobs=True,
        enable_prompt_logprobs=True,
    )
    opted_in = _scheduler_request("opted-in", [b"a" * 32])
    opted_in.sampling_params = SimpleNamespace(
        routed_experts_prompt_start=0,
        extra_args={"aux_output_replay": True},
        num_logprobs=2,
        prompt_logprobs=2,
        logprob_token_ids=None,
        prompt_logprob_token_ids=None,
    )
    regular = _scheduler_request("regular", [b"b" * 32])
    regular.sampling_params = SimpleNamespace(
        routed_experts_prompt_start=0,
        extra_args=None,
        num_logprobs=2,
        prompt_logprobs=2,
        logprob_token_ids=None,
        prompt_logprob_token_ids=None,
    )

    metadata = connector.build_connector_meta(
        _step_output([opted_in.request_id, regular.request_id], [0, 0], [4, 4]),
        {opted_in.request_id: opted_in, regular.request_id: regular},
    )

    assert metadata.logprobs.keys() == {opted_in.request_id}
    assert metadata.prompt_logprobs.keys() == {opted_in.request_id}
    assert metadata.block_hashes.keys() == {opted_in.request_id}
    assert metadata.logprob_block_hashes.keys() == {opted_in.request_id}


def test_scheduler_rejects_missing_prompt_logprob_artifact():
    connector = AuxOutputSchedulerConnector(
        enable_routed_experts=False,
        enable_prompt_logprobs=True,
    )
    request = _scheduler_request("request", [b"a" * 32])
    request.sampling_params = SimpleNamespace(
        routed_experts_prompt_start=0,
        extra_args={"aux_output_replay": True},
        num_logprobs=2,
        prompt_logprobs=2,
        logprob_token_ids=None,
        prompt_logprob_token_ids=None,
    )

    with pytest.raises(RuntimeError, match="prompt logprobs artifact is missing"):
        connector.take_prompt_logprobs(
            request, {request.request_id: AuxRequestOutput(0)}
        )


def test_prompt_logprob_delivery_latch_follows_request_lifetime():
    connector = AuxOutputSchedulerConnector(
        enable_routed_experts=False,
        enable_prompt_logprobs=True,
    )

    def make_request():
        request = _scheduler_request("reused-id", [b"a" * 32])
        request.sampling_params = SimpleNamespace(
            routed_experts_prompt_start=0,
            extra_args={"aux_output_replay": True},
            num_logprobs=None,
            prompt_logprobs=2,
            logprob_token_ids=None,
            prompt_logprob_token_ids=None,
        )
        return request

    artifact = LogprobsTensors(
        torch.empty((0, 2), dtype=torch.int32),
        torch.empty((0, 2), dtype=torch.float32),
        torch.empty(0, dtype=torch.int32),
    )
    first = make_request()
    output = {first.request_id: AuxRequestOutput(0, prompt_logprobs=artifact)}

    assert connector.take_prompt_logprobs(first, output) is artifact
    assert connector.take_prompt_logprobs(first, None) is None
    connector.release_request(first)
    assert connector.take_prompt_logprobs(first, output) is artifact

    del first
    gc.collect()

    second = make_request()
    assert connector.take_prompt_logprobs(
        second,
        {second.request_id: AuxRequestOutput(0, prompt_logprobs=artifact)},
    ) is artifact


def test_scheduler_takes_generated_logprobs_without_routed_experts():
    connector = AuxOutputSchedulerConnector(
        enable_routed_experts=False,
        enable_logprobs=True,
    )
    request = _scheduler_request("request", [b"a" * 32])
    request.sampling_params = SimpleNamespace(
        routed_experts_prompt_start=0,
        extra_args={"aux_output_replay": True},
        num_logprobs=2,
        prompt_logprobs=None,
        logprob_token_ids=None,
        prompt_logprob_token_ids=None,
    )
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )

    rows = _logprob_rows(4, 1)
    value = LogprobsLists(rows.token_ids, rows.values, rows.ranks)
    output = {request.request_id: AuxRequestOutput(0, logprobs=value)}

    assert connector.take_logprobs(request, output) is value


def test_logprob_artifact_identity_includes_boundary_token():
    connector = AuxOutputSchedulerConnector(
        enable_routed_experts=False,
        enable_logprobs=True,
        hash_block_size=_BLOCK_SIZE,
    )
    requests = []
    for request_id, boundary_token_id in (("first", 10), ("second", 20)):
        request = _scheduler_request(request_id, [b"a" * 32], num_tokens=5)
        request.token_ids = [1, 2, 3, 4, boundary_token_id]
        request.sampling_params = SimpleNamespace(
            routed_experts_prompt_start=0,
            extra_args={"aux_output_replay": True},
            num_logprobs=2,
            prompt_logprobs=None,
            logprob_token_ids=None,
            prompt_logprob_token_ids=None,
        )
        requests.append(request)

    metadata = connector.build_connector_meta(
        _step_output([request.request_id for request in requests], [0, 0], [5, 5]),
        {request.request_id: request for request in requests},
    )

    worker = _make_logprob_worker()
    first_key = worker._logprob_keys(
        metadata.logprob_block_hashes["first"],
        metadata.logprob_boundary_token_ids["first"],
        metadata.logprobs["first"],
    )
    second_key = worker._logprob_keys(
        metadata.logprob_block_hashes["second"],
        metadata.logprob_boundary_token_ids["second"],
        metadata.logprobs["second"],
    )

    assert first_key != second_key
    worker.close()


def test_scheduler_sends_pending_logprob_hash_without_boundary_token():
    connector = AuxOutputSchedulerConnector(
        enable_routed_experts=False,
        enable_logprobs=True,
        hash_block_size=_BLOCK_SIZE,
    )
    request = _scheduler_request("request", [b"a" * 32], num_tokens=_BLOCK_SIZE)
    request.token_ids = [1, 2, 3, 4]
    request.sampling_params = SimpleNamespace(
        routed_experts_prompt_start=0,
        extra_args={"aux_output_replay": True},
        num_logprobs=2,
        prompt_logprobs=None,
        logprob_token_ids=None,
        prompt_logprob_token_ids=None,
    )

    first = connector.build_connector_meta(
        _step_output([request.request_id], [0], [_BLOCK_SIZE]),
        {request.request_id: request},
    )
    assert list(first.logprob_block_hashes[request.request_id]) == [b"a" * 32]
    assert first.logprob_boundary_token_ids[request.request_id] == (None,)

    request.num_tokens += 1
    request.token_ids.append(99)
    second = connector.build_connector_meta(
        _step_output([request.request_id], [_BLOCK_SIZE], [1]),
        {request.request_id: request},
    )

    assert request.request_id not in second.logprob_block_hashes


def test_scheduler_connector_preserves_request_finish_order():
    connector = _make_connector()
    first = _scheduler_request("first", [b"a" * 32])
    second = _scheduler_request("second", [b"b" * 32])
    connector.build_connector_meta(
        _step_output([first.request_id, second.request_id], [0, 0], [0, 0]),
        {first.request_id: first, second.request_id: second},
    )
    connector.request_finished(second)
    connector.request_finished(first)

    metadata = connector.build_connector_meta(_step_output([], [], []), {})

    assert metadata.finished_requests == (second.request_id, first.request_id)
    assert not metadata.block_hashes


def test_worker_preemption_terminal_keeps_inflight_hashes_for_publish():
    worker = _make_worker(1)
    block_hashes = [b"a" * 32, b"b" * 32]
    logical = np.arange(2 * _BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(
        2 * _BLOCK_SIZE, 3, 2
    )
    first = _metadata(
        0,
        [
            _request_metadata(
                "request",
                0,
                _BLOCK_SIZE,
                0,
                block_hashes[:1],
            )
        ],
        {},
    )
    _process_output(
        worker,
        first,
        logical[:_BLOCK_SIZE],
        ["request"],
        np.array([0]),
    )
    inflight = _metadata(
        0,
        [
            _request_metadata(
                "request",
                _BLOCK_SIZE,
                _BLOCK_SIZE,
                0,
                block_hashes[1:],
            )
        ],
        {},
    )
    _process_output(
        worker,
        inflight,
        logical[_BLOCK_SIZE:],
        ["request"],
        np.array([0]),
        np.zeros(1, dtype=np.int32),
        token_starts=(_BLOCK_SIZE,),
        num_tokens=(_BLOCK_SIZE,),
    )
    _begin_step(
        worker,
        _metadata(
            0,
            [],
            {"request": []},
        ),
    )

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys(block_hashes[:1], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[:_BLOCK_SIZE],
    )
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys(block_hashes[1:], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical[_BLOCK_SIZE:],
    )
    worker.close()


def test_worker_releases_terminal_pins_before_same_step_publish():
    worker = _make_worker(2, max_store_blocks=1)
    victim_hash = b"a" * 32
    producer_hash = b"b" * 32
    victim_rows = np.zeros((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE)
    producer_rows = np.ones((_BLOCK_SIZE, *_SHAPE), dtype=_DTYPE)
    running = _metadata(
        0,
        [
            _request_metadata("victim", 0, _BLOCK_SIZE, 0, [victim_hash]),
            _request_metadata("producer", 0, _BLOCK_SIZE, 0, []),
        ],
        {},
    )
    _process_output(
        worker,
        running,
        np.concatenate((victim_rows, producer_rows)),
        ["victim", "producer"],
        np.zeros(2, dtype=np.int32),
        np.zeros(2, dtype=np.int32),
    )

    _begin_step(
        worker,
        _metadata(
            0,
            [],
            {"victim": []},
            {"producer": [producer_hash]},
        ),
    )

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([producer_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        producer_rows,
    )
    with pytest.raises(BlockObjectStoreError, match="does not exist"):
        worker._store.get_concatenated(routed_experts_keys([victim_hash], "0"))
    worker.close()


def test_terminal_request_publishes_hash_discovered_after_last_schedule():
    connector = _make_connector()
    worker = _make_worker(1)
    block_hashes: list[bytes] = []
    request = _scheduler_request("request", block_hashes, num_tokens=5)
    request.num_computed_tokens = 0
    metadata = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    logical = np.arange(_BLOCK_SIZE * 3 * 2, dtype=np.uint8).reshape(_BLOCK_SIZE, 3, 2)

    _process_output(
        worker,
        metadata,
        logical,
        [request.request_id],
        np.array([0]),
        token_starts=(0,),
        num_tokens=(_BLOCK_SIZE,),
    )
    assert worker._requests[request.request_id].pending_blocks

    block_hash = b"a" * 32
    block_hashes.append(block_hash)
    request.num_computed_tokens = _BLOCK_SIZE
    connector.request_finished(request)
    cleanup = connector.build_connector_meta(_step_output([], [], []), {})
    worker.begin_step(cleanup)

    assert worker._store is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(
            worker._store,
            routed_experts_keys([block_hash], "0"),
            shape_per_token=_SHAPE,
            dtype=_DTYPE,
        ),
        logical,
    )
    worker.close()


def test_scheduler_connector_recreates_preempted_request_state():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    connector.request_finished(request)
    cleanup = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )
    assert request.request_id in cleanup.finished_requests
    assert request.request_id not in cleanup.block_hashes

    resumed = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    assert list(resumed.block_hashes[request.request_id]) == [b"a" * 32]


def test_scheduler_consumes_ordered_stale_aux_outputs():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=6)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    first_rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
    second_rows = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2) + 40
    first = {"request": AuxRequestOutput(0, first_rows)}
    second = {"request": AuxRequestOutput(4, second_rows)}
    connector.request_finished(request)

    request.num_tokens = 5
    np.testing.assert_array_equal(connector.take_output(request, first), first_rows)
    request.num_tokens = 6
    np.testing.assert_array_equal(
        connector.take_output(request, second), second_rows[:1]
    )


def test_scheduler_rejects_stale_output_without_aux_outputs():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=1)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [1]),
        {request.request_id: request},
    )

    with pytest.raises(
        RuntimeError, match="auxiliary output worker output is missing"
    ):
        connector.take_output(
            request,
            {},
        )


def test_scheduler_rejects_missing_accepted_aux_output_rows():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=2)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [1]),
        {request.request_id: request},
    )
    output = {
        "request": AuxRequestOutput(
            0,
            np.empty((0, *_SHAPE), dtype=_DTYPE),
        )
    }

    with pytest.raises(RuntimeError, match="invalid token range"):
        connector.take_output(request, output)


def test_scheduler_rejects_aux_output_past_finished_request():
    connector = _make_connector()
    request = _scheduler_request("request", [], num_tokens=1)
    request.finished = True
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [1]),
        {request.request_id: request},
    )
    output = {
        "request": AuxRequestOutput(
            1,
            np.empty((0, *_SHAPE), dtype=_DTYPE),
        )
    }

    with pytest.raises(
        RuntimeError, match="finished auxiliary output has no accepted"
    ):
        connector.take_output(request, output)


@pytest.mark.parametrize("chunk_size", [2, 4])
@pytest.mark.parametrize("max_tokens", [1, 2])
def test_prompt_end_offset_through_worker_and_scheduler(chunk_size, max_tokens):
    """Omit prompt R3 without losing the empty first output or later decode rows."""
    connector = _make_connector()
    worker = _make_worker(1)
    request = _scheduler_request(
        "request", [b"a" * 32], num_tokens=4, num_output_tokens=0, prompt_start=4
    )
    rows = np.arange(5 * 3 * 2, dtype=_DTYPE).reshape(5, *_SHAPE)
    try:
        for start in range(0, 4, chunk_size):
            metadata = connector.build_connector_meta(
                _step_output([request.request_id], [start], [chunk_size]),
                {request.request_id: request},
            )
            output = _process_output(
                worker,
                metadata,
                rows[start : start + chunk_size],
                [request.request_id],
                np.array([0]),
                num_sampled=np.array([int(start + chunk_size == 4)]),
                token_starts=(start,),
                num_tokens=(chunk_size,),
            )
            if start + chunk_size < 4:
                assert output == {}
        request.num_tokens = 5
        request.num_output_tokens = 1
        request.finished = max_tokens == 1
        result = connector.take_output(request, output)
        np.testing.assert_array_equal(result, rows[:0])

        if max_tokens == 2:
            metadata = connector.build_connector_meta(
                _step_output([request.request_id], [4], [1]),
                {request.request_id: request},
            )
            output = _process_output(
                worker,
                metadata,
                rows[4:],
                [request.request_id],
                np.array([0]),
                token_starts=(4,),
                num_tokens=(1,),
            )
            request.num_tokens = 6
            request.finished = True
            np.testing.assert_array_equal(
                connector.take_output(request, output), rows[4:]
            )
    finally:
        worker.close()


def test_scheduler_connector_sends_each_block_hash_once():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32, b"b" * 32], num_tokens=12)
    scheduler_output = _step_output([request.request_id], [0], [4])

    first = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )
    request.block_hashes.append(b"c" * 32)
    second = connector.build_connector_meta(
        _step_output([request.request_id], [4], [4]),
        {request.request_id: request},
    )
    third = connector.build_connector_meta(
        _step_output([request.request_id], [8], [4]),
        {request.request_id: request},
    )

    assert list(first.block_hashes[request.request_id]) == [b"a" * 32, b"b" * 32]
    assert list(second.block_hashes[request.request_id]) == [b"c" * 32]
    assert request.request_id not in third.block_hashes


def test_scheduler_connector_sends_unscheduled_hash_update():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=4)
    connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )

    request.block_hashes.append(b"b" * 32)
    request.num_tokens = 9
    request.num_computed_tokens = 8
    metadata = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )

    assert not metadata.requests
    assert list(metadata.block_hashes[request.request_id]) == [b"b" * 32]


def test_scheduler_connector_reset_derives_emit_start_from_request():
    connector = _make_connector()
    request = _scheduler_request("request", [b"a" * 32], num_tokens=5)
    request.num_computed_tokens = 0
    before_reset = connector.build_connector_meta(
        _step_output([request.request_id], [0], [4]),
        {request.request_id: request},
    )
    assert list(before_reset.block_hashes[request.request_id]) == [b"a" * 32]

    connector.reset()
    empty = connector.build_connector_meta(
        _step_output([], [], []),
        {request.request_id: request},
    )
    assert not empty.block_hashes
    request.num_computed_tokens = 4
    scheduler_output = _step_output([request.request_id], [4], [1])
    metadata = connector.build_connector_meta(
        scheduler_output, {request.request_id: request}
    )

    assert metadata.generation == 1
    assert metadata.requests[request.request_id] == 4
    assert list(metadata.block_hashes[request.request_id]) == [b"a" * 32]
