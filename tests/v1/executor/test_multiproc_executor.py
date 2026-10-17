# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from functools import partial
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vllm.distributed.aux_output_connector.connector import (
    AuxOutputAggregator,
    AuxRequestOutput,
)
from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import (
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
)


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


@pytest.mark.parametrize("stalled", [False, True])
@pytest.mark.parametrize(
    "method,result",
    [
        ("take_draft_token_ids", DraftTokenIds(["req"], [[1, 2, 3]])),
        ("execute_dummy_batch", None),
    ],
)
def test_model_rpc_uses_execute_model_timeout(monkeypatch, method, result, stalled):
    """A wedged model RPC must not block EngineCore forever. For
    execute_dummy_batch this is an idle DP rank stuck in a collective with a
    hung peer."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7")

    def dequeue(*, timeout=None):
        assert timeout is not None and 0 < timeout <= 7
        if stalled:
            raise TimeoutError
        return WorkerProc.ResponseStatus.SUCCESS, result

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.output_rank = 1
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda payload: None)
    executor.response_mqs = [None, SimpleNamespace(dequeue=dequeue)]

    if stalled:
        with pytest.raises(TimeoutError, match=f"{method} timed out"):
            getattr(executor, method)()
    else:
        assert getattr(executor, method)() is result


@pytest.mark.parametrize(
    ("use_kv", "use_ec"),
    [(False, False), (True, False), (False, True)],
    ids=["aux_only", "aux_kv", "aux_ec"],
)
def test_collective_rpc_aggregates_pp_leaders_with_other_connectors(use_kv, use_ec):
    """PP=3, TP=2 uses stage leaders for R3 and all ranks for KV/EC outputs."""
    stage_ranks = (0, 2, 4)
    layer_ranges = ((0, 2), (2, 5), (5, 7))
    logical = np.arange(2 * 7 * 2, dtype=np.uint16).reshape(2, 7, 2)
    outputs = []
    for rank in range(6):
        start, end = layer_ranges[rank // 2]
        outputs.append(
            ModelRunnerOutput(
                req_ids=["request"],
                req_id_to_index={"request": 0},
                sampled_token_ids=[[rank]],
                aux_output_connector_output=(
                    {"request": AuxRequestOutput(3, logical[:, start:end])}
                    if rank in stage_ranks
                    else None
                ),
                kv_connector_output=(
                    KVConnectorOutput(
                        finished_sending={"kv-transfer"}, invalid_block_ids={rank}
                    )
                    if use_kv
                    else None
                ),
                ec_connector_output=(
                    ECConnectorOutput(finished_sending={f"encoder-{rank}"})
                    if use_ec
                    else None
                ),
            )
        )

    received_ranks: list[int] = []

    def dequeue(rank, *, timeout=None):
        received_ranks.append(rank)
        return WorkerProc.ResponseStatus.SUCCESS, outputs[rank]

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.futures_queue = deque()
    requests: list[Any] = []
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=requests.append)
    executor.response_mqs = [
        SimpleNamespace(dequeue=partial(dequeue, rank)) for rank in range(6)
    ]
    output = executor.collective_rpc(
        "sample_tokens",
        unique_reply_rank=4,
        non_block=True,
        aux_output_aggregator=AuxOutputAggregator(stage_ranks),
        kv_output_aggregator=KVOutputAggregator(6) if use_kv else None,
        ec_output_aggregator=ECOutputAggregator() if use_ec else None,
    )
    assert received_ranks == []
    output = output.result()

    expected_reply_ranks = None if use_kv or use_ec else stage_ranks
    assert requests[0][3] == expected_reply_ranks
    assert received_ranks == list(range(6) if use_kv or use_ec else stage_ranks)
    assert output is outputs[4]
    assert output.req_ids == ["request"]
    assert output.sampled_token_ids == [[4]]
    assert output.aux_output_connector_output is not None
    result = output.aux_output_connector_output["request"]
    assert result.token_start == 3
    np.testing.assert_array_equal(result.rows, logical)
    if use_kv:
        assert output.kv_connector_output is not None
        assert output.kv_connector_output.finished_sending == {"kv-transfer"}
        assert output.kv_connector_output.invalid_block_ids == set(range(6))
    if use_ec:
        assert output.ec_connector_output is not None
        assert output.ec_connector_output.finished_sending == {
            f"encoder-{rank}" for rank in range(6)
        }
