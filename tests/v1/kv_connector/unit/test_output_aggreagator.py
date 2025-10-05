# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import Future
from typing import Optional

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class DummyModelRunnerOutput(ModelRunnerOutput):
    def __init__(
        self,
        finished_sending: Optional[set[str]] = None,
        finished_recving: Optional[set[str]] = None,
        invalid_block_ids: Optional[set[int]] = None,
    ):
        self.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
        )

    def __repr__(self):
        return (
            f"DummyModelRunnerOutput("
            f"finished_sending={self.kv_connector_output.finished_sending},"
            f"finished_recving={self.kv_connector_output.finished_recving})"
            f"invalid_block_ids={self.kv_connector_output.invalid_block_ids})"
        )


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(world_size=2)

    output1 = DummyModelRunnerOutput()
    output2 = DummyModelRunnerOutput()

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert not aggregated.invalid_block_ids

    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, finished_recving={"req2"}
    )
    output2 = DummyModelRunnerOutput(invalid_block_ids={1})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {1}

    output1 = DummyModelRunnerOutput(invalid_block_ids={2})
    output2 = DummyModelRunnerOutput(finished_sending={"req1"})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {"req1"}
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {2}

    output1 = DummyModelRunnerOutput(invalid_block_ids={3, 4})
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, invalid_block_ids={4, 5}
    )

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {"req2"}
    assert aggregated.invalid_block_ids == {3, 4, 5}


def test_async_aggregate_workers_output():
    aggregator = KVOutputAggregator(world_size=2)

    future1: Future[DummyModelRunnerOutput] = Future()
    future2: Future[DummyModelRunnerOutput] = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput()
    output2 = DummyModelRunnerOutput()
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert not aggregated.invalid_block_ids

    future1 = Future()
    future2 = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, finished_recving={"req2"}
    )
    output2 = DummyModelRunnerOutput(invalid_block_ids={1})
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {1}

    future1 = Future()
    future2 = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput(invalid_block_ids={2})
    output2 = DummyModelRunnerOutput(finished_sending={"req1"})
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {"req1"}
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {2}

    future1 = Future()
    future2 = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput(invalid_block_ids={3, 4})
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, invalid_block_ids={4, 5}
    )
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {"req2"}
    assert aggregated.invalid_block_ids == {3, 4, 5}
