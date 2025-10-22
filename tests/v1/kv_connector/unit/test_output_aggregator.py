# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import Future

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class DummyModelRunnerOutput(ModelRunnerOutput):
    def __init__(
        self,
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
        invalid_block_ids: set[int] | None = None,
        expected_finished_count: int = 0,
    ):
        self.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
            expected_finished_count=expected_finished_count,
        )

    def __repr__(self):
        return (
            f"DummyModelRunnerOutput("
            f"finished_sending={self.kv_connector_output.finished_sending},"
            f"finished_recving={self.kv_connector_output.finished_recving})"
            f"invalid_block_ids={self.kv_connector_output.invalid_block_ids})"
        )


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(expected_finished_count=2)

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
    aggregator = KVOutputAggregator(expected_finished_count=2)

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


def test_aggregate_workers_output_with_expected_finished_count():
    # We create the aggregator expecting to collect from 4 workers
    aggregator = KVOutputAggregator(expected_finished_count=4)
    assert aggregator._expected_finished_count == 4
    # Some request with default expected finished requests
    output1 = DummyModelRunnerOutput(finished_sending={"req1"})
    aggregated = aggregator.aggregate([output1])
    # still expecting to collect from 4 workers
    assert aggregator._send_remaining_count["req1"] == 3
    assert not aggregated.kv_connector_output.finished_sending
    assert not aggregated.kv_connector_output.finished_recving

    # Workers discover and find that in this setup they only need to
    # collect from 2
    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, expected_finished_count=2
    )
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, expected_finished_count=2
    )
    output3 = DummyModelRunnerOutput(finished_recving={"req2"})
    # Req2 only needs 2 acks
    aggregated = aggregator.aggregate([output1, output2, output3])
    assert aggregated.kv_connector_output.expected_finished_count == 2

    assert not aggregated.kv_connector_output.finished_sending

    # Req2 is finished
    assert "req2" not in aggregator._recv_remaining_count
    assert aggregated.kv_connector_output.finished_recving == {"req2"}

    # Req1 is still waiting for 2 more acks (expected_finished_count has no effect)
    # NOTE: This is to showcase dynamic update. Workers are responsible for
    # ensuring "req1" termination in this case
    assert aggregator._send_remaining_count["req1"] == 2
