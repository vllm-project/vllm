# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import Future
from typing import Optional

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


class DummyModelRunnerOutput(ModelRunnerOutput):

    def __init__(self,
                 finished_sending: Optional[set[str]] = None,
                 finished_recving: Optional[set[str]] = None):
        self.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        )

    def __repr__(self):
        return (
            f"DummyModelRunnerOutput("
            f"finished_sending={self.kv_connector_output.finished_sending},"
            f"finished_recving={self.kv_connector_output.finished_recving})")


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(world_size=2)

    output1 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'})
    output2 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving=None)

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {'req1'}
    assert aggregated.finished_recving is None

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {'req2'}


def test_async_aggregate_workers_output():
    aggregator = KVOutputAggregator(world_size=2)

    future1: Future[DummyModelRunnerOutput] = Future()
    future2: Future[DummyModelRunnerOutput] = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'})
    output2 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None

    future1 = Future()
    future2 = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving=None)
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {'req1'}
    assert aggregated.finished_recving is None

    future1 = Future()
    future2 = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'})
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {'req2'}
