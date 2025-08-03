# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import Future
from typing import Optional

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import ModelRunnerOutput


class DummyModelRunnerOutput(ModelRunnerOutput):

    def __init__(self,
                 finished_sending: Optional[set[str]] = None,
                 finished_recving: Optional[set[str]] = None,
                 finished_loading_dict: Optional[dict[str, int]] = None):
        self.finished_sending = finished_sending
        self.finished_recving = finished_recving
        self.finished_loading_dict = finished_loading_dict


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(world_size=2)

    output1 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'},
                                     finished_loading_dict={'req2': 256})
    output2 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None,
                                     finished_loading_dict=None)

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.finished_loading_dict is None

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None,
                                     finished_loading_dict=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving=None,
                                     finished_loading_dict=None)

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    assert aggregated.finished_sending == {'req1'}
    assert aggregated.finished_recving is None
    assert aggregated.finished_loading_dict is None

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'},
                                     finished_loading_dict={'req2': 128})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {'req2'}
    assert aggregated.finished_loading_dict == {'req2': 128}


def test_async_aggregate_workers_output():
    aggregator = KVOutputAggregator(world_size=2)

    future1: Future[DummyModelRunnerOutput] = Future()
    future2: Future[DummyModelRunnerOutput] = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'},
                                     finished_loading_dict={'req2': 256})
    output2 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None,
                                     finished_loading_dict=None)
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.finished_loading_dict is None

    future1 = Future()
    future2 = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None,
                                     finished_loading_dict=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving=None,
                                     finished_loading_dict=None)
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    assert aggregated.finished_sending == {'req1'}
    assert aggregated.finished_recving is None
    assert aggregated.finished_loading_dict is None

    future1 = Future()
    future2 = Future()
    result_future = aggregator.async_aggregate([future1, future2])

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None,
                                     finished_loading_dict=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'},
                                     finished_loading_dict={'req2': 128})
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {'req2'}
    assert aggregated.finished_loading_dict == {'req2': 128}
