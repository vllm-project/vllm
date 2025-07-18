# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections import defaultdict
from concurrent.futures import Future
from typing import Optional

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.outputs import ModelRunnerOutput


class DummyMultiprocExecutor(MultiprocExecutor):

    def __init__(self, output_rank, world_size):
        # Manually initialize minimal required fields
        self.output_rank = output_rank
        self.world_size = world_size
        self._send_remaining_count = defaultdict[str,
                                                 int](lambda: self.world_size)
        self._recv_remaining_count = defaultdict[str,
                                                 int](lambda: self.world_size)
        self.io_thread_pool = None
        self.shutdown_event = threading.Event()


class DummyModelRunnerOutput(ModelRunnerOutput):

    def __init__(self,
                 finished_sending: Optional[set[str]] = None,
                 finished_recving: Optional[set[str]] = None):
        self.finished_sending = finished_sending
        self.finished_recving = finished_recving


def test_aggregate_workers_output():
    executor = DummyMultiprocExecutor(output_rank=0, world_size=2)

    output1 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'})
    output2 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)

    aggregated = executor._aggregate_workers_output([output1, output2])

    assert aggregated is output1
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving=None)

    aggregated = executor._aggregate_workers_output([output1, output2])

    assert aggregated is output1
    assert aggregated.finished_sending == {'req1'}
    assert aggregated.finished_recving is None

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'})

    aggregated = executor._aggregate_workers_output([output1, output2])

    assert aggregated is output1
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {'req2'}


def test_async_aggregate_workers_output():
    executor = DummyMultiprocExecutor(output_rank=0, world_size=2)

    future1: Future[DummyModelRunnerOutput] = Future()
    future2: Future[DummyModelRunnerOutput] = Future()
    result_future = executor._async_aggregate_workers_output(
        [future1, future2])

    output1 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'})
    output2 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None

    future1 = Future()
    future2 = Future()
    result_future = executor._async_aggregate_workers_output(
        [future1, future2])

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving=None)
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    assert aggregated.finished_sending == {'req1'}
    assert aggregated.finished_recving is None

    future1 = Future()
    future2 = Future()
    result_future = executor._async_aggregate_workers_output(
        [future1, future2])

    output1 = DummyModelRunnerOutput(finished_sending=None,
                                     finished_recving=None)
    output2 = DummyModelRunnerOutput(finished_sending={'req1'},
                                     finished_recving={'req2'})
    future1.set_result(output1)
    future2.set_result(output2)

    assert result_future.done()
    aggregated = result_future.result()
    assert aggregated is output1
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {'req2'}
