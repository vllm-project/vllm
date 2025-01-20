import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import sleep
from typing import Any, List, Tuple

import pytest

from vllm.config import VllmConfig
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)
from vllm.worker.worker_base import WorkerWrapperBase


class DummyWorkerWrapper(WorkerWrapperBase):
    """Dummy version of vllm.worker.worker.Worker"""

    def worker_method(self, worker_input: Any) -> Tuple[int, Any]:
        sleep(0.05)

        if isinstance(worker_input, Exception):
            # simulate error case
            raise worker_input

        return self.rpc_rank, input


def _start_workers() -> Tuple[List[ProcessWorkerWrapper], WorkerMonitor]:
    result_handler = ResultHandler()
    vllm_config = VllmConfig()
    workers = [
        ProcessWorkerWrapper(result_handler, DummyWorkerWrapper, vllm_config,
                             rank) for rank in range(8)
    ]

    worker_monitor = WorkerMonitor(workers, result_handler)
    assert not worker_monitor.is_alive()

    result_handler.start()
    worker_monitor.start()
    assert worker_monitor.is_alive()

    return workers, worker_monitor


def test_local_workers() -> None:
    """Test workers with sync task submission"""

    workers, worker_monitor = _start_workers()

    def execute_workers(worker_input: str) -> None:
        worker_outputs = [
            worker.execute_method("worker_method", worker_input)
            for worker in workers
        ]

        for rank, output in enumerate(worker_outputs):
            assert output.get() == (rank, input)

    executor = ThreadPoolExecutor(max_workers=4)

    # Test concurrent submission from different threads
    futures = [
        executor.submit(partial(execute_workers, f"thread {thread_num}"))
        for thread_num in range(4)
    ]

    for future in futures:
        future.result()

    # Test error case
    exception = ValueError("fake error")
    result = workers[0].execute_method("worker_method", exception)
    try:
        result.get()
        pytest.fail("task should have failed")
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "fake error"

    # Test cleanup when a worker fails
    assert worker_monitor.is_alive()
    workers[3].process.kill()

    # Other workers should get shut down here
    worker_monitor.join(20)

    # Ensure everything is stopped
    assert not worker_monitor.is_alive()
    assert all(not worker.process.is_alive() for worker in workers)

    # Further attempts to submit tasks should fail
    try:
        _result = workers[0].execute_method("worker_method", "test")
        pytest.fail("task should fail once workers have been shut down")
    except Exception as e:
        assert isinstance(e, ChildProcessError)


def test_local_workers_clean_shutdown() -> None:
    """Test clean shutdown"""

    workers, worker_monitor = _start_workers()

    assert worker_monitor.is_alive()
    assert all(worker.process.is_alive() for worker in workers)

    # Clean shutdown
    worker_monitor.close()

    worker_monitor.join(20)

    # Ensure everything is stopped
    assert not worker_monitor.is_alive()
    assert all(not worker.process.is_alive() for worker in workers)

    # Further attempts to submit tasks should fail
    try:
        _result = workers[0].execute_method("worker_method", "test")
        pytest.fail("task should fail once workers have been shut down")
    except Exception as e:
        assert isinstance(e, ChildProcessError)


@pytest.mark.asyncio
async def test_local_workers_async() -> None:
    """Test local workers with async task submission"""

    workers, worker_monitor = _start_workers()

    async def execute_workers(worker_input: str) -> None:
        worker_coros = [
            worker.execute_method_async("worker_method", worker_input)
            for worker in workers
        ]

        results = await asyncio.gather(*worker_coros)
        for rank, result in enumerate(results):
            assert result == (rank, input)

    tasks = [
        asyncio.create_task(execute_workers(f"task {task_num}"))
        for task_num in range(4)
    ]

    for task in tasks:
        await task

    # Test error case
    exception = ValueError("fake error")
    try:
        _result = await workers[0].execute_method_async(
            "worker_method", exception)
        pytest.fail("task should have failed")
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == "fake error"

    # Test cleanup when a worker fails
    assert worker_monitor.is_alive()
    workers[3].process.kill()

    # Other workers should get shut down here
    worker_monitor.join(20)

    # Ensure everything is stopped
    assert not worker_monitor.is_alive()
    assert all(not worker.process.is_alive() for worker in workers)

    # Further attempts to submit tasks should fail
    try:
        _result = await workers[0].execute_method_async(
            "worker_method", "test")
        pytest.fail("task should fail once workers have been shut down")
    except Exception as e:
        assert isinstance(e, ChildProcessError)
