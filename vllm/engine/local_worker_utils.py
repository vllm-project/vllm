import asyncio
import multiprocessing
import traceback
import threading
import uuid
from multiprocessing.connection import wait
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Union, List, Dict

from vllm.logger import init_logger

logger = init_logger(__name__)

T = TypeVar('T')

_TERMINATE = "TERMINATE"  # sentinel

# Use dedicated multiprocess context for workers.
# Both spawn and fork work
mp = multiprocessing.get_context("fork")


@dataclass
class Result(Generic[T]):
    """Result of task dispatched to worker"""

    task_id: uuid.UUID = None
    value: Optional[T] = None
    exception: Optional[BaseException] = None


class ResultFuture(threading.Event, Generic[T]):
    """Synchronous future for non-async case"""

    def __init__(self):
        super().__init__()
        self.result: Optional[Result[T]] = None

    def set_result(self, result: Result[T]):
        self.result = result
        self.set()

    def get(self) -> T:
        self.wait()
        if self.result.exception is not None:
            raise self.result.exception
        return self.result.value


def _set_future_result(future: Union[ResultFuture, asyncio.Future],
                       result: Result):
    if isinstance(future, ResultFuture):
        future.set_result(result)
        return
    loop = future.get_loop()
    if result.exception is not None:
        loop.call_soon_threadsafe(future.set_exception, result.exception)
    else:
        loop.call_soon_threadsafe(future.set_result, result.value)


class ResultHandler(threading.Thread):
    """Handle results from all workers (in background thread)"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.result_queue = mp.Queue()
        self.tasks: Dict[uuid.UUID, Union[ResultFuture, asyncio.Future]] = {}

    def run(self):
        for result in iter(self.result_queue.get, _TERMINATE):
            future = self.tasks.pop(result.task_id)
            _set_future_result(future, result)
        # Ensure that all waiters will receive an exception
        for future in self.tasks.values():
            _set_future_result(
                future, Result(exception=ChildProcessError("worker died")))

    def close(self):
        self.result_queue.put(_TERMINATE)


class WorkerMonitor(threading.Thread):
    """Monitor worker status (in background thread)"""

    def __init__(self, workers: List['LocalWorkerVllm'],
                 result_handler: ResultHandler):
        super().__init__(daemon=True)
        self.workers = workers
        self.result_handler = result_handler
        self._close = False

    def run(self) -> None:
        # Blocks until any worker exits
        dead_sentinels = wait([p.sentinel for p in self.workers])
        if self._close:
            return
        self._close = True

        # Kill / cleanup all workers
        for worker in self.workers:
            if worker.sentinel in dead_sentinels:
                worker.join(1)
            if worker.exitcode is not None and worker.exitcode != 0:
                logger.error(f"Worker {worker.name} pid {worker.pid} died, "
                             f"exit code: {worker.exitcode}")
        # Cleanup any remaining workers
        logger.info("Killing local vLLM worker processes")
        for worker in self.workers:
            worker.kill_worker()
        # Must be done after worker task queues are all closed
        self.result_handler.close()

    def close(self):
        if self._close:
            return
        self._close = True
        logger.info("Terminating local vLLM worker processes")
        for worker in self.workers:
            worker.terminate_worker()
        # Must be done after worker task queues are all closed
        self.result_handler.close()


class LocalWorkerVllm(mp.Process):
    """Local process wrapper for vllm.worker.Worker 
    for handling single-node multi-GPU tensor parallel."""

    def __init__(self, result_handler: ResultHandler, *args, **kwargs) -> None:
        super().__init__(daemon=True)
        self._task_queue = mp.Queue()
        self.result_queue = result_handler.result_queue
        self.tasks = result_handler.tasks
        self.worker_args = args
        self.worker_kwargs = kwargs
        self.worker = None

    def _enqueue_task(self, future: Union[ResultFuture, asyncio.Future],
                      method: str, args, kwargs):
        task_id = uuid.uuid4()
        self.tasks[task_id] = future
        try:
            self._task_queue.put((task_id, method, args, kwargs))
        except BaseException as e:
            del self.tasks[task_id]
            raise ChildProcessError("worker died") from e

    def execute_method(self, method: str, *args, **kwargs):
        future = ResultFuture()
        self._enqueue_task(future, method, args, kwargs)
        return future

    async def execute_method_async(self, method: str, *args, **kwargs):
        future = asyncio.get_running_loop().create_future()
        self._enqueue_task(future, method, args, kwargs)
        return await future

    def terminate_worker(self):
        try:
            self._task_queue.put(_TERMINATE)
        except ValueError:
            self.kill()
        self._task_queue.close()

    def kill_worker(self):
        self._task_queue.close()
        self.kill()

    def run(self) -> None:
        # Re-init logger in forked process, to include worker-specific prefix
        global logger
        logger = init_logger(__name__)

        del self.tasks  # Not used in forked process
        from vllm.worker.worker import Worker
        self.worker = Worker(*self.worker_args, **self.worker_kwargs)
        del self.worker_args
        del self.worker_kwargs

        # Accept tasks from the engine in task_queue
        # and return task output in result_queue
        logger.info("Worker ready; awaiting tasks")
        try:
            for items in iter(self._task_queue.get, _TERMINATE):
                output = None
                exception = None
                task_id, method, args, kwargs = items
                try:
                    executor = getattr(self.worker, method)
                    output = executor(*args, **kwargs)
                except BaseException as e:
                    tb = traceback.format_exc()
                    logger.error(
                        f"Exception in worker {mp.current_process().name} "
                        f"while processing method {method}: {e}, {tb}")
                    exception = e
                self.result_queue.put(
                    Result(task_id=task_id, value=output, exception=exception))
        except KeyboardInterrupt:
            pass
        except Exception:
            logger.exception("Worker failed")

        logger.info("Worker exiting")
