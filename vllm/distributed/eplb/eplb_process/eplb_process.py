from queue import Queue, Empty
from typing import Optional, Any, Callable,List
from contextlib import suppress
import threading

from vllm.logger import logger


class EplbProcess:
    """
    Encapsulates lifecycle management for asynchronous expert
    rearrangement threads
    """

    def __init__(self, target_func: Callable, num_wait_worker_iterations: int):
        """
        Initialize asynchronous thread manager
        Args:
            target_func: Target function to execute in asynchronous thread
                (e.g., rebalance_experts)
            num_wait_worker_iterations: Number of steps to wait before
                checking results
        """
        self.target_func = target_func
        self._num_wait_worker_iterations = num_wait_worker_iterations

        # Thread management related
        self._thread: Optional[threading.Thread] = None
        self._input_queue: Optional[Queue] = None
        self._result_queue: Optional[Queue] = None
        self._exception_queue: Optional[Queue] = None
        self._step_counter = 0
        self._result: Optional[tuple] = None
        self._args: Optional[tuple] = None
        self._is_running = False

        # Save parameters needed for post-processing
        self._post_process_args: Optional[dict[str, Any]] = None

    def start(self, args: tuple, post_process_args: dict[str, Any]) -> bool:
        """
        Start asynchronous thread
        Args:
            args: Tuple of arguments to pass to the target function
            post_process_args: Parameters needed for subsequent
                processing (e.g., model, ep_group)
        Returns:
            True if thread started successfully, False otherwise
        """
        # Ensure previous thread is cleaned up
        self.cleanup()

        try:
            # Initialize queues
            self._input_queue = Queue()
            self._result_queue = Queue()
            self._exception_queue = Queue()
            self._step_counter = 0
            self._result = None
            self._args = args
            self._post_process_args = post_process_args

            # Put arguments and start thread
            self._input_queue.put(args)
            self._thread = threading.Thread(target=self._worker,
                                            args=(self._input_queue,
                                                  self._result_queue,
                                                  self._exception_queue),
                                            daemon=True)
            self._thread.start()
            self._is_running = True
            return True
        except Exception as e:
            logger.error("Failed to start asynchronous thread: %s", str(e))
            self.cleanup()
            return False

    def _worker(self, input_queue: Queue, output_queue: Queue,
                exception_queue: Queue) -> None:
        """Subthread worker function"""
        try:
            # Get arguments
            args = input_queue.get()

            # Execute target function
            result = self.target_func(*args)
            output_queue.put(result)
        except Exception as e:
            output_queue.put(None)
            if hasattr(e, "add_note"):
                import traceback
                e.add_note(traceback.format_exc())
            exception_queue.put(e)
            logger.exception("Asynchronous thread execution failed")

    def step(self) -> bool:
        """
        Increment step counter and check if results need processing
        Returns:
            Whether results have been processed
        """
        if not self._is_running:
            return False

        self._step_counter += 1

        # Check for exceptions first
        if self._exception_queue and not self._exception_queue.empty():
            try:
                error_msg = self._exception_queue.get_nowait()
                self.cleanup()
                raise RuntimeError(f"Asynchronous thread failed: {error_msg}")
            except Empty:
                pass

        # Check if processing conditions are met
        if self._should_process():
            self._fetch_result()
            self.cleanup()
            return True

        return False

    def _should_process(self) -> bool:
        """Determine if results need processing"""
        if not self._thread or not self._result_queue:
            return True

        return (self._step_counter >= self._num_wait_worker_iterations
                or not self._thread.is_alive()
                or not self._result_queue.empty())

    def _fetch_result(self) -> None:
        """Retrieve subthread results"""
        if self._result_queue and not self._result_queue.empty():
            try:
                self._result = self._result_queue.get_nowait()
            except Empty:
                self._result = None
        else:
            self._result = None
            logger.warning(
                "Asynchronous thread completed but no result was returned")
    
    def get_at_index(self,i) -> List[Any]:
        if not self._result_queue or self._result_queue.empty():
            raise ValueError("Queue is empty, cannot retrieve element")
        size = self._result_queue.qsize()
        # check if queue length matches the of layers
        if size != 94: #layer of qwen
            raise ValueError(f"Queue length {size} does not match the expected layer numbers of qwen")
        if i <=0 or i > size:
            raise ValueError(f"Index {i} out of range for queue of size {size}")
        result_result_queue = list(self._result_queue)
        return result_result_queue[i-1]
    
    def cleanup(self) -> None:
        """Clean up thread resources"""
        # Threads can't be terminated, so we just mark it as not running
        self._is_running = False
        self._thread = None

        # Clear queues
        for q in (self._input_queue, self._result_queue,
                  self._exception_queue):
            if q:
                with suppress(Empty):
                    while not q.empty():
                        q.get_nowait()

        self._input_queue = None
        self._result_queue = None
        self._exception_queue = None

    @property
    def is_running(self) -> bool:
        """Return whether the thread is running"""
        if not self._is_running or self._thread is None:
            return False
        return self._thread.is_alive()

    @property
    def result(self) -> Optional[tuple]:
        """Return processing results"""
        return self._result

    @property
    def post_process_args(self) -> Optional[dict[str, Any]]:
        """Return post-processing arguments"""
        return self._post_process_args