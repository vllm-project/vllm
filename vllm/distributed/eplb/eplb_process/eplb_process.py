from queue import Queue
from typing import Optional, Any, Callable
import multiprocessing as mp

from vllm.logger import logger


class EPLBProcess:
    """
    Encapsulates lifecycle management for asynchronous expert
    rearrangement processes
    """

    def __init__(self, target_func: Callable, num_wait_worker_iterations: int):
        """
        Initialize asynchronous process manager

        Args:
            target_func: Target function to execute in asynchronous process
                (e.g., rebalance_experts)
            num_wait_worker_iterations: Number of steps to wait before
                checking results
        """
        self.target_func = target_func
        self._num_wait_worker_iterations = num_wait_worker_iterations

        # Process management related
        self._process: Optional[mp.Process] = None
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
        Start asynchronous process

        Args:
            args: Tuple of arguments to pass to the target function
            post_process_args: Parameters needed for subsequent
                processing (e.g., model, ep_group)

        Returns:
            True if process started successfully, False otherwise
        """
        # Ensure previous process is cleaned up
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

            # Put arguments and start process
            self._input_queue.put(args)
            self._process = mp.Process(target=self._worker,
                                       args=(self._input_queue,
                                             self._result_queue,
                                             self._exception_queue),
                                       daemon=True)
            self._process.start()
            self._is_running = True
            return True

        except Exception as e:
            logger.error("Failed to start asynchronous process: {}", str(e))
            self.cleanup()
            return False

    def _worker(self, input_queue: Queue, output_queue: Queue,
                exception_queue: Queue) -> None:
        """Subprocess worker function"""
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
            logger.exception("Asynchronous process execution failed")

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
            error_msg = self._exception_queue.get()
            self.cleanup()
            raise RuntimeError("Asynchronous process failed: {}", error_msg)

        # Check if processing conditions are met
        if self._should_process():
            self._fetch_result()
            self.cleanup()
            return True

        return False

    def _should_process(self) -> bool:
        """Determine if results need processing"""
        if not self._process or not self._result_queue:
            return True

        return (self._step_counter >= self._num_wait_worker_iterations
                or not self._process.is_alive()
                or not self._result_queue.empty())

    def _fetch_result(self) -> None:
        """Retrieve subprocess results"""
        if self._result_queue and not self._result_queue.empty():
            self._result = self._result_queue.get()
        else:
            self._result = None
            logger.warning(
                "Asynchronous process completed but no result was returned")

    def cleanup(self) -> None:
        """Clean up process resources"""
        if self._process:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5.0)
            self._process = None

        for q in (self._input_queue, self._result_queue,
                  self._exception_queue):
            if q:
                q.close()
                q.join_thread()
        self._input_queue = None
        self._result_queue = None
        self._exception_queue = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Return whether the process is running"""
        return self._is_running

    @property
    def result(self) -> Optional[tuple]:
        """Return processing results"""
        return self._result

    @property
    def post_process_args(self) -> Optional[dict[str, Any]]:
        """Return post-processing arguments"""
        return self._post_process_args

    def __del__(self):
        """Ensure resource cleanup when object is destroyed"""
        self.cleanup()
