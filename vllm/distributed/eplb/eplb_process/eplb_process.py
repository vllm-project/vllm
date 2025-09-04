# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp
from contextlib import suppress
from multiprocessing import Queue
from typing import Any, Callable, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


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
        self._has_pending_task = False
        self._is_post_processing = False

        # Save parameters needed for post-processing
        self._post_process_args: Optional[dict[str, Any]] = None

        # Initialize process and queues
        self._initialize_process()

    def _initialize_process(self) -> None:
        """Initialize the background process and queues"""
        try:
            # Initialize queues
            self._input_queue = Queue()
            self._result_queue = Queue()
            self._exception_queue = Queue()

            # Start the process
            self._process = mp.Process(target=self._worker_loop,
                                       name="EPLBProcess",
                                       args=(self._input_queue,
                                             self._result_queue,
                                             self._exception_queue))
            self._process.start()
            self._is_running = True
            logger.debug("EPLB background process started")

        except Exception as e:
            logger.error("Failed to start EPLB background process: {}", str(e))
            self.cleanup()
            raise

    def _worker_loop(self, input_queue: Queue, output_queue: Queue,
                     exception_queue: Queue) -> None:
        """Subprocess worker loop that processes tasks continuously"""
        try:
            while True:
                # Get arguments from input queue
                try:
                    args = input_queue.get(timeout=1.0)
                    if args is None:  # Sentinel value to stop the process
                        break

                    # Execute target function
                    result = self.target_func(*args)
                    output_queue.put(result)
                except Exception as e:
                    output_queue.put(None)
                    if hasattr(e, "add_note"):
                        import traceback
                        e.add_note(traceback.format_exc())
                    exception_queue.put(e)
                    logger.exception("Task execution failed in worker process")

        except Exception as e:
            exception_queue.put(e)
            logger.exception("Worker process encountered fatal error")
        finally:
            logger.debug("EPLB worker process exiting")

    def submit_task(self, args: tuple, post_process_args: dict[str,
                                                               Any]) -> bool:
        """
        Submit a task to the asynchronous process

        Args:
            args: Tuple of arguments to pass to the target function
            post_process_args: Parameters needed for subsequent
                processing (e.g., model, ep_group)

        Returns:
            True if task submitted successfully, False otherwise
        """
        if not self._is_running:
            logger.error("Cannot submit task: process is not running")
            return False

        if self._has_pending_task:
            logger.warning("Cannot submit task: already has a pending task")
            return False

        if not self._input_queue:
            logger.error("Cannot submit task: input queue is not initialized")
            return False

        try:
            # Put arguments to the input queue
            self._input_queue.put(args)
            self._args = args
            self._post_process_args = post_process_args
            self._has_pending_task = True
            self._step_counter = 0
            self._result = None
            return True

        except Exception as e:
            logger.error("Failed to submit task to asynchronous process: {}",
                         str(e))
            return False

    def step(self) -> bool:
        """
        Increment step counter and check if results need processing

        Returns:
            Whether results have been processed
        """
        if not self._is_running or not self._has_pending_task:
            return False

        self._step_counter += 1

        # Check for exceptions first
        if self._exception_queue and not self._exception_queue.empty():
            error_msg = self._exception_queue.get()
            self._has_pending_task = False
            raise RuntimeError("Asynchronous process failed: {}", error_msg)

        # Check if processing conditions are met
        if self._should_process():
            self._fetch_result()
            self._has_pending_task = False
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
        # Send sentinel value to stop the process
        if self._input_queue:
            with suppress(Exception):
                self._input_queue.put(None)

        if self._process:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None

        for q in (self._input_queue, self._result_queue,
                  self._exception_queue):
            if q:
                with suppress(Exception):
                    q.close()
                with suppress(Exception):
                    q.join_thread()

        self._input_queue = None
        self._result_queue = None
        self._exception_queue = None
        self._is_running = False
        self._has_pending_task = False

    @property
    def is_running(self) -> bool:
        """Return whether the process is running"""
        return self._is_running

    @property
    def has_pending_task(self) -> bool:
        """Return whether there is a pending task"""
        return self._has_pending_task

    @property
    def is_post_processing(self) -> bool:
        return self._is_post_processing

    @is_post_processing.setter
    def is_post_processing(self, value: bool):
        self._is_post_processing = value

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
