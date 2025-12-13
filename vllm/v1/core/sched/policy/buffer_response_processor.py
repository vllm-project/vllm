from typing import Any, Optional, Callable, Dict, Union
from dataclasses import dataclass, field
from rwlock import RWLock

import time
import threading
import queue
import _queue

SECOND_TO_MS = 1000

@dataclass
class GlobalSLORequirement:
    ttft_slo: Optional[Union[int, float]] = 1000 # ms
    tpot_slo: Optional[Union[int, float]] = 50  # ms

@dataclass
class BufferedResponse:
    request_id: str
    req_slo_requirement: Optional[Union[GlobalSLORequirement, None]] = None
    output: Union[queue.Queue, Any] = None
    is_ended: Optional[bool] = False
    have_sent_prefill: Optional[bool] = False
    last_processed_time: Optional[float] = 0.0
    engine_index: Optional[int] = 0
    is_aborted : Optional[bool] = False

class BufferResponseProcessor():
    def __init__(self,
            process_callback: Callable[[Any], Any],
            global_slo_requirement: Optional[GlobalSLORequirement] = GlobalSLORequirement(),
            engine_num: Optional[int] = 1
        ):
        """
        Initializes the BufferResponseProcessor.

        This object belongs to async_llm.

        Args:
            process_callback (Callable[[Any], Any]): Function to release responses to request when it meets SLO requirements.
            global_slo_requirement (Optional[GlobalSLORequirement]): Global SLO requirements. Defaults to GlobalSLORequirement().
            engine_num (Optional[int]): Record the engine number for saving corresponding logs to different loggers in async_llm. Defaults to 1.
        """
        self.process_callback = process_callback
        self.slo_send_factor = 0.95
        self.default_slo_requirement = global_slo_requirement
        self.engine_num = engine_num
        self.response_container : Dict[str, BufferedResponse] = {}
        self._rw_lock = RWLock()
        self._running = True
        self._buffer_response_thread = threading.Thread(
            target=self._process_buffered_response,
            daemon=True
        )
        self._buffer_response_thread.start()

    def add_response(self, response: BufferedResponse) -> None:
        """
        Adds a BufferedResponse to the BufferResponseProcessor.

        Args:
            response (BufferedResponse): A BufferedResponse object containing request_id, output, and optional SLO requirement.
        """
        with self._rw_lock.writer_lock:
            if response.request_id in self.response_container:
                # update output, engine_index(DP), is_ended for the request in response_container
                self.response_container[response.request_id].output.put_nowait(response.output)
                self.response_container[response.request_id].engine_index = response.engine_index
                self.response_container[response.request_id].is_ended = response.is_ended
            else:
                # add new request to response_container
                if not response.req_slo_requirement:
                    response.req_slo_requirement = self.default_slo_requirement
                self.response_container[response.request_id] = response

    def abort_request(self, request_id: str) -> None:
        """
        Removes the request from response_container once it is aborted.

        Args:
            request_id (str): The ID of the request to abort.
        """
        with self._rw_lock.writer_lock:
            if request_id in self.response_container:
                self.response_container[request_id].is_aborted = True

    def _slo_checker(self) -> Dict[int, list[Any]]:
        """
        Filters outputs that are approaching SLO requirements.

        Returns:
            Dict[int, list[Any]]: A dictionary mapping engine indices to lists of outputs that are approaching their SLO requirements.
        """
        global SECOND_TO_MS
        to_send_outputs = {i: [] for i in range(self.engine_num)}
        to_update_response = []
        with self._rw_lock.reader_lock:
            for req_id, req_response in self.response_container.items():
                if req_response.is_aborted or req_response.is_ended:
                    to_update_response.append((req_id, req_response.engine_index))
                else:
                    processing_slo = "tpot_slo" if req_response.have_sent_prefill else "ttft_slo"

                    if (((time.time() - req_response.last_processed_time) * SECOND_TO_MS >
                            self.slo_send_factor * getattr(req_response.req_slo_requirement, processing_slo))
                            and req_response.output.qsize() > 0):
                        to_update_response.append((req_id, req_response.engine_index))

        for id_index_pair in to_update_response:
            outputs = self._update_response_container_and_get_output(id_index_pair[0])
            to_send_outputs[id_index_pair[1]].extend(outputs)
        return to_send_outputs

    def _process_buffered_response(self) -> None:
        """
        Continuously checks SLO requirements in response_container and releases buffered responses.

        This method runs in a separate thread and processes responses based on their SLO requirements.
        """
        while self._running:
            to_send_output = self._slo_checker()
            for engine_index in range(self.engine_num):
                if len(to_send_output[engine_index]) > 0:
                    self.process_callback(outputs = to_send_output[engine_index], engine_index = engine_index)
            time.sleep(0.001)

    def _update_response_container_and_get_output(self, req_id: str):
        """
        Updates the request's information in response_container and retrieves output.

        Args:
            req_id (str): The request ID to update and retrieve output from.

        Returns:
            list[Any]: A list of outputs from the response queue.
        """
        with self._rw_lock.writer_lock:
            response = self.response_container[req_id]
            result = []
            if response.is_aborted:
                del self.response_container[req_id]
            elif response.is_ended:
                while True:
                    try:
                        result.append(response.output.get_nowait())
                    except (_queue.Empty, queue.Empty):
                        break
                del self.response_container[req_id]
            # update whether send the first token
            else:
                # ensure the queue is not empty in _slo_checker
                try:
                    result.append(response.output.get_nowait())
                    if not response.have_sent_prefill:
                        self.response_container[req_id].have_sent_prefill = True
                    self.response_container[req_id].last_processed_time = time.time()
                except (_queue.Empty, queue.Empty):
                    pass
        return result

    def stop(self) -> None:
        """
        Stops the buffer_response_thread and cleans up resources.

        This method terminates the background thread that processes buffered responses.
        """
        self._running = False
        if self._buffer_response_thread and self._buffer_response_thread.is_alive():
            self._buffer_response_thread.join(timeout=10.0)

@staticmethod
def bind_fixed_params(*fixed_args: Any, **fixed_kwargs: Any):
    """
    Decorator to bind fixed parameters or arguments to a callback function.

    Args:
        *fixed_args (Any): Fixed positional arguments to bind to the callback function.
        **fixed_kwargs (Any): Fixed keyword arguments to bind to the callback function.

    Returns:
        Callable[..., None]: A wrapped callback function with the fixed parameters bound.
    """
    def decorator(callback: Callable[..., None]) -> Callable[..., None]:
        def wrapped(*dynamic_args: Any, **dynamic_kwargs: Any) -> None:
            callback(*fixed_args, *dynamic_args, **fixed_kwargs, **dynamic_kwargs)
        return wrapped
    return decorator
