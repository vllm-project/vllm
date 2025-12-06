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
        Init BufferResponseProcessor and object is belonged to async_llm
        :param process_callback: func to release responses to request when it meets slo requirements
        :param engine_num: Optional, record the engine num for saving corresponding logs to different loggers in async_llm
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
        Add BufferedResponse to the BufferResponseProcessor.
        :param response: class BufferedResponse with request_id, output and slo_requirement(optional)
        :return: None
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
        Remove the request from response_container once it is aborted
        :param request_id: str
        :return: None
        """
        with self._rw_lock.writer_lock:
            if request_id in self.response_container:
                self.response_container[request_id].is_aborted = True

    def _slo_checker(self) -> list[Any]:
        """
        To filter outputs that are approaching to SLO requirements
        :return: list[Any]
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
        Loop to check slo in response_container and release buffered responses
        :return: None
        """
        while self._running:
            to_send_output = self._slo_checker()
            for engine_index in range(self.engine_num):
                if len(to_send_output[engine_index]) > 0:
                    self.process_callback(outputs = to_send_output[engine_index], engine_index = engine_index)
            time.sleep(0.001)

    def _update_response_container_and_get_output(self, req_id: str):
        """
        Update the request's info in response_container
        :param req_id: str, request id
        :return: Union None or outputs
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
        End buffer_response_thread
        :return: None
        """
        self._running = False
        if self._buffer_response_thread and self._buffer_response_thread.is_alive():
            self._buffer_response_thread.join(timeout=10.0)

@staticmethod
def bind_fixed_params(*fixed_args: Any, **fixed_kwargs: Any):
    """
    Decorator to bind fixed parameters or arguments to callback function
    :param fixed_args: fixed positional arguments
    :param fixed_kwargs: fixed named arguments
    :return: callback func with fixed parameters or arguments
    """
    def decorator(callback: Callable[..., None]) -> Callable[..., None]:
        def wrapped(*dynamic_args: Any, **dynamic_kwargs: Any) -> None:
            callback(*fixed_args, *dynamic_args, **fixed_kwargs, **dynamic_kwargs)
        return wrapped
    return decorator
