from queue import Queue
from typing import Optional, Any, Callable,List
import threading

class EplbProcess:
    def __init__(self, target_func: Callable, num_wait_worker_iterations: int, adaptor):
        self.target_func = target_func
        self._num_wait_worker_iterations = num_wait_worker_iterations
        self.adaptor = adaptor
        self._thread: Optional[threading.Thread] = None
        self._result_queue: Optional[Queue] = None
        self._step_counter = 0        
    
    def _should_process(self) -> bool:
        return True
    
    def get_at_index(self,*args) -> List[Any]:
        return []
    
