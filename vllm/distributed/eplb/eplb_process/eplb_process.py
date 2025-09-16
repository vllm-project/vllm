from queue import Queue
from typing import Optional, Any
import threading

class EplbProcess:
    def __init__(self, *agrs, **kargs):
        self._thread: Optional[threading.Thread] = None
        self._result_queue: Optional[Queue] = None
    
    def _should_process(self) -> bool:
        return True
    
    def get_at_index(self,*args) -> list[Any]:
        return []