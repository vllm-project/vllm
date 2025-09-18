# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from queue import Queue
from typing import Optional, Any


class EplbProcess:
    
    def __init__(self, *agrs, **kargs):
        self._thread: Optional[threading.Thread] = None
        self._result_queue: Optional[Queue] = None
    
    def _should_process(self) -> bool:
        return True
    
    def get_at_index(self,*args) -> list[Any]:
        return []