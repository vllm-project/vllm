import gc
from typing import Callable, Optional

import torch


class PlatformMemoryProfiler:

    def __init__(self,
                 current_memory_usage_func: Callable[[torch.types.Device],
                                                     float],
                 device: Optional[torch.types.Device] = None):
        self.device = device
        self.current_memory_usage_func = current_memory_usage_func

    def __enter__(self):
        self.initial_memory = self.current_memory_usage_func(self.device)
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage_func(self.device)
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()
