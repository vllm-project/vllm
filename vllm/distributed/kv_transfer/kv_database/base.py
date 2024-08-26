
from abc import ABC, abstractmethod
from typing import Optional
import torch


class KV_Database(ABC):
    
    @abstractmethod
    def insert(self, input_tokens, kv, roi):
        raise NotImplementedError
    
    @abstractmethod
    def drop_select(self, input_tokens, roi) -> Optional[torch.Tensor]:
        raise NotImplementedError
    