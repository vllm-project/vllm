
from abc import ABC, abstractmethod
from typing import Optional
import torch


class KVLookupBufferBase(ABC):
    
    @abstractmethod
    def insert(self,
               input_tokens: torch.Tensor,
               kv: torch.Tensor, roi) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def drop_select(self, input_tokens, roi) -> Optional[torch.Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def close(self):
        """
        Close the buffer, release resources.
        """
        raise NotImplementedError
