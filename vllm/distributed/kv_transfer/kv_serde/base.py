
import torch
from abc import ABC, abstractmethod

class KV_serde(ABC):
    
    @abstractmethod
    def serialize(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def deserialize(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError