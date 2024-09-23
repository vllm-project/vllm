from abc import ABC, abstractmethod
from typing import Optional

import torch


class KVPipeBase(ABC):

    @abstractmethod
    def send_tensor(self, tensor: Optional[torch.Tensor], tensor_key: str = "") -> None:
        raise NotImplementedError

    @abstractmethod
    def recv_tensor(self, tensor_key: str = "") -> Optional[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
