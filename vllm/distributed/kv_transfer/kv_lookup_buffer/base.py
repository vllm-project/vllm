from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class KVLookupBufferBase(ABC):

    @abstractmethod
    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def drop_select(self, input_tokens: torch.Tensor,
                    roi: torch.Tensor) -> List[Optional[torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Close the buffer, release resources.
        """
        raise NotImplementedError
