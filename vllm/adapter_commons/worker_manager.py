from abc import ABC, abstractmethod
from typing import Any, Optional, Set

import torch


class AbstractWorkerManager(ABC):

    def __init__(self, device: torch.device):
        self.device = device

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        ...

    @abstractmethod
    def set_active_adapters(self, requests: Set[Any],
                            mapping: Optional[Any]) -> None:
        ...

    @abstractmethod
    def add_adapter(self, adapter_request: Any) -> bool:
        ...

    @abstractmethod
    def remove_adapter(self, adapter_id: int) -> bool:
        ...

    @abstractmethod
    def remove_all_adapters(self):
        ...

    @abstractmethod
    def list_adapters(self) -> Set[int]:
        ...
