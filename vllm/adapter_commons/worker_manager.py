from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Optional, Set

import torch


class AbstractWorkerManager(ABC):

    def __init__(self, device: torch.device):
        self.device = device

    @abstractproperty
    def _model_manager(self):
        ...

    @abstractproperty
    def is_enabled(self) -> bool:
        ...

    @abstractmethod
    def create_manager(self, model: torch.nn.Module) -> Any:
        ...

    def set_active_adapters(self, requests: Set[Any],
                            mapping: Optional[Any]) -> None:
        self._apply_adapters(requests)
        self._model_manager.set_adapter_mapping(mapping)

    @abstractmethod
    def add_dummy_adapter(self, request: Any) -> bool:
        ...

    @abstractmethod
    def _load_adapter(self, request: Any) -> Any:
        ...

    def add_adapter(self, adapter_request: Any) -> bool:
        if adapter_request.adapter_id in self.list_adapters():
            return False
        loaded_adapter = self._load_adapter(adapter_request)
        loaded = self._model_manager.add_adapter(loaded_adapter)
        self._model_manager.activate_adapter(loaded_adapter.id)
        return loaded

    def _apply_adapters(self, adapter_requests: Set[Any]) -> None:
        models_that_exist = self.list_adapters()
        models_map = {
            adapter_request.adapter_id: adapter_request
            for adapter_request in adapter_requests if adapter_request
        }
        if len(models_map) > self._model_manager.adapter_slots:
            raise RuntimeError(
                f"Number of requested models ({len(models_map)}) is greater "
                "than the number of GPU model slots "
                f"({self._model_manager.adapter_slots}).")

        new_models = set(models_map)
        models_to_add = new_models - models_that_exist
        models_to_remove = models_that_exist - new_models

        for adapter_id in models_to_remove:
            self.remove_adapter(adapter_id)

        for adapter_id in models_to_add:
            self.add_adapter(models_map[adapter_id])

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._model_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._model_manager.remove_all_adapters()

    def list_adapters(self) -> Set[int]:
        return set(self._model_manager.list_adapters())
