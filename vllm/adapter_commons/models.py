from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Hashable, Optional, TypeVar

from torch import nn

from vllm.logger import init_logger
from vllm.utils import LRUCache

logger = init_logger(__name__)


class AdapterModel(ABC):

    def __init__(self, model_id=None):
        self.id = model_id

    @abstractmethod
    def from_local_checkpoint(cls, model_dir, model_id=None, **kwargs):
        # Common initialization code
        # Load weights or embeddings from local checkpoint
        raise NotImplementedError("Subclasses must implement this method.")


T = TypeVar('T')


class AdapterLRUCache(LRUCache[T]):

    def __init__(self, capacity: int, deactivate_fn: Callable[[Hashable],
                                                              None]):
        super().__init__(capacity)
        self.deactivate_fn = deactivate_fn

    def _on_remove(self, key: Hashable, value: T):
        logger.debug("Removing adapter int id: %d", key)
        self.deactivate_fn(key)
        return super()._on_remove(key, value)


class AdapterModelManager(ABC):

    def __init__(
        self,
        model: nn.Module,
    ):
        """Create a AdapterModelManager and adapter for a given model.
        Args:
            model: the model to be adapted.
        """
        self.model: nn.Module = model
        self._registered_adapters: Dict[int, Any] = {}
        # Dict instead of a Set for compatibility with LRUCache.
        self._active_adapters: Dict[int, None] = {}
        self.adapter_type = 'Adapter'
        self._last_mapping = None

    def __len__(self) -> int:
        return len(self._registered_adapters)

    @property
    @abstractmethod
    def adapter_slots(self):
        ...

    @property
    @abstractmethod
    def capacity(self):
        ...

    @abstractmethod
    def _deactivate_adapter(self, adapter_id: int):
        raise NotImplementedError("Subclasses must implement this method.")

    def deactivate_adapter(self, adapter_id: int) -> bool:
        if adapter_id in self._active_adapters:
            self._deactivate_adapter(adapter_id)
            self._active_adapters.pop(adapter_id)
            return True
        return False

    @abstractmethod
    def _add_adapter(self, adapter: Any):
        raise NotImplementedError("Subclasses must implement this method.")

    def add_adapter(self, adapter: Any) -> bool:
        if adapter.id not in self._registered_adapters:
            if len(self._registered_adapters) >= self.capacity:
                raise RuntimeError(f'No free {self.adapter_type} slots.')
            self._add_adapter(adapter)
            return True
        return False

    def set_adapter_mapping(self, mapping: Any) -> None:
        if self._last_mapping != mapping:
            self._set_adapter_mapping(mapping)
        self._last_mapping = mapping

    @abstractmethod
    def _set_adapter_mapping(self, mapping: Any) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    def remove_adapter(self, adapter_id: int) -> bool:
        self.deactivate_adapter(adapter_id)
        return bool(self._registered_adapters.pop(adapter_id, None))

    def list_adapters(self) -> Dict[int, Any]:
        return dict(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        return self._registered_adapters.get(adapter_id, None)
