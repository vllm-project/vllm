from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TypeVar

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


class AdapterLRUCache(LRUCache[int, T]):

    def __init__(self, capacity: int, deactivate_fn: Callable[[int], object]):
        super().__init__(capacity)
        self.deactivate_fn = deactivate_fn

    def _on_remove(self, key: int, value: Optional[T]):
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
    def adapter_slots(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def capacity(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def activate_adapter(self, adapter_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def deactivate_adapter(self, adapter_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def add_adapter(self, adapter: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def set_adapter_mapping(self, mapping: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_adapter(self, adapter_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove_all_adapters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def list_adapters(self) -> Dict[int, Any]:
        raise NotImplementedError

    @abstractmethod
    def pin_adapter(self, adapter_id: int) -> bool:
        raise NotImplementedError
