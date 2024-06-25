from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union

import torch

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.models import (LoRAModel, LoRAModelManager,
                              LRUCacheLoRAModelManager, create_lora_manager)
from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


class AbstractWorkerLoRAManager(ABC):
    """Abstract class for managing LoRA models on the worker side."""

    def __init__(self,
                 max_num_seqs: int,
                 max_num_batched_tokens: int,
                 vocab_size: int,
                 lora_config: LoRAConfig,
                 device: torch.device,
                 max_position_embeddings: Optional[int] = None):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.device = device
        self.lora_config = lora_config

        # If False, do not cache. If None, cache is empty.
        self._cached_dummy_lora: Union[None, Literal[False], LoRAModel] = False

    @contextmanager
    def dummy_lora_cache(self):
        """Use this context manager to reuse the dummy lora model
        to avoid creating it repeatedly."""
        self._cached_dummy_lora = None
        yield
        self._cached_dummy_lora = False

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        ...

    @abstractmethod
    def create_lora_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        ...

    @abstractmethod
    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        ...

    @abstractmethod
    def add_lora(self, lora_request: LoRARequest) -> bool:
        ...

    @abstractmethod
    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        ...

    @abstractmethod
    def remove_lora(self, lora_id: int) -> bool:
        ...

    @abstractmethod
    def remove_all_loras(self):
        ...

    @abstractmethod
    def list_loras(self) -> Set[int]:
        ...


class WorkerLoRAManager(AbstractWorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Every request, the requested LoRAs will be loaded (unless they are already
    loaded), and every other LoRA will be unloaded."""

    _lora_manager_cls: Type[LoRAModelManager] = LoRAModelManager

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        embedding_modules: Dict[str, str],
        embedding_padding_modules: List[str],
        lora_model_cls: Type[LoRAModel] = LoRAModel,
        max_position_embeddings: Optional[int] = None,
    ):
        self._lora_model_cls = lora_model_cls
        self.embedding_modules = embedding_modules
        self.embedding_padding_modules = embedding_padding_modules
        # Lazily initialized by create_lora_manager.
        self._lora_manager: LoRAModelManager
        super().__init__(
            max_num_seqs,
            max_num_batched_tokens,
            vocab_size,
            lora_config,
            device,
            max_position_embeddings=max_position_embeddings,
        )

    @property
    def is_enabled(self) -> bool:
        return True

    def create_lora_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            lora_manager_cls=self._lora_manager_cls,
        )
        self._lora_manager = lora_manager
        return lora_manager.model

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        self._apply_loras(lora_requests)
        self._lora_manager.set_lora_mapping(lora_mapping)

    def _apply_loras(self, lora_requests: Set[LoRARequest]) -> None:
        loras_that_exist = self.list_loras()
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._lora_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._lora_manager.lora_slots}).")

        new_loras = set(loras_map)
        loras_to_add = new_loras - loras_that_exist
        loras_to_remove = loras_that_exist - new_loras

        for lora_id in loras_to_remove:
            self.remove_lora(lora_id)

        for lora_id in loras_to_add:
            self.add_lora(loras_map[lora_id])

    def _load_lora(self, lora_request: LoRARequest) -> LoRAModel:
        try:
            model = self._lora_manager.model
            supported_lora_modules = model.supported_lora_modules
            packed_modules_mapping = model.packed_modules_mapping
            expected_lora_modules: List[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_modules.extend(
                        packed_modules_mapping[module])
                else:
                    expected_lora_modules.append(module)
            lora = self._lora_model_cls.from_local_checkpoint(
                lora_request.lora_local_path,
                expected_lora_modules,
                max_position_embeddings=self.max_position_embeddings,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                target_embedding_padding=self.vocab_size +
                self.lora_config.lora_extra_vocab_size,
                embedding_modules=self.embedding_modules,
                embedding_padding_modules=self.embedding_padding_modules,
            )
        except Exception as e:
            raise RuntimeError(
                f"Loading lora {lora_request.lora_local_path} failed") from e
        if lora.rank > self.lora_config.max_lora_rank:
            raise ValueError(
                f"LoRA rank {lora.rank} is greater than max_lora_rank "
                f"{self.lora_config.max_lora_rank}.")
        if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
            raise ValueError(f"LoRA added vocab size {lora.extra_vocab_size} "
                             f"is greater than lora_extra_vocab_size "
                             f"{self.lora_config.lora_extra_vocab_size}.")
        return lora

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        if lora_request.lora_int_id in self.list_loras():
            return False
        if isinstance(self._cached_dummy_lora, LoRAModel):
            dummy_lora = self._cached_dummy_lora.clone(
                lora_request.lora_int_id)
        else:
            dummy_lora = self._lora_manager.create_dummy_lora(
                lora_request.lora_int_id, rank, 1, self.embedding_modules)
            if self._cached_dummy_lora is None:
                self._cached_dummy_lora = dummy_lora
        return self._lora_manager.add_lora(dummy_lora)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id in self.list_loras():
            return False
        lora = self._load_lora(lora_request)
        loaded = self._lora_manager.add_lora(lora)
        self._lora_manager.activate_lora(lora.id)
        return loaded

    def remove_lora(self, lora_id: int) -> bool:
        return self._lora_manager.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self._lora_manager.pin_lora(lora_id)

    def remove_all_loras(self):
        self._lora_manager.remove_all_loras()

    def list_loras(self) -> Set[int]:
        return set(self._lora_manager.list_loras())


class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Uses an LRU Cache. Every request, the requested LoRAs will be loaded
    (unless they are already loaded) and least recently used LoRAs will
    be unloaded if the cache is above capacity."""

    _lora_manager_cls: Type[
        LRUCacheLoRAModelManager] = LRUCacheLoRAModelManager

    def create_lora_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            lora_manager_cls=self._lora_manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._lora_manager = lora_manager
        return lora_manager.model

    def _apply_loras(self, lora_requests: Set[LoRARequest]) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._lora_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._lora_manager.lora_slots}).")
        for lora in loras_map.values():
            self.add_lora(lora)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id not in self.list_loras():
            # Remove before we load the new lora to save memory
            if len(self._lora_manager) + 1 > self._lora_manager.capacity:
                assert isinstance(self._lora_manager, LRUCacheLoRAModelManager)
                self._lora_manager.remove_oldest_lora()
            lora = self._load_lora(lora_request)
            loaded = self._lora_manager.add_lora(lora)
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            loaded = self._lora_manager.get_lora(
                lora_request.lora_int_id) is not None
        self._lora_manager.activate_lora(lora_request.lora_int_id)
        return loaded
