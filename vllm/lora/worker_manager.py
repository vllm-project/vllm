from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union

import torch

from vllm.adapter_commons.utils import (add_adapter_worker,
                                        apply_adapters_worker,
                                        list_adapters_worker,
                                        set_active_adapters_worker)
from vllm.adapter_commons.worker_manager import AbstractWorkerManager
from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.models import (LoRAModel, LoRAModelManager,
                              LRUCacheLoRAModelManager, create_lora_manager)
from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


class WorkerLoRAManager(AbstractWorkerManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Every request, the requested LoRAs will be loaded (unless they are already
    loaded), and every other LoRA will be unloaded."""

    _manager_cls: Type[LoRAModelManager] = LoRAModelManager

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
        self._cached_dummy_lora: Union[None, Literal[False], LoRAModel] = False
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vocab_size = vocab_size
        self.lora_config = lora_config
        self.max_position_embeddings = max_position_embeddings
        super().__init__(device)
        # Lazily initialized by create_lora_manager.
        self._adapter_manager: LoRAModelManager

    @contextmanager
    def dummy_lora_cache(self):
        """Use this context manager to reuse the dummy lora model
        to avoid creating it repeatedly."""
        self._cached_dummy_lora = None
        yield
        self._cached_dummy_lora = False

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
            lora_manager_cls=self._manager_cls,
        )
        self._adapter_manager = lora_manager
        return lora_manager.model

    def _load_adapter(self, lora_request: LoRARequest) -> LoRAModel:
        try:
            model = self._adapter_manager.model
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
        if lora_request.lora_int_id in self.list_adapters():
            return False
        if isinstance(self._cached_dummy_lora, LoRAModel):
            dummy_lora = self._cached_dummy_lora.clone(
                lora_request.lora_int_id)
        else:
            dummy_lora = self._adapter_manager.create_dummy_lora(
                lora_request.lora_int_id, rank, 1, self.embedding_modules)
            if self._cached_dummy_lora is None:
                self._cached_dummy_lora = dummy_lora
        return self._adapter_manager.add_adapter(dummy_lora)

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: Set[Any],
                            mapping: Optional[Any]) -> None:
        set_active_adapters_worker(requests, mapping, self._apply_adapters,
                                   self._adapter_manager.set_adapter_mapping)

    def _apply_adapters(self, adapter_requests: Set[Any]) -> None:
        apply_adapters_worker(adapter_requests, self.list_adapters,
                              self._adapter_manager.adapter_slots,
                              self.remove_adapter, self.add_adapter)

    def add_adapter(self, adapter_request: Any) -> bool:
        return add_adapter_worker(adapter_request, self.list_adapters,
                                  self._load_adapter,
                                  self._adapter_manager.add_adapter,
                                  self._adapter_manager.activate_adapter)

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> Set[int]:
        return list_adapters_worker(self._adapter_manager.list_adapters)


class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Uses an LRU Cache. Every request, the requested LoRAs will be loaded
    (unless they are already loaded) and least recently used LoRAs will
    be unloaded if the cache is above capacity."""

    _manager_cls: Type[LRUCacheLoRAModelManager] = LRUCacheLoRAModelManager

    def create_lora_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            lora_manager_cls=self._manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._adapter_manager = lora_manager
        return lora_manager.model

    def _apply_adapters(self, lora_requests: Set[LoRARequest]) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._adapter_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._adapter_manager.lora_slots}).")
        for lora in loras_map.values():
            self.add_adapter(lora)

    def add_adapter(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id not in self.list_adapters():
            # Remove before we load the new lora to save memory
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager,
                                  LRUCacheLoRAModelManager)
                self._adapter_manager.remove_oldest_adapter()
            lora = self._load_adapter(lora_request)
            loaded = self._adapter_manager.add_adapter(lora)
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            loaded = self._adapter_manager.get_adapter(
                lora_request.lora_int_id) is not None
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded
