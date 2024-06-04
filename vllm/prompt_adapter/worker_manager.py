import logging
from typing import Any, Optional, Set, Type

import torch

from vllm.adapter_commons.worker_manager import AbstractWorkerManager
from vllm.config import PromptAdapterConfig
from vllm.prompt_adapter.models import (LRUCachePromptAdapterModelManager,
                                        PromptAdapterModel,
                                        PromptAdapterModelManager,
                                        create_prompt_adapter_manager)
from vllm.prompt_adapter.request import PromptAdapterRequest

logger = logging.getLogger(__name__)


class WorkerPromptAdapterManager(AbstractWorkerManager):
    """WorkerPromptAdapterManager that manages 
    prompt_adapter models on the worker side.

    Every request, the requested prompt_adapters will be 
    loaded (unless they are already loaded), 
    and every other prompt_adapter will be unloaded."""

    _manager_cls: Type[PromptAdapterModelManager] = PromptAdapterModelManager

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        device: torch.device,
        prompt_adapter_config: PromptAdapterConfig,
        prompt_adapter_model_cls: Type[PromptAdapterModel] = PromptAdapterModel
    ):
        self._prompt_adapter_manager: Optional[
            PromptAdapterModelManager] = None
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self._prompt_adapter_model_cls = prompt_adapter_model_cls
        self.prompt_adapter_config = prompt_adapter_config
        super().__init__(device)

    @property
    def is_enabled(self) -> bool:
        return True

    def reset_adapter(self):
        self._prompt_adapter_manager.reset_adapter()

    def create_prompt_adapter_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        prompt_adapter_manager = create_prompt_adapter_manager(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            prompt_adapter_config=self.prompt_adapter_config,
            prompt_adapter_manager_cls=self._manager_cls,
        )
        self._prompt_adapter_manager = prompt_adapter_manager
        return prompt_adapter_manager.model

    @property
    def set_active_prompt_adapters(self):
        return self.set_active_adapters

    def _load_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest
    ) -> PromptAdapterModel:
        try:
            prompt_adapter = self._prompt_adapter_model_cls\
                .from_local_checkpoint(
                    prompt_adapter_request.prompt_adapter_local_path,
                    prompt_adapter_id=prompt_adapter_request.prompt_adapter_id,
                    torch_device=str(self.device)
                )
        except Exception as e:
            raise RuntimeError(
                f"Loading prompt_adapter "
                f"{prompt_adapter_request.prompt_adapter_local_path}"
                f" failed") from e
        return prompt_adapter

    def add_dummy_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return True

    @property
    def add_dummy_adapter(self):
        return self.add_dummy_prompt_adapter

    @property
    def create_manager(self):
        return self.create_prompt_adapter_manager

    @property
    def _load_adapter(self):
        return self._load_prompt_adapter

    @property
    def _model_manager(self):
        return self._prompt_adapter_manager

    @property
    def add_prompt_adapter(self):
        return self.add_adapter

    @property
    def remove_prompt_adapter(self):
        return self.remove_adapter

    @property
    def remove_all_prompt_adapters(self):
        return self.remove_all_adapters

    @property
    def list_prompt_adapters(self):
        return self.list_adapters

    @property
    def _apply_prompt_adapters(self):
        return self._apply_adapters


class LRUCacheWorkerPromptAdapterManager(WorkerPromptAdapterManager):
    """WorkerPromptAdapterManager that manages 
    prompt_adapter models on the worker side.

    Uses an LRU Cache. Every request, the requested 
    prompt_adapters will be loaded (unless they are already loaded) 
    and least recently used prompt_adapters will
    be unloaded if the cache is above capacity."""

    _prompt_adapter_manager_cls: Type[
        LRUCachePromptAdapterModelManager] = LRUCachePromptAdapterModelManager

    def create_prompt_adapter_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        prompt_adapter_manager = create_prompt_adapter_manager(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            prompt_adapter_config=self.prompt_adapter_config,
            prompt_adapter_manager_cls=self._prompt_adapter_manager_cls)
        self._prompt_adapter_manager: \
            LRUCachePromptAdapterModelManager = prompt_adapter_manager
        return prompt_adapter_manager.model

    def _apply_adapters(
            self, prompt_adapter_requests: Set[PromptAdapterRequest]) -> None:
        prompt_adapters_map = {
            prompt_adapter_request.prompt_adapter_id: prompt_adapter_request
            for prompt_adapter_request in prompt_adapter_requests
            if prompt_adapter_request
        }
        if len(prompt_adapters_map
               ) > self._prompt_adapter_manager.prompt_adapter_slots:
            raise RuntimeError(
                f"Number of requested prompt_adapters "
                f"({len(prompt_adapters_map)}) is greater "
                "than the number of GPU prompt_adapter slots "
                f"({self._prompt_adapter_manager.prompt_adapter_slots}).")
        for prompt_adapter in prompt_adapters_map.values():
            self.add_prompt_adapter(prompt_adapter)

    def add_adapter(self,
                    prompt_adapter_request: PromptAdapterRequest) -> bool:
        if prompt_adapter_request.prompt_adapter_id not in \
            self.list_prompt_adapters():
            # Remove before we load the new prompt_adapter to save memory
            if len(self._prompt_adapter_manager
                   ) + 1 > self._prompt_adapter_manager.capacity:
                self._prompt_adapter_manager.remove_oldest_prompt_adapter()
            prompt_adapter = self._load_prompt_adapter(prompt_adapter_request)
            loaded = self._prompt_adapter_manager.add_prompt_adapter(
                prompt_adapter)
        else:
            # If the prompt_adapter is already loaded, just touch it to
            # update its position in the caches
            loaded = self._prompt_adapter_manager.get_prompt_adapter(
                prompt_adapter_request.prompt_adapter_id)
        self._prompt_adapter_manager.activate_prompt_adapter(
            prompt_adapter_request.prompt_adapter_id)
        return loaded
