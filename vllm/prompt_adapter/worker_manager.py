# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional, Set, Type

import torch

from vllm.adapter_commons.utils import (add_adapter_worker,
                                        apply_adapters_worker,
                                        list_adapters_worker,
                                        set_active_adapters_worker)
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
        self._adapter_manager: PromptAdapterModelManager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self._prompt_adapter_model_cls = prompt_adapter_model_cls
        self.prompt_adapter_config = prompt_adapter_config
        super().__init__(device)

    @property
    def is_enabled(self) -> bool:
        return True

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
        self._adapter_manager = prompt_adapter_manager
        return prompt_adapter_manager.model

    def _load_adapter(
            self, prompt_adapter_request: PromptAdapterRequest
    ) -> PromptAdapterModel:
        try:
            prompt_adapter = (
                self._prompt_adapter_model_cls.from_local_checkpoint(
                    prompt_adapter_request.prompt_adapter_local_path,
                    prompt_adapter_id=prompt_adapter_request.prompt_adapter_id,
                    num_virtual_tokens=prompt_adapter_request.
                    prompt_adapter_num_virtual_tokens,
                    config=self.prompt_adapter_config,
                    device=str(self.device),
                ))
        except Exception as e:
            raise RuntimeError(
                f"Loading prompt_adapter "
                f"{prompt_adapter_request.prompt_adapter_local_path}"
                f" failed") from e
        return prompt_adapter

    def add_dummy_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return True

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: Set[Any],
                            mapping: Optional[Any]) -> None:
        set_active_adapters_worker(requests, mapping, self._apply_adapters,
                                   self._adapter_manager.set_adapter_mapping)

    def add_adapter(self, adapter_request: Any) -> bool:
        return add_adapter_worker(adapter_request, self.list_adapters,
                                  self._load_adapter,
                                  self._adapter_manager.add_adapter,
                                  self._adapter_manager.activate_adapter)

    def _apply_adapters(self, adapter_requests: Set[Any]) -> None:
        apply_adapters_worker(adapter_requests, self.list_adapters,
                              self._adapter_manager.adapter_slots,
                              self.remove_adapter, self.add_adapter)

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> Set[int]:
        return list_adapters_worker(self._adapter_manager.list_adapters)


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
        self._adapter_manager: LRUCachePromptAdapterModelManager = (
            prompt_adapter_manager)
        return prompt_adapter_manager.model

    def _apply_adapters(
            self, prompt_adapter_requests: Set[PromptAdapterRequest]) -> None:
        prompt_adapters_map = {
            prompt_adapter_request.prompt_adapter_id: prompt_adapter_request
            for prompt_adapter_request in prompt_adapter_requests
            if prompt_adapter_request
        }
        if len(prompt_adapters_map
               ) > self._adapter_manager.prompt_adapter_slots:
            raise RuntimeError(
                f"Number of requested prompt_adapters "
                f"({len(prompt_adapters_map)}) is greater "
                "than the number of GPU prompt_adapter slots "
                f"({self._adapter_manager.prompt_adapter_slots}).")
        for prompt_adapter in prompt_adapters_map.values():
            self.add_adapter(prompt_adapter)

    def add_adapter(self,
                    prompt_adapter_request: PromptAdapterRequest) -> bool:
        if prompt_adapter_request.prompt_adapter_id not in self.list_adapters(
        ):
            # Remove before we load the new prompt_adapter to save memory
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                self._adapter_manager.remove_oldest_adapter()
            prompt_adapter = self._load_adapter(prompt_adapter_request)
            loaded = self._adapter_manager.add_adapter(prompt_adapter)
        else:
            # If the prompt_adapter is already loaded, just touch it to
            # update its position in the caches
            loaded = self._adapter_manager.get_adapter(
                prompt_adapter_request.prompt_adapter_id) is not None
        self._adapter_manager.activate_adapter(
            prompt_adapter_request.prompt_adapter_id)
        return loaded
