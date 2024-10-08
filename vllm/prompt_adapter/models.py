import logging
import math
from typing import Any, Callable, Dict, List, Optional, Type

import torch
from torch import nn

from vllm.adapter_commons.models import (AdapterLRUCache, AdapterModel,
                                         AdapterModelManager)
from vllm.adapter_commons.utils import (add_adapter, deactivate_adapter,
                                        get_adapter, list_adapters,
                                        remove_adapter, set_adapter_mapping)
from vllm.config import PromptAdapterConfig
from vllm.prompt_adapter.layers import (
    VocabParallelEmbeddingWithPromptAdapter)  # yapf: disable
from vllm.prompt_adapter.layers import PromptAdapterMapping
from vllm.prompt_adapter.utils import load_peft_weights

logger = logging.getLogger(__name__)

_GLOBAL_PROMPT_ADAPTER_ID = 0


def get_prompt_adapter_id():
    global _GLOBAL_PROMPT_ADAPTER_ID
    _GLOBAL_PROMPT_ADAPTER_ID += 1
    return _GLOBAL_PROMPT_ADAPTER_ID


def convert_to_embedding_indices(indices):
    embedding_indices = []
    count = 0

    for value in indices:
        if value == -1:
            count = 0
        else:
            embedding_indices.append([value, count])
            count += 1

    return torch.tensor(embedding_indices)


def convert_mapping(
    mapping: PromptAdapterMapping,
    prompt_adapter_index_to_id: List[Optional[int]],
) -> torch.Tensor:
    """Converts PromptAdapterMapping to index tensors.

    Args:
        mapping: PromptAdapterMapping mapping rows in a 
                batch to PromptAdapter ids.
        prompt_adapter_index_to_id: List mapping PromptAdapter 
                ids to PromptAdapter indices.
        
    Returns:
        pa_indices: Tensor of shape [batch_size] mapping batch rows to
            PromptAdapter indices.
    """
    id_to_index = {
        id_: idx
        for idx, id_ in enumerate(prompt_adapter_index_to_id)
        if id_ is not None
    }
    pa_indices = ([
        id_to_index.get(id_, -1) if id_ > 0 else -1
        for id_ in mapping.index_mapping
    ])

    pa_embedding_mapping = convert_to_embedding_indices(pa_indices)
    pa_indices = torch.tensor(pa_indices)
    return pa_indices, pa_embedding_mapping


class PromptAdapterModel(AdapterModel):

    def __init__(self,
                 prompt_adapter_id=None,
                 num_virtual_tokens=None,
                 prompt_embedding=None) -> None:
        self.id = prompt_adapter_id
        self.prompt_embedding = prompt_embedding
        self.num_virtual_tokens = num_virtual_tokens

    @classmethod
    def from_local_checkpoint(
        cls,
        adapter_model_path: str,
        prompt_adapter_id: int,
        num_virtual_tokens: int,
        config: PromptAdapterConfig,
        device: str = "cuda",
    ) -> "PromptAdapterModel":

        if num_virtual_tokens > config.max_prompt_adapter_token:
            raise ValueError(
                f'num_virtual_tokens ({num_virtual_tokens}) should be <= '
                f'max_prompt_adapter_token({config.max_prompt_adapter_token})')

        adapters_weights = load_peft_weights(adapter_model_path, device)
        prompt_embedding = adapters_weights["prompt_embeddings"].to(
            config.prompt_adapter_dtype)

        return cls(prompt_adapter_id, num_virtual_tokens, prompt_embedding)


class PromptAdapterModelManager(AdapterModelManager):
    """A manager that manages multiple Prompt Adapter models."""

    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        prompt_adapter_config: PromptAdapterConfig,
    ):
        """Create a PromptAdapterModel and adapter for a given model.

        Args:
            model: the model to be adapted.
            max_num_seqs: the maximum number of sequences model can run in a
                single batch.
            max_num_batched_tokens: the maximum number of tokens model can run
                in a single batch.
            prompt_adapter_config: the PromptAdapter config,
        """
        self.model: nn.Module = model
        # Dict instead of a Set for compatibility with LRUCache.
        self.prompt_adapter_index_to_id: List[
            Optional[int]] = [None] * self.prompt_adapter_slots
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.prompt_adapter_config = prompt_adapter_config
        self.model.prompt_adapter_manager = self
        self.adapter_type = 'PromptAdapter'

        self.base_indices = torch.tensor([-1])
        self.base_embedding_indices = torch.tensor([])

        self.modules: Dict[str, nn.Module] = {}
        self._create_prompt_adapter_modules()
        self._last_mapping: Optional[PromptAdapterMapping] = None

    @property
    def prompt_adapter_slots(self) -> int:
        return self.prompt_adapter_config.max_prompt_adapters

    @property
    def adapter_slots(self) -> int:
        return self.prompt_adapter_slots

    @property
    def capacity(self) -> int:
        return self.prompt_adapter_config.max_cpu_prompt_adapters

    def activate_adapter(
        self,
        prompt_adapter_id: int,
    ) -> bool:
        """Move PromptAdapter into a GPU buffer 
            to be used in the forward pass."""
        if prompt_adapter_id in self._active_adapters:
            return False
        first_free_slot = next(
            ((i, prompt_adapter_id) for i, prompt_adapter_id in enumerate(
                self.prompt_adapter_index_to_id) if prompt_adapter_id is None),
            None)
        if first_free_slot is None:
            raise ValueError("No free prompt_adapter slots")
        index, _ = first_free_slot
        self._active_adapters[prompt_adapter_id] = None
        prompt_adapter_model = (self._registered_adapters[prompt_adapter_id])
        logger.debug("Activating prompt_adapter. int id: %d, slot index: %d",
                     prompt_adapter_model.id, index)
        self.prompt_adapter_index_to_id[index] = prompt_adapter_model.id
        for _, v in self.modules.items():
            v.set_prompt_adapter(index, prompt_adapter_model.prompt_embedding)
        return True

    def _deactivate_adapter(self, prompt_adapter_id: int):
        try:
            index = self.prompt_adapter_index_to_id.index(prompt_adapter_id)
            self.prompt_adapter_index_to_id[index] = None
            for _, v in self.modules.items():
                v.reset_prompt_adapter(index)
        except ValueError:
            pass

    def _add_adapter(self, prompt_adapter: PromptAdapterModel):
        self._registered_adapters[prompt_adapter.id] = prompt_adapter

    def _set_adapter_mapping(self, mapping: PromptAdapterMapping) -> None:
        base_indices, base_embedding_indices = convert_mapping(
            mapping, self.prompt_adapter_index_to_id)
        for k, v in self.modules.items():
            v.set_mapping(base_indices, base_embedding_indices)

    def _create_prompt_adapter_modules(self):
        for module_name, module in self.model.named_modules(
                remove_duplicate=False):
            if "VocabParallel" in module.__class__.__name__:
                new_module = VocabParallelEmbeddingWithPromptAdapter(module)
                new_module.create_prompt_adapter_weights(
                    self.prompt_adapter_config)
                replaced_module = self.replace_submodule(
                    self.model, module_name, new_module)
                self.register_module(module.__class__.__name__,
                                     replaced_module)
                replaced_module.set_mapping(self.base_indices,
                                            self.base_embedding_indices)
                break

    def replace_submodule(self, model: nn.Module, module_name: str,
                          new_module: nn.Module) -> nn.Module:
        """Replace a submodule in a model with a new module."""
        parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
        target_name = module_name.split(".")[-1]
        setattr(parent, target_name, new_module)
        return new_module

    def register_module(self, module_name: str, module: nn.Module):
        self.modules[module_name] = module

    def pin_adapter(self, prompt_adapter_id: int) -> bool:
        """Pin a PromptAdapterModel in the manager cache."""
        raise NotImplementedError(
            "Pinning is not supported in PromptAdapterModelManager."
            "Use LRUCachePromptAdapterModelManager for pinning"
        )  # type: ignore

    def remove_all_adapters(self):
        """Remove all PromptAdapterModel from the manager."""
        self._registered_adapters.clear()
        self.prompt_adapter_index_to_id = [None] * self.prompt_adapter_slots
        self._active_adapters.clear()

    def deactivate_adapter(self, adapter_id: int) -> bool:
        return deactivate_adapter(adapter_id, self._active_adapters,
                                  self._deactivate_adapter)

    def add_adapter(self, adapter: PromptAdapterModel) -> bool:
        return add_adapter(adapter, self._registered_adapters, self.capacity,
                           self._add_adapter)

    def set_adapter_mapping(self, mapping: PromptAdapterMapping) -> None:
        self._last_mapping = set_adapter_mapping(mapping, self._last_mapping,
                                                 self._set_adapter_mapping)

    def remove_adapter(self, adapter_id: int) -> bool:
        return remove_adapter(adapter_id, self._registered_adapters,
                              self.deactivate_adapter)

    def list_adapters(self) -> Dict[int, Any]:
        return list_adapters(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        return get_adapter(adapter_id, self._registered_adapters)


class PromptAdapterLRUCache(AdapterLRUCache[PromptAdapterModel]):

    def __init__(self, capacity: int,
                 deactivate_prompt_adapter_fn: Callable[[int], bool]):
        super().__init__(capacity, deactivate_prompt_adapter_fn)


class LRUCachePromptAdapterModelManager(PromptAdapterModelManager):
    """A model manager that manages multiple prompt_adapters with LRU cache."""

    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        prompt_adapter_config: PromptAdapterConfig,
    ):
        self.prompt_adapter_config = prompt_adapter_config
        super().__init__(model, max_num_seqs, max_num_batched_tokens,
                         prompt_adapter_config)
        self._registered_adapters = PromptAdapterLRUCache(
            self.capacity, self.deactivate_adapter)
        self._active_adapters = PromptAdapterLRUCache(
            self.prompt_adapter_slots, self._deactivate_adapter)

    def list_adapters(self) -> Dict[int, PromptAdapterModel]:
        """List all registered PromptAdapterModel."""
        return dict(self._registered_adapters.cache)

    def add_adapter(self, prompt_adapter: PromptAdapterModel) -> bool:
        """Add a PromptAdapterModel to the manager."""
        if prompt_adapter.id not in self._registered_adapters:
            self._add_adapter(prompt_adapter)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_adapters.touch(prompt_adapter.id)
            was_added = False
        return was_added

    def activate_adapter(
        self,
        prompt_adapter_id: int,
    ) -> bool:
        if prompt_adapter_id not in self._active_adapters and len(
                self._active_adapters) >= self.prompt_adapter_slots:
            self._active_adapters.remove_oldest()
        result = super().activate_adapter(prompt_adapter_id)
        # We always touch to update the LRU cache order
        self._active_adapters.touch(prompt_adapter_id)
        return result

    def remove_oldest_adapter(self) -> bool:
        if len(self._registered_adapters) > 0:
            self._registered_adapters.remove_oldest()
            return True
        return False

    def pin_adapter(self, prompt_adapter_id: int) -> bool:
        """Pin a PromptAdapterModel in the manager cache."""
        self._pin_prompt_adapter_in_cpu_cache(prompt_adapter_id)
        self._pin_prompt_adapter_in_gpu_cache(prompt_adapter_id)
        return True

    def _pin_prompt_adapter_in_cpu_cache(self, prompt_adapter_id: int):
        try:
            self._registered_adapters.pin(prompt_adapter_id)
        except ValueError as err:
            raise ValueError(
                "Pinning failed. "
                f"Prompt Adapter {prompt_adapter_id} is not registered."
            ) from err

    def _pin_prompt_adapter_in_gpu_cache(self, prompt_adapter_id: int):
        if prompt_adapter_id not in self._active_adapters:
            # move adapter to gpu if not already active
            self.activate_adapter(prompt_adapter_id)
        self._active_adapters.pin(prompt_adapter_id)


def create_prompt_adapter_manager(
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        prompt_adapter_config: PromptAdapterConfig,
        prompt_adapter_manager_cls: Type[
            PromptAdapterModelManager] = PromptAdapterModelManager,
        **kwargs) -> PromptAdapterModelManager:
    """Create a PromptAdapterModel for a given model."""
    prompt_adapter_manager = prompt_adapter_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        prompt_adapter_config=prompt_adapter_config,
        **kwargs)
    return prompt_adapter_manager
