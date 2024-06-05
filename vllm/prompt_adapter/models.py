import logging
import math
from typing import Callable, Dict, List, Optional, Type

from peft.utils import load_peft_weights
from torch import nn

from vllm.adapter_commons.models import (AdapterLRUCache, AdapterModel,
                                         AdapterModelManager)
from vllm.config import PromptAdapterConfig
from vllm.prompt_adapter.layers import (PromptAdapterMapping, 
                                        VocabParallelEmbeddingWithPromptAdapter)

logger = logging.getLogger(__name__)

_GLOBAL_PROMPT_ADAPTER_ID = 0


def get_prompt_adapter_id():
    global _GLOBAL_PROMPT_ADAPTER_ID
    _GLOBAL_PROMPT_ADAPTER_ID += 1
    return _GLOBAL_PROMPT_ADAPTER_ID


class PromptAdapterModel(AdapterModel):

    def __init__(self,
                 prompt_adapter_id=None,
                 num_virtual_tokens=None,
                 prompt_embedding=None) -> None:
        self.id = prompt_adapter_id
        self.kv_cache = None
        self.prompt_embedding = prompt_embedding
        self.num_virtual_tokens = num_virtual_tokens

    @classmethod
    def from_local_checkpoint(cls,
                              adapter_model_and_path,
                              prompt_adapter_id,
                              torch_device='cuda') -> "PromptAdapterModel":
        adapters_weights = load_peft_weights(adapter_model_and_path,
                                             torch_device)
        prompt_embedding = adapters_weights["prompt_embeddings"].half()
        num_virtual_tokens = prompt_embedding.shape[0]
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
        """
        self.model: nn.Module = model
        # Dict instead of a Set for compatibility with LRUCache.
        self.prompt_adapter_index_to_id: List[Optional[int]] =\
                                         [None] * self.prompt_adapter_slots
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.prompt_adapter_config = prompt_adapter_config
        self.model.prompt_adapter_manager = self
        self.adapter_type = 'PromptAdapter'

        self.base_indices = [0]
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

    def activate_prompt_adapter(
        self,
        prompt_adapter_id: int,
    ) -> bool:
        """Move PromptAdapter into a GPU buffer 
            to be used in the forward pass."""
        if prompt_adapter_id in self._active_adapters:
            return False
        first_free_slot = next(
            ((i, prompt_adapter_id) for i, prompt_adapter_id in \
                            enumerate(self.prompt_adapter_index_to_id)
             if prompt_adapter_id is None), None)
        if first_free_slot is None:
            raise ValueError("No free prompt_adapter slots")
        index, _ = first_free_slot
        self._active_adapters[prompt_adapter_id] = None
        prompt_adapter_model = \
            self._registered_adapters[prompt_adapter_id]
        logger.debug("Activating prompt_adapter. int id: %d, slot index: %d",
                     prompt_adapter_model.id, index)
        self.prompt_adapter_index_to_id[index] = prompt_adapter_model.id
        for _, v in self.modules.items():
            v.set_prompt_adapter(prompt_adapter_id, prompt_adapter_model)
        return True

    @property
    def activate_adapter(self):
        return self.activate_prompt_adapter

    def _deactivate_prompt_adapter(self, prompt_adapter_id: int):
        try:
            index = self.prompt_adapter_index_to_id.index(prompt_adapter_id)
            self.prompt_adapter_index_to_id[index] = None
            for _, v in self.modules.items():
                v.reset_prompt_adapter(prompt_adapter_id)
        except ValueError:
            pass

    @property
    def _deactivate_adapter(self):
        return self._deactivate_prompt_adapter

    @property
    def deactivate_prompt_adapter(self):
        return self.deactivate_adapter

    def _add_prompt_adapter(self, prompt_adapter: PromptAdapterModel):
        self._registered_adapters[prompt_adapter.id] = prompt_adapter

    @property
    def _add_adapter(self):
        return self._add_prompt_adapter

    @property
    def add_prompt_adapter(self):
        return self.add_adapter

    @property
    def remove_prompt_adapter(self):
        return self.remove_adapter

    def _set_prompt_adapter_mapping(self,
                                    mapping: PromptAdapterMapping) -> None:
        for k, v in self.modules.items():
            v.set_mapping(mapping.index_mapping)

    def _create_prompt_adapter_modules(self):
        for module_name, module in self.model.named_modules(
                remove_duplicate=False):
            if "VocabParallel" in module.__class__.__name__:
                new_module = VocabParallelEmbeddingWithPromptAdapter(module)
                replaced_module = self.replace_submodule(
                    self.model, module_name, new_module)
                self.register_module(module.__class__.__name__,
                                     replaced_module)
                replaced_module.set_mapping(self.base_indices)

    def replace_submodule(self, model: nn.Module, module_name: str,
                          new_module: nn.Module) -> nn.Module:
        """Replace a submodule in a model with a new module."""
        parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
        target_name = module_name.split(".")[-1]
        setattr(parent, target_name, new_module)
        return new_module

    def register_module(self, module_name: str, module: nn.Module):
        self.modules[module_name] = module

    @property
    def set_prompt_adapter_mapping(self):
        return self.set_adapter_mapping

    @property
    def _set_adapter_mapping(self):
        return self._set_prompt_adapter_mapping

    @property
    def list_prompt_adapters(self):
        return self.list_adapters

    @property
    def get_prompt_adapter(self):
        return self.get_adapter

    def remove_all_prompt_adapters(self):
        """Remove all PromptAdapterModel from the manager."""
        self._registered_adapters.clear()
        self.prompt_adapter_index_to_id = [None] * self.prompt_adapter_slots
        self._active_adapters.clear()

    @property
    def remove_all_adapters(self):
        return self.remove_all_prompt_adapters


class PromptAdapterLRUCache(AdapterLRUCache[PromptAdapterModel]):

    def __init__(self, capacity: int,
                 deactivate_prompt_adapter_fn: Callable[[int], None]):
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
        super().__init__(model, max_num_seqs, \
                         max_num_batched_tokens, prompt_adapter_config)
        self._registered_adapters: PromptAdapterLRUCache = \
                PromptAdapterLRUCache(self.capacity,
                                    self.deactivate_prompt_adapter)
        self._active_adapters: PromptAdapterLRUCache = \
                PromptAdapterLRUCache(self.prompt_adapter_slots,
                                  self._deactivate_prompt_adapter)

    def list_adapters(self) -> Dict[int, PromptAdapterModel]:
        """List all registered PromptAdapterModel."""
        return dict(self._registered_adapters.cache)

    def add_adapter(self, prompt_adapter: PromptAdapterModel) -> bool:
        """Add a PromptAdapterModel to the manager."""
        if prompt_adapter.id not in self._registered_adapters:
            self._add_prompt_adapter(prompt_adapter)
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
        result = super().activate_prompt_adapter(prompt_adapter_id)
        # We always touch to update the LRU cache order
        self._active_adapters.touch(prompt_adapter_id)
        return result

    def remove_oldest_prompt_adapter(self) -> bool:
        if len(self._registered_adapters) > 0:
            self._registered_adapters.remove_oldest()
            return True
        return False


def create_prompt_adapter_manager(
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        prompt_adapter_config: PromptAdapterConfig,
        prompt_adapter_manager_cls: Type[PromptAdapterModelManager] \
                                    = PromptAdapterModelManager,
        **kwargs) -> PromptAdapterModelManager:
    """Create a PromptAdapterModel for a given model."""
    prompt_adapter_manager = prompt_adapter_manager_cls(
        model=model, max_num_seqs=max_num_seqs, \
        max_num_batched_tokens=max_num_batched_tokens, \
        prompt_adapter_config=prompt_adapter_config, **kwargs)
    return prompt_adapter_manager
