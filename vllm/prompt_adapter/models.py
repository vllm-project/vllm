import logging
import math
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from peft.utils import load_peft_weights
from torch import nn

from vllm.adapter_commons.models import (AdapterLRUCache, AdapterModel,
                                         AdapterModelManager)
from vllm.config import PromptAdapterConfig
from vllm.prompt_adapter.layers import PromptAdapterMapping

logger = logging.getLogger(__name__)

_GLOBAL_PROMPT_ADAPTER_ID = 0


def get_prompt_adapter_id():
    global _GLOBAL_PROMPT_ADAPTER_ID
    _GLOBAL_PROMPT_ADAPTER_ID += 1
    return _GLOBAL_PROMPT_ADAPTER_ID


def convert_mapping(
    mapping: PromptAdapterMapping,
    prompt_adapter_index_to_id: List[Optional[int]], max_prompt_adapters: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Converts PromptAdapterMapping to index tensors.

    Args:
        mapping: PromptAdapterMapping mapping rows in a batch to ids.
        prompt_adapter_index_to_id: List mapping PromptAdapter ids to indices.
        max_prompt_adapters: Maximum number of PromptAdapters.
    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                PromptAdapter indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                PromptAdapter indices for sampler. For generation, this will be
                same as base_indicies. For prefill, this will map requests
                to PromptAdapter indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to PromptAdapter indices for sampler with padding.
                Same as sampler_indicies, but -1 is replaced with
                max_promt_adapters.
            indices_len: List of lengths of the above tensors.
                Used to index into each tensor. It contains length for
                (base_indices, sampler_indices, sampler_indices_padded).
    """
    index_mapping_indices: List[int] = list(mapping.index_mapping).copy()
    prompt_adapter_indices = index_mapping_indices.copy()
    prompt_mapping: List[int] = [
        prompt_adapter_index_to_id.index(x) if x > 0 else -1
        for x in mapping.prompt_mapping
    ]
    prompt_adapter_idx = None
    for i in range(len(index_mapping_indices)):
        # TODO index can be slow. optimize
        prompt_adapter_idx = (prompt_adapter_index_to_id.index(
            index_mapping_indices[i]) if index_mapping_indices[i] > 0 else -1)
        prompt_adapter_indices[i] = prompt_adapter_idx

    indices_list: List[Union[List[int], torch.Tensor]] = [
        index_mapping_indices, prompt_adapter_indices
    ]
    indices = torch.tensor(indices_list, dtype=torch.long, device="cuda")
    prompt_mapping_tensor = torch.tensor(prompt_mapping,
                                         device="cuda",
                                         dtype=torch.long)
    base_indices = indices[1]
    sampler_indices = prompt_mapping_tensor
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded[sampler_indices_padded ==
                           -1] = max_prompt_adapters - 1
    sampler_indices_padded = (
        torch.arange(
            0, len(sampler_indices_padded), device="cuda", dtype=torch.long) +
        (sampler_indices_padded * len(sampler_indices_padded)))
    # Contain length of indices tensors. Used to index into each tensor.
    indices_len = [
        base_indices.shape[-1], sampler_indices.shape[-1],
        sampler_indices_padded.shape[-1]
    ]
    return (base_indices, sampler_indices, sampler_indices_padded, indices_len)


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

        self.base_indices = torch.empty(self.max_num_batched_tokens,
                                        dtype=torch.long,
                                        device="cuda")
        self.sampler_indices = torch.empty(self.max_num_batched_tokens,
                                           dtype=torch.long,
                                           device="cuda")
        self.sampler_indices_padded = torch.empty(self.max_num_batched_tokens,
                                                  dtype=torch.long,
                                                  device="cuda")
        self.indices_len: List[Optional[int]] = [None] * 3
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
        for module_name, module in self.model.named_modules():
            if 'Model' in (module.__class__.__name__):
                module.prefix_encoder = prompt_adapter_model
                break
        return True

    @property
    def activate_adapter(self):
        return self.activate_prompt_adapter

    def _deactivate_prompt_adapter(self, prompt_adapter_id: int):
        try:
            index = self.prompt_adapter_index_to_id.index(prompt_adapter_id)
            self.prompt_adapter_index_to_id[index] = None
            for module_name, module in self.model.named_modules():
                if 'Model' in (module.__class__.__name__):
                    module.prefix_encoder = None
                    break
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
        (base_indices, sampler_indices, sampler_indices_padded,
         indices_len) = convert_mapping(mapping,
                                        self.prompt_adapter_index_to_id,
                                        self.prompt_adapter_slots + 1)
        self.base_indices[:base_indices.shape[0]].copy_(base_indices)
        self.sampler_indices[:sampler_indices.shape[0]].copy_(sampler_indices)
        self.sampler_indices_padded[:sampler_indices_padded.shape[0]].copy_(
            sampler_indices_padded)
        # Maintain the reference
        self.indices_len[:] = indices_len

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
