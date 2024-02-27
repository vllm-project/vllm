import copy
import json
import logging
import math
import os
import re
from typing import (Any, Callable, Dict, Hashable, List, Optional, Tuple, Type)

import safetensors.torch
import torch
from torch import nn

from vllm.config import LoRAConfig
from vllm.utils import LRUCache, in_wsl

from vllm.lora.layers import BaseLayerWithLoRA, LoRAMapping, from_layer, from_layer_sampler
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.utils import parse_fine_tuned_lora_name, replace_submodule

logger = logging.getLogger(__name__)

_GLOBAL_LORA_ID = 0


def convert_mapping(
    mapping: LoRAMapping, lora_index_to_id: List[Optional[int]],
    max_loras: int, vocab_size: int, extra_vocab_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Converts LoRAMapping to index tensors.

    Args:
        mapping: LoRAMapping mapping rows in a batch to LoRA ids.
        lora_index_to_id: List mapping LoRA ids to LoRA indices.
        max_loras: Maximum number of LoRAs.
        vocab_size: Model vocab size.
        extra_vocab_size: Extra vocab size each LoRA can have.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                LoRA indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                LoRA indices for sampler. For generation, this will be the
                same as base_indicies. For prefill, this will map requests
                to LoRA indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to LoRA indices for sampler with padding.
                Same as sampler_indicies, but -1 is replaced with
                max_loras.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the LoRAs, second row is for the LoRA.lora_a
                embeddings.
            indices_len: List of lengths of the above tensors.
    """
    indices = list(mapping.index_mapping).copy()
    embedding_indices = indices.copy()
    lora_indices = indices.copy()
    prompt_mapping = [
        lora_index_to_id.index(x) if x > 0 else -1
        for x in mapping.prompt_mapping
    ]
    lora_idx = None
    for i in range(len(indices)):
        # TODO index can be slow. optimize
        lora_idx = (lora_index_to_id.index(indices[i])
                    if indices[i] > 0 else -1)
        embedding_indices[i] = lora_idx if indices[i] > 0 else 0
        indices[i] = i
        lora_indices[i] = lora_idx

    indices = torch.tensor([indices, lora_indices, embedding_indices],
                           dtype=torch.long,
                           device="cuda")
    prompt_mapping = torch.tensor(prompt_mapping,
                                  device="cuda",
                                  dtype=torch.long)
    embeddings_indices = torch.stack([
        indices[2] * extra_vocab_size,
        indices[2] * (vocab_size + extra_vocab_size)
    ])
    embeddings_indices[embeddings_indices == -1] = max_loras - 1
    base_indices = indices[1]
    sampler_indices = prompt_mapping
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded[sampler_indices_padded == -1] = max_loras - 1
    sampler_indices_padded = (
        torch.arange(
            0, len(sampler_indices_padded), device="cuda", dtype=torch.long) +
        (sampler_indices_padded * len(sampler_indices_padded)))
    indices_len = (base_indices.shape[-1], sampler_indices.shape[-1],
                   sampler_indices_padded.shape[-1],
                   embeddings_indices.shape[-1])

    return (base_indices, sampler_indices, sampler_indices_padded,
            embeddings_indices, indices_len)


def get_lora_id():
    global _GLOBAL_LORA_ID
    _GLOBAL_LORA_ID += 1
    return _GLOBAL_LORA_ID


class LoRAModel:
    """A LoRA fine-tuned model."""

    def __init__(
        self,
        lora_model_id: int,
        rank: int,
        loras: Dict[str, LoRALayerWeights],
    ) -> None:
        self.id = lora_model_id
        assert (lora_model_id >
                0), f"a valid lora id should be greater than 0, got {self.id}"
        self.rank = rank
        self.loras: Dict[str, LoRALayerWeights] = loras

    @property
    def extra_vocab_size(self) -> int:
        return max(lora.extra_vocab_size
                   for lora in self.loras.values()) if self.loras else 0

    def get_lora(self, module_name: str) -> Optional[LoRALayerWeights]:
        """Get LoRA for a given module by name"""
        return self.loras.get(module_name, None)

    # (yard1): TODO see if we can derive target_embedding_padding automatically
    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id: int,
        rank: int,
        lora_alpha: int,
        tensors: Dict[str, torch.Tensor],
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        embeddings: Optional[Dict[str, torch.Tensor]] = None,
        target_embedding_padding: Optional[int] = None,
        embedding_modules: Optional[Dict[str, str]] = None,
        embedding_padding_modules: Optional[List[str]] = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a dictionary of tensors."""
        pin_memory = str(device) == "cpu" and not in_wsl()
        loras: Dict[str, LoRALayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_name)
            if module_name not in loras:
                lora_embeddings_tensor = None
                if embeddings:
                    embeddings_module = next(
                        (k for k in embedding_modules if k in module_name),
                        None)
                    if embeddings_module:
                        lora_embeddings_tensor = embeddings[
                            embedding_modules[embeddings_module]].to(
                                device=device, dtype=dtype)
                        if pin_memory:
                            lora_embeddings_tensor = (
                                lora_embeddings_tensor.pin_memory())
                loras[module_name] = LoRALayerWeights(module_name, rank,
                                                      lora_alpha, None, None,
                                                      lora_embeddings_tensor)
            if is_lora_a:
                loras[module_name].lora_a = tensor.to(device=device,
                                                      dtype=dtype).t()
                if pin_memory:
                    loras[module_name].lora_a = loras[
                        module_name].lora_a.pin_memory()
            else:
                loras[module_name].lora_b = tensor.to(device=device,
                                                      dtype=dtype).t()
                if any(name in module_name
                       for name in embedding_padding_modules
                       ) and target_embedding_padding is not None:
                    lora_b = loras[module_name].lora_b
                    assert target_embedding_padding >= lora_b.shape[1]
                    addition = target_embedding_padding - lora_b.shape[1]
                    loras[module_name].lora_b = torch.nn.functional.pad(
                        lora_b, (0, addition))
                if pin_memory:
                    loras[module_name].lora_b = loras[
                        module_name].lora_b.pin_memory()

        for lora in loras.values():
            lora.optimize()
        return cls(lora_model_id, rank, loras)

    @classmethod
    def from_local_checkpoint(
        cls,
        lora_dir: str,
        lora_model_id: Optional[int] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        target_embedding_padding: Optional[int] = None,
        embedding_modules: Optional[Dict[str, str]] = None,
        embedding_padding_modules: Optional[List[str]] = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint."""
        lora_config_path = os.path.join(lora_dir, "adapter_config.json")
        lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
        new_embeddings_tensor_path = os.path.join(
            lora_dir, "new_embeddings.safetensors")
        new_embeddings_bin_file_path = os.path.join(lora_dir,
                                                    "new_embeddings.bin")
        if os.path.isfile(lora_tensor_path):
            tensors = safetensors.torch.load_file(lora_tensor_path)
        elif os.path.isfile(lora_bin_file_path):
            tensors = torch.load(lora_bin_file_path)
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        embeddings = None
        if os.path.isfile(new_embeddings_tensor_path):
            embeddings = safetensors.torch.load_file(
                new_embeddings_tensor_path)
        elif os.path.isfile(new_embeddings_bin_file_path):
            embeddings = torch.load(new_embeddings_bin_file_path)

        with open(lora_config_path) as f:
            config = json.load(f)
        rank = config["r"]
        lora_alpha = config["lora_alpha"]
        return cls.from_lora_tensors(
            lora_model_id=get_lora_id()
            if lora_model_id is None else lora_model_id,
            rank=rank,
            lora_alpha=lora_alpha,
            tensors=tensors,
            device=device,
            dtype=dtype,
            embeddings=embeddings,
            target_embedding_padding=target_embedding_padding,
            embedding_modules=embedding_modules,
            embedding_padding_modules=embedding_padding_modules,
        )


class LoRAModelManager:
    """A manager that manages multiple LoRA-fine-tuned models."""

    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
    ):
        """Create a LoRAModelManager and adapter for a given model.

        Args:
            model: the model to be adapted.
            max_num_seqs: the maximum number of sequences model can run in a
                single batch.
            max_num_batched_tokens: the maximum number of tokens model can run
                in a single batch.
            vocab_size: the vocab size of the model.
            lora_config: the LoRA configuration.
        """
        self.lora_config = lora_config
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.lora_slots
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.lora_index_to_id: List[Optional[int]] = [None] * self.lora_slots
        self.vocab_size = vocab_size
        self.base_indices = torch.empty(self.max_num_batched_tokens,
                                        dtype=torch.long,
                                        device="cuda")
        self.sampler_indices = torch.empty(self.max_num_batched_tokens,
                                           dtype=torch.long,
                                           device="cuda")
        self.sampler_indices_padded = torch.empty(self.max_num_batched_tokens,
                                                  dtype=torch.long,
                                                  device="cuda")
        self.embeddings_indices = torch.empty(2,
                                              self.max_num_batched_tokens,
                                              dtype=torch.long,
                                              device="cuda")
        self.offsets = []
        # 4 is the number of indicies tensors defined above
        # base_indices, sampler_indices, sampler_indices_padded,
        # embeddings_indices
        self.indices_len = [None] * 4

        self.model: nn.Module = model
        if hasattr(self.model, "supported_lora_modules"):
            self.supported_lora_modules = copy.deepcopy(
                self.model.supported_lora_modules)
            self.packed_modules_mapping = copy.deepcopy(
                self.model.packed_modules_mapping)
        self.packed_modules: Dict[str, List[str]] = {}
        self.modules: Dict[str, "BaseLayerWithLoRA"] = {}
        self._registered_loras: Dict[int, LoRAModel] = {}
        # Dict instead of a Set for compatibility with LRUCache.
        self._active_loras: Dict[int, None] = {}
        self._last_mapping = None
        self._create_lora_modules()
        self.model.lora_manager = self

    @property
    def capacity(self) -> int:
        return self.lora_config.max_cpu_loras

    @property
    def lora_slots(self) -> int:
        return self.lora_config.max_loras

    def __len__(self) -> int:
        return len(self._registered_loras)

    def activate_lora(
        self,
        lora_id: int,
    ) -> bool:
        """Move LoRA into a GPU buffer to be used in the forward pass."""
        if lora_id in self._active_loras:
            return False
        first_free_slot = next(
            ((i, lora_id) for i, lora_id in enumerate(self.lora_index_to_id)
             if lora_id is None), None)
        if first_free_slot is None:
            raise ValueError("No free lora slots")
        index, _ = first_free_slot
        self._active_loras[lora_id] = None
        lora_model = self._registered_loras[lora_id]
        logger.debug(
            f"Activating LoRA. int id: {lora_model.id}, slot index: {index}")
        self.lora_index_to_id[index] = lora_model.id
        for module_name, module in self.modules.items():
            module_lora = lora_model.get_lora(module_name)
            if module_lora:
                module_lora.optimize()
                module.set_lora(index, module_lora.lora_a, module_lora.lora_b,
                                module_lora.embeddings_tensor)
            else:
                module.reset_lora(index)
        return True

    def _deactivate_lora(self, lora_id: int):
        try:
            index = self.lora_index_to_id.index(lora_id)
            self.lora_index_to_id[index] = None
        except ValueError:
            pass

    def deactivate_lora(self, lora_id: int) -> bool:
        """Remove a LoRA from a GPU buffer."""
        if lora_id in self._active_loras:
            self._deactivate_lora(lora_id)
            self._active_loras.pop(lora_id)
            return True
        return False

    def _add_lora(self, lora: LoRAModel) -> bool:
        self._create_merged_loras_inplace(lora)
        self._registered_loras[lora.id] = lora

    def add_lora(self, lora: LoRAModel) -> bool:
        """Add a LoRAModel to the manager CPU cache."""
        if lora.id not in self._registered_loras:
            if len(self._registered_loras) >= self.capacity:
                raise RuntimeError("No free LoRA slots.")
            self._add_lora(lora)
            return True
        return False

    def remove_lora(self, lora_id: int) -> bool:
        """Remove a LoRAModel from the manager CPU cache."""
        # TODO: should we check active lora?
        self.deactivate_lora(lora_id)
        return bool(self._registered_loras.pop(lora_id, None))

    # TODO see if this can be vectorized
    def _set_lora_mapping(self, mapping: LoRAMapping) -> None:
        (base_indices, sampler_indices, sampler_indices_padded,
         embeddings_indices,
         indices_len) = convert_mapping(mapping, self.lora_index_to_id,
                                        self.lora_slots + 1, self.vocab_size,
                                        self.lora_config.lora_extra_vocab_size)
        self.base_indices[:base_indices.shape[0]].copy_(base_indices)
        self.sampler_indices[:sampler_indices.shape[0]].copy_(sampler_indices)
        self.sampler_indices_padded[:sampler_indices_padded.shape[0]].copy_(
            sampler_indices_padded)
        self.embeddings_indices[:embeddings_indices.
                                shape[0], :embeddings_indices.shape[1]].copy_(
                                    embeddings_indices)
        # Maintain the reference
        self.indices_len[:] = indices_len

    def set_lora_mapping(self, lora_mapping: LoRAMapping) -> None:
        if self._last_mapping != lora_mapping:
            self._set_lora_mapping(lora_mapping)
        self._last_mapping = lora_mapping

    def list_loras(self) -> Dict[int, LoRAModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_loras)

    def get_lora(self, lora_id: int) -> Optional[LoRAModel]:
        return self._registered_loras.get(lora_id, None)

    def remove_all_loras(self) -> bool:
        """Remove all LoRAModels from the manager."""
        self._registered_loras.clear()
        self.lora_index_to_id = [None] * self.lora_slots
        self._active_loras.clear()

    def _create_lora_modules(self):
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name):
                continue

            new_module = replace_submodule(
                self.model, module_name,
                from_layer(module, self.lora_slots, self.lora_config,
                           self.model.config))
            # (yard1): TODO make this more robust
            if "lm_head" in module_name:
                sampler_module = self.model.get_submodule("sampler")
                new_module = replace_submodule(
                    self.model, "sampler",
                    from_layer_sampler(sampler_module, module, self.lora_slots,
                                       self.lora_config, self.model.config))
            self.register_module(module_name, new_module)
            self._register_packed_modules(module_name)
            new_module.set_mapping(self.base_indices, self.sampler_indices,
                                   self.sampler_indices_padded,
                                   self.embeddings_indices, self.indices_len)

    def register_module(self, module_name: str, module: "BaseLayerWithLoRA"):
        assert isinstance(module, BaseLayerWithLoRA)
        self.modules[module_name] = module

    def create_dummy_lora(
            self,
            lora_id: int,
            rank: int,
            embedding_modules: Optional[Dict[str, str]] = None) -> LoRAModel:
        """Create zero-initialized LoRAModel for warmup."""
        model = LoRAModel(lora_id, rank, {})
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name) or not isinstance(
                    module, BaseLayerWithLoRA):
                continue
            parts = module_name.split(".")
            if module_name not in self.packed_modules:
                if parts[-1] in embedding_modules:
                    input_dim = (module.base_layer.org_vocab_size +
                                 self.lora_config.lora_extra_vocab_size if
                                 hasattr(module.base_layer, "org_vocab_size")
                                 else module.base_layer.weight.shape[1])
                    output_dim = module.base_layer.embedding_dim if hasattr(
                        module.base_layer,
                        "embedding_dim") else module.base_layer.weight.shape[0]
                    embeddings_tensor_dim = (module.base_layer.embedding_dim if
                                             hasattr(module.base_layer,
                                                     "embedding_dim") else
                                             module.base_layer.weight.shape[1])
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        input_dim,
                        output_dim,
                        rank,
                        module.lora_a_stacked.dtype,
                        "cpu",
                        embeddings_tensor_dim=embeddings_tensor_dim)
                else:
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        module.lora_a_stacked.shape[-1],
                        module.lora_b_stacked.shape[-2],
                        rank,
                        module.lora_a_stacked.dtype,
                        "cpu",
                    )
                lora.optimize()
            else:
                parts = module_name.split(".")
                replacements = self.packed_modules_mapping[parts[-1]]
                subloras = []
                for i, r in enumerate(replacements):
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name + "." + r,
                        module.lora_a_stacked[i].shape[-1],
                        module.lora_b_stacked[i].shape[-2],
                        rank,
                        module.lora_a_stacked[i].dtype,
                        "cpu",
                    )
                    lora.optimize()
                    subloras.append(lora)
                lora = PackedLoRALayerWeights.pack(subloras)
            model.loras[module_name] = lora
        return model

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module),
                module_name) or target_module == module_name
            for target_module in self.supported_lora_modules)

    def _register_packed_modules(self, module_full_name: str) -> None:
        parts = module_full_name.split(".")
        module_name = parts[-1]
        replacements = self.packed_modules_mapping.get(module_name)
        if not replacements:
            return
        prefix = ".".join(parts[:-1])
        self.packed_modules[module_full_name] = [
            prefix + "." + r if prefix else r for r in replacements
        ]

    def _create_merged_loras_inplace(self, lora_model: LoRAModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_loras = []
            has_replacement = False
            for r in new_module_names:
                lora = lora_model.get_lora(r)
                replacement_loras.append(lora)
                if lora:
                    has_replacement = True
            if not has_replacement:
                continue
            for i in range(len(replacement_loras)):
                if replacement_loras[i]:
                    continue
                replacement_loras[i] = None
            lora_model.loras[module_name] = PackedLoRALayerWeights.pack(
                replacement_loras)


class LoRALRUCache(LRUCache):

    def __init__(self, capacity: int, deactivate_lora_fn: Callable[[Hashable],
                                                                   None]):
        super().__init__(capacity)
        self.deactivate_lora_fn = deactivate_lora_fn

    def _on_remove(self, key: Hashable, value: Any):
        logger.debug(f"Removing LoRA. int id: {key}")
        self.deactivate_lora_fn(key)
        return super()._on_remove(key, value)


class LRUCacheLoRAModelManager(LoRAModelManager):
    """A model manager that manages multiple LoRAs with LRU cache."""

    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
    ):
        super().__init__(model, max_num_seqs, max_num_batched_tokens,
                         vocab_size, lora_config)
        self._registered_loras: LoRALRUCache = LoRALRUCache(
            self.capacity, self.deactivate_lora)
        self._active_loras: LoRALRUCache = LoRALRUCache(
            self.lora_slots, self._deactivate_lora)

    def list_loras(self) -> Dict[int, LoRAModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_loras.cache)

    def add_lora(self, lora: LoRAModel) -> bool:
        """Add a LoRAModel to the manager."""
        if lora.id not in self._registered_loras:
            self._add_lora(lora)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_loras.touch(lora.id)
            was_added = False
        return was_added

    def activate_lora(
        self,
        lora_id: int,
    ) -> bool:
        if lora_id not in self._active_loras and len(
                self._active_loras) >= self.lora_slots:
            self._active_loras.remove_oldest()
        result = super().activate_lora(lora_id)
        # We always touch to update the LRU cache order
        self._active_loras.touch(lora_id)
        return result

    def remove_oldest_lora(self) -> bool:
        if len(self._registered_loras) > 0:
            self._registered_loras.remove_oldest()
            return True
        return False


def create_lora_manager(
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        lora_manager_cls: Type[LoRAModelManager] = LoRAModelManager,
        **kwargs) -> LoRAModelManager:
    """Create a LoRA adapter for a given model."""
    if not hasattr(model, "supported_lora_modules"):
        raise ValueError(f"Model {type(model)} is not supported for LoRA.")
    lora_manager = lora_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        lora_config=lora_config,
        **kwargs)
    return lora_manager
