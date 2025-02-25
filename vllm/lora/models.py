# SPDX-License-Identifier: Apache-2.0

import copy
import math
import os
import re
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Sequence, Set, Type,
                    Union)

import safetensors.torch
import torch
from torch import nn

from vllm.adapter_commons.models import (AdapterLRUCache, AdapterModel,
                                         AdapterModelManager)
from vllm.adapter_commons.utils import (add_adapter, deactivate_adapter,
                                        get_adapter, list_adapters,
                                        remove_adapter, set_adapter_mapping)
from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import (BaseLayerWithLoRA,
                              LinearScalingRotaryEmbeddingWithLora,
                              LoRAMapping)
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.lora.utils import (from_layer, from_layer_logits_processor,
                             get_supported_lora_modules,
                             is_regex_target_modules,
                             parse_fine_tuned_lora_name, replace_submodule)
from vllm.model_executor.models import SupportsLoRA, supports_multimodal
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import PPMissingLayer, WeightsMapper
from vllm.utils import is_pin_memory_available

logger = init_logger(__name__)

_GLOBAL_LORA_ID = 0


@dataclass
class LongContextLoRAContext:
    """Context for lora adapters that support long context."""
    # The scaling factors to support long context lora fine tuned models.
    scaling_factors: List[float]
    # dimension to apply rotary embedding.
    rot_dim: int
    # offsets to the sin_cos_cache for each lora_id loaded.
    # This value is dynamically modified.
    offsets_by_lora_id: Dict[int, int] = field(default_factory=dict)


def get_lora_id():
    global _GLOBAL_LORA_ID
    _GLOBAL_LORA_ID += 1
    return _GLOBAL_LORA_ID


class LoRAModel(AdapterModel):
    """A LoRA fine-tuned model."""

    def __init__(
        self,
        lora_model_id: int,
        rank: int,
        loras: Dict[str, LoRALayerWeights],
        scaling_factor: Optional[float] = None,
    ) -> None:
        """
        Args:
            lora_model_id: The integer id for the lora model.
            rank: lora rank.
            loras: module name -> weights for lora-replaced layers.
            scaling_factor: Scaling factor to support long context lora model.
                None if the lora is not tuned for long context support.
        """
        self.id = lora_model_id
        # Scaling factor for long context lora model. None if it is not
        # fine tuned for the long context.
        self.scaling_factor = scaling_factor
        assert (
            lora_model_id
            > 0), f"a valid lora id should be greater than 0, got {self.id}"
        self.rank = rank
        self.loras: Dict[str, LoRALayerWeights] = loras

    def clone(self, lora_model_id: int) -> "LoRAModel":
        """Return a copy of the object with different ids.

        Will share the underlying tensors."""
        return self.__class__(
            lora_model_id,
            rank=self.rank,
            loras=self.loras.copy(),
        )

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
        tensors: Dict[str, torch.Tensor],
        peft_helper: PEFTHelper,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        embeddings: Optional[Dict[str, torch.Tensor]] = None,
        target_embedding_padding: Optional[int] = None,
        embedding_modules: Optional[Dict[str, str]] = None,
        embedding_padding_modules: Optional[List[str]] = None,
        weights_mapper: Optional[WeightsMapper] = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a dictionary of tensors."""
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        loras: Dict[str, LoRALayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            module_name, is_lora_a, is_bias = parse_fine_tuned_lora_name(
                tensor_name, weights_mapper)
            if module_name not in loras:
                lora_embeddings_tensor = None
                if embeddings:
                    assert embedding_modules is not None
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
                loras[module_name] = LoRALayerWeights.from_config(
                    module_name, peft_helper, lora_embeddings_tensor)

            if is_bias:
                loras[module_name].bias = tensor.to(device=device,
                                                    dtype=dtype).t()
                bias = tensor.to(device=device, dtype=dtype).t()
                if pin_memory:
                    bias = bias.pin_memory()
                loras[module_name].bias = bias
            elif is_lora_a:
                loras[module_name].lora_a = tensor.to(device=device,
                                                      dtype=dtype).t()
                if pin_memory:
                    loras[module_name].lora_a = loras[
                        module_name].lora_a.pin_memory()
            else:
                loras[module_name].lora_b = tensor.to(device=device,
                                                      dtype=dtype).t()
                assert embedding_padding_modules is not None
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

        return cls(lora_model_id,
                   peft_helper.r,
                   loras,
                   scaling_factor=peft_helper.vllm_long_context_scaling_factor)

    @classmethod
    def from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: List[str],
        peft_helper: PEFTHelper,
        *,
        lora_model_id: Optional[int] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        target_embedding_padding: Optional[int] = None,
        embedding_modules: Optional[Dict[str, str]] = None,
        embedding_padding_modules: Optional[List[str]] = None,
        weights_mapper: Optional[WeightsMapper] = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint.
        
        Args:
            lora_dir: The local path that has lora data.
            expected_lora_modules: Name of modules that are expected to be
                replaced by lora.
            peft_helper: Loaded lora configuration information.
            lora_model_id: Lora model id. If not given, automatically set by
                a global counter.
            device: Device where the lora model is loaded.
            dtype: dtype of the lora model weights.

        Returns:
            Loaded LoRA Model.
        """
        lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
        new_embeddings_tensor_path = os.path.join(
            lora_dir, "new_embeddings.safetensors")
        new_embeddings_bin_file_path = os.path.join(lora_dir,
                                                    "new_embeddings.bin")

        unexpected_modules: List[Union[list[str], str]]
        if os.path.isfile(lora_tensor_path):
            tensors: Dict[str, torch.Tensor] = {}
            # Find unexpected modules.
            # Use safetensor key as a source of truth to find expected modules.
            # in peft if you have target_modules A, B, C and C does not exist
            # in the model it won’t error and model will be trained with A, B
            # loraified. C won’t exist in the safetensor but it will exist in
            # the target_modules of the adapter_config.json.
            unexpected_modules = []
            with safetensors.safe_open(lora_tensor_path,
                                       framework="pt") as f:  # type: ignore
                for lora_module in f.keys():  # noqa
                    module_name, _, _ = parse_fine_tuned_lora_name(
                        lora_module, weights_mapper)
                    part_name = module_name.split(".")[-1]
                    if part_name not in expected_lora_modules:
                        unexpected_modules.append(module_name)
                if unexpected_modules:
                    raise ValueError(
                        f"While loading {lora_dir}, expected"
                        f" target modules in {expected_lora_modules}"
                        f" but received {unexpected_modules}."
                        f" Please verify that the loaded LoRA module is correct"
                    )
                # Load tensors if there are only expected modules.
                for module in f.keys():  # noqa
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(lora_bin_file_path):
            # When a bin file is provided, we rely on config to find unexpected
            # modules.
            unexpected_modules = []
            target_modules = peft_helper.target_modules
            if not isinstance(target_modules, list):
                target_modules = [target_modules]
            for module in target_modules:
                # Compatible with more modules,
                # such as:layers.11.self_attn.k_proj
                part_name = module.split(".")[-1]
                if part_name not in expected_lora_modules:
                    unexpected_modules.append(module)
            # loaded lora's target modules must be a subset of
            # expected_lora_modules. It is not reliable. See
            # https://github.com/vllm-project/vllm/pull/5909. But there's no
            # other better mechanism.
            if unexpected_modules and not is_regex_target_modules(
                    peft_helper.target_modules, expected_lora_modules):
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct")
            tensors = torch.load(lora_bin_file_path, map_location=device)
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        embeddings = None
        if os.path.isfile(new_embeddings_tensor_path):
            embeddings = safetensors.torch.load_file(
                new_embeddings_tensor_path)
        elif os.path.isfile(new_embeddings_bin_file_path):
            embeddings = torch.load(new_embeddings_bin_file_path,
                                    map_location=device,
                                    weights_only=True)

        return cls.from_lora_tensors(
            lora_model_id=get_lora_id()
            if lora_model_id is None else lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            embeddings=embeddings,
            target_embedding_padding=target_embedding_padding,
            embedding_modules=embedding_modules,
            embedding_padding_modules=embedding_padding_modules,
            weights_mapper=weights_mapper)


class LoRAModelManager(AdapterModelManager):
    """A manager that manages multiple LoRA-fine-tuned models."""

    def __init__(
        self,
        model: SupportsLoRA,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
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
        self.device = device
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.lora_slots
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.lora_index_to_id: List[Optional[int]] = [None] * self.lora_slots
        self.vocab_size = vocab_size
        self.long_lora_context: Optional[LongContextLoRAContext] = None
        self.punica_wrapper = get_punica_wrapper(max_num_batched_tokens,
                                                 max_batches=self.max_num_seqs,
                                                 device=self.device)
        # Scaling factor -> offset to the sin_cos_cache to it.
        # Used for long context lora.
        self.scaling_factor_to_offset: Dict[float, int] = {}
        super().__init__(model)
        self.supported_lora_modules = get_supported_lora_modules(self.model)
        assert self.supported_lora_modules, "No supported LoRA modules found in"
        f"{self.model.__class__.__name__}."
        if lora_config.long_lora_scaling_factors:
            # We need to replace rotary emb layer to do batch computation
            # for long lora.
            self.supported_lora_modules.append("rotary_emb")
        self.packed_modules_mapping = copy.deepcopy(
            self.model.packed_modules_mapping)
        # Used to indicate whether the model is a multimodal model
        self.supports_mm: bool = (
            supports_multimodal(self.model)
            # In case the model only supports LoRA for
            # text modules (e.g. ChatGLM)
            and hasattr(self.model, "get_mm_mapping"))
        self.packed_modules: Dict[str, List[str]] = {}
        self.modules: Dict[str, BaseLayerWithLoRA] = {}
        # Dict instead of a Set for compatibility with LRUCache.
        self._last_mapping: Optional[LoRAMapping] = None
        self._create_lora_modules()
        self.model.lora_manager = self
        self.adapter_type = 'LoRa'

    @property
    def capacity(self) -> int:
        return self.lora_config.max_cpu_loras

    @property
    def lora_slots(self) -> int:
        return self.lora_config.max_loras

    @property
    def adapter_slots(self) -> int:
        return self.lora_slots

    def activate_adapter(
        self,
        lora_id: int,
    ) -> bool:
        """Move LoRA into a GPU buffer to be used in the forward pass."""
        if lora_id in self._active_adapters:
            return False
        first_free_slot = next(
            ((i, lora_id) for i, lora_id in enumerate(self.lora_index_to_id)
             if lora_id is None), None)
        if first_free_slot is None:
            raise ValueError("No free lora slots")
        index, _ = first_free_slot
        self._active_adapters[lora_id] = None
        lora_model = self._registered_adapters[lora_id]
        logger.debug("Activating LoRA. int id: %d, slot index: %d",
                     lora_model.id, index)
        self.lora_index_to_id[index] = lora_model.id
        for module_name, module in self.modules.items():
            module_lora = lora_model.get_lora(module_name)
            if module_lora:
                module_lora.optimize()
                # Bias is not explicitly enabled with the flag enable_lora_bias.
                bias = module_lora.bias
                if ((torch.is_tensor(bias) or
                     (isinstance(bias, Sequence) and any(b is not None
                                                         for b in bias)))
                        and not self.lora_config.bias_enabled):
                    module_lora.bias = None
                    raise ValueError(
                        f"Adapter bias cannot be used for {module_name}"
                        " without --enable-lora-bias.")
                module.set_lora(index, module_lora.lora_a, module_lora.lora_b,
                                module_lora.embeddings_tensor,
                                module_lora.bias)
            else:
                module.reset_lora(index)
        return True

    def _deactivate_adapter(self, lora_id: int):
        try:
            index = self.lora_index_to_id.index(lora_id)
            self.lora_index_to_id[index] = None
        except ValueError:
            pass

    def _set_long_lora_context(self, lora: LoRAModel):
        if self.long_lora_context is None:
            return

        if lora.scaling_factor is None:
            return

        if (lora.scaling_factor not in self.scaling_factor_to_offset):
            raise ValueError(f"Long LoRA scaling factor {lora.scaling_factor}"
                             " has not been initialized.")

        offsets = self.scaling_factor_to_offset.get(lora.scaling_factor)
        if offsets:
            self.long_lora_context.offsets_by_lora_id[lora.id] = offsets

    def _add_adapter(self, lora: LoRAModel):
        self._create_merged_loras_inplace(lora)
        self._registered_adapters[lora.id] = lora
        self._set_long_lora_context(lora)

    def pin_adapter(self, lora_id: int) -> bool:
        """Pin a LoRAModel in the manager cache."""
        raise NotImplementedError(
            "Pinning is not supported in LoRAModelManager. "
            "Use LRUCacheLoRAModelManager for pinning")  # type: ignore

    def _set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        # update lora states
        self.punica_wrapper.update_metadata(
            mapping,
            self.lora_index_to_id,
            self.lora_slots + 1,
            self.vocab_size,
            self.lora_config.lora_extra_vocab_size,
            self.long_lora_context,
        )

    def remove_all_adapters(self):
        """Remove all LoRAModels from the manager."""
        self._registered_adapters.clear()
        self.lora_index_to_id = [None] * self.lora_slots
        self._active_adapters.clear()

    def _create_lora_modules(self):
        for module_name, module in self.model.named_modules(
                remove_duplicate=False):
            if isinstance(module, PPMissingLayer):
                continue
            if not self._match_target_modules(module_name):
                continue
            # A temporary approach for multimodal models to support LoRA
            # TODO: Remove this restriction
            if self._filter_unsupported_mm_module(module_name):
                logger.warning(
                    "Regarding multimodal models, vLLM currently only supports "
                    "adding LoRA to language model, %s will be ignored.",
                    module_name,
                )
                continue
            parts = module_name.split(".")[-1]
            packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
            new_module = replace_submodule(
                self.model, module_name,
                from_layer(module, self.lora_slots, self.lora_config,
                           packed_moduled_lst, self.model.config))

            # LinearScalingRotaryEmbeddingWithLora is used to handle
            # long context lora. Register relevant metadata.
            if isinstance(new_module, LinearScalingRotaryEmbeddingWithLora):
                self.long_lora_context = LongContextLoRAContext(
                    new_module.scaling_factors, new_module.rotary_dim)
                self.scaling_factor_to_offset = \
                    new_module.scaling_factor_to_offset
            # (yard1): TODO make this more robust
            if "lm_head" in module_name:
                logits_processor_module = self.model.get_submodule(
                    "logits_processor")
                new_module = replace_submodule(
                    self.model, "logits_processor",
                    from_layer_logits_processor(logits_processor_module,
                                                module, self.lora_slots,
                                                self.lora_config,
                                                self.model.config))

            # In some models, especially multimodal ones, layers with the same
            # name may have different types, such as nn.Linear and
            # ReplicatedLinear. The nn.Linear layers cannot be replaced with
            # LoRA layers, leading to assertion error. The following check
            # aims to prevent this error
            if self.supports_mm and not isinstance(new_module,
                                                   BaseLayerWithLoRA):
                continue
            self.register_module(module_name, new_module)
            self._register_packed_modules(module_name)
            # All lora layers share the same punica_wrapper based on reference.
            new_module.set_mapping(self.punica_wrapper)

    def register_module(self, module_name: str, module: "BaseLayerWithLoRA"):
        assert isinstance(module, BaseLayerWithLoRA)
        self.modules[module_name] = module

    def create_dummy_lora(
            self,
            lora_id: int,
            rank: int,
            scaling_factor: Optional[float],
            embedding_modules: Optional[Dict[str, str]] = None) -> LoRAModel:
        """Create zero-initialized LoRAModel for warmup."""
        model = LoRAModel(lora_id, rank, {}, scaling_factor)
        for module_name, module in self.model.named_modules():
            bias_enabled = self.lora_config.bias_enabled
            if (not self._match_target_modules(module_name)
                    or not isinstance(module, BaseLayerWithLoRA)
                    or isinstance(module, LinearScalingRotaryEmbeddingWithLora)
                    or self._filter_unsupported_mm_module(module_name)):
                continue
            parts = module_name.split(".")
            if module_name not in self.packed_modules:
                assert embedding_modules is not None
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
                        module.lora_a_stacked[0].dtype,
                        "cpu",
                        embeddings_tensor_dim=embeddings_tensor_dim,
                        bias_enabled=bias_enabled)
                else:
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        module.lora_a_stacked[0].shape[-1],
                        module.lora_b_stacked[0].shape[-2],
                        rank,
                        module.lora_a_stacked[0].dtype,
                        "cpu",
                        bias_enabled=bias_enabled,
                    )
                lora.optimize()
            else:
                parts = module_name.split(".")
                replacements = self.packed_modules_mapping[parts[-1]]
                subloras: List[Optional[LoRALayerWeights]] = []
                for i, r in enumerate(replacements):
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name + "." + r,
                        module.lora_a_stacked[i].shape[-1],
                        module.lora_b_stacked[i].shape[-2],
                        rank,
                        module.lora_a_stacked[i].dtype,
                        "cpu",
                        bias_enabled=bias_enabled,
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

    def _filter_unsupported_mm_module(self, module_name: str) -> bool:
        """
        Regarding multimodal models, vLLM currently only supports adding LoRA to
        language model. LoRA for other modules, such as the vision tower, will 
        be filtered out.
        """
        if self.supports_mm:
            module_mapping: MultiModelKeys = self.model.get_mm_mapping()
            prefix_lst = module_mapping.connector + module_mapping.tower_model
            return any(
                [module_name.startswith(prefix) for prefix in prefix_lst])
        return False

    def _register_packed_modules(self, module_full_name: str) -> None:
        parts = module_full_name.split(".")
        module_name = parts[-1]
        replacements = self.packed_modules_mapping.get(module_name, [])
        # When replacements is less than or equal to 1, it indicates that this
        # module is not a packed module.
        if len(replacements) <= 1:
            return
        prefix = ".".join(parts[:-1])
        self.packed_modules[module_full_name] = [
            prefix + "." + r if prefix else r for r in replacements
        ]

    def _create_merged_loras_inplace(self, lora_model: LoRAModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_loras: List[Optional[LoRALayerWeights]] = []
            replaced_module: Set[str] = set()
            has_replacement = False
            for r in new_module_names:
                lora = lora_model.get_lora(r)
                replacement_loras.append(lora)
                if lora:
                    has_replacement = True
                    replaced_module.add(r)
            if not has_replacement:
                continue
            for i in range(len(replacement_loras)):
                if replacement_loras[i]:
                    continue
                replacement_loras[i] = None
            lora_model.loras[module_name] = PackedLoRALayerWeights.pack(
                replacement_loras)
            # Remove the modules that have been replaced.
            for module in replaced_module:
                lora_model.loras.pop(module, None)

    def deactivate_adapter(self, adapter_id: int) -> bool:
        return deactivate_adapter(adapter_id, self._active_adapters,
                                  self._deactivate_adapter)

    def add_adapter(self, adapter: LoRAModel) -> bool:
        logger.debug(
            "Adding lora. Model id: %d, "
            "int id: %d, "
            "scaling factor: %s", adapter.id, adapter.id,
            adapter.scaling_factor)
        return add_adapter(adapter, self._registered_adapters, self.capacity,
                           self._add_adapter)

    def set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        self._last_mapping = set_adapter_mapping(mapping, self._last_mapping,
                                                 self._set_adapter_mapping)

    def remove_adapter(self, adapter_id: int) -> bool:
        return remove_adapter(adapter_id, self._registered_adapters,
                              self.deactivate_adapter)

    def list_adapters(self) -> Dict[int, Any]:
        return list_adapters(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        return get_adapter(adapter_id, self._registered_adapters)


class LoRALRUCache(AdapterLRUCache[LoRAModel]):

    def __init__(self, capacity: int, deactivate_lora_fn: Callable[[int],
                                                                   bool]):
        super().__init__(capacity, deactivate_lora_fn)


class LRUCacheLoRAModelManager(LoRAModelManager):
    """A model manager that manages multiple LoRAs with LRU cache."""

    def __init__(self, model: nn.Module, max_num_seqs: int,
                 max_num_batched_tokens: int, vocab_size: int,
                 lora_config: LoRAConfig, device: torch.device):
        super().__init__(model, max_num_seqs, max_num_batched_tokens,
                         vocab_size, lora_config, device)
        self._registered_adapters: LoRALRUCache = LoRALRUCache(
            self.capacity, self.deactivate_adapter)
        self._active_adapters: LoRALRUCache = LoRALRUCache(
            self.lora_slots, self._deactivate_adapter)

    def list_adapters(self) -> Dict[int, LoRAModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_adapters.cache)

    def add_adapter(self, lora: LoRAModel) -> bool:
        """Add a LoRAModel to the manager."""
        logger.debug(
            "Adding lora. Model id: %d, "
            "int id: %d, "
            "scaling factor: %s", lora.id, lora.id, lora.scaling_factor)
        if lora.id not in self._registered_adapters:
            self._add_adapter(lora)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_adapters.touch(lora.id)
            was_added = False
        return was_added

    def activate_adapter(
        self,
        lora_id: int,
    ) -> bool:
        if lora_id not in self._active_adapters and len(
                self._active_adapters) >= self.lora_slots:
            self._active_adapters.remove_oldest()
        result = super().activate_adapter(lora_id)
        # We always touch to update the LRU cache order
        self._active_adapters.touch(lora_id)
        return result

    def remove_oldest_adapter(self) -> bool:
        if len(self._registered_adapters) > 0:
            self._registered_adapters.remove_oldest()
            return True
        return False

    def pin_adapter(self, lora_id: int) -> bool:
        """Pin a LoRAModel in the manager cache."""
        self._pin_lora_in_cpu_cache(lora_id)
        self._pin_lora_in_gpu_cache(lora_id)
        return True

    def _pin_lora_in_cpu_cache(self, lora_id: int):
        try:
            self._registered_adapters.pin(lora_id)
        except ValueError as err:
            raise ValueError("Pinning failed. "
                             f"LoRA {lora_id} is not registered.") from err

    def _pin_lora_in_gpu_cache(self, lora_id: int):
        if lora_id not in self._active_adapters:
            # move lora to gpu if not already active
            self.activate_adapter(lora_id)

        self._active_adapters.pin(lora_id)


def create_lora_manager(
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        lora_manager_cls: Type[LoRAModelManager] = LoRAModelManager,
        **kwargs) -> LoRAModelManager:
    """Create a LoRA adapter for a given model."""
    if not hasattr(model, "packed_modules_mapping"):
        raise ValueError(f"Model {type(model)} is not supported for LoRA.")
    lora_manager = lora_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        lora_config=lora_config,
        device=device,
        **kwargs)
    return lora_manager
