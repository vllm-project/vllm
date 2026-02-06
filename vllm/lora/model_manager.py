# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable
from typing import TypeVar

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import (
    BaseLayerWithLoRA,
    FusedMoE3DWithLoRA,
    LoRAMapping,
    LoRAMappingType,
)
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.punica_wrapper import PunicaWrapperBase, get_punica_wrapper
from vllm.lora.utils import (
    from_layer,
    from_layer_logits_processor,
    get_supported_lora_modules,
    is_moe_model,
    process_packed_modules_mapping,
    replace_submodule,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models import SupportsLoRA, supports_multimodal
from vllm.model_executor.models.interfaces import is_pooling_model
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.budget import MultiModalBudget
from vllm.utils.cache import LRUCache
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

T = TypeVar("T")
DEFAULT_LANGUAGE_WRAPPER_KEY = "language_model"


class AdapterLRUCache(LRUCache[int, T]):
    def __init__(self, capacity: int, deactivate_fn: Callable[[int], object]):
        super().__init__(capacity)
        self.deactivate_fn = deactivate_fn

    def _on_remove(self, key: int, value: T | None):
        logger.debug("Removing adapter int id: %d", key)
        self.deactivate_fn(key)
        return super()._on_remove(key, value)


class LoRAModelManager:
    """A manager that manages multiple LoRA-fine-tuned models."""

    def __init__(
        self,
        model: SupportsLoRA,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        vllm_config: VllmConfig | None = None,
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
        self.model: SupportsLoRA = model
        self.supported_lora_modules = get_supported_lora_modules(self.model)
        assert self.supported_lora_modules, (
            f"No supported LoRA modules found in {self.model.__class__.__name__}."
        )

        self._registered_adapters: dict[int, LoRAModel] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._active_adapters: dict[int, None] = {}
        self.adapter_type = "LoRA"
        self.lora_config = lora_config
        self.device = device
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.lora_slots
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.lora_index_to_id: list[int | None] = [None] * self.lora_slots
        self.vocab_size = vocab_size
        self.packed_modules_mapping = process_packed_modules_mapping(self.model)

        self.is_pooling_model = is_pooling_model(self.model)
        self.packed_modules: dict[str, list[str]] = {}
        self.modules: dict[str, BaseLayerWithLoRA] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._last_mapping: LoRAMapping | None = None
        is_moe = is_moe_model(self.model)
        self._is_3d_moe_model = is_moe and self.model.is_3d_moe_weight
        self._is_non_gated_moe = is_moe and self.model.is_non_gated_moe
        self._init_punica_wrapper(max_num_batched_tokens, vllm_config)
        self._create_lora_modules()

        self.model.lora_manager = self

    def _init_punica_wrapper(
        self, max_num_batched_tokens: int, vllm_config: VllmConfig
    ) -> None:
        # Used to indicate whether the model is a multimodal model
        self.supports_mm: bool = (
            supports_multimodal(self.model)
            # In case the model only supports LoRA for
            # text modules (e.g. ChatGLM)
            and hasattr(self.model, "get_mm_mapping")
        )
        self.punica_wrapper_mapping: dict[str, PunicaWrapperBase] = {}
        if self.supports_mm:
            self._maybe_init_mm(vllm_config, max_num_batched_tokens)
        else:
            llm_punica_wrapper = get_punica_wrapper(
                max_num_batched_tokens,
                max_batches=self.max_num_seqs,
                device=self.device,
                lora_config=self.lora_config,
            )

            self.punica_wrapper_mapping[DEFAULT_LANGUAGE_WRAPPER_KEY] = (
                llm_punica_wrapper
            )

    def _maybe_init_mm(
        self,
        vllm_config: VllmConfig,
        max_num_batched_tokens: int,
    ) -> None:
        mm_registry = MULTIMODAL_REGISTRY

        self.supports_tower_connector_lora = False
        self.mm_mapping: MultiModelKeys = self.model.get_mm_mapping()

        # Only one language model can be included in the model.
        assert len(self.mm_mapping.language_model) == 1

        # Language model punica wrapper
        llm_punica_wrapper = get_punica_wrapper(
            max_num_batched_tokens,
            max_batches=self.max_num_seqs,
            device=self.device,
            lora_config=self.lora_config,
        )
        lm_prefix = self.mm_mapping.language_model[0]
        self.punica_wrapper_mapping[lm_prefix] = llm_punica_wrapper

        if self.lora_config.enable_tower_connector_lora:
            self.supports_tower_connector_lora = self.supports_mm and hasattr(
                self.model, "get_num_mm_encoder_tokens"
            )
        if not self.supports_tower_connector_lora:
            return

        logger.warning(
            "LoRA for the tower and connector of multimodal models is "
            "experimental and may contain bugs. Please report any related issues on "
            "GitHub if you encounter them."
        )

        mm_budget = MultiModalBudget(vllm_config, mm_registry)
        limit_per_prompt = max(mm_budget.mm_max_items_per_prompt.values())
        num_encoder_tokens = self.model.get_num_mm_encoder_tokens(
            mm_budget.get_encoder_budget()
        )

        # Tower wrappers for each modality separately
        for prefix in self.mm_mapping.tower_model:
            tower_punica_wrapper = get_punica_wrapper(
                num_encoder_tokens,
                max_batches=self.max_num_seqs * limit_per_prompt,
                device=self.device,
                lora_config=self.lora_config,
            )
            self.punica_wrapper_mapping[prefix] = tower_punica_wrapper

        # Use wrapper for connector if present.
        if self.mm_mapping.connector:
            if hasattr(self.model, "get_num_mm_connector_tokens"):
                connector_tokens = self.model.get_num_mm_connector_tokens(
                    num_encoder_tokens
                )
                connector_punica_wrapper = get_punica_wrapper(
                    connector_tokens,
                    max_batches=self.max_num_seqs * limit_per_prompt,
                    device=self.device,
                    lora_config=self.lora_config,
                )
                for prefix in self.mm_mapping.connector:
                    self.punica_wrapper_mapping[prefix] = connector_punica_wrapper
            else:
                logger.warning_once(
                    "Connector LoRA support disabled: model does not implement "
                    "get_num_mm_connector_tokens(). This method is required to "
                    "determine the connector's token budget for LoRA operations."
                )

    def __len__(self) -> int:
        return len(self._registered_adapters)

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
            (
                (i, lora_id)
                for i, lora_id in enumerate(self.lora_index_to_id)
                if lora_id is None
            ),
            None,
        )
        if first_free_slot is None:
            raise ValueError("No free lora slots")
        index, _ = first_free_slot
        self._active_adapters[lora_id] = None
        lora_model = self._registered_adapters[lora_id]
        logger.debug(
            "Activating LoRA. int id: %d, slot index: %d", lora_model.id, index
        )
        self.lora_index_to_id[index] = lora_model.id
        for module_name, module in self.modules.items():
            module_lora = self._get_lora_layer_weights(lora_model, module_name)
            if not module_lora:
                module.reset_lora(index)
                continue

            module.set_lora(
                index,
                module_lora.lora_a,
                module_lora.lora_b,
            )

        return True

    def _deactivate_adapter(self, lora_id: int):
        try:
            index = self.lora_index_to_id.index(lora_id)
            self.lora_index_to_id[index] = None
        except ValueError:
            pass

    def _add_adapter(self, lora: LoRAModel):
        self._create_merged_loras_inplace(lora)
        self._registered_adapters[lora.id] = lora

    def pin_adapter(self, lora_id: int) -> bool:
        """Pin a LoRAModel in the manager cache."""
        raise NotImplementedError(
            "Pinning is not supported in LoRAModelManager. "
            "Use LRUCacheLoRAModelManager for pinning"
        )  # type: ignore

    def _set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        if mapping.target_prefix is not None:
            target_prefix = mapping.target_prefix
        elif not (self.supports_mm and self.supports_tower_connector_lora):
            target_prefix = (
                self.mm_mapping.language_model[0]
                if self.supports_mm
                else DEFAULT_LANGUAGE_WRAPPER_KEY
            )
        elif mapping.type == LoRAMappingType.TOWER and self.mm_mapping.tower_model:
            target_prefix = self.mm_mapping.tower_model[0]
        elif mapping.type == LoRAMappingType.CONNECTOR and self.mm_mapping.connector:
            target_prefix = self.mm_mapping.connector[0]
        else:
            target_prefix = self.mm_mapping.language_model[0]

        punica_wrapper = self._get_punica_wrapper(target_prefix)
        assert punica_wrapper is not None, (
            f"No punica wrapper found for prefix '{target_prefix}'. "
            f"Available prefixes: {list(self.punica_wrapper_mapping.keys())}"
        )

        punica_wrapper.update_metadata(
            mapping,
            self.lora_index_to_id,
            self.lora_slots + 1,
            self.vocab_size,
        )

    def remove_all_adapters(self):
        """Remove all LoRAModels from the manager."""
        self._registered_adapters.clear()
        self.lora_index_to_id = [None] * self.lora_slots
        self._active_adapters.clear()

    def _create_lora_modules(self):
        def _parent_module(module_name: str) -> str:
            # module name is a dot separated name.
            # for example:
            #  - given an input 'x.y.z' return 'x.y'
            #  - given an input 'x' return ''
            return module_name.rpartition(".")[0]

        for module_name, module in self.model.named_modules(remove_duplicate=False):
            if isinstance(module, PPMissingLayer):
                continue

            if not self._match_target_modules(module_name):
                continue

            punica_wrapper = self._get_punica_wrapper(module_name)
            if punica_wrapper is None:
                logger.warning(
                    "Regarding %s, vLLM currently only supports adding LoRA to"
                    " language model, %s will be ignored.",
                    self.model.__class__.__name__,
                    module_name,
                )
                continue

            # TODO: Remove this restriction
            # peft error when generating LoRA adapter with "gate" module:
            # "Target module NemotronHTopkRouter() is not supported."
            # Working LoRA adapter was created using peft with:
            # LoraConfig(target_modules="all-linear", ...)
            if self._is_non_gated_moe and module_name.endswith("mixer.gate"):
                logger.debug_once(
                    "LoRA is not supported for non-gated MoE gate module."
                    " %s will be ignored.",
                    module_name,
                    scope="local",
                )
                continue

            parts = module_name.split(".")[-1]
            packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
            if isinstance(module, FusedMoE):
                # packed_moduled_lst is used here to just determine whether to
                # instantiate FusedMoE3DWithLoRA or FusedMoEWithLoRA, and the
                # difference between these two LoRA layers is whether the
                # LoRA weights of w1 and w3 have already been fused on disk.

                packed_moduled_lst = ["w13"] if self._is_3d_moe_model else ["w1", "w3"]
            new_module = replace_submodule(
                self.model,
                module_name,
                from_layer(
                    module,
                    self.lora_slots,
                    self.lora_config,
                    packed_moduled_lst,
                    self.model.config,
                ),
            )

            # (yard1): TODO make this more robust
            if "lm_head" in module_name:
                logits_processor_module_name = "logits_processor"
                parent_module = _parent_module(module_name)
                if parent_module:
                    logits_processor_module_name = (
                        f"{parent_module}.{logits_processor_module_name}"
                    )

                logits_processor_module = self.model.get_submodule(
                    logits_processor_module_name
                )

                new_module = replace_submodule(
                    self.model,
                    logits_processor_module_name,
                    from_layer_logits_processor(
                        logits_processor_module,
                        module,
                        self.lora_slots,
                        self.lora_config,
                        self.model.config,
                    ),
                )

            # In some models, especially multimodal ones, layers with the same
            # name may have different types, such as nn.Linear and
            # ReplicatedLinear. The nn.Linear layers cannot be replaced with
            # LoRA layers, leading to assertion error. The following check
            # aims to prevent this error
            if self.supports_mm and not isinstance(new_module, BaseLayerWithLoRA):
                continue
            self.register_module(module_name, new_module)

            self._register_packed_modules(module_name)
            # All lora layers share the same punica_wrapper based on reference.
            new_module.set_mapping(punica_wrapper)

    def register_module(self, module_name: str, module: "BaseLayerWithLoRA"):
        assert isinstance(module, BaseLayerWithLoRA), (
            f"Module {module_name} must be a BaseLayerWithLoRA instance, "
            f"got {type(module)}"
        )
        self.modules[module_name] = module

    @staticmethod
    def _pad_lora_pairs_to_triplets(
        loras: list[LoRALayerWeights | None],
    ) -> list[LoRALayerWeights | None]:
        """Pad LoRA weight pairs to triplets for non-gated MoE.

        For non-gated MoE, each expert has 2 entries (w1, w2) that need to be
        padded to triplets (w1, w2, None) to match pack_moe expectations.
        """
        assert len(loras) % 2 == 0, "Expected pairs of LoRA weights for non-gated MoE."
        padded: list[LoRALayerWeights | None] = []
        for i in range(0, len(loras), 2):
            padded.extend(loras[i : i + 2])
            padded.append(None)
        return padded

    def create_dummy_lora(
        self,
        lora_id: int,
        rank: int,
        embedding_modules: dict[str, str] | None = None,
    ) -> LoRAModel:
        """Create zero-initialized LoRAModel for warmup."""
        model = LoRAModel(lora_id, rank, {})
        for module_name, module in self.model.named_modules():
            if (
                not self._match_target_modules(module_name)
                or not isinstance(module, BaseLayerWithLoRA)
                or self._get_punica_wrapper(module_name) is None
            ):
                continue
            parts = module_name.split(".")
            if module_name not in self.packed_modules:
                assert embedding_modules is not None
                if parts[-1] in embedding_modules:
                    # Special-case lm_head: wrapped by LogitsProcessorWithLoRA.
                    # LoRA input dim is hidden_size, output dim is vocab size.
                    # LogitsProcessorWithLoRA handles extra vocab size directly.
                    if parts[-1] == "lm_head":
                        input_dim = module.lora_a_stacked[0].shape[-1]
                        output_dim = module.lora_b_stacked[0].shape[-2]
                    else:
                        input_dim = (
                            module.base_layer.org_vocab_size
                            if hasattr(module.base_layer, "org_vocab_size")
                            else module.base_layer.weight.shape[1]
                        )
                        output_dim = (
                            module.base_layer.embedding_dim
                            if hasattr(module.base_layer, "embedding_dim")
                            else module.base_layer.weight.shape[0]
                        )
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        input_dim,
                        output_dim,
                        rank,
                        module.lora_a_stacked[0].dtype,
                        "cpu",
                    )
                    model.loras[module_name] = lora
                elif module.__class__.__name__ == "FusedMoE3DWithLoRA":
                    # Case for 3D moe model
                    # w2
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        module.w2_input_size,
                        module.w2_output_size,
                        rank * module.w2_lora_a_stacked[0].shape[1],  # rank*num_experts
                        module.w2_lora_a_stacked[0].dtype,
                        "cpu",
                    )
                    model.loras[module_name] = lora
                    # w13
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        module.w13_input_size,
                        module.w13_output_size,
                        rank
                        * module.w13_lora_a_stacked[0].shape[1],  # rank*num_experts
                        module.w13_lora_a_stacked[0].dtype,
                        "cpu",
                    )
                    model.loras[module_name + ".base_layer"] = lora
                else:
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        module.lora_a_stacked[0].shape[-1],
                        module.lora_b_stacked[0].shape[-2],
                        rank,
                        module.lora_a_stacked[0].dtype,
                        "cpu",
                    )
                    model.loras[module_name] = lora
            else:
                parts = module_name.split(".")
                replacements = self.packed_modules_mapping[parts[-1]]
                subloras: list[LoRALayerWeights | None] = []
                for i, r in enumerate(replacements):
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name + "." + r,
                        module.lora_a_stacked[i].shape[-1],
                        module.lora_b_stacked[i].shape[-2],
                        rank,
                        module.lora_a_stacked[i].dtype,
                        "cpu",
                    )
                    subloras.append(lora)
                if module.__class__.__name__ == "FusedMoEWithLoRA":
                    # For non-gated MoE, pad subloras to 3 elements per expert
                    # to match pack_moe expectations (w1, w2, None for w3)
                    if self._is_non_gated_moe and len(subloras) > 0:
                        subloras = self._pad_lora_pairs_to_triplets(subloras)
                    lora = PackedLoRALayerWeights.pack_moe(
                        subloras, module_name, is_non_gated_moe=self._is_non_gated_moe
                    )
                else:
                    lora = PackedLoRALayerWeights.pack(subloras)
                model.loras[module_name] = lora
        return model

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module), module_name
            )
            or target_module == module_name
            for target_module in self.supported_lora_modules
        )

    def _get_punica_wrapper(self, module_name: str) -> PunicaWrapperBase | None:
        """
        Determine whether this module supports LoRA and which wrapper to use.
        """
        # For language model (early return)
        if not self.supports_mm:
            return self.punica_wrapper_mapping[DEFAULT_LANGUAGE_WRAPPER_KEY]

        # For multimodal model
        # NOTE Sort by prefix length (descending) to match the longest prefix first
        # e.g., 'visual.merger' should match 'visual.merger' instead of 'visual.'
        for prefix in sorted(self.punica_wrapper_mapping.keys(), key=len, reverse=True):
            if module_name.startswith(prefix):
                return self.punica_wrapper_mapping[prefix]

        return None

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
            replacement_loras: list[LoRALayerWeights | None] = []
            replaced_module: set[str] = set()
            has_replacement = False
            for r in new_module_names:
                lora = self._get_lora_layer_weights(lora_model, r)
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
            # HACK Temporary solution for the pool model.
            if self.is_pooling_model and not lora_model.check_lora_name(module_name):
                replaced_module_name = module_name.replace("model.", "")
                if lora_model.check_lora_name(module_name):
                    module_name = replaced_module_name
            if module_name.endswith(".experts"):
                if self._is_non_gated_moe and len(replacement_loras) > 0:
                    replacement_loras = self._pad_lora_pairs_to_triplets(
                        replacement_loras
                    )
                lora_model.loras[module_name] = PackedLoRALayerWeights.pack_moe(
                    replacement_loras,
                    module_name,
                    is_non_gated_moe=self._is_non_gated_moe,
                )
            else:
                lora_model.loras[module_name] = PackedLoRALayerWeights.pack(
                    replacement_loras
                )
            # Remove the modules that have been replaced.
            for module in replaced_module:
                lora_model.loras.pop(module, None)

        for lora in lora_model.loras.values():
            lora.optimize()

        for module_name, module in self.modules.items():
            if isinstance(module, FusedMoE3DWithLoRA):
                self._stack_moe_lora_weights(lora_model, module, module_name)

        first_lora: LoRALayerWeights = next(iter(lora_model.loras.values()))
        assert first_lora.lora_a is not None
        if isinstance(first_lora.lora_a, list):
            lora_device = next(iter(first_lora.lora_a))
        else:
            lora_device = first_lora.lora_a.device
        # Execute pin_memory after LoRA weight merging, mainly because:
        # 1. Some MoE models have a large number of LoRA weights. If we
        # perform # pin_memory immediately after loading weights, the
        # overhead is significant.
        # 2. The weight packing above (e.g., pack_moe) may invalidate the
        # pin_memory allocation, so we execute it after packing.

        pin_memory = str(lora_device) == "cpu" and is_pin_memory_available()
        if pin_memory:
            for lora in lora_model.loras.values():
                if isinstance(lora.lora_a, list):
                    for index in range(len(lora.lora_a)):
                        if lora.lora_a[index] is None:
                            continue
                        lora.lora_a[index] = lora.lora_a[index].pin_memory()
                        lora.lora_b[index] = lora.lora_b[index].pin_memory()
                else:
                    lora.lora_a = lora.lora_a.pin_memory()
                    lora.lora_b = lora.lora_b.pin_memory()

    def _stack_moe_lora_weights(
        self, lora_model: LoRAModel, module: FusedMoE3DWithLoRA, module_name: str
    ):
        module_lora = self._get_lora_layer_weights(lora_model, module_name)

        # Note (gnovack) - If MOE lora weights are not split into
        # num_experts chunks, we split them here
        if module_lora and torch.is_tensor(module_lora.lora_a):
            # Handle PEFT file format where experts.base_layer is the
            # gate_up_proj and experts is the down_proj
            gate_up_proj_lora = self._get_lora_layer_weights(
                lora_model, module_name + ".base_layer"
            )
            down_proj_lora = module_lora
            # FIXME Edge case where LoRA is not added to gate_up_proj
            # or down_proj
            assert gate_up_proj_lora is not None
            assert down_proj_lora is not None
            if self._is_3d_moe_model:
                num_experts = module.w13_lora_a_stacked[0].shape[1]

                # (num_experts,rank,input_size)
                gate_up_proj_lora.lora_a = gate_up_proj_lora.lora_a.reshape(
                    num_experts, -1, gate_up_proj_lora.lora_a.shape[-1]
                )
                down_proj_lora.lora_a = down_proj_lora.lora_a.reshape(
                    num_experts, -1, down_proj_lora.lora_a.shape[-1]
                )

                # (output_size,rank,num_experts)
                gate_up_proj_lora.lora_b = gate_up_proj_lora.lora_b.reshape(
                    gate_up_proj_lora.lora_b.shape[0], -1, num_experts
                )
                down_proj_lora.lora_b = down_proj_lora.lora_b.reshape(
                    down_proj_lora.lora_b.shape[0], -1, num_experts
                )

                # (num_experts,output_size,rank)
                gate_up_proj_lora.lora_b = gate_up_proj_lora.lora_b.permute(
                    2, 0, 1
                ).contiguous()
                down_proj_lora.lora_b = down_proj_lora.lora_b.permute(
                    2, 0, 1
                ).contiguous()

                module_lora.lora_a = [
                    gate_up_proj_lora.lora_a,
                    down_proj_lora.lora_a,
                ]
                module_lora.lora_b = [
                    gate_up_proj_lora.lora_b,
                    down_proj_lora.lora_b,
                ]
            else:
                # Some 3D MoE models haven't added the `is_3d_moe_weight`
                # attribute yet, so fallback here
                num_experts = module_lora.lora_a.shape[0] // module_lora.rank

                gate_proj_a = gate_up_proj_lora.lora_a.chunk(num_experts, dim=0)
                up_proj_a = gate_up_proj_lora.lora_a.chunk(num_experts, dim=0)

                gate_proj_b = gate_up_proj_lora.lora_b[::2, ...].chunk(
                    num_experts, dim=-1
                )
                up_proj_b = gate_up_proj_lora.lora_b[1::2, ...].chunk(
                    num_experts, dim=-1
                )

                down_proj_a = down_proj_lora.lora_a.chunk(num_experts, dim=0)
                down_proj_b = down_proj_lora.lora_b.chunk(num_experts, dim=-1)

                lora_a = []
                lora_b = []
                for i in range(num_experts):
                    lora_a.append(gate_proj_a[i])
                    lora_a.append(down_proj_a[i])
                    lora_a.append(up_proj_a[i])

                    lora_b.append(gate_proj_b[i])
                    lora_b.append(down_proj_b[i])
                    lora_b.append(up_proj_b[i])

                module_lora.lora_a = lora_a
                module_lora.lora_b = lora_b

    def _get_lora_layer_weights(
        self, lora_model: LoRAModel, module_name: str
    ) -> LoRALayerWeights | None:
        org_module_name = module_name
        if self.is_pooling_model and not lora_model.check_lora_name(module_name):
            # If it's a pool model, and the layer name is not found,
            # remove the prefix 'model.' and search again.
            module_name = module_name.replace("model.", "")
            if lora_model.check_lora_name(module_name):
                org_module_name = module_name
                logger.info_once(
                    "For the pool model, successfully loaded the LoRA weights "
                    "after removing the prefix 'model.'."
                )
        return lora_model.get_lora(org_module_name)

    def deactivate_adapter(self, adapter_id: int) -> bool:
        if adapter_id not in self._active_adapters:
            return False
        self._deactivate_adapter(adapter_id)
        self._active_adapters.pop(adapter_id, None)
        return True

    def add_adapter(self, adapter: LoRAModel) -> bool:
        logger.debug("Adding lora. Model id: %d, int id: %d", adapter.id, adapter.id)
        if adapter.id in self._registered_adapters:
            return False
        if len(self._registered_adapters) >= self.capacity:
            raise RuntimeError("No free adapter slots.")
        self._add_adapter(adapter)
        return True

    def set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        if self._last_mapping != mapping:
            self._set_adapter_mapping(mapping)
            self._last_mapping = mapping

    def remove_adapter(self, adapter_id: int) -> bool:
        self.deactivate_adapter(adapter_id)
        if adapter_id not in self._registered_adapters:
            return False
        self._registered_adapters.pop(adapter_id, None)
        return True

    def list_adapters(self) -> dict[int, LoRAModel]:
        return dict(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> LoRAModel | None:
        return self._registered_adapters.get(adapter_id)


class LoRALRUCache(AdapterLRUCache[LoRAModel]):
    def __init__(self, capacity: int, deactivate_lora_fn: Callable[[int], bool]):
        super().__init__(capacity, deactivate_lora_fn)


class LRUCacheLoRAModelManager(LoRAModelManager):
    """A model manager that manages multiple LoRAs with LRU cache."""

    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        vllm_config: VllmConfig | None = None,
    ):
        super().__init__(
            model,
            max_num_seqs,
            max_num_batched_tokens,
            vocab_size,
            lora_config,
            device,
            vllm_config,
        )
        self._registered_adapters: LoRALRUCache = LoRALRUCache(
            self.capacity, self.deactivate_adapter
        )
        self._active_adapters: LoRALRUCache = LoRALRUCache(
            self.lora_slots, self._deactivate_adapter
        )

    def list_adapters(self) -> dict[int, LoRAModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_adapters.cache)

    def add_adapter(self, lora: LoRAModel) -> bool:
        """Add a LoRAModel to the manager."""
        logger.debug("Adding lora. Model id: %d, int id: %d", lora.id, lora.id)
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
        if (
            lora_id not in self._active_adapters
            and len(self._active_adapters) >= self.lora_slots
        ):
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
            raise ValueError(
                f"Pinning failed. LoRA {lora_id} is not registered."
            ) from err

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
    vllm_config: VllmConfig,
    device: torch.device,
    lora_manager_cls: type[LoRAModelManager] = LoRAModelManager,
    **kwargs,
) -> LoRAModelManager:
    """Create a LoRA adapter for a given model."""
    if not isinstance(model, SupportsLoRA):
        raise ValueError(f"Model {type(model)} is not supported for LoRA.")
    lora_manager = lora_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        lora_config=lora_config,
        vllm_config=vllm_config,
        device=device,
        **kwargs,
    )
    return lora_manager
