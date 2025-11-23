# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from collections.abc import Callable
from typing import TypeVar

import regex as re
import safetensors.torch
import torch
from torch import nn

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA, FusedMoEWithLoRA, LoRAMapping
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.lora.utils import (
    from_layer,
    from_layer_logits_processor,
    get_supported_lora_modules,
    is_base_embeddding_weights,
    is_moe_model,
    is_regex_target_modules,
    parse_fine_tuned_lora_name,
    process_packed_modules_mapping,
    replace_submodule,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models import SupportsLoRA, supports_multimodal
from vllm.model_executor.models.interfaces import is_pooling_model
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import PPMissingLayer, WeightsMapper
from vllm.utils.cache import LRUCache
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

T = TypeVar("T")


class AdapterLRUCache(LRUCache[int, T]):
    def __init__(self, capacity: int, deactivate_fn: Callable[[int], object]):
        super().__init__(capacity)
        self.deactivate_fn = deactivate_fn

    def _on_remove(self, key: int, value: T | None):
        logger.debug("Removing adapter int id: %d", key)
        self.deactivate_fn(key)
        return super()._on_remove(key, value)


_GLOBAL_LORA_ID = 0


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
        loras: dict[str, LoRALayerWeights],
    ) -> None:
        """
        Args:
            lora_model_id: The integer id for the lora model.
            rank: lora rank.
            loras: module name -> weights for lora-replaced layers.

        """
        self.id = lora_model_id

        assert lora_model_id > 0, (
            f"a valid lora id should be greater than 0, got {self.id}"
        )
        self.rank = rank
        self.loras: dict[str, LoRALayerWeights] = loras

    def clone(self, lora_model_id: int) -> "LoRAModel":
        """Return a copy of the object with different ids.

        Will share the underlying tensors."""
        return self.__class__(
            lora_model_id,
            rank=self.rank,
            loras=self.loras.copy(),
        )

    def get_lora(self, module_name: str) -> LoRALayerWeights | None:
        """Get LoRA for a given module by name"""
        return self.loras.get(module_name, None)

    def check_lora_name(self, lora_name: str) -> bool:
        return lora_name in self.loras

    # (yard1): TODO see if we can derive target_embedding_padding automatically
    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id: int,
        tensors: dict[str, torch.Tensor],
        peft_helper: PEFTHelper,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        target_embedding_padding: int | None = None,
        embedding_modules: dict[str, str] | None = None,
        embedding_padding_modules: list[str] | None = None,
        weights_mapper: WeightsMapper | None = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a dictionary of tensors."""
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        loras: dict[str, LoRALayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            if is_base_embeddding_weights(tensor_name):
                continue
            module_name, is_lora_a = parse_fine_tuned_lora_name(
                tensor_name, weights_mapper
            )
            if module_name not in loras:
                loras[module_name] = LoRALayerWeights.from_config(
                    module_name, peft_helper
                )

            if is_lora_a:
                loras[module_name].lora_a = tensor.to(device=device, dtype=dtype)
                if pin_memory:
                    loras[module_name].lora_a = loras[module_name].lora_a.pin_memory()
            else:
                loras[module_name].lora_b = tensor.to(device=device, dtype=dtype)
                assert embedding_padding_modules is not None
                if (
                    any(name in module_name for name in embedding_padding_modules)
                    and target_embedding_padding is not None
                ):
                    lora_b = loras[module_name].lora_b
                    assert target_embedding_padding >= lora_b.shape[0]
                    addition = target_embedding_padding - lora_b.shape[0]
                    loras[module_name].lora_b = torch.nn.functional.pad(
                        lora_b, (0, 0, 0, addition)
                    )
                if pin_memory:
                    loras[module_name].lora_b = loras[module_name].lora_b.pin_memory()

        for lora in loras.values():
            lora.optimize()

        return cls(lora_model_id, peft_helper.r, loras)

    @classmethod
    def from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: list[str],
        peft_helper: PEFTHelper,
        *,
        lora_model_id: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        target_embedding_padding: int | None = None,
        embedding_modules: dict[str, str] | None = None,
        embedding_padding_modules: list[str] | None = None,
        weights_mapper: WeightsMapper | None = None,
        tensorizer_config_dict: dict | None = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint.

        Args:
            lora_dir: The local path that has lora data.
            expected_lora_modules: Name of modules that are expected to be
                replaced by lora.
            peft_helper: Loaded lora configuration information.
            lora_model_id: LoRA model id. If not given, automatically set by
                a global counter.
            device: Device where the lora model is loaded.
            dtype: dtype of the lora model weights.

        Returns:
            Loaded LoRA Model.
        """
        lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
        lora_pt_file_path = os.path.join(lora_dir, "adapter_model.pt")
        # new_embeddings_tensor_path = os.path.join(
        #     lora_dir, "new_embeddings.safetensors"
        # )
        # new_embeddings_bin_file_path = os.path.join(lora_dir, "new_embeddings.bin")
        tensors: dict[str, torch.Tensor] = {}
        unexpected_modules: list[list[str] | str] = []

        def check_unexpected_modules(modules: dict):
            for lora_module in modules.keys():  # noqa
                if is_base_embeddding_weights(lora_module):
                    continue
                module_name, _ = parse_fine_tuned_lora_name(lora_module, weights_mapper)
                # Handle FSDP file format where experts.base_layer is the
                # gate_up_proj and experts is the down_proj
                if "base_layer" in lora_module:
                    continue
                # Case for expert lora weights
                if ".experts" in module_name:
                    if not any(
                        module_name.endswith(ele) for ele in expected_lora_modules
                    ):
                        unexpected_modules.append(module_name)
                elif module_name.split(".")[-1] not in expected_lora_modules:
                    unexpected_modules.append(module_name)

            if unexpected_modules:
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct"
                )

        if tensorizer_config_dict:
            from tensorizer import TensorDeserializer

            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            lora_tensor_path = os.path.join(
                tensorizer_config.tensorizer_dir, "adapter_model.tensors"
            )
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(
                lora_tensor_path,
                dtype=tensorizer_config.dtype,
                **tensorizer_args.deserialization_kwargs,
            )
            check_unexpected_modules(tensors)

        elif os.path.isfile(lora_tensor_path):
            # Find unexpected modules.
            # Use safetensor key as a source of truth to find expected modules.
            # in peft if you have target_modules A, B, C and C does not exist
            # in the model it won’t error and model will be trained with A, B
            # loraified. C won’t exist in the safetensor but it will exist in
            # the target_modules of the adapter_config.json.
            unexpected_modules = []
            with safetensors.safe_open(lora_tensor_path, framework="pt") as f:  # type: ignore
                # Load tensors if there are only expected modules.
                check_unexpected_modules(f)
                for module in f.keys():  # noqa
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(lora_bin_file_path) or os.path.isfile(lora_pt_file_path):
            # When a bin/pt file is provided, we rely on config to find
            # unexpected modules.
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
                peft_helper.target_modules, expected_lora_modules
            ):
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct"
                )
            lora_file_path = (
                lora_bin_file_path
                if os.path.isfile(lora_bin_file_path)
                else lora_pt_file_path
            )
            tensors = torch.load(lora_file_path, map_location=device, weights_only=True)
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        return cls.from_lora_tensors(
            lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            target_embedding_padding=target_embedding_padding,
            embedding_modules=embedding_modules,
            embedding_padding_modules=embedding_padding_modules,
            weights_mapper=weights_mapper,
        )


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
        self.punica_wrapper = get_punica_wrapper(
            max_num_batched_tokens,
            max_batches=self.max_num_seqs,
            device=self.device,
            max_loras=self.lora_config.max_loras,
        )

        self.supported_lora_modules = get_supported_lora_modules(self.model)
        assert self.supported_lora_modules, "No supported LoRA modules found in"
        f" {self.model.__class__.__name__}."

        self.packed_modules_mapping = process_packed_modules_mapping(self.model)
        # Used to indicate whether the model is a multimodal model
        self.supports_mm: bool = (
            supports_multimodal(self.model)
            # In case the model only supports LoRA for
            # text modules (e.g. ChatGLM)
            and hasattr(self.model, "get_mm_mapping")
        )
        self.is_pooling_model = is_pooling_model(self.model)
        self.packed_modules: dict[str, list[str]] = {}
        self.modules: dict[str, BaseLayerWithLoRA] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._last_mapping: LoRAMapping | None = None
        self._is_3d_moe_model = is_moe_model(self.model) and hasattr(
            self.model, "is_3d_moe_weight"
        )
        self._create_lora_modules()

        self.model.lora_manager = self

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
            # Note (gnovack) - If MOE lora weights are not split into
            # num_experts chunks, we split them here
            if isinstance(module, FusedMoEWithLoRA) and torch.is_tensor(
                module_lora.lora_a
            ):
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
        # update lora states
        self.punica_wrapper.update_metadata(
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
            new_module.set_mapping(self.punica_wrapper)
        pass

    def register_module(self, module_name: str, module: "BaseLayerWithLoRA"):
        assert isinstance(module, BaseLayerWithLoRA), (
            f"Module {module_name} must be a BaseLayerWithLoRA instance,"
        )
        f" got {type(module)}"
        self.modules[module_name] = module

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
                or self._filter_unsupported_mm_module(module_name)
            ):
                continue
            parts = module_name.split(".")
            if module_name not in self.packed_modules:
                assert embedding_modules is not None
                if parts[-1] in embedding_modules:
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

    def _filter_unsupported_mm_module(self, module_name: str) -> bool:
        """
        Regarding multimodal models, vLLM currently only supports adding LoRA to
        language model. LoRA for other modules, such as the vision tower, will
        be filtered out.
        """
        if self.supports_mm:
            module_mapping: MultiModelKeys = self.model.get_mm_mapping()
            prefix_lst = module_mapping.connector + module_mapping.tower_model
            return any([module_name.startswith(prefix) for prefix in prefix_lst])
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
            lora_model.loras[module_name] = PackedLoRALayerWeights.pack(
                replacement_loras
            )
            # Remove the modules that have been replaced.
            for module in replaced_module:
                lora_model.loras.pop(module, None)

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
    ):
        super().__init__(
            model, max_num_seqs, max_num_batched_tokens, vocab_size, lora_config, device
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
        device=device,
        **kwargs,
    )
    return lora_manager
