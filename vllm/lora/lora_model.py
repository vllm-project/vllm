# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import safetensors.torch
import torch

from vllm.logger import init_logger
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import (
    get_lora_id,
    is_base_embeddding_weights,
    parse_fine_tuned_lora_name,
)
from vllm.envs import SLAB_OPTIMIZATION
from typing import TypeVar, Optional
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.lora.slab_helper import create_slab_optimized_lora_model, process_slab_activation_loop
from vllm.config.lora import LoRAConfig

logger = init_logger(__name__)


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

    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id: int,
        tensors: dict[str, torch.Tensor],
        peft_helper: PEFTHelper,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        embedding_modules: dict[str, str] | None = None,
        embedding_padding_modules: list[str] | None = None,
        weights_mapper: WeightsMapper | None = None,
        lora_dir: Optional[str] = None,
        target_modules_dict: dict | None = None,
        target_lora_config: LoRAConfig | None = None,
        slab_path: Optional[str] = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a dictionary of tensors."""
        if not SLAB_OPTIMIZATION:
            pin_memory = str(device) == "cpu" and is_pin_memory_available()
            loras: dict[str, LoRALayerWeights] = {}
            
            # Track sizes for logging
            module_sizes: dict[str, dict[str, int]] = {}
            total_lora_bytes = 0
            
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
                    module_sizes[module_name] = {'lora_a': 0, 'lora_b': 0}

                if is_lora_a:
                    if (
                        "lora_embedding_A" in tensor_name
                        and model_vocab_size is not None
                        and model_vocab_size != tensor.shape[1]
                    ):
                        raise RuntimeError(
                            f"The embedding LoRA size({tensor.shape[1]}) must be consistent"
                            f" with the base model's vocabulary size({model_vocab_size})."
                        )
                    loras[module_name].lora_a = tensor.to(device=device, dtype=dtype)
                    if pin_memory:
                        loras[module_name].lora_a = loras[module_name].lora_a.pin_memory()
                    
                    # Track size
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    module_sizes[module_name]['lora_a'] = tensor_bytes
                    total_lora_bytes += tensor_bytes
                else:
                    loras[module_name].lora_b = tensor.to(device=device, dtype=dtype)

                    if pin_memory:
                        loras[module_name].lora_b = loras[module_name].lora_b.pin_memory()
                    
                    # Track size
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    module_sizes[module_name]['lora_b'] = tensor_bytes
                    total_lora_bytes += tensor_bytes

            # Log LoRA weight sizes (basic path - no slab optimization)
            logger.info(f"[BASIC_LORA_PATH] Loading LoRA without slab optimization")
            logger.info(f"[BASIC_LORA_PATH] Total modules: {len(loras)}")
            logger.info(f"[BASIC_LORA_PATH] Total LoRA weight size: {total_lora_bytes / (1024**2):.2f} MB ({total_lora_bytes:,} bytes)")
            
            # Log individual module sizes (first 20 modules)
            logger.info(f"[BASIC_LORA_PATH] Module sizes (first 20):")
            for idx, (mod_name, sizes) in enumerate(sorted(module_sizes.items())[:20]):
                lora_a_mb = sizes['lora_a'] / (1024**2)
                lora_b_mb = sizes['lora_b'] / (1024**2)
                total_mod_mb = (sizes['lora_a'] + sizes['lora_b']) / (1024**2)
                logger.info(f"  {idx+1}. {mod_name}: lora_a={lora_a_mb:.2f}MB, lora_b={lora_b_mb:.2f}MB, total={total_mod_mb:.2f}MB")
            
            if len(module_sizes) > 20:
                logger.info(f"  ... and {len(module_sizes) - 20} more modules")
            
            # Log breakdown by module type
            moe_size = 0
            attention_size = 0
            other_size = 0
            for mod_name, sizes in module_sizes.items():
                mod_total = sizes['lora_a'] + sizes['lora_b']
                if 'mlp.experts' in mod_name:
                    moe_size += mod_total
                elif 'attn' in mod_name:
                    attention_size += mod_total
                else:
                    other_size += mod_total
            
            logger.info(f"[BASIC_LORA_PATH] Size breakdown:")
            logger.info(f"  - MoE layers: {moe_size / (1024**2):.2f} MB ({100*moe_size/total_lora_bytes:.1f}%)")
            logger.info(f"  - Attention layers: {attention_size / (1024**2):.2f} MB ({100*attention_size/total_lora_bytes:.1f}%)")
            logger.info(f"  - Other layers: {other_size / (1024**2):.2f} MB ({100*other_size/total_lora_bytes:.1f}%)")

            return cls(lora_model_id, peft_helper.r, loras)
        else:
            # Use experimental slab optimization for improved performance
            logger.debug("Using slab-based LoRA tensor optimization")
            
            lora_model, gpu_slab, metadata = create_slab_optimized_lora_model(
                lora_model_id=lora_model_id,
                tensors=tensors,
                peft_helper=peft_helper,
                device=device,
                dtype=dtype,
                embeddings=False,
                target_embedding_padding=model_vocab_size,
                embedding_modules=embedding_modules,
                embedding_padding_modules=embedding_padding_modules,
                weights_mapper=weights_mapper,
                lora_dir=lora_dir,
                lora_config=peft_helper,
                target_modules_dict=target_modules_dict,  # Now pass target modules from LoRAManager
                target_lora_config=target_lora_config,     # Pass target lora_config with fully_sharded_loras
                slab_path=slab_path,  # Pass slab path for disk save/load
            )
            
            # EFFICIENT: Update only MoE modules with slab views (most critical for performance)
            if gpu_slab is not None and metadata is not None and target_modules_dict is not None:
                logger.info(f"[EFFICIENT_SLAB_UPDATE] Updating critical modules with slab views")
                
                # Pre-cache metadata lookup once for all modules
                if not hasattr(metadata, '_lookup_cache'):
                    metadata._lookup_cache = {info.module_name: info for info in metadata.tensor_infos}
                
                # Instead of calling expensive create_lora_weights, just cache slab references
                for module_name, module in target_modules_dict.items():
                    if hasattr(module, 'create_lora_weights'):
                        # Cache slab references without triggering expensive operations
                        module._gpu_slab_ref = gpu_slab
                        module._slab_metadata_ref = metadata
                        module._slab_ready = True
                        # Mark as using slab views to enable ultra-fast SetLoRA path
                        module._using_slab_views = True
                
                logger.info(f"[ULTRA_EFFICIENT_SLAB_UPDATE] Cached slab references for all {len(target_modules_dict)} modules - zero H2D transfers")
            
            # Add any post-processing after slab model creation
            torch.cuda.synchronize()  # Check for pending GPU operations
                
            return lora_model            
         

    @classmethod
    def from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: set[str],
        peft_helper: PEFTHelper,
        *,
        lora_model_id: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        embedding_modules: dict[str, str] | None = None,
        embedding_padding_modules: list[str] | None = None,
        weights_mapper: WeightsMapper | None = None,
        tensorizer_config_dict: dict | None = None,
        target_modules_dict: dict | None = None,
        target_lora_config: LoRAConfig | None = None,
        slab_path: Optional[str] = None,
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

        tensors: dict[str, torch.Tensor] = {}
        unexpected_modules: list[list[str] | str] = []

        def check_unexpected_modules(modules: dict):
            for lora_module in modules.keys():  # noqa
                if is_base_embeddding_weights(lora_module):
                    continue
                # Handle PEFT file format where experts.base_layer is the
                # gate_up_proj and experts is the down_proj
                if "base_layer" in lora_module:
                    continue
                module_name, _ = parse_fine_tuned_lora_name(lora_module, weights_mapper)
                # Case for expert lora weights
                if ".experts" in module_name:
                    expert_idx = module_name.find(".experts")
                    expert_suffix = module_name[expert_idx + 1 :]
                    if expert_suffix not in expected_lora_modules:
                        unexpected_modules.append(module_name)

                elif module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
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
            lora_file_path = (
                lora_bin_file_path
                if os.path.isfile(lora_bin_file_path)
                else lora_pt_file_path
            )
            tensors = torch.load(lora_file_path, map_location=device, weights_only=True)
            check_unexpected_modules(tensors)
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        # return cls.from_lora_tensors(
        #     lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id,
        #     tensors=tensors,
        #     peft_helper=peft_helper,
        #     device=device,
        #     dtype=dtype,
        #     model_vocab_size=model_vocab_size,
        #     weights_mapper=weights_mapper,
        #     lora_dir=lora_dir,
        #     target_modules_dict=target_modules_dict,
        #     target_lora_config=target_lora_config,
        #     slab_path=slab_path,

        # )

        return cls.from_lora_tensors(
            lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            model_vocab_size=model_vocab_size,
            embedding_modules=embedding_modules,
            embedding_padding_modules=embedding_padding_modules,
            weights_mapper=weights_mapper,
            lora_dir=lora_dir,
            target_modules_dict=target_modules_dict,
            target_lora_config=target_lora_config,
            slab_path=slab_path,
        )

