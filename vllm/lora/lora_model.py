# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass

import safetensors
import torch

from vllm.logger import init_logger
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import (
    get_lora_id,
    is_base_embedding_weights,
    parse_fine_tuned_lora_name,
)
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.utils.torch_utils import PIN_MEMORY

logger = init_logger(__name__)


@dataclass(frozen=True)
class MoEEPLoadSpec:
    """Per-expert-parallel slicing metadata for one FusedMoE LoRA module.

    Threaded into the LoRA loader so per-expert weights from EP ranks
    other than this one can be skipped before they ever hit CPU memory.
    """

    ep_rank: int
    local_num_experts: int
    global_num_experts: int


_EXPERTS_SEPARATOR = ".experts."


def _is_remote_expert_key(raw_name: str, spec: "MoEEPLoadSpec") -> bool:
    """
    Decide whether a checkpoint key belongs to a non-local expert.
    """
    pos = raw_name.find(_EXPERTS_SEPARATOR)
    if pos < 0:
        return False
    idx_start = pos + len(_EXPERTS_SEPARATOR)
    idx_end = raw_name.find(".", idx_start)
    if idx_end < 0:
        return False
    idx_str = raw_name[idx_start:idx_end]
    if not idx_str.isdigit():
        return False
    expert_idx = int(idx_str)
    local_start = spec.ep_rank * spec.local_num_experts
    return not (local_start <= expert_idx < local_start + spec.local_num_experts)


class LoRAModel:
    """A LoRA fine-tuned model."""

    def __init__(
        self,
        lora_model_id: int,
        rank: int,
        loras: dict[str, LoRALayerWeights],
        is_3d_lora_weight: bool = False,
    ) -> None:
        """
        Args:
            lora_model_id: The integer id for the lora model.
            rank: lora rank.
            loras: module name -> weights for lora-replaced layers.
            is_3d_lora_weight: Whether the on-disk MoE adapter is in the 3D
                fused (gate_up_proj / down_proj) layout. Propagated from the
                originating LoRARequest. Only consulted by the LoRA model
                manager when enable_mixed_moe_lora_format is on.

        """
        self.id = lora_model_id

        assert lora_model_id > 0, (
            f"a valid lora id should be greater than 0, got {self.id}"
        )
        self.rank = rank
        self.loras: dict[str, LoRALayerWeights] = loras
        self.is_3d_lora_weight = is_3d_lora_weight

    def clone(self, lora_model_id: int) -> "LoRAModel":
        """Return a copy of the object with different ids.

        Will share the underlying tensors."""
        return self.__class__(
            lora_model_id,
            rank=self.rank,
            loras=self.loras.copy(),
            is_3d_lora_weight=self.is_3d_lora_weight,
        )

    def get_lora(self, module_name: str) -> LoRALayerWeights | None:
        """Get LoRA for a given module by name"""
        return self.loras.get(module_name, None)

    def check_lora_name(self, lora_name: str) -> bool:
        return lora_name in self.loras

    @staticmethod
    def _should_skip_module(module_name: str, skip_prefixes: list[str]) -> bool:
        """Check if a module should be skipped based on skip prefixes"""
        for prefix in skip_prefixes:
            if f".{prefix}" in module_name or module_name.startswith(prefix):
                return True
        return False

    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id: int,
        tensors: dict[str, torch.Tensor],
        peft_helper: PEFTHelper,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        weights_mapper: WeightsMapper | None = None,
        skip_prefixes: list[str] | None = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a dictionary of tensors."""
        pin_memory = str(device) == "cpu" and PIN_MEMORY
        loras: dict[str, LoRALayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            if is_base_embedding_weights(tensor_name):
                continue
            # Skip modules based on model-defined prefixes (e.g., MTP layers)
            if skip_prefixes and cls._should_skip_module(tensor_name, skip_prefixes):
                continue
            module_name, is_lora_a = parse_fine_tuned_lora_name(
                tensor_name, weights_mapper
            )
            if module_name not in loras:
                loras[module_name] = LoRALayerWeights.from_config(
                    module_name, peft_helper
                )

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
            else:
                loras[module_name].lora_b = tensor.to(device=device, dtype=dtype)

                if pin_memory:
                    loras[module_name].lora_b = loras[module_name].lora_b.pin_memory()

        return cls(lora_model_id, peft_helper.r, loras)

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
        weights_mapper: WeightsMapper | None = None,
        tensorizer_config_dict: dict | None = None,
        skip_prefixes: list[str] | None = None,
        moe_ep_spec: MoEEPLoadSpec | None = None,
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
            skip_prefixes: List of module name prefixes to skip during loading.
                Models can define this to skip modules not used in inference
                (e.g., MTP layers). Format: ["mtp."]
            moe_ep_spec: When 2D FusedMoE LoRA modules are present with
                expert parallelism enabled, the (ep_rank, local, global)
                slicing metadata shared across all MoE layers. Non-local
                expert weights are skipped at read time instead of being
                loaded and discarded later.

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
                if is_base_embedding_weights(lora_module):
                    continue
                # Handle PEFT file format where experts.base_layer is the
                # gate_up_proj and experts is the down_proj
                if "base_layer" in lora_module:
                    continue
                # Skip modules based on model-defined prefixes
                if skip_prefixes and cls._should_skip_module(
                    lora_module, skip_prefixes
                ):
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
            tensorizer_dir = tensorizer_config.tensorizer_dir
            if tensorizer_dir is None:
                raise ValueError("tensorizer_dir must be set in tensorizer config.")
            lora_tensor_path = os.path.join(tensorizer_dir, "adapter_model.tensors")
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(
                lora_tensor_path,
                dtype=tensorizer_config.dtype,
                device=device,
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
                    if moe_ep_spec is not None and _is_remote_expert_key(
                        module, moe_ep_spec
                    ):
                        continue
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(lora_bin_file_path) or os.path.isfile(lora_pt_file_path):
            lora_file_path = (
                lora_bin_file_path
                if os.path.isfile(lora_bin_file_path)
                else lora_pt_file_path
            )
            tensors = torch.load(lora_file_path, map_location=device, weights_only=True)
            check_unexpected_modules(tensors)
            if moe_ep_spec is not None:
                # `.bin`/`.pt` adapters can't be lazy-loaded, but pruning
                # the dict here still frees the non-local expert tensors
                # before the dtype cast / pin_memory work that follows.
                tensors = {
                    k: v
                    for k, v in tensors.items()
                    if not _is_remote_expert_key(k, moe_ep_spec)
                }
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        return cls.from_lora_tensors(
            lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            model_vocab_size=model_vocab_size,
            weights_mapper=weights_mapper,
            skip_prefixes=skip_prefixes,
        )
