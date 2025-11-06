# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe_permute_experts_unpermute import (
    FusedMoEPermuteExpertsUnpermuteWithLoRA,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.layer import FusedMoEModularMethod
from vllm.model_executor.layers.fused_moe.mk_fused_experts_lora_support import (
    mk_fused_experts_supports_lora,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEModularKernel,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)


class FusedMoEWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: FusedMoE) -> None:
        super().__init__()
        self.base_layer = base_layer

        assert not self.base_layer.use_ep, (
            "EP support for Fused MoE LoRA is not implemented yet."
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = base_layer.w2_weight.device
        self._inject_lora_into_fused_moe()

        assert isinstance(self.base_layer.quant_method, FusedMoEModularMethod)
        self.fused_experts: FusedMoEPermuteExpertsUnpermuteWithLoRA = (
            self.base_layer.quant_method.fused_experts.fused_experts
        )
        assert isinstance(self.fused_experts, FusedMoEPermuteExpertsUnpermuteWithLoRA)

    @staticmethod
    def _use_modular_kernel(layer: FusedMoE) -> FusedMoEModularMethod:
        assert not isinstance(layer.quant_method, FusedMoEModularMethod)
        qm = layer.quant_method
        prepare_finalize = MoEPrepareAndFinalizeNoEP()
        mk_experts = qm.select_gemm_impl(prepare_finalize, layer)
        mk = FusedMoEModularKernel(prepare_finalize, mk_experts, layer.shared_experts)
        return FusedMoEModularMethod(layer.quant_method, mk)

    def _inject_lora_into_fused_moe(self):
        # TODO: Support EP
        assert not self.base_layer.use_ep

        self.base_layer.ensure_moe_quant_config_init()

        # Replace base-layer quant_method with ModularKernel
        if not isinstance(self.base_layer.quant_method, FusedMoEModularMethod):
            self.base_layer.quant_method = FusedMoEWithLoRA._use_modular_kernel(
                self.base_layer
            )
        assert isinstance(self.base_layer.quant_method, FusedMoEModularMethod)

        assert isinstance(
            self.base_layer.quant_method.fused_experts, FusedMoEModularKernel
        )
        fused_experts = self.base_layer.quant_method.fused_experts.fused_experts
        assert mk_fused_experts_supports_lora(fused_experts)
        # TODO: Support batched activation format
        assert fused_experts.activation_formats == (
            FusedMoEActivationFormat.Standard,
            FusedMoEActivationFormat.Standard,
        )

        # Override fused_experts implementation.
        self.base_layer.quant_method.fused_experts.fused_experts = (
            FusedMoEPermuteExpertsUnpermuteWithLoRA(fused_experts)
        )

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""

        self.w1_lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                lora_config.max_lora_rank,
                self.base_layer.hidden_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.w1_lora_b_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                self.base_layer.intermediate_size_per_partition,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

        self.w2_lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                lora_config.max_lora_rank,
                self.base_layer.intermediate_size_per_partition,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.w2_lora_b_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                self.base_layer.hidden_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

        self.w3_lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                lora_config.max_lora_rank,
                self.base_layer.hidden_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.w3_lora_b_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                self.base_layer.intermediate_size_per_partition,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

        self.fused_experts.set_lora_weights(
            self.w1_lora_a_stacked,
            self.w1_lora_b_stacked,
            self.w3_lora_a_stacked,
            self.w3_lora_b_stacked,
            self.w2_lora_a_stacked,
            self.w2_lora_b_stacked,
        )

        # They will be used by 'LoRALayerWeights.create_dummy_lora_weights'
        # to create a dummy LoRA weights.
        self.lora_a_stacked = []
        self.lora_b_stacked = []
        for lora_id in range(max_loras):
            for experts_id in range(self.base_layer.local_num_experts):
                # gate_proj,down_proj,up_proj
                self.lora_a_stacked.append(self.w1_lora_a_stacked[lora_id][experts_id])
                self.lora_a_stacked.append(self.w2_lora_a_stacked[lora_id][experts_id])
                self.lora_a_stacked.append(self.w3_lora_a_stacked[lora_id][experts_id])

                self.lora_b_stacked.append(self.w1_lora_b_stacked[lora_id][experts_id])
                self.lora_b_stacked.append(self.w2_lora_b_stacked[lora_id][experts_id])
                self.lora_b_stacked.append(self.w3_lora_b_stacked[lora_id][experts_id])

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        self.w1_lora_a_stacked[index] = 0
        self.w1_lora_b_stacked[index] = 0
        self.w3_lora_a_stacked[index] = 0
        self.w3_lora_b_stacked[index] = 0
        self.w2_lora_a_stacked[index] = 0
        self.w2_lora_b_stacked[index] = 0
        self.fused_experts.set_adapter_enabled(index, 0)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: torch.Tensor | None,
        bias: torch.Tensor | None = None,
    ):
        """Overwrites lora tensors at index."""
        self.reset_lora(index)
        self.fused_experts.set_adapter_enabled(index, 1)
        for eid in range(len(lora_a) // 3):
            w1_lora_a = lora_a[eid * 3]
            w2_lora_a = lora_a[eid * 3 + 1]
            w3_lora_a = lora_a[eid * 3 + 2]
            w1_lora_b = lora_b[eid * 3]
            w2_lora_b = lora_b[eid * 3 + 1]
            w3_lora_b = lora_b[eid * 3 + 2]

            # Handle the case of adding LoRA to only a subset of experts
            if w1_lora_a is None or w2_lora_a is None or w3_lora_a is None:
                continue

            if self.tp_size > 1:
                shard_size = self.base_layer.intermediate_size_per_partition
                start_idx = self.tp_rank * shard_size
                end_idx = (self.tp_rank + 1) * shard_size

                w1_lora_b = w1_lora_b[start_idx:end_idx, :]
                w3_lora_b = w3_lora_b[start_idx:end_idx, :]
                w2_lora_a = w2_lora_a[:, start_idx:end_idx]

            self.w1_lora_a_stacked[
                index, eid, : w1_lora_a.shape[0], : w1_lora_a.shape[1]
            ].copy_(w1_lora_a, non_blocking=True)

            self.w3_lora_a_stacked[
                index, eid, : w3_lora_a.shape[0], : w3_lora_a.shape[1]
            ].copy_(w3_lora_a, non_blocking=True)

            self.w2_lora_b_stacked[
                index, eid, : w2_lora_b.shape[0], : w2_lora_b.shape[1]
            ].copy_(w2_lora_b, non_blocking=True)

            self.w1_lora_b_stacked[
                index, eid, : w1_lora_b.shape[0], : w1_lora_b.shape[1]
            ].copy_(w1_lora_b, non_blocking=True)
            self.w3_lora_b_stacked[
                index, eid, : w3_lora_b.shape[0], : w3_lora_b.shape[1]
            ].copy_(w3_lora_b, non_blocking=True)
            self.w2_lora_a_stacked[
                index, eid, : w2_lora_a.shape[0], : w2_lora_a.shape[1]
            ].copy_(w2_lora_a, non_blocking=True)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        # return type(source_layer) is FusedMoE
        return isinstance(source_layer, FusedMoE)

    def forward(self, *args, **kwargs):
        return self.base_layer.forward(*args, **kwargs)

    def maybe_all_reduce_tensor_model_parallel(self, *args, **kwargs):
        return self.base_layer.maybe_all_reduce_tensor_model_parallel(*args, **kwargs)

    def set_mapping(
        self,
        punica_wrapper,
    ):
        super().set_mapping(punica_wrapper)
        self.fused_experts.set_mapping(self.punica_wrapper)

    @property
    def _shared_experts(self):
        return self.base_layer._shared_experts

    @property
    def quant_method(self):
        return self.base_layer.quant_method

    @property
    def is_internal_router(self) -> bool:
        return self.base_layer.is_internal_router
