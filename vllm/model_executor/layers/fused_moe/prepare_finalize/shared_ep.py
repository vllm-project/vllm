# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare and finalize owner-resident SharedEP objects."""

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.shared_ep import get_shared_ep_memory


class SharedEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        quant_dtype: str,
        group: dist.ProcessGroup,
        device: torch.device,
        max_tokens: int = 32,
    ) -> None:
        self.memory = get_shared_ep_memory(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            top_k=top_k,
            quant_dtype=quant_dtype,
            group=group,
            device=device,
        )
        self._num_tokens = 0

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.memory.max_tokens

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.memory.world_size

    def output_is_reduced(self) -> bool:
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        if apply_router_weight_on_input:
            raise ValueError("SharedEP applies router weights during W2")
        if defer_input_quant:
            raise ValueError("SharedEP owns native activation publication")
        if self.memory.quant_dtype == "nvfp4":
            if quant_config.quant_dtype != "nvfp4":
                raise ValueError("Native NVFP4 SharedEP requires NVFP4 quantization")
            if quant_config.a1_gscale is None:
                raise ValueError("Native NVFP4 SharedEP requires an input global scale")
            self._num_tokens = self.memory.publish_input(
                a1,
                topk_ids,
                topk_weights,
                input_global_scale=quant_config.a1_gscale,
            )
            activations, scales, global_ids, global_weights = (
                self.memory.gather_nvfp4_inputs()
            )
            return (
                activations,
                scales,
                None,
                global_ids,
                global_weights,
            )
        if quant_config.quant_dtype != "mxfp8":
            raise ValueError("Native MXFP4 SharedEP requires MXFP8 activations")
        self._num_tokens = self.memory.publish_input(
            a1,
            topk_ids,
            topk_weights,
        )
        activations, scales, global_ids, global_weights = (
            self.memory.gather_mxfp8_inputs()
        )
        return (
            activations,
            scales,
            None,
            global_ids,
            global_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if self.memory.quant_dtype == "nvfp4":
            self.memory.reduce_direct_output(output, self._num_tokens)
        else:
            self.memory.reduce_output(output, self._num_tokens)
