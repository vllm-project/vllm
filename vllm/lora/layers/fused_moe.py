# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm import envs
from vllm.config.lora import LoRAConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import (FusedMoEQuantConfig,
                                                         _get_config_dtype_str)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    modular_triton_fused_moe, try_get_optimal_moe_config)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_lora_align_block_size)


class FusedMoEWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: FusedMoE) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = base_layer.w2_weight.device
        self._inject_lora_into_fused_moe()

    def _inject_lora_into_fused_moe(self):
        base_layer = self.base_layer
        base_layer._lora = {}
        top_k = base_layer.top_k
        quant_config = base_layer.quant_config

        def fwd_decorator(layer, func):

            def wrapper(*args, **kwargs):
                self.base_layer._lora["hidden_states"] = kwargs[
                    "hidden_states"]
                self.base_layer._lora["topk_ids"] = kwargs["topk_ids"]
                self.base_layer._lora["topk_weights"] = kwargs["topk_weights"]
                self.base_layer._lora["global_num_experts"] = kwargs[
                    "global_num_experts"]
                self.base_layer._lora["expert_map"] = kwargs["expert_map"]
                self.base_layer._lora["apply_router_weight_on_input"] = kwargs[
                    "apply_router_weight_on_input"]
                result = func(*args, **kwargs)
                return result

            return wrapper

        def act_decorator(layer, func):

            def wrapper(*args, **kwargs):
                _, output, input = args

                hidden_states = layer._lora["hidden_states"]
                topk_weights = layer._lora["topk_weights"]
                curr_topk_ids = layer._lora["topk_ids"]
                global_num_experts = layer._lora["global_num_experts"]
                expert_map = layer._lora["expert_map"]

                (token_lora_mapping, _, _, _, _,
                 _) = layer.punica_wrapper.token_mapping_meta.meta_args(
                     hidden_states.size(0))
                config_dtype = _get_config_dtype_str(use_fp8_w8a8=False,
                                                     use_int8_w8a16=False,
                                                     use_int4_w4a16=False,
                                                     use_mxfp4_w4a4=False,
                                                     dtype=hidden_states.dtype)
                CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
                num_tokens = hidden_states.size(0)
                M = min(num_tokens, CHUNK_SIZE)

                get_config_func = functools.partial(
                    try_get_optimal_moe_config,
                    layer.w13_weight.size(),
                    layer.w2_weight.size(),
                    top_k,
                    config_dtype,
                    block_shape=layer.quant_method.moe_quant_config.
                    block_shape,
                )

                config = get_config_func(M)
                (sorted_token_ids_lora, expert_ids_lora,
                 num_tokens_post_padded_lora) = (moe_lora_align_block_size(
                     curr_topk_ids, token_lora_mapping, config['BLOCK_SIZE_M'],
                     global_num_experts, curr_topk_ids.shape[-1], expert_map))

                layer._lora["sorted_token_ids_lora"] = sorted_token_ids_lora
                layer._lora["expert_ids_lora"] = expert_ids_lora
                layer._lora[
                    "num_tokens_post_padded_lora"] = num_tokens_post_padded_lora

                w1_lora_a_stacked = layer.w1_lora_a_stacked
                w1_lora_b_stacked = layer.w1_lora_b_stacked
                w3_lora_a_stacked = layer.w3_lora_a_stacked
                w3_lora_b_stacked = layer.w3_lora_b_stacked

                max_lora_rank = w1_lora_a_stacked.shape[-2]
                w13_lora_a_stacked = [w1_lora_a_stacked, w3_lora_a_stacked]
                w13_lora_b_stacked = [w1_lora_b_stacked, w3_lora_b_stacked]
                expert_ids_lora = expert_ids_lora.view(curr_topk_ids.shape[-1],
                                                       -1)
                sorted_token_ids_lora = sorted_token_ids_lora.view(
                    curr_topk_ids.shape[-1], -1)

                layer.punica_wrapper.add_lora_fused_moe(
                    input.view(-1, top_k, input.shape[-1]),
                    hidden_states,
                    w13_lora_a_stacked,
                    w13_lora_b_stacked,
                    topk_weights,
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                    max_lora_rank,
                    top_k,
                    config,
                )

                result = func(*args, **kwargs)

                layer._lora["intermediate_cache2"] = output
                return result

            return wrapper

        def finalize_weight_and_reduce_impl_decorator(layer, func):
            def wrapper(*args, **kwargs):
                topk_weight_and_reduce = func(*args, **kwargs)

                def apply_decorator(func):
                    def wrapper(*args, **kwargs):
                        # _,fused_expert_output,_,_,_=args
                        fused_expert_output = kwargs["fused_expert_output"]

                        hidden_states = layer._lora["hidden_states"]
                        topk_weights = layer._lora["topk_weights"]
                        curr_topk_ids = layer._lora["topk_ids"]

                        config_dtype = _get_config_dtype_str(use_fp8_w8a8=False,
                                                     use_int8_w8a16=False,
                                                     use_int4_w4a16=False,
                                                     use_mxfp4_w4a4=False,
                                                     dtype=hidden_states.dtype)
                        
                        CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
                        num_tokens = hidden_states.size(0)
                        M = min(num_tokens, CHUNK_SIZE)

                        get_config_func = functools.partial(
                            try_get_optimal_moe_config,
                            layer.w13_weight.size(),
                            layer.w2_weight.size(),
                            top_k,
                            config_dtype,
                            block_shape=layer.quant_method.moe_quant_config.
                            block_shape,
                        )

                        config = get_config_func(M)
                        w1_lora_a_stacked = layer.w1_lora_a_stacked
                        w2_lora_a_stacked = layer.w2_lora_a_stacked
                        w2_lora_b_stacked = layer.w2_lora_b_stacked

                        max_lora_rank = w1_lora_a_stacked.shape[-2]

                        sorted_token_ids_lora = layer._lora["sorted_token_ids_lora"]
                        expert_ids_lora = layer._lora["expert_ids_lora"]
                        num_tokens_post_padded_lora = layer._lora[
                            "num_tokens_post_padded_lora"]

                        expert_ids_lora = expert_ids_lora.view(curr_topk_ids.shape[-1],
                                                            -1)
                        sorted_token_ids_lora = sorted_token_ids_lora.view(
                            curr_topk_ids.shape[-1], -1)
                        intermediate_cache2 = layer._lora["intermediate_cache2"]

                        layer.punica_wrapper.add_lora_fused_moe(
                            fused_expert_output, intermediate_cache2,
                            [w2_lora_a_stacked], [w2_lora_b_stacked], topk_weights,
                            sorted_token_ids_lora, expert_ids_lora,
                            num_tokens_post_padded_lora, max_lora_rank, top_k, config,
                            True)

                        result = func(*args, **kwargs)
                        return result

                    return wrapper

                topk_weight_and_reduce.apply = apply_decorator(topk_weight_and_reduce.apply)

                return topk_weight_and_reduce
            return wrapper


        m_fused_moe_fn = modular_triton_fused_moe(
            quant_config
            if quant_config is not None else FusedMoEQuantConfig.make(),
            shared_experts=base_layer.shared_experts)

        fused_experts = m_fused_moe_fn.fused_experts

        m_fused_moe_fn.forward = fwd_decorator(base_layer,
                                               m_fused_moe_fn.forward)
        fused_experts.activation = act_decorator(base_layer,
                                                 fused_experts.activation)

        fused_experts.finalize_weight_and_reduce_impl = \
            finalize_weight_and_reduce_impl_decorator(base_layer,fused_experts.finalize_weight_and_reduce_impl)

        base_layer.quant_method.old_fused_experts = \
            base_layer.quant_method.fused_experts
        base_layer.quant_method.fused_experts = m_fused_moe_fn

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """Initializes lora matrices."""
        self.w1_lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.global_num_experts,
                lora_config.max_lora_rank,
                self.base_layer.hidden_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.w1_lora_b_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.global_num_experts,
                self.base_layer.intermediate_size_per_partition,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

        self.w2_lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.global_num_experts,
                lora_config.max_lora_rank,
                self.base_layer.intermediate_size_per_partition,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.w2_lora_b_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.global_num_experts,
                self.base_layer.hidden_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

        self.w3_lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.global_num_experts,
                lora_config.max_lora_rank,
                self.base_layer.hidden_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.w3_lora_b_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.global_num_experts,
                self.base_layer.intermediate_size_per_partition,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

        self.base_layer.w1_lora_a_stacked = self.w1_lora_a_stacked
        self.base_layer.w1_lora_b_stacked = self.w1_lora_b_stacked
        self.base_layer.w2_lora_a_stacked = self.w2_lora_a_stacked
        self.base_layer.w2_lora_b_stacked = self.w2_lora_b_stacked
        self.base_layer.w3_lora_a_stacked = self.w3_lora_a_stacked
        self.base_layer.w3_lora_b_stacked = self.w3_lora_b_stacked
        # They will be used by 'LoRALayerWeights.create_dummy_lora_weights'
        # to create a dummy LoRA weights.
        self.lora_a_stacked = []
        self.lora_b_stacked = []
        for lora_id in range(max_loras):
            for experts_id in range(self.base_layer.global_num_experts):
                # gate_proj,down_proj,up_proj
                self.lora_a_stacked.append(
                    self.w1_lora_a_stacked[lora_id][experts_id])
                self.lora_a_stacked.append(
                    self.w2_lora_a_stacked[lora_id][experts_id])
                self.lora_a_stacked.append(
                    self.w3_lora_a_stacked[lora_id][experts_id])

                self.lora_b_stacked.append(
                    self.w1_lora_b_stacked[lora_id][experts_id])
                self.lora_b_stacked.append(
                    self.w2_lora_b_stacked[lora_id][experts_id])
                self.lora_b_stacked.append(
                    self.w3_lora_b_stacked[lora_id][experts_id])

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        self.w1_lora_a_stacked[index] = 0
        self.w1_lora_b_stacked[index] = 0
        self.w3_lora_a_stacked[index] = 0
        self.w3_lora_b_stacked[index] = 0
        self.w2_lora_a_stacked[index] = 0
        self.w2_lora_b_stacked[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        """Overwrites lora tensors at index."""
        for eid in range(len(lora_a) // 3):
            w1_lora_a = lora_a[eid * 3]
            w2_lora_a = lora_a[eid * 3 + 1]
            w3_lora_a = lora_a[eid * 3 + 2]
            w1_lora_b = lora_b[eid * 3]
            w2_lora_b = lora_b[eid * 3 + 1]
            w3_lora_b = lora_b[eid * 3 + 2]

            if self.tp_size > 1:

                shard_size = self.base_layer.intermediate_size_per_partition
                start_idx = self.tp_rank * shard_size
                end_idx = (self.tp_rank + 1) * shard_size

                w1_lora_b = w1_lora_b[start_idx:end_idx,:]
                w3_lora_b = w3_lora_b[start_idx:end_idx,:]
                w2_lora_a = w2_lora_a[:,start_idx:end_idx]

            self.w1_lora_a_stacked[
                index, eid, :w1_lora_a.shape[0], :w1_lora_a.shape[1]].copy_(
                    w1_lora_a, non_blocking=True)

            self.w3_lora_a_stacked[
                index, eid, :w3_lora_a.shape[0], :w3_lora_a.shape[1]].copy_(
                    w3_lora_a, non_blocking=True)

            self.w2_lora_b_stacked[
                index, eid, :w2_lora_b.shape[0], :w2_lora_b.shape[1]].copy_(
                    w2_lora_b, non_blocking=True)

            self.w1_lora_b_stacked[
                index, eid, :w1_lora_b.shape[0], :w1_lora_b.shape[1]].copy_(
                    w1_lora_b, non_blocking=True)
            self.w3_lora_b_stacked[
                index, eid, :w3_lora_b.shape[0], :w3_lora_b.shape[1]].copy_(
                    w3_lora_b, non_blocking=True)
            self.w2_lora_a_stacked[
                index, eid, :w2_lora_a.shape[0], :w2_lora_a.shape[1]].copy_(
                    w2_lora_a, non_blocking=True)

    def set_mapping(
        self,
        punica_wrapper,
    ):
        self.punica_wrapper: PunicaWrapperBase = punica_wrapper
        self.base_layer.punica_wrapper = self.punica_wrapper

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        # return type(source_layer) is FusedMoE
        return isinstance(source_layer, FusedMoE)

    def forward(self, *args, **kwargs):
        return self.base_layer.forward(*args, **kwargs)

    def maybe_all_reduce_tensor_model_parallel(self, *args, **kwargs):
        return self.base_layer.maybe_all_reduce_tensor_model_parallel(
            *args, **kwargs)

    @property
    def _shared_experts(self):
        return self.base_layer._shared_experts

    @property
    def quant_method(self):
        return self.base_layer.quant_method
