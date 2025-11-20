# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm import envs
from vllm.config.lora import LoRAConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.utils import divide
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import (
    _get_config_dtype_str,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    modular_marlin_fused_moe,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    modular_triton_fused_moe,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
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

    def _normalize_keys(self, config: dict[str, int | None]) -> dict[str, int | None]:
        normalized_config = {}
        for key, value in config.items():
            if key.islower():
                if key.startswith("block_"):
                    normalized_key = "BLOCK_SIZE_" + key.split("_")[-1].upper()
                else:
                    normalized_key = key.upper()
            else:
                normalized_key = key
            normalized_config[normalized_key] = value
        return normalized_config

    def _get_lora_moe_configs(
        self,
        op_prefix: str,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        num_slices: int,
        M: int,
        layer: FusedMoE,
        top_k: int,
        config_dtype: str,
    ):
        if envs.VLLM_TUNED_CONFIG_FOLDER:
            shrink_config = get_lora_op_configs(
                op_type=f"fused_moe_lora_{op_prefix}_shrink",
                max_loras=lora_a_stacked.shape[0],
                batch=M,
                hidden_size=lora_a_stacked.shape[-1],
                rank=lora_a_stacked.shape[-2],
                num_slices=num_slices,
                moe_intermediate_size=lora_b_stacked.shape[-2],
            )
            expand_config = get_lora_op_configs(
                op_type=f"fused_moe_lora_{op_prefix}_expand",
                max_loras=lora_a_stacked.shape[0],
                batch=M,
                hidden_size=lora_a_stacked.shape[-1],
                rank=lora_a_stacked.shape[-2],
                num_slices=num_slices,
                moe_intermediate_size=lora_b_stacked.shape[-2],
            )
        else:  # fall back to the default config
            get_config_func = functools.partial(
                try_get_optimal_moe_config,
                layer.w13_weight.size(),
                layer.w2_weight.size(),
                top_k,
                config_dtype,
                block_shape=layer.quant_method.moe_quant_config.block_shape,
            )
            shrink_config = get_config_func(M)
            expand_config = get_config_func(M)
        shrink_config = self._normalize_keys(shrink_config)
        expand_config = self._normalize_keys(expand_config)
        return shrink_config, expand_config

    def _inject_lora_into_fused_moe(self):
        moe_state_dict = {}
        top_k = self.base_layer.top_k

        self.base_layer.ensure_moe_quant_config_init()
        quant_config = self.base_layer.quant_method.moe_quant_config

        m_fused_moe_fn = (
            modular_triton_fused_moe(
                quant_config, shared_experts=self.base_layer.shared_experts
            )
            if not quant_config.use_mxfp4_w4a16
            else modular_marlin_fused_moe(
                quant_config, shared_experts=self.base_layer.shared_experts
            )
        )

        def fwd_decorator(layer, func):
            def wrapper(*args, **kwargs):
                moe_state_dict["hidden_states"] = kwargs["hidden_states"]
                moe_state_dict["topk_ids"] = kwargs["topk_ids"]
                moe_state_dict["topk_weights"] = kwargs["topk_weights"]
                moe_state_dict["expert_map"] = kwargs["expert_map"]
                moe_state_dict["apply_router_weight_on_input"] = kwargs[
                    "apply_router_weight_on_input"
                ]
                result = func(*args, **kwargs)
                return result

            return wrapper

        def act_decorator(layer, func):
            def wrapper(*args, **kwargs):
                _, output, input = args

                hidden_states = moe_state_dict["hidden_states"]
                topk_weights = moe_state_dict["topk_weights"]
                curr_topk_ids = moe_state_dict["topk_ids"]

                expert_map = moe_state_dict["expert_map"]

                config_dtype = _get_config_dtype_str(
                    dtype=hidden_states.dtype,
                    use_fp8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                )
                CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
                num_tokens = hidden_states.size(0)
                M = min(num_tokens, CHUNK_SIZE)

                shrink_config, expand_config = self._get_lora_moe_configs(
                    op_prefix="w13",
                    lora_a_stacked=self.w1_lora_a_stacked,
                    lora_b_stacked=self.w1_lora_b_stacked,
                    num_slices=2,
                    M=M,
                    layer=layer,
                    top_k=top_k,
                    config_dtype=config_dtype,
                )

                # get the block size of m from customized config or default config
                max_loras = self.w1_lora_a_stacked.shape[0]
                (
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                ) = self.punica_wrapper.moe_lora_align_block_size(
                    curr_topk_ids,
                    num_tokens,
                    shrink_config["BLOCK_SIZE_M"],
                    self.base_layer.local_num_experts,
                    max_loras,
                    self.adapter_enabled,
                    expert_map,
                )

                moe_state_dict["sorted_token_ids_lora"] = sorted_token_ids_lora
                moe_state_dict["expert_ids_lora"] = expert_ids_lora
                moe_state_dict["num_tokens_post_padded_lora"] = (
                    num_tokens_post_padded_lora
                )

                w13_lora_a_stacked = [self.w1_lora_a_stacked, self.w3_lora_a_stacked]
                w13_lora_b_stacked = [self.w1_lora_b_stacked, self.w3_lora_b_stacked]
                max_lora_rank = self.w1_lora_a_stacked.shape[-2]
                expert_ids_lora = expert_ids_lora.view(max_loras, -1)
                sorted_token_ids_lora = sorted_token_ids_lora.view(max_loras, -1)

                self.punica_wrapper.add_lora_fused_moe(
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
                    shrink_config,  ## pass the shrink config
                    expand_config,  ## pass the expand config
                    self.adapter_enabled,
                    fully_sharded=self.fully_sharded,
                )

                result = func(*args, **kwargs)

                moe_state_dict["intermediate_cache2"] = output
                return result

            return wrapper

        def moe_sum_decorator(layer, func):
            def wrapper(*args, **kwargs):
                hidden_states = moe_state_dict["hidden_states"]
                topk_weights = moe_state_dict["topk_weights"]

                config_dtype = _get_config_dtype_str(
                    dtype=hidden_states.dtype,
                    use_fp8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                )
                CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
                num_tokens = hidden_states.size(0)
                M = min(num_tokens, CHUNK_SIZE)

                shrink_config, expand_config = self._get_lora_moe_configs(
                    op_prefix="w2",
                    lora_a_stacked=self.w2_lora_a_stacked,
                    lora_b_stacked=self.w2_lora_b_stacked,
                    num_slices=1,
                    M=M,
                    layer=layer,
                    top_k=top_k,
                    config_dtype=config_dtype,
                )

                sorted_token_ids_lora = moe_state_dict["sorted_token_ids_lora"]
                expert_ids_lora = moe_state_dict["expert_ids_lora"]
                num_tokens_post_padded_lora = moe_state_dict[
                    "num_tokens_post_padded_lora"
                ]
                max_loras = self.w1_lora_a_stacked.shape[0]
                expert_ids_lora = expert_ids_lora.view(max_loras, -1)
                sorted_token_ids_lora = sorted_token_ids_lora.view(max_loras, -1)
                intermediate_cache2 = moe_state_dict["intermediate_cache2"]
                intermediate_cache3 = args[0]
                max_lora_rank = self.w2_lora_a_stacked.shape[-2]

                shard_size_w2 = divide(self.base_layer.hidden_size, self.tp_size)

                self.punica_wrapper.add_lora_fused_moe(
                    intermediate_cache3,
                    intermediate_cache2,
                    [self.w2_lora_a_stacked],
                    [self.w2_lora_b_stacked],
                    topk_weights,
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                    max_lora_rank,
                    top_k,
                    shrink_config,  ## pass the shrink config
                    expand_config,  ## pass the expand config
                    self.adapter_enabled,
                    True,
                    fully_sharded=self.fully_sharded,
                    offset=shard_size_w2 * self.tp_rank if self.fully_sharded else 0,
                )

                result = func(*args, **kwargs)
                return result

            return wrapper

        fused_experts = m_fused_moe_fn.fused_experts

        m_fused_moe_fn.forward = fwd_decorator(self.base_layer, m_fused_moe_fn.forward)
        fused_experts.activation = act_decorator(
            self.base_layer, fused_experts.activation
        )
        fused_experts.moe_sum = moe_sum_decorator(
            self.base_layer, fused_experts.moe_sum
        )

        self.base_layer.quant_method = FusedMoEModularMethod(
            self.base_layer.quant_method, m_fused_moe_fn
        )

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""
        self.fully_sharded = lora_config.fully_sharded_loras

        self.adapter_enabled = torch.tensor(
            [0] * (max_loras + 1), dtype=torch.int, device=self.device
        )

        self.w1_lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                lora_config.max_lora_rank
                if not self.fully_sharded
                else divide(lora_config.max_lora_rank, self.tp_size),
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
                self.base_layer.hidden_size
                if not self.fully_sharded
                else divide(self.base_layer.hidden_size, self.tp_size),
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

        self.w3_lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.local_num_experts,
                lora_config.max_lora_rank
                if not self.fully_sharded
                else divide(lora_config.max_lora_rank, self.tp_size),
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
        self.adapter_enabled[index] = 0

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
        self.adapter_enabled[index] = 1
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

                if self.fully_sharded:
                    # Based on S-LoRA, we slice W1 and W3 A along the rank dim,
                    # and W2 B along the hidden_size dim.
                    w13_shard_size = self.w1_lora_a_stacked[index, eid].shape[0]
                    w13_start_idx = self.tp_rank * w13_shard_size
                    w13_end_idx = (self.tp_rank + 1) * w13_shard_size
                    w1_lora_a = w1_lora_a[w13_start_idx:w13_end_idx, :]
                    w3_lora_a = w3_lora_a[w13_start_idx:w13_end_idx, :]

                    w2_shard_size = self.w2_lora_b_stacked[index, eid].shape[0]
                    w2_start_idx = self.tp_rank * w2_shard_size
                    w2_end_idx = (self.tp_rank + 1) * w2_shard_size
                    w2_lora_b = w2_lora_b[w2_start_idx:w2_end_idx, :]

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

    @property
    def _shared_experts(self):
        return self.base_layer._shared_experts

    @property
    def quant_method(self):
        return self.base_layer.quant_method

    @property
    def is_internal_router(self) -> bool:
        return self.base_layer.is_internal_router
