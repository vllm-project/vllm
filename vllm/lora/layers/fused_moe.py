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
    MarlinExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
    UnfusedOAITritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)

from .utils import _get_lora_device, try_get_optimal_moe_lora_config


class FusedMoEWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: FusedMoE) -> None:
        super().__init__()
        self.base_layer = base_layer

        assert not self.base_layer.use_ep, (
            "EP support for Fused MoE LoRA is not implemented yet."
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = _get_lora_device(base_layer)
        # For non-gated MoE (is_act_and_mul=False), only 1 slice is needed
        # since there's only up_proj (w1), not gate_proj + up_proj (w1 + w3)
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
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
        num_loras: int,
        rank: int,
        num_slices: int,
        M: int,
        layer: FusedMoE,
        top_k: int,
        config_dtype: str,
    ):
        if envs.VLLM_TUNED_CONFIG_FOLDER:
            hidden_size = layer.hidden_size
            intermediate_size = layer.intermediate_size_per_partition
            shrink_config = get_lora_op_configs(
                op_type=f"fused_moe_lora_{op_prefix}_shrink",
                max_loras=num_loras,
                batch=M,
                hidden_size=hidden_size,
                rank=rank,
                num_slices=num_slices,
                moe_intermediate_size=intermediate_size,
            )
            expand_config = get_lora_op_configs(
                op_type=f"fused_moe_lora_{op_prefix}_expand",
                max_loras=num_loras,
                batch=M,
                hidden_size=hidden_size,  # lora_a_stacked.shape[-1],
                rank=rank,
                num_slices=num_slices,
                moe_intermediate_size=intermediate_size,  # lora_b_stacked.shape[-2],
            )
        else:  # fall back to the default config
            get_config_func = functools.partial(
                try_get_optimal_moe_lora_config,
                w1_shape=layer.w13_weight.size(),
                w2_shape=layer.w2_weight.size(),
                rank=rank,
                top_k=top_k,
                dtype=config_dtype,
                M=M,
                block_shape=layer.quant_method.moe_quant_config.block_shape,
            )
            shrink_config = get_config_func(
                op_type=f"fused_moe_lora_{op_prefix}_shrink"
            )
            expand_config = get_config_func(
                op_type=f"fused_moe_lora_{op_prefix}_expand"
            )
        shrink_config = self._normalize_keys(shrink_config)
        expand_config = self._normalize_keys(expand_config)
        return shrink_config, expand_config

    def _inject_lora_into_fused_moe(self):
        moe_state_dict = {}
        top_k = self.base_layer.top_k

        self.base_layer.ensure_moe_quant_config_init()
        quant_config = self.base_layer.quant_method.moe_quant_config

        prepare_finalize = MoEPrepareAndFinalizeNoEP()
        m_fused_moe_fn = FusedMoEModularKernel(
            prepare_finalize,
            self.base_layer.quant_method.select_gemm_impl(
                prepare_finalize, self.base_layer
            ),
            self.base_layer.shared_experts,
        )
        if quant_config.use_mxfp4_w4a16:
            assert isinstance(
                m_fused_moe_fn.fused_experts, (MarlinExperts, UnfusedOAITritonExperts)
            )
        else:
            assert isinstance(m_fused_moe_fn.fused_experts, TritonExperts)

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
                max_lora_rank = self.w13_lora_a_stacked[0].shape[-2]
                shrink_config, expand_config = self._get_lora_moe_configs(
                    op_prefix="w13",
                    num_loras=self.max_loras,
                    rank=max_lora_rank,
                    num_slices=self._w13_slices,
                    M=M,
                    layer=layer,
                    top_k=top_k,
                    config_dtype=config_dtype,
                )

                # SPARSITY_FACTOR is a heuristic margin ensuring tokens * top_k
                # activates only a small fraction of total experts * loras.
                SPARSITY_FACTOR = 8
                naive_block_assignment = (
                    expert_map is None
                    and num_tokens * top_k * SPARSITY_FACTOR
                    <= self.base_layer.local_num_experts * self.max_loras
                )

                # get the block size of m from customized config or default config
                # token_lora_mapping and lora_ids are available from meta_args()
                # in add_lora_fused_moe(), so they're not returned here
                (
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                ) = self.punica_wrapper.moe_lora_align_block_size(
                    curr_topk_ids,
                    num_tokens,
                    shrink_config["BLOCK_SIZE_M"],
                    self.base_layer.local_num_experts,
                    self.max_loras,
                    self.adapter_enabled,
                    expert_map,
                    naive_block_assignment=naive_block_assignment,
                )

                moe_state_dict["sorted_token_ids_lora"] = sorted_token_ids_lora
                moe_state_dict["expert_ids_lora"] = expert_ids_lora
                moe_state_dict["num_tokens_post_padded_lora"] = (
                    num_tokens_post_padded_lora
                )

                self.punica_wrapper.add_lora_fused_moe(
                    input.view(-1, top_k, input.shape[-1]),
                    hidden_states,
                    self.w13_lora_a_stacked,
                    self.w13_lora_b_stacked,
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
                max_lora_rank = self.w2_lora_a_stacked[0].shape[-2]
                shrink_config, expand_config = self._get_lora_moe_configs(
                    op_prefix="w2",
                    num_loras=self.max_loras,
                    rank=max_lora_rank,
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

                intermediate_cache2 = moe_state_dict["intermediate_cache2"]
                intermediate_cache3 = args[0]

                shard_size_w2 = divide(self.base_layer.hidden_size, self.tp_size)

                self.punica_wrapper.add_lora_fused_moe(
                    intermediate_cache3,
                    intermediate_cache2,
                    self.w2_lora_a_stacked,
                    self.w2_lora_b_stacked,
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

    def _create_lora_a_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
    ):
        self.w13_lora_a_stacked: tuple[torch.Tensor, ...] = tuple(
            torch.zeros(
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
            for _ in range(self._w13_slices)
        )
        self.w2_lora_a_stacked: tuple[torch.Tensor, ...] = (
            torch.zeros(
                (
                    max_loras,
                    self.base_layer.local_num_experts,
                    lora_config.max_lora_rank,
                    self.base_layer.intermediate_size_per_partition,
                ),
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
        )

    def _create_lora_b_weights(self, max_loras: int, lora_config: LoRAConfig):
        self.w13_lora_b_stacked: tuple[torch.Tensor, ...] = tuple(
            torch.zeros(
                (
                    max_loras,
                    self.base_layer.local_num_experts,
                    self.base_layer.intermediate_size_per_partition,
                    lora_config.max_lora_rank,
                ),
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self._w13_slices)
        )
        self.w2_lora_b_stacked: tuple[torch.Tensor, ...] = (
            torch.zeros(
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
            ),
        )

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""
        self.max_loras = lora_config.max_loras
        self.fully_sharded = lora_config.fully_sharded_loras

        self.adapter_enabled = torch.tensor(
            [0] * (max_loras + 1), dtype=torch.int, device=self.device
        )

        self._create_lora_a_weights(max_loras, lora_config)
        self._create_lora_b_weights(max_loras, lora_config)
        # They will be used by 'LoRALayerWeights.create_dummy_lora_weights'
        # to create a dummy LoRA weights.
        # TODO Optimize this section
        self.lora_a_stacked = []
        self.lora_b_stacked = []
        for lora_id in range(max_loras):
            for experts_id in range(self.base_layer.local_num_experts):
                # For gated MoE: gate_proj (w1), down_proj (w2), up_proj (w3)
                # For non-gated MoE: up_proj (w1), down_proj (w2)
                self.lora_a_stacked.append(
                    self.w13_lora_a_stacked[0][lora_id][experts_id]
                )
                self.lora_a_stacked.append(
                    self.w2_lora_a_stacked[0][lora_id][experts_id]
                )

                self.lora_b_stacked.append(
                    self.w13_lora_b_stacked[0][lora_id][experts_id]
                )
                self.lora_b_stacked.append(
                    self.w2_lora_b_stacked[0][lora_id][experts_id]
                )

                # Only add w3 (up_proj) for gated MoE (_w13_slices == 2)
                if self._w13_slices == 2:
                    self.lora_a_stacked.append(
                        self.w13_lora_a_stacked[1][lora_id][experts_id]
                    )
                    self.lora_b_stacked.append(
                        self.w13_lora_b_stacked[1][lora_id][experts_id]
                    )

    def _slice_w13_a(self, w13_lora_a: torch.Tensor) -> torch.Tensor:
        """
        Applies to FusedMoEWithLoRA and FusedMoE3DWithLoRA
        """
        if self.tp_size == 1 or not self.fully_sharded:
            return w13_lora_a

        # w13_lora_a shape (num_experts,rank,input_size)
        current_lora_rank = w13_lora_a.shape[1]
        assert current_lora_rank % self.tp_size == 0
        # Based on S-LoRA, we slice W13/W1/W3 A along the rank dim.
        shard_size = self.w13_lora_a_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        return w13_lora_a[:, start_idx:end_idx, :]

    def _slice_w13_b(self, w13_lora_b: torch.Tensor):
        if self.tp_size == 1:
            return w13_lora_b

        # w13_lora_b shape (num_experts,output_size,rank)
        shard_size = self.base_layer.intermediate_size_per_partition
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size

        return w13_lora_b[:, start_idx:end_idx, :]

    def _slice_w2_a(self, w2_lora_a: torch.Tensor) -> torch.Tensor:
        """
        Applies to FusedMoEWithLoRA and FusedMoE3DWithLoRA
        """
        if self.tp_size == 1:
            return w2_lora_a
        # w2_lora_a shape (num_experts,rank,input_size)
        shard_size = self.base_layer.intermediate_size_per_partition
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size

        return w2_lora_a[:, :, start_idx:end_idx]

    def _slice_w2_b(self, w2_lora_b: torch.Tensor) -> torch.Tensor:
        """
        Applies to FusedMoEWithLoRA and FusedMoE3DWithLoRA
        """
        if self.tp_size == 1 or not self.fully_sharded:
            return w2_lora_b
        # Based on S-LoRA, we slice W2 B along the hidden_size dim.
        # w2_lora_b shape (num_experts,output_size,rank)
        shard_size = self.w2_lora_b_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size

        return w2_lora_b[:, start_idx:end_idx, :]

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        for pos in range(self._w13_slices):
            self.w13_lora_a_stacked[pos][index] = 0
            self.w13_lora_b_stacked[pos][index] = 0

        self.w2_lora_a_stacked[0][index] = 0
        self.w2_lora_b_stacked[0][index] = 0
        self.adapter_enabled[index] = 0

    #

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        """Overwrites lora tensors at index."""
        # Make mypy happy
        assert isinstance(lora_a, list)
        assert isinstance(lora_b, list)

        self.reset_lora(index)
        self.adapter_enabled[index] = 1

        num_experts = self.w13_lora_a_stacked[0].shape[1]

        w1_lora_a, w2_lora_a, w3_lora_a = lora_a
        w1_lora_b, w2_lora_b, w3_lora_b = lora_b
        assert (
            num_experts
            == w1_lora_a.shape[0]
            == w2_lora_a.shape[0]
            == w3_lora_a.shape[0]
        )

        slliced_w1_lora_a = self._slice_w13_a(w1_lora_a)
        slliced_w1_lora_b = self._slice_w13_b(w1_lora_b)

        sliced_w2_lora_a = self._slice_w2_a(w2_lora_a)
        sliced_w2_lora_b = self._slice_w2_b(w2_lora_b)

        self.w13_lora_a_stacked[0][
            index, :, : slliced_w1_lora_a.shape[1], : slliced_w1_lora_a.shape[2]
        ].copy_(slliced_w1_lora_a, non_blocking=True)

        self.w13_lora_b_stacked[0][
            index, :, : slliced_w1_lora_b.shape[1], : slliced_w1_lora_b.shape[2]
        ].copy_(slliced_w1_lora_b, non_blocking=True)

        # Only copy w3 (up_proj) for gated MoE (_w13_slices == 2)
        if self._w13_slices == 2:
            slliced_w3_lora_a = self._slice_w13_a(w3_lora_a)
            slliced_w3_lora_b = self._slice_w13_b(w3_lora_b)

            self.w13_lora_a_stacked[1][
                index, :, : slliced_w3_lora_a.shape[1], : slliced_w3_lora_a.shape[2]
            ].copy_(slliced_w3_lora_a, non_blocking=True)

            self.w13_lora_b_stacked[1][
                index, :, : slliced_w3_lora_b.shape[1], : slliced_w3_lora_b.shape[2]
            ].copy_(slliced_w3_lora_b, non_blocking=True)

        self.w2_lora_a_stacked[0][
            index, :, : sliced_w2_lora_a.shape[1], : sliced_w2_lora_a.shape[2]
        ].copy_(sliced_w2_lora_a, non_blocking=True)

        self.w2_lora_b_stacked[0][
            index, :, : sliced_w2_lora_b.shape[1], : sliced_w2_lora_b.shape[2]
        ].copy_(sliced_w2_lora_b, non_blocking=True)

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

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""

        # source_layer is FusedMoE or SharedFusedMoE
        return isinstance(source_layer, FusedMoE) and len(packed_modules_list) == 2


class FusedMoE3DWithLoRA(FusedMoEWithLoRA):
    def __init__(self, base_layer):
        super().__init__(base_layer)
        self._w13_slices = 1

    def _create_lora_b_weights(self, max_loras, lora_config):
        self.w13_lora_b_stacked: tuple[torch.Tensor] = tuple(
            torch.zeros(
                (
                    max_loras,
                    self.base_layer.local_num_experts,
                    self.base_layer.intermediate_size_per_partition * 2,
                    lora_config.max_lora_rank,
                ),
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self._w13_slices)
        )
        self.w2_lora_b_stacked: tuple[torch.Tensor] = (
            torch.zeros(
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
            ),
        )

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""

        assert isinstance(model_config, PretrainedConfig)
        self._base_model = model_config.architectures[0]
        self.max_loras = lora_config.max_loras
        self.fully_sharded = lora_config.fully_sharded_loras

        self.adapter_enabled = torch.tensor(
            [0] * (max_loras + 1), dtype=torch.int, device=self.device
        )

        self._create_lora_a_weights(max_loras, lora_config)
        self._create_lora_b_weights(max_loras, lora_config)

    def _slice_w13_b(self, w13_lora_b: torch.Tensor):
        if self.tp_size == 1:
            return w13_lora_b

        # w13_lora_b shape (num_experts,output_size,rank)
        shard_size = self.base_layer.intermediate_size_per_partition
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        # HACK: Currently, only GPT-OSS is in interleaved order
        if self._base_model == "GptOssForCausalLM":
            # For models like GPT-OSS, the weights of w1 (gate_proj) and w3 (up_proj)
            # in the interleaved order, and corresponding LoRA need to be processed.
            w1_lora_b = w13_lora_b[:, ::2, :]
            w3_lora_b = w13_lora_b[:, 1::2, :]
            sliced_w1_lora_b = w1_lora_b[:, start_idx:end_idx, :]
            sliced_w3_lora_b = w3_lora_b[:, start_idx:end_idx, :]

            return torch.stack([sliced_w1_lora_b, sliced_w3_lora_b], dim=2).flatten(
                1, 2
            )
        else:
            slice_size = w13_lora_b.shape[1] // 2
            w1_lora_b = w13_lora_b[:, :slice_size, :]
            w3_lora_b = w13_lora_b[:, slice_size:, :]
            sliced_w1_lora_b = w1_lora_b[:, start_idx:end_idx, :]
            sliced_w3_lora_b = w3_lora_b[:, start_idx:end_idx, :]

            return torch.cat([sliced_w1_lora_b, sliced_w3_lora_b], dim=1)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        """Overwrites lora tensors at index."""
        # Make mypy happy
        assert isinstance(lora_a, list)
        assert isinstance(lora_b, list)
        assert len(lora_a) == len(lora_b) == 2

        self.reset_lora(index)
        self.adapter_enabled[index] = 1

        w13_lora_a, w2_lora_a = lora_a
        w13_lora_b, w2_lora_b = lora_b

        sliced_w13_lora_a = self._slice_w13_a(w13_lora_a)
        sliced_w13_lora_b = self._slice_w13_b(w13_lora_b)

        sliced_w2_lora_a = self._slice_w2_a(w2_lora_a)
        sliced_w2_lora_b = self._slice_w2_b(w2_lora_b)

        self.w13_lora_a_stacked[0][
            index, :, : sliced_w13_lora_a.shape[1], : sliced_w13_lora_a.shape[2]
        ].copy_(sliced_w13_lora_a, non_blocking=True)
        self.w2_lora_a_stacked[0][
            index, :, : sliced_w2_lora_a.shape[1], : sliced_w2_lora_a.shape[2]
        ].copy_(sliced_w2_lora_a, non_blocking=True)

        self.w13_lora_b_stacked[0][
            index, :, : sliced_w13_lora_b.shape[1], : sliced_w13_lora_b.shape[2]
        ].copy_(sliced_w13_lora_b, non_blocking=True)
        self.w2_lora_b_stacked[0][
            index, :, : sliced_w2_lora_b.shape[1], : sliced_w2_lora_b.shape[2]
        ].copy_(sliced_w2_lora_b, non_blocking=True)

    @property
    def w13_input_size(self):
        """
        Full size
        """
        return self.w13_lora_a_stacked[0].shape[-1]

    @property
    def w13_output_size(self):
        """
        Full size
        """
        return self.w13_lora_b_stacked[0].shape[-2] * self.tp_size

    @property
    def w2_input_size(self):
        """
        Full size
        """
        return self.w2_lora_a_stacked[0].shape[-1] * self.tp_size

    @property
    def w2_output_size(self):
        """
        Full size
        """
        return self.base_layer.hidden_size

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        # source_layer is FusedMoE or SharedFusedMoE
        return isinstance(source_layer, FusedMoE) and len(packed_modules_list) == 1
