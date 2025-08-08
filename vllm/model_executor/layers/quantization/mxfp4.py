# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
                                                  FusedMoEMethodBase)
from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
    triton_kernel_moe_forward)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    _can_support_mxfp4, _swizzle_mxfp4)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import next_power_of_2, round_up

if (envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8
        or envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16):
    # from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer import (mxfp8_quantize, shuffle_matrix_a,
                            shuffle_matrix_sf_a, trtllm_fp4_block_scale_moe)


class Mxfp4Config(QuantizationConfig):

    def __init__(self, ignored_layers: Optional[list[str]] = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            raise NotImplementedError("Mxfp4 linear layer is not implemented")
        elif isinstance(layer, FusedMoE):
            return Mxfp4MoEMethod(layer.moe_config)
        elif isinstance(layer, Attention):
            raise NotImplementedError(
                "Mxfp4 attention layer is not implemented")
        return None


class Mxfp4MoEMethod(FusedMoEMethodBase):

    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.topk_indices_dtype = None
        self.moe = moe

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        # FIXME (zyongye): ship after torch and safetensors support mxfp4
        # is_torch_mxfp4_available = (
        #     hasattr(torch, "float4_e2m1fn_x2") and
        #     hasattr(torch, "float8_e8m0fnu"))
        # if is_torch_mxfp4_available:
        #     weight_dtype = torch.float4_e2m1fn_x2
        #     scale_dtype = torch.float8_e8m0fnu

        mxfp4_block = 32

        intermediate_size_per_partition_after_pad = \
            intermediate_size_per_partition
        # pad the intermediate size to be a multiple of 2 * mxfp4_block
        # for to hold non-uniform sharded tensor as well as swizzling
        # other padding to increase performance
        if (envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8
                or envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16):
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 256)
            hidden_size = round_up(hidden_size, 256)
        elif current_platform.is_rocm():
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 128)
        else:
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 64)

        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        if (envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8
                or envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16):
            layer.gemm1_alpha = Parameter(torch.tensor(
                [1.702] * self.num_experts, dtype=torch.float32).cuda(),
                                          requires_grad=False)
            layer.gemm1_beta = Parameter(torch.tensor(
                [1.0] * self.num_experts, dtype=torch.float32).cuda(),
                                         requires_grad=False)
            layer.gemm1_clamp_limit = Parameter(torch.tensor(
                [7.0] * self.num_experts, dtype=torch.float32).cuda(),
                                                requires_grad=False)
            sf_block_size = 32  # mxfp4 block size

            assert (layer.w13_weight.dim() == 3
                    and layer.w13_weight.shape[0] == self.num_experts
                    and layer.w13_weight.shape[1] == self.intermediate_size * 2
                    and layer.w13_weight.shape[2] == self.hidden_size // 2)
            assert (layer.w13_weight_scale.dim() == 3
                    and layer.w13_weight_scale.shape[0] == self.num_experts
                    and layer.w13_weight_scale.shape[1]
                    == self.intermediate_size * 2
                    and layer.w13_weight_scale.shape[2]
                    == self.hidden_size // sf_block_size)
            assert (layer.w2_weight.dim() == 3
                    and layer.w2_weight.shape[0] == self.num_experts
                    and layer.w2_weight.shape[1] == self.hidden_size and
                    layer.w2_weight.shape[2] == self.intermediate_size // 2)
            assert (layer.w2_weight_scale.dim() == 3
                    and layer.w2_weight_scale.shape[1] == self.hidden_size
                    and layer.w2_weight_scale.shape[2]
                    == self.intermediate_size // sf_block_size)
            assert (layer.w13_bias.dim() == 2
                    and layer.w13_bias.shape[0] == self.num_experts
                    and layer.w13_bias.shape[1] == self.intermediate_size * 2)
            assert (layer.w2_bias.dim() == 2
                    and layer.w2_bias.shape[0] == self.num_experts
                    and layer.w2_bias.shape[1] == self.hidden_size)

            w13_weight_scale = layer.w13_weight_scale.data
            w2_weight_scale = layer.w2_weight_scale.data
            w13_weight = layer.w13_weight.data
            w2_weight = layer.w2_weight.data
            w13_bias = layer.w13_bias.data.to(torch.float32)
            w2_bias = layer.w2_bias.data.to(torch.float32)

            # Swap w1 and w3 as the defenition of
            # swiglu is different in the trtllm-gen
            def swap_every_two_rows(x, axis=-1):
                shape = x.shape
                if axis < 0:
                    axis = len(shape) + axis

                # Create a new shape with pairs swapped along specified axis
                new_shape = list(shape)
                new_shape[axis] = shape[axis] // 2
                new_shape.insert(axis + 1, 2)

                # Reshape to expose pairs, swap them, and reshape back
                x = x.reshape(*new_shape)
                x = x.flip(axis + 1)
                new_shape = list(shape)
                return x.reshape(*new_shape)

            w13_weight_scale = swap_every_two_rows(w13_weight_scale, -2)
            w13_weight = swap_every_two_rows(w13_weight, -2)
            w13_bias = swap_every_two_rows(w13_bias, -1)

            # Do not interleave as the checkpoint is already interleaved

            # Shuffle weights and scaling factors for transposed mma output
            gemm1_weights_mxfp4_shuffled = []
            gemm1_scales_mxfp4_shuffled = []
            gemm2_weights_mxfp4_shuffled = []
            gemm2_scales_mxfp4_shuffled = []
            gemm1_bias_shuffled = []
            gemm2_bias_shuffled = []
            epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
            for i in range(self.num_experts):
                gemm1_weights_mxfp4_shuffled.append(
                    shuffle_matrix_a(w13_weight[i].view(torch.uint8),
                                     epilogue_tile_m))
                gemm1_scales_mxfp4_shuffled.append(
                    shuffle_matrix_sf_a(w13_weight_scale[i].view(torch.uint8),
                                        epilogue_tile_m))
                gemm1_bias_shuffled.append(
                    shuffle_matrix_a(w13_bias[i].clone().reshape(-1, 1),
                                     epilogue_tile_m))

                gemm2_weights_mxfp4_shuffled.append(
                    shuffle_matrix_a(w2_weight[i].view(torch.uint8),
                                     epilogue_tile_m))
                gemm2_scales_mxfp4_shuffled.append(
                    shuffle_matrix_sf_a(w2_weight_scale[i].view(torch.uint8),
                                        epilogue_tile_m))
                gemm2_bias_shuffled.append(
                    shuffle_matrix_a(w2_bias[i].clone().reshape(-1, 1),
                                     epilogue_tile_m))

            w13_weight = torch.stack(gemm1_weights_mxfp4_shuffled)
            w13_weight_scale = torch.stack(
                gemm1_scales_mxfp4_shuffled).reshape(
                    self.num_experts, 2 * self.intermediate_size,
                    self.hidden_size // sf_block_size).view(
                        torch.float8_e4m3fn)

            w2_weight = torch.stack(gemm2_weights_mxfp4_shuffled)
            w2_weight_scale = torch.stack(gemm2_scales_mxfp4_shuffled).reshape(
                self.num_experts, self.hidden_size, self.intermediate_size //
                sf_block_size).view(torch.float8_e4m3fn)

            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = Parameter(w13_weight_scale,
                                               requires_grad=False)
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = Parameter(w2_weight_scale,
                                              requires_grad=False)
            layer.w13_bias = Parameter(
                torch.stack(gemm1_bias_shuffled).reshape(self.num_experts, -1),
                requires_grad=False)
            layer.w2_bias = Parameter(torch.stack(gemm2_bias_shuffled).reshape(
                self.num_experts, -1),
                                      requires_grad=False)
        else:
            from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

            w13_bias = layer.w13_bias.to(torch.float32)
            w2_bias = layer.w2_bias.to(torch.float32)

            layer.w13_bias = Parameter(w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(w2_bias, requires_grad=False)

            # FIXME warp need to be adjusted based on batch size
            # only apply to  batched mode
            if self.moe.use_ep:
                num_warps = 4 if envs.VLLM_MOE_DP_CHUNK_SIZE <= 512 else 8
            else:
                num_warps = 8

            w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
                layer.w13_weight, layer.w13_weight_scale, num_warps)
            w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
                layer.w2_weight, layer.w2_weight_scale, num_warps)

            self.w13_precision_config = PrecisionConfig(
                weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex))
            self.w2_precision_config = PrecisionConfig(
                weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex))

            self.w13_weight_triton_tensor = w13_weight
            self.w2_weight_triton_tensor = w2_weight

            # need to delete the original weights to save memory on single GPU
            del layer.w13_weight
            del layer.w2_weight
            layer.w13_weight = None
            layer.w2_weight = None
            torch.cuda.empty_cache()

    def _get_tile_tokens_dim(self, x: torch.Tensor, top_k: int):
        # Number of tokens in the input tensor.
        num_tokens = x.shape[0]
        # Factor to account for the imbalance of the experts.
        # factor equals to the
        # max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
        # - 1.0 means perfect expert distribution.
        # - > 1.0 means some experts have more
        #     tokens than the perfect distribution.
        # - < 1.0 does not make sense.
        imbalance_factor = 1.3
        # Calculate the number of tokens per expert
        # assuming perfect distribution.
        num_tokens_per_expert = (num_tokens * top_k) // self.num_experts
        # Apply the imbalance factor.
        num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
        # And pad the number to the next power of 2.
        tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
        # Cap to 8-64 tokens per CTA tile
        # as it's the range supported by the kernel.
        tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

        return tile_tokens_dim

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        assert _can_support_mxfp4(
            use_grouped_topk, topk_group, num_expert_group, expert_map,
            custom_routing_function, e_score_correction_bias,
            apply_router_weight_on_input, scoring_func, activation,
            expert_load_view, logical_to_physical_map,
            logical_replica_count), ("MXFP4 are not supported\
                                      with this configuration.")

        if (envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8
                or envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16):
            assert not self.moe.use_ep, (
                "EP is not supported for flashinfer mxfp4 moe backend yet.")
            if envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16:
                assert x.dtype == torch.bfloat16
                x_quant = x
                x_scale = None
            else:
                x_quant, x_scale = mxfp8_quantize(x, False)  # to mxfp8
                x_scale = x_scale.view(torch.float8_e4m3fn).reshape(-1)
            trtllm_gen_output = trtllm_fp4_block_scale_moe(
                router_logits.to(torch.bfloat16),
                None,  # routing_bias
                x_quant,
                x_scale,
                layer.w13_weight,  # uint8 (e2m1 x 2)
                layer.w13_weight_scale,  # uint8 (e4m3 x 2)
                layer.w13_bias,  # fp32 per expert per channel
                layer.gemm1_alpha,  # fp32 per expert
                layer.gemm1_beta,  # fp32 per expert
                layer.gemm1_clamp_limit,  # fp32 per expert
                layer.w2_weight,  # uint8 (e2m1 x 2)
                layer.w2_weight_scale,  # ue8m0
                layer.w2_bias,  # fp32 per expert per channel
                None,  # output1_scale_scalar
                None,  # output1_scale_gate_scalar
                None,  # output2_scale_scalar
                self.num_experts,
                top_k,
                None,  # n_group
                None,  # topk_group
                self.intermediate_size,  # padded to multiple of 256
                0,  # local_expert_offset
                self.num_experts,  # local num experts
                None,
                self._get_tile_tokens_dim(x, top_k),
                1 if renormalize else 0,  # routing_method_type, renormalize
                True,  # do finalize
            )[0]
            return trtllm_gen_output
        else:
            return triton_kernel_moe_forward(
                hidden_states=x,
                w1=self.w13_weight_triton_tensor,
                w2=self.w2_weight_triton_tensor,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_precision=self.w13_precision_config,
                w2_precision=self.w2_precision_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
