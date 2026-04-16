# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
)

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.utils import set_weight_attrs
from aiter.ops.shuffle import shuffle_weight

logger = init_logger(__name__)

def _pack_shuffled_int8_to_packed_int4_no_perm(x_shuf_i8: torch.Tensor) -> torch.Tensor:
    """Pack a preshuffled int8 tensor (values in [-8, 7]) into packed int4 bytes.

    Each contiguous 8-value block [v0..v7] -> 4 bytes:
      b0=(v4<<4)|v0, b1=(v5<<4)|v1, b2=(v6<<4)|v2, b3=(v7<<4)|v3.

    This matches the 7-op in-kernel unpack sequence and avoids any v_perm.
    """
    flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    out[:, 0] = u[:, 0] | (u[:, 4] << 4)
    out[:, 1] = u[:, 1] | (u[:, 5] << 4)
    out[:, 2] = u[:, 2] | (u[:, 6] << 4)
    out[:, 3] = u[:, 3] | (u[:, 7] << 4)
    return out.view(-1).to(torch.int8)

def _unpack_gptq_int32_to_signed_int4(w_int32):
    """Unpack GPTQ int32 [E, K//8, N] to signed int4 values [E, N, K] (as int8).

    Shared by both the packed-int4 and bf16-dequant paths.
    """
    E = w_int32.shape[0]
    # [E, K//8, N] -> transpose -> [E, N, K//8]
    w = w_int32.transpose(1, 2).contiguous()
    N = w.shape[1]
    K_div8 = w.shape[2]
    K = K_div8 * 8

    # Unpack int32 -> 8 x uint4 values along K
    w_expanded = w.unsqueeze(-1).expand(E, N, K_div8, 8)  # [E, N, K//8, 8]
    shifts = torch.arange(8, device=w.device) * 4  # [0, 4, 8, ..., 28]
    nibbles = ((w_expanded >> shifts) & 0xF).to(torch.int8)  # [E, N, K//8, 8]
    nibbles = nibbles.reshape(E, N, K)  # [E, N, K] unsigned int4 as int8

    # Convert unsigned [0,15] to signed [-8,7]
    signed = nibbles.to(torch.int16) - 8
    signed = signed.to(torch.int8)  # [E, N, K] signed int4 as int8
    return signed

def _gptq_int32_to_flydsl_packed(w_int32):
    """Convert GPTQ int32 [E, K//8, N] to FlyDSL shuffled packed int4 [E, N, K//2].

    Steps:
    1. Unpack int32 to individual signed int4 values (as int8)
    2. Apply FlyDSL preshuffle (on individual int8 values)
    3. Pack with FlyDSL's interleaved int4 packing
    """
    signed = _unpack_gptq_int32_to_signed_int4(w_int32)
    E, N, K = signed.shape

    # FlyDSL preshuffle (operates on individual values)
    shuffled = shuffle_weight(signed, layout=(16, 16))

    # FlyDSL interleaved int4 packing
    packed = _pack_shuffled_int8_to_packed_int4_no_perm(shuffled).contiguous()
    return packed.view(E, N, K // 2)

class CompressedTensorsW4A16FlydslMoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        # Extract properties from weight_quant
        assert weight_quant.num_bits == 4
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        # channelwise is not supported by this kernel
        assert weight_quant.strategy == "group"
        assert weight_quant.group_size in (-1, 32)
        self.group_size = weight_quant.group_size
        # grouped actorder isn't supported by this kernel
        assert weight_quant.actorder != "group"
        assert weight_quant.symmetric, (
            "Only symmetric quantization is supported for MoE"
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        self.num_experts = num_experts
        self.inter_dim = intermediate_size_per_partition
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                w13_num_shards * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_scales_size = intermediate_size_per_partition

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": False})

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reconfigure packed weights and scales to match flydsl_w4a16 format

        # Convert w13 weights
        w13 = layer.w13_weight_packed.data
        w13 = _gptq_int32_to_flydsl_packed(w13)
        w13 = w13.view(-1).contiguous()
        layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)

        # Convert w2 weights
        w2 = layer.w2_weight_packed.data
        w2 = _gptq_int32_to_flydsl_packed(w2)
        w2 = w2.view(-1).contiguous()
        layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)

        # Convert scales for FlyDSL:
        #   per-row:   [E, 1, N] -> squeeze -> [E, N]
        #   groupwise: [E, K//gs, N] -> keep as-is (Opt 0: cache-friendly layout)
        w13_scale = layer.w13_weight_scale.data
        if self.group_size > 0 and w13_scale.dim() == 3 and w13_scale.shape[1] > 1:
            # Groupwise: keep [E, K//gs, N] layout (Opt 0: stride-1 access for adjacent threads)
            E, G, N = w13_scale.shape
            w13_scale = w13_scale.view(E, G // 2, 2, N).permute(0, 1, 3, 2).contiguous().view(-1).contiguous()
        elif w13_scale.dim() == 3 and w13_scale.shape[1] == 1:
            # Per-row: squeeze [E, 1, N] -> [E, N]
            w13_scale = w13_scale.squeeze(1)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale.contiguous(), requires_grad=False)

        w2_scale = layer.w2_weight_scale.data
        if self.group_size > 0 and w2_scale.dim() == 3 and w2_scale.shape[1] > 1:
            # Groupwise: keep [E, K//gs, N] layout (Opt 0: stride-1 access for adjacent threads)
            # w2_scale = w2_scale.contiguous()
            E, G, N = w2_scale.shape
            w2_scale = w2_scale.view(E, G // 2, 2, N).permute(0, 1, 3, 2).contiguous().view(-1).contiguous()
        elif w2_scale.dim() == 3 and w2_scale.shape[1] == 1:
            # Per-row: squeeze [E, 1, N] -> [E, N]
            w2_scale = w2_scale.squeeze(1)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale.contiguous(), requires_grad=False)

        layer.w13_weight_packed.is_shuffled = True
        layer.w2_weight_packed.is_shuffled = True
        layer.is_aiter_converted = True

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        assert self.num_bits == 4
        return int4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular:
        if self.moe.is_lora_enabled:
            assert self.moe_quant_config is not None
            from vllm.triton_utils import HAS_TRITON

            if HAS_TRITON:
                from vllm.model_executor.layers.fused_moe import TritonWNA16Experts

                layer.w13_weight = layer.w13_weight_packed
                layer.w2_weight = layer.w2_weight_packed
                return TritonWNA16Experts(
                    moe_config=self.moe, quant_config=self.moe_quant_config
                )
            else:
                raise NotImplementedError(
                    "TritonExperts requires Triton. "
                    "Install triton or disable LoRA for MoE."
                )

        raise NotImplementedError

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.fused_flydsl_moe import fused_flydsl_moe
        return fused_flydsl_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            self.num_experts,
            self.inter_dim,
            topk_weights,
            topk_ids,
            w1_scale=self.moe_quant_config.w1_scale,
            w2_scale=self.moe_quant_config.w2_scale,
            topk=topk_weights.shape[-1],
            group_size=self.group_size,
            doweight_stage1=layer.apply_router_weight_on_input,
            scale_is_bf16=(self.moe_quant_config.w1_scale.dtype == torch.bfloat16)
        )