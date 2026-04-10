# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
)

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4afp8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    convert_bf16_scales_to_fp8,
    convert_packed_uint4b8_to_signed_int4_inplace,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class CompressedTensorsW4A8Fp8MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        self.group_size = self.weight_quant.group_size
        self.num_bits = self.weight_quant.num_bits
        self.packed_factor = 32 // self.num_bits

        assert self.weight_quant.symmetric, (
            "Only symmetric quantization is supported for W4A8 MoE"
        )
        assert self.weight_quant.actorder != "group"
        assert self.group_size == 128, "Only group size 128 supported for W4A8 MoE"

        self.disable_expert_map = False
        self.layer_name = layer_name

        from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            GroupShape,
        )

        self.quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # requirement for CUTLASS reorder_tensor
        assert hidden_size % 256 == 0, f"{hidden_size=} must be divisible by 256"
        assert intermediate_size_per_partition % 256 == 0, (
            f"{intermediate_size_per_partition=} must be divisible by 256"
        )
        # storage type, pack 8xint4 into int32
        params_dtype = torch.int32

        # WEIGHTS
        w13_weight_packed = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.packed_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight_packed)
        set_weight_attrs(w13_weight_packed, extra_weight_attrs)

        w2_weight_packed = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.packed_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight_packed)
        set_weight_attrs(w2_weight_packed, extra_weight_attrs)

        # SCALES
        # weight_scale refers to the group-wise scales
        # they are initially loaded as bf16, we will convert to fp8
        # after loading
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=layer.orig_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=layer.orig_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-GROUP quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # weight shapes
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

        # don't use input scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer):
        device = layer.w13_weight_packed.device

        # STRIDES
        # A, C
        self.a_strides1_c_strides2 = torch.full(
            (layer.local_num_experts,),
            layer.hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.a_strides2 = torch.full(
            (layer.local_num_experts,),
            layer.intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides1 = torch.full(
            (layer.local_num_experts,),
            2 * layer.intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )

        # S (group-wise scales)
        # sizeof(StrideS) = 16 bytes, so we need to use 2xint64 to encode it
        self.s_strides1 = torch.zeros(
            (layer.local_num_experts, 2), device=device, dtype=torch.int64
        )
        self.s_strides1[:, 0] = 2 * layer.intermediate_size_per_partition

        self.s_strides2 = torch.zeros(
            (layer.local_num_experts, 2), device=device, dtype=torch.int64
        )
        self.s_strides2[:, 0] = layer.hidden_size

        # encode and reorder weight tensors, and get the layout to pass to
        # the grouped gemm kernel. `b_strides1/2` specifies the entire layout
        convert_packed_uint4b8_to_signed_int4_inplace(layer.w13_weight_packed)
        w13_weight_shuffled, self.b_strides1 = (
            ops.cutlass_encode_and_reorder_int4b_grouped(layer.w13_weight_packed)
        )
        replace_parameter(layer, "w13_weight_packed", w13_weight_shuffled)
        convert_packed_uint4b8_to_signed_int4_inplace(layer.w2_weight_packed)
        w2_weight_shuffled, self.b_strides2 = (
            ops.cutlass_encode_and_reorder_int4b_grouped(layer.w2_weight_packed)
        )
        replace_parameter(layer, "w2_weight_packed", w2_weight_shuffled)

        # convert bf16 scales to (fp8_scales, channel_scales)
        w13_weight_scale, w13_weight_chan_scale = convert_bf16_scales_to_fp8(
            self.quant_fp8, layer.w13_weight_scale
        )
        w2_weight_scale, w2_weight_chan_scale = convert_bf16_scales_to_fp8(
            self.quant_fp8, layer.w2_weight_scale
        )

        # register channel scales
        layer.register_parameter(
            "w13_weight_chan_scale",
            torch.nn.Parameter(w13_weight_chan_scale, requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_chan_scale",
            torch.nn.Parameter(w2_weight_chan_scale, requires_grad=False),
        )

        # The scales are stored as (E, N, K // 128) but the kernel expects
        # (E, K // 128, N) in row-major format, so we need to permute the last 2 dims
        # and make it contiguous
        w13_weight_scale_packed = ops.cutlass_pack_scale_fp8(
            w13_weight_scale.permute(0, 2, 1).contiguous()
        )
        replace_parameter(layer, "w13_weight_scale", w13_weight_scale_packed)
        w2_weight_scale_packed = ops.cutlass_pack_scale_fp8(
            w2_weight_scale.permute(0, 2, 1).contiguous()
        )
        replace_parameter(layer, "w2_weight_scale", w2_weight_scale_packed)

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        return super().maybe_make_prepare_finalize(routing_tables)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # Store quantization scales; both per-group and per-channel
        # Note we haven't specified the group size here because
        # the quant config logic assumes group-wise scaling
        # and channel-wise scaling are exclusive.
        return int4_w4afp8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,  # group scale
            w2_scale=layer.w2_weight_scale,  # group scale
            g1_alphas=layer.w13_weight_chan_scale,
            g2_alphas=layer.w2_weight_chan_scale,
            per_act_token_quant=True,  # always use dynamic per-token
            per_out_ch_quant=True,  # always use per-channel
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular:
        assert self.moe_quant_config is not None
        assert (
            prepare_finalize.activation_format == FusedMoEActivationFormat.Standard
        ), "BatchedExperts not supported"

        from vllm.model_executor.layers.fused_moe import CutlassExpertsW4A8Fp8

        experts: FusedMoEExpertsModular

        logger.debug("CutlassExpertsW4A8Fp8(%s)", self.__class__.__name__)
        experts = CutlassExpertsW4A8Fp8(
            out_dtype=self.moe.in_dtype,
            a_strides1=self.a_strides1_c_strides2,
            a_strides2=self.a_strides2,
            b_strides1=self.b_strides1,
            b_strides2=self.b_strides2,
            c_strides1=self.c_strides1,
            c_strides2=self.a_strides1_c_strides2,
            s_strides1=self.s_strides1,
            s_strides2=self.s_strides2,
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            group_size=self.group_size,
        )

        num_dispatchers = prepare_finalize.num_dispatchers()
        self.disable_expert_map = (
            num_dispatchers > 1 or not experts.supports_expert_map()
        )

        return experts

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if layer.enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `CompressedTensorsW4A8Fp8MoEMethod` yet."
            )
        assert self.moe_quant_config is not None

        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            cutlass_moe_w4a8_fp8,
        )

        return cutlass_moe_w4a8_fp8(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights,
            topk_ids,
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=None if self.disable_expert_map else layer.expert_map,
            a_strides1=self.a_strides1_c_strides2,
            a_strides2=self.a_strides2,
            b_strides1=self.b_strides1,
            b_strides2=self.b_strides2,
            c_strides1=self.c_strides1,
            c_strides2=self.a_strides1_c_strides2,
            s_strides1=self.s_strides1,
            s_strides2=self.s_strides2,
            group_size=self.group_size,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )

    @property
    def supports_eplb(self) -> bool:
        return False
