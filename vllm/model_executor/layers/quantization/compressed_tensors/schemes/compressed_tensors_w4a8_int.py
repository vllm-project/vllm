# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import CpuArchEnum, current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A8Int", "CompressedTensorsW4A8Int8MoEMethod"]
W4A8_SUPPORTED_TYPES_MAP = {
    4: scalar_types.int4,
}
W4A8_SUPPORTED_BITS = list(W4A8_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsW4A8Int(CompressedTensorsScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        is_static_input_scheme: bool = False,
        input_symmetric: bool = True,
    ):
        self.strategy = strategy
        self.group_size = -1 if group_size is None else group_size
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

        if num_bits not in W4A8_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}."
                f"Supported num_bits = {W4A8_SUPPORTED_TYPES_MAP.keys()}"
            )
        self.quant_type = W4A8_SUPPORTED_TYPES_MAP[num_bits]

    @classmethod
    def get_min_capability(cls) -> int:
        return 1

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        row_parallel = input_size != input_size_per_partition

        # Compute effective group_size
        if self.group_size == -1:
            effective_group_size = (
                input_size_per_partition if row_parallel else input_size
            )
        else:
            effective_group_size = self.group_size

        # Ensure group_size divides input_size_per_partition
        assert input_size_per_partition % effective_group_size == 0, (
            f"input_size_per_partition {input_size_per_partition}"
            f" not divisible by group_size {effective_group_size}"
        )

        # Determine scale partitioning
        is_channelwise = self.group_size == -1
        repeat_scales = is_channelwise and row_parallel
        partition_scales = not repeat_scales

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=effective_group_size,
            zero_points=False,
            has_g_idx=False,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for CompressedTensorsW4A8Int", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        scales_and_zp_size = input_size_per_partition // effective_group_size

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition, scales_and_zp_size, dtype=params_dtype
            ),
        }

        if partition_scales:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )
        else:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name=None,
            w_gidx_param_name=None,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class CompressedTensorsW4A8Int8MoEMethod(FusedMoEMethodBase):
    """
    CPU-only MoE method using dynamic 4-bit matmul kernels on Arm Platform
    - Weights: int4 (stored as int8 values in [-8,7], packed to uint8 nibbles)
    - Scales: Fp32 for Channelwise , bf16 for groupwise quantization
    - Bias: Same data type as original weights
    - Activations: FP32/Bf16 dynamic per-token (A8 Int),
      quantized inside the kernel
    """

    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias
        self.quant_config = quant_config

        # Validate scheme: weights=W4 (channel or group),
        # activations=dynamic TOKEN (A8)
        wq = self.quant_config.target_scheme_map["Linear"].get("weights")
        aq = self.quant_config.target_scheme_map["Linear"].get("input_activations")

        # Must be dynamic per-token activations
        if aq.strategy != QuantizationStrategy.TOKEN or not aq.dynamic:
            raise ValueError(
                "W4A8-int MoE needs dynamic per-token activation quantization."
            )

        # Weight can be channel-wise (group_size=None) or group-wise
        self.group_size = wq.group_size if (wq.group_size is not None) else -1
        if wq.num_bits != 4:
            raise ValueError("This method only supports 4-bit weights (num_bits=4).")

        # CPU only
        if not current_platform.is_cpu():
            raise ValueError("CompressedTensorsW4A8Int8MoEMethod is CPU-only.")

        # Arm: check _dyn ops availability
        if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
            try:
                _ = torch.ops.aten._dyn_quant_matmul_4bit
                _ = torch.ops.aten._dyn_quant_pack_4bit_weight
            except AttributeError as err:
                raise RuntimeError(
                    f"""PyTorch {torch.__version__} lacks _dyn_quant_* 4bit ops;
                    install a newer build."""
                ) from err
        self.static_input_scales = False  # always dynamic per token

    # ---- parameter creation ----
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Shapes per local rank (TP/EP):
        #   w13: [E, 2*I_local, H]  int8  (int4 values in [-8,7])
        #   w2 : [E, H, I_local]    int8
        # Scales:
        #   channel-wise: group_size=-1 -> per-output-row, single scale per row
        #   group-wise  : group_size=g   ->
        #   per-output-row, (in_features/g) scales

        E = num_experts
        H = hidden_size
        IN = intermediate_size_per_partition
        g = self.group_size

        # Per-row scale columns
        def _n_scale_cols(in_features: int) -> int:
            return 1 if g == -1 else (in_features // g)

        # Register unpacked int4-as-int8 weights the loader will fill.
        w13 = torch.nn.Parameter(
            torch.empty(E, 2 * IN, H, dtype=torch.int8), requires_grad=False
        )
        set_weight_attrs(w13, extra_weight_attrs)
        layer.register_parameter("w13_weight", w13)

        w2 = torch.nn.Parameter(
            torch.empty(E, H, IN, dtype=torch.int8), requires_grad=False
        )
        set_weight_attrs(w2, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2)

        # Register scales
        # KleidiAI groupwise kernels accepts float32 scales
        # KleidiAI groupwise kernels accepts bfloat16 scales
        scale_dtype = torch.float32 if g == -1 else torch.bfloat16

        w13_s = torch.nn.Parameter(
            torch.ones(E, 2 * IN, _n_scale_cols(H), dtype=scale_dtype),
            requires_grad=False,
        )
        set_weight_attrs(
            w13_s,
            {"quant_method": "channel" if g == -1 else "group", **extra_weight_attrs},
        )
        layer.register_parameter("w13_weight_scale", w13_s)

        w2_s = torch.nn.Parameter(
            torch.ones(E, H, _n_scale_cols(IN), dtype=scale_dtype), requires_grad=False
        )
        set_weight_attrs(
            w2_s,
            {"quant_method": "channel" if g == -1 else "group", **extra_weight_attrs},
        )
        layer.register_parameter("w2_weight_scale", w2_s)

        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(E, 2 * IN, dtype=params_dtype), requires_grad=False
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        # Placeholders for packed weights (will be replaced after packing)
        layer.register_parameter(
            "w13_weight_packed", torch.nn.Parameter(torch.empty(0), requires_grad=False)
        )
        set_weight_attrs(layer.w13_weight_packed, extra_weight_attrs)

        layer.register_parameter(
            "w2_weight_packed", torch.nn.Parameter(torch.empty(0), requires_grad=False)
        )
        set_weight_attrs(layer.w2_weight_packed, extra_weight_attrs)

        # dims for 4 bit fused matmuls
        layer.w13_in_features = H
        layer.w13_out_features = 2 * IN
        layer.w2_in_features = IN
        layer.w2_out_features = H
        layer.group_size = g

    # post-load packing to dyn-4bit KleidiAI kernel's format
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        E = layer.w13_weight.shape[0]
        H = layer.w13_in_features
        I2 = layer.w13_out_features
        IN = layer.w2_in_features
        g = layer.group_size

        def _pack_matrix(
            int4_as_int8_2d: torch.Tensor,
            scales_2d: torch.Tensor,
            bias_1d: torch.Tensor | None,
            in_features: int,
            out_features: int,
        ) -> torch.Tensor:
            # int4 values are stored as int8 in [-8,7].
            # Shift to unsigned nibble and pack pairs along input-dim.
            tmp = int4_as_int8_2d.add(8)  # [out, in]
            uint8_nibbles = ((tmp[:, 1::2] << 4) | tmp[:, ::2]).to(
                torch.uint8
            )  # [out, in//2]

            # KleidiAI groupwise kernels accepts float32 scales
            # KleidiAI groupwise kernels accepts bfloat16 scales
            scale_dtype = torch.float32 if g == -1 else torch.bfloat16
            scales = scales_2d.to(scale_dtype)
            bias = None if bias_1d is None else bias_1d.to(torch.float32)
            return torch.ops.aten._dyn_quant_pack_4bit_weight(
                uint8_nibbles,
                scales,
                bias,
                g if g != -1 else in_features,
                in_features,
                out_features,
            )

        # Pack per expert
        w13_packed_list = []
        w2_packed_list = []

        has_w13_bias = hasattr(layer, "w13_bias") and layer.w13_bias is not None
        has_w2_bias = hasattr(layer, "w2_bias") and layer.w2_bias is not None

        for e in range(E):
            w13_packed_list.append(
                _pack_matrix(
                    layer.w13_weight[e],  # [2I, H]
                    layer.w13_weight_scale[e],  # [2I, H/g or 1]
                    layer.w13_bias[e] if has_w13_bias else None,  # [2I]
                    H,
                    I2,
                )
            )
            w2_packed_list.append(
                _pack_matrix(
                    # w2 shape is [H, IN]; we need [out, in] == [H, IN].
                    layer.w2_weight[e],  # [H, IN]
                    layer.w2_weight_scale[e],  # [H, IN/g or 1]
                    layer.w2_bias[e] if has_w2_bias else None,  # [H]
                    IN,
                    layer.w2_out_features,  # in_features=IN, out_features=H
                )
            )

        # each packed tensor has identical shape per expert; stack on dim 0
        w13_packed = torch.stack(w13_packed_list, dim=0)
        w2_packed = torch.stack(w2_packed_list, dim=0)

        replace_parameter(
            layer,
            "w13_weight_packed",
            torch.nn.Parameter(w13_packed, requires_grad=False),
        )
        replace_parameter(
            layer,
            "w2_weight_packed",
            torch.nn.Parameter(w2_packed, requires_grad=False),
        )

        # free raw tensors/scales/bias now that they're packed into the payload.
        replace_parameter(
            layer, "w13_weight", torch.nn.Parameter(torch.empty(0), requires_grad=False)
        )
        replace_parameter(
            layer, "w2_weight", torch.nn.Parameter(torch.empty(0), requires_grad=False)
        )
        replace_parameter(
            layer,
            "w13_weight_scale",
            torch.nn.Parameter(torch.empty(0), requires_grad=False),
        )
        replace_parameter(
            layer,
            "w2_weight_scale",
            torch.nn.Parameter(torch.empty(0), requires_grad=False),
        )
        if has_w13_bias:
            replace_parameter(
                layer,
                "w13_bias",
                torch.nn.Parameter(torch.empty(0), requires_grad=False),
            )
        if has_w2_bias:
            replace_parameter(
                layer,
                "w2_bias",
                torch.nn.Parameter(torch.empty(0), requires_grad=False),
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # CPU dynamic 4-bit MoE path does not use modular kernels or
        # fused_experts; quant config is not needed.
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert not enable_eplb, "EPLB not supported for W4A8-int MoE yet."
        assert activation in ("silu", "swigluoai", "swiglu"), (
            "Only SiLU/SwiGLUGU/SwiGLUUG are supported."
        )
        assert expert_map is None, """expert_map/EP not implemented
        for CPU dyn-4bit MoE."""

        def _act_kind(s: str) -> int:
            # 0 = SwiGLU_Gu (SiLU(g)*u), 1 = SwiGLU_Ug (SiLU(u)*g), 2 = SiLU
            if s == "swiglu":
                return 0
            if s == "swigluoai":
                return 1
            if s == "silu":
                return 2
            raise ValueError(f"Unknown activation '{s}'")

        # Apply topk softmax on router output
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        return torch.ops._C.dynamic_4bit_int_moe(
            x,
            topk_ids.to(torch.long),
            topk_weights,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w2_out_features,
            layer.w2_in_features,
            layer.w13_out_features,
            layer.group_size,
            apply_router_weight_on_input,
            int(_act_kind(activation)),
        )
