# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from enum import Enum

import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
)

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoeWeightScaleSupported,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
    int4_w4afp8_moe_quant_config,
    int8_w8a8_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    BatchedMarlinExperts,
    MarlinExperts,
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    is_global_sf_supported_for_nvfp4_backend,
    make_mxfp4_moe_quant_config,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS,
    WNA16_SUPPORTED_TYPES_MAP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    flashinfer_trtllm_fp4_moe,
    flashinfer_trtllm_fp4_routed_moe,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (
    flashinfer_trtllm_mxint4_moe,
    is_flashinfer_mxint4_moe_available,
    prepare_static_weights_for_trtllm_mxint4_moe,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    apply_fi_trtllm_fp8_per_tensor_moe,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_tensor_strategy_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    convert_bf16_scales_to_fp8,
    convert_packed_uint4b8_to_signed_int4_inplace,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import CpuArchEnum, current_platform

logger = init_logger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


__all__ = [
    "CompressedTensorsMoEMethod",
    "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsW8A8Int8MoEMethod",
    "CompressedTensorsWNA16MarlinMoEMethod",
    "CompressedTensorsWNA16MoEMethod",
    "CompressedTensorsW4A4Nvfp4MoEMethod",
    "CompressedTensorsW4A8Int8MoEMethod",
]


class CompressedTensorsMoEMethod(FusedMoEMethodBase):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> FusedMoEMethodBase:
        # FusedMoE was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        # multiple schemes found
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )

        if scheme_dict is None:  # ignored layer
            return UnquantizedFusedMoEMethod(layer.moe_config)

        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")
        format = scheme_dict.get("format")

        if quant_config._is_mxfp4(weight_quant):
            return CompressedTensorsW4A4Mxfp4MoEMethod(layer.moe_config)

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            # group_size=None means channelwise
            group_size = weight_quant.group_size or -1

            valid_format_and_bits = (
                weight_quant.num_bits in WNA16_SUPPORTED_BITS
                and format == CompressionFormat.pack_quantized.value
            )

            if not valid_format_and_bits:
                raise ValueError(
                    "For Fused MoE layers, only format: ",
                    f"{CompressionFormat.pack_quantized.value} ",
                    f" and bits: {WNA16_SUPPORTED_BITS} is supported ",
                    f"but got format: {CompressionFormat.pack_quantized.value} "
                    f" and bits: {weight_quant.num_bits}",
                )

            # Prefer to use the MarlinMoE kernel when it is supported.
            if (
                not check_moe_marlin_supports_layer(layer, group_size)
                or current_platform.is_rocm()
            ):
                if (
                    weight_quant.strategy == QuantizationStrategy.GROUP
                    and weight_quant.actorder
                    in (ActivationOrdering.GROUP, ActivationOrdering.DYNAMIC)
                ):
                    raise ValueError(
                        "WNA16MoE is not supported with actorder=group/dynamic."
                    )
                logger.info_once("Using CompressedTensorsWNA16MoEMethod")
                return CompressedTensorsWNA16MoEMethod(
                    weight_quant, input_quant, layer.moe_config
                )
            else:
                logger.info_once("Using CompressedTensorsWNA16MarlinMoEMethod")
                return CompressedTensorsWNA16MarlinMoEMethod(
                    weight_quant, input_quant, layer.moe_config
                )
        elif quant_config._is_nvfp4_format(weight_quant):
            _is_valid_nvfp4_activations = (
                quant_config._is_nvfp4_format(input_quant) or input_quant is None
            )
            if not _is_valid_nvfp4_activations:
                raise ValueError(
                    "For NVFP4 weights, input quantization must also be NVFP4 format ",
                    f"or None for NVFP4A16, found {input_quant}",
                )
            return CompressedTensorsW4A4Nvfp4MoEMethod(
                layer.moe_config, layer_name, use_a16=(input_quant is None)
            )
        elif (
            quant_config._is_fp8_w8a8_sm90(weight_quant, input_quant)
            or quant_config._is_fp8_w8a8_sm100(weight_quant, input_quant)
            or quant_config._is_fp8_w8a8(weight_quant, input_quant)
        ):
            return CompressedTensorsW8A8Fp8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_fp8_w4a8_sm90(weight_quant, input_quant):
            logger.info_once("Using CompressedTensorsW4A8Fp8MoEMethod")
            return CompressedTensorsW4A8Fp8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
            return CompressedTensorsW4A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )


class CompressedTensorsW4A4Mxfp4MoEMethod(CompressedTensorsMoEMethod):
    def __init__(self, moe):
        super().__init__(moe)
        self.group_size = 32
        self.mxfp4_backend = NvFp4MoeBackend.MARLIN
        self.experts_cls = MarlinExperts

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
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return make_mxfp4_moe_quant_config(
            w13_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale
        )

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        prepare_moe_fp4_layer_for_marlin(layer)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None:
            self.moe_mk = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                shared_experts=layer.shared_experts,
                routing_tables=layer._maybe_init_expert_routing_tables(),
            )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.moe_mk is not None
        return self.moe_mk(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            inplace=False,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )


class CompressedTensorsW4A4Nvfp4MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
        use_a16: bool = False,
    ):
        super().__init__(moe)
        self.group_size = 16

        # Select experts implementation.
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=None if use_a16 else kNvfp4Dynamic,
        )

        self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
            self.nvfp4_backend
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
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales
        w13_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        """
        Convert NVFP4 MoE weights into kernel format and setup the kernel.
        """
        # NOTE(rob): wN_weight_packed -> wN_weight is because ModularKernelMethod
        # requires this naming convention. However, the name change breaks
        # reloading because the state dict no longer matches disk. Once we
        # remove MKM, we should revert this change to ensure compatibility.
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected.",
            )
        w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()

        # Shuffle weights into the NvFp4 kernel format.
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=(1.0 / w13_weight_global_scale),
            a13_scale=(1.0 / layer.w13_input_global_scale),
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=(1.0 / layer.w2_weight_global_scale),
            a2_scale=(1.0 / layer.w2_input_global_scale),
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        layer.w13_weight_scale_2 = w13_scale_2
        layer.w2_weight_scale_2 = w2_scale_2
        layer.w13_input_scale = a13_scale
        layer.w2_input_scale = a2_scale

        # Setup modular kernel for TP case and naive DP/EP case.
        # In non-naive DP/EP case, we will create a ModularKernelMethod.
        # TODO(rob): unify these so FP8MoEMethod owns the ModularKernel
        # in both cases.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_mk = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                shared_experts=layer.shared_experts,
                routing_tables=layer._maybe_init_expert_routing_tables(),
            )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalize | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    @property
    def is_monolithic(self) -> bool:
        return (
            self.nvfp4_backend == NvFp4MoeBackend.FLASHINFER_TRTLLM
            and not self.moe.moe_parallel_config.enable_eplb
        )

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.is_monolithic
        assert layer.activation == MoEActivation.SILU, (
            f"Only SiLU activation is supported, not {layer.activation}."
        )
        assert (
            self.nvfp4_backend == NvFp4MoeBackend.FLASHINFER_TRTLLM
            and not layer.enable_eplb
        )
        return flashinfer_trtllm_fp4_moe(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=layer.top_k,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            custom_routing_function=layer.custom_routing_function,
            e_score_correction_bias=layer.e_score_correction_bias,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert not self.is_monolithic
        assert layer.activation == MoEActivation.SILU, (
            f"Only SiLU activation is supported, not {layer.activation}."
        )

        # EPLB path
        if self.nvfp4_backend == NvFp4MoeBackend.FLASHINFER_TRTLLM:
            assert layer.enable_eplb
            return flashinfer_trtllm_fp4_routed_moe(
                layer=layer,
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                top_k=layer.top_k,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
            )
        else:
            assert self.moe_mk is not None
            return self.moe_mk(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                inplace=False,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
            )


class CompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsMoEMethod):
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

        per_tensor = (
            self.weight_quant.strategy == QuantizationStrategy.TENSOR
            and self.input_quant.strategy == QuantizationStrategy.TENSOR
        )
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not (per_tensor or per_channel):
            assert self.weight_quant.strategy == QuantizationStrategy.BLOCK
            self.weight_block_size = self.weight_quant.block_structure
            assert self.weight_quant.dynamic is not None
        else:
            self.weight_block_size = None
        self.block_quant = self.weight_block_size is not None

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization."
            )

        ct2vllm_weight = {
            QuantizationStrategy.CHANNEL: kFp8StaticChannelSym,
            QuantizationStrategy.TENSOR: kFp8StaticTensorSym,
            QuantizationStrategy.BLOCK: kFp8Static128BlockSym,
        }
        ct2vllm_act = {
            QuantizationStrategy.TOKEN: kFp8DynamicTokenSym,
            QuantizationStrategy.TENSOR: (
                kFp8StaticTensorSym if self.static_input_scales else kFp8Dynamic128Sym
            ),
        }
        weight_key = ct2vllm_weight[self.weight_quant.strategy]
        if weight_key == kFp8Static128BlockSym:
            activation_key = kFp8Dynamic128Sym
        else:
            activation_key = ct2vllm_act[self.input_quant.strategy]

        # Select Fp8 MoE backend
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=True,
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
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        params_dtype = torch.float8_e4m3fn
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # For gated MoE, allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            # For non-gated MoE, allocate 1 scale for w13.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, w13_num_shards, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSOR quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.weight_quant.strategy == QuantizationStrategy.BLOCK:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    w13_num_shards
                    * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        # Allow for accessing weights and scales in standard way.
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_input_scale = layer.w13_input_scale
        w2_input_scale = layer.w2_input_scale

        # MI300x and MI325x use FNUZ format for FP8. Convert if needed.
        if current_platform.is_fp8_fnuz():
            w13, w13_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w13, w13_scale, w13_input_scale
            )
            w2, w2_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w2, w2_scale, w2_input_scale
            )

        # Per tensor kernels require single activation scale. Use the max.
        if self.static_input_scales:
            assert self.input_quant.strategy == QuantizationStrategy.TENSOR
            assert w13_input_scale is not None and w2_input_scale is not None
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                w13_input_scale, w2_input_scale
            )
            replace_parameter(layer, "w13_input_scale", w13_input_scale)
            replace_parameter(layer, "w2_input_scale", w2_input_scale)

        # Per-tensor kernels use a single scale, for W13, but on disk there
        # is a separate scale for W1 and W3. Requantize with the max scale.
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13,
                w13_scale,
                shard_size=layer.intermediate_size_per_partition,
                num_experts=layer.num_local_experts,
                is_act_and_mul=self.moe.is_act_and_mul,
            )

        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )

        # Replace parameters with updated versions. Note that this helper
        # function ensures the replacement is compatible with RL weight reloads.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)

        # Setup modular kernel for TP case and naive DP/EP case.
        # In non-naive DP/EP case, we will create a ModularKernelMethod.
        # TODO(rob): unify these so FP8MoEMethod owns the ModularKernel
        # in both cases.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_mk, self.use_inplace = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalize | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        w1_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        a1_scale = layer.w13_input_scale
        a2_scale = layer.w2_input_scale

        return make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            per_act_token_quant=(
                self.input_quant.strategy == QuantizationStrategy.TOKEN
            ),
            per_out_ch_quant=(self.input_quant.strategy == QuantizationStrategy.TOKEN),
            block_shape=self.weight_block_size,
        )

    @property
    def is_monolithic(self) -> bool:
        return self.fp8_backend == Fp8MoeBackend.FLASHINFER_TRTLLM

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.is_monolithic
        assert self.fp8_backend == Fp8MoeBackend.FLASHINFER_TRTLLM
        assert layer.activation == MoEActivation.SILU, (
            f"Only SiLU activation is supported, not {layer.activation}."
        )

        if self.block_quant:
            import vllm.model_executor.layers.fused_moe.flashinfer_trtllm_moe  # noqa: E501, F401

            return torch.ops.vllm.flashinfer_fused_moe_blockscale_fp8(
                routing_logits=router_logits,
                routing_bias=layer.e_score_correction_bias,
                x=x,
                w13_weight=layer.w13_weight,
                w13_weight_scale_inv=layer.w13_weight_scale,
                w2_weight=layer.w2_weight,
                w2_weight_scale_inv=layer.w2_weight_scale,
                global_num_experts=layer.global_num_experts,
                top_k=layer.top_k,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                intermediate_size=layer.intermediate_size_per_partition,
                expert_offset=layer.ep_rank * layer.local_num_experts,
                local_num_experts=layer.local_num_experts,
                block_shape=self.weight_block_size,
                routing_method_type=layer.routing_method_type,
                routed_scaling=layer.routed_scaling_factor,
            )
        else:
            return apply_fi_trtllm_fp8_per_tensor_moe(
                layer=layer,
                hidden_states=x,
                router_logits=router_logits,
                routing_bias=layer.e_score_correction_bias,
                global_num_experts=layer.global_num_experts,
                top_k=layer.top_k,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
            )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert not self.is_monolithic
        assert self.moe_mk is not None
        return self.moe_mk(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            inplace=self.use_inplace,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            # TODO(rob): investigate the disable_expert_map introduced by:
            # https://github.com/vllm-project/vllm/commit/84166fee9770e6fba71a96978b3e7d149392fb28 # noqa: E501
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )

    @property
    def supports_eplb(self) -> bool:
        return True


class CompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):
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

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
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
        params_dtype = torch.int8
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe import fused_experts

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )


class CompressedTensorsWNA16MarlinMoEMethod(CompressedTensorsMoEMethod):
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
        assert weight_quant.symmetric, (
            "Only symmetric quantization is supported for MoE"
        )
        # Extract properties from weight_quant
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size
        self.actorder = weight_quant.actorder

        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[self.num_bits]

        self.marlin_input_dtype = get_marlin_input_dtype(layer_name)
        self.use_flashinfer_mxint4_moe = (
            is_flashinfer_mxint4_moe_available()
            and self.group_size == 32
            and weight_quant.num_bits == 4
        )
        self.kernel_backend = (
            "Flashinfer" if self.use_flashinfer_mxint4_moe else "Marlin"
        )
        logger.info_once(
            f"Using {self.kernel_backend} backend for WNA16 MoE "
            f"(group_size={self.group_size}, num_bits={self.num_bits})",
            scope="local",
        )

    def get_weight_shape(
        self,
        weight_name: str,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        num_groups_w2: int | None = None,
        num_groups_w13: int | None = None,
    ) -> tuple[int, int, int]:
        """
        Get the shape of the weight based on the weight name, number of experts
        hidden size, intermediate size per partition, number of groups for w2,
        and number of groups for w13. Pass in num_groups_w2 and num_groups_w13
        for weight scales.
        """
        if weight_name == "w13_scale":
            assert num_groups_w13 is not None, (
                "num_groups_w13 must be provided for weight scales"
            )
        if weight_name == "w2_scale":
            assert num_groups_w2 is not None, (
                "num_groups_w2 must be provided for weight scales"
            )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        shape_map = {
            "w13_weight": {
                "Flashinfer": (
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    hidden_size // self.packed_factor,
                ),
                "Marlin": (
                    num_experts,
                    hidden_size // self.packed_factor,
                    w13_num_shards * intermediate_size_per_partition,
                ),
            },
            "w13_scale": {
                "Flashinfer": (
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    num_groups_w13,
                ),
                "Marlin": (
                    num_experts,
                    num_groups_w13,
                    w13_num_shards * intermediate_size_per_partition,
                ),
            },
            "w2_weight": {
                "Flashinfer": (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.packed_factor,
                ),
                "Marlin": (
                    num_experts,
                    intermediate_size_per_partition // self.packed_factor,
                    hidden_size,
                ),
            },
            "w2_scale": {
                "Flashinfer": (num_experts, hidden_size, num_groups_w2),
                "Marlin": (num_experts, num_groups_w2, hidden_size),
            },
        }
        return shape_map[weight_name][self.kernel_backend]

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        is_transposed = self.kernel_backend != "Flashinfer"
        extra_weight_attrs.update(
            {"is_transposed": is_transposed, "quant_method": self.strategy}
        )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                *self.get_weight_shape(
                    "w13_weight",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                ),
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                *self.get_weight_shape(
                    "w2_weight",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                ),
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1
        w2_scales_size = (
            intermediate_size_full if load_full_w2 else intermediate_size_per_partition
        )

        self.is_k_full = (not self.actorder) or (
            intermediate_size_per_partition == intermediate_size_full
        )

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        w13_scale = torch.nn.Parameter(
            torch.ones(
                *self.get_weight_shape(
                    "w13_scale",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    num_groups_w13=num_groups_w13,
                ),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(
                *self.get_weight_shape(
                    "w2_scale",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    num_groups_w2=num_groups_w2,
                ),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

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
        layer.marlin_state = GPTQMarlinState.REPACK

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device
        if self.kernel_backend == "Flashinfer":
            dict_weights_mxint4 = prepare_static_weights_for_trtllm_mxint4_moe(
                layer.w13_weight_packed,
                layer.w13_weight_scale,
                layer.w2_weight_packed,
                layer.w2_weight_scale,
            )
            replace_parameter(
                layer, "w13_weight_packed", dict_weights_mxint4["gemm1_weights"]
            )
            replace_parameter(
                layer, "w13_weight_scale", dict_weights_mxint4["gemm1_scales"]
            )
            replace_parameter(
                layer, "w2_weight_packed", dict_weights_mxint4["gemm2_weights"]
            )
            replace_parameter(
                layer, "w2_weight_scale", dict_weights_mxint4["gemm2_scales"]
            )
            return None

        is_a_8bit = (
            self.marlin_input_dtype is not None
            and self.marlin_input_dtype.itemsize == 1
        )

        if self.marlin_input_dtype == torch.float8_e4m3fn:
            # NOTE: for non-zp quantization format only
            ops.marlin_int4_fp8_preprocess(layer.w13_weight_packed, inplace=True)
            ops.marlin_int4_fp8_preprocess(layer.w2_weight_packed, inplace=True)
            layer.w13_weight_scale.data = layer.w13_weight_scale.data * 512
            layer.w2_weight_scale.data = layer.w2_weight_scale.data * 512

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]
                ]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w13_weight_packed", marlin_w13_qweight)

        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w2_weight_packed", marlin_w2_qweight)

        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_weight_scale,
            size_k=layer.w13_weight_packed.shape[2],
            size_n=layer.w13_weight_scale.shape[2],
            group_size=self.group_size,
            is_a_8bit=is_a_8bit,
        )
        if self.marlin_input_dtype == torch.int8 and layer.num_groups_w13 > 1:
            marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
                marlin_w13_scales
            )
            layer.register_parameter(
                "w13_input_global_scale",
                torch.nn.Parameter(w13_input_global_scale, requires_grad=False),
            )
        replace_parameter(layer, "w13_weight_scale", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_weight_scale,
            size_k=layer.w2_weight_scale.shape[1]
            * (self.group_size if self.group_size != -1 else self.packed_factor),
            size_n=layer.w2_weight_scale.shape[2],
            group_size=self.group_size,
            is_a_8bit=is_a_8bit,
        )
        if self.marlin_input_dtype == torch.int8 and layer.num_groups_w2 > 1:
            marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
                marlin_w2_scales
            )
            layer.register_parameter(
                "w2_input_global_scale",
                torch.nn.Parameter(w2_input_global_scale, requires_grad=False),
            )
        replace_parameter(layer, "w2_weight_scale", marlin_w2_scales)

        layer.workspace = marlin_make_workspace_new(device, 4)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.num_bits != 4:
            return None
        return int4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        assert self.num_bits == 4, "only supporting w4"
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed
        assert all([w is not None for w in [layer.w13_weight, layer.w2_weight]])
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None
            return BatchedMarlinExperts(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=layer.w13_weight_g_idx,
                w2_g_idx=layer.w2_weight_g_idx,
                w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
            )
        else:
            return MarlinExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=layer.w13_weight_g_idx,
                w2_g_idx=layer.w2_weight_g_idx,
                w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
            )

    @property
    def is_monolithic(self) -> bool:
        return self.kernel_backend == "Flashinfer"

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.kernel_backend == "Flashinfer"
        return flashinfer_trtllm_mxint4_moe(
            x=x,
            router_logits=router_logits,
            w13_weight_packed=layer.w13_weight_packed,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight_packed=layer.w2_weight_packed,
            w2_weight_scale=layer.w2_weight_scale,
            global_num_experts=layer.global_num_experts,
            top_k=layer.top_k,
            intermediate_size_per_partition=layer.intermediate_size_per_partition,
            local_num_experts=layer.local_num_experts,
            ep_rank=layer.ep_rank,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routing_method_type=layer.routing_method_type,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.kernel_backend == "Marlin"
        return fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            None,
            None,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            input_global_scale1=getattr(layer, "w13_input_global_scale", None),
            input_global_scale2=getattr(layer, "w2_input_global_scale", None),
            quant_type_id=self.quant_type.id,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            activation=layer.activation,
            expert_map=layer.expert_map,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            workspace=layer.workspace,
            input_dtype=self.marlin_input_dtype,
            is_k_full=self.is_k_full,
        )


class CompressedTensorsWNA16MoEMethod(CompressedTensorsMoEMethod):
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
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        # channelwise is not supported by this kernel
        assert weight_quant.strategy == "group"
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
        # Reconfigure packed weights and scales to match moe_wna16 format
        layer.w13_weight_packed = torch.nn.Parameter(
            layer.w13_weight_packed.transpose(1, 2).contiguous().view(torch.uint8),
            requires_grad=False,
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            layer.w2_weight_packed.transpose(1, 2).contiguous().view(torch.uint8),
            requires_grad=False,
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.transpose(1, 2).contiguous(), requires_grad=False
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        assert self.num_bits == 4 or self.num_bits == 8
        config_builder = (
            int4_w4a16_moe_quant_config
            if self.num_bits == 4
            else int8_w8a16_moe_quant_config
        )

        return config_builder(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe import fused_experts

        return fused_experts(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )

    @property
    def supports_eplb(self) -> bool:
        return True


class CompressedTensorsW4A8Int8MoEMethod(CompressedTensorsMoEMethod):
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
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        # Validate scheme: weights=W4 (channel or group),
        # activations=dynamic TOKEN (A8)

        # Must be dynamic per-token activations
        if (
            input_quant.strategy != QuantizationStrategy.TOKEN
            or not input_quant.dynamic
        ):
            raise ValueError(
                "W4A8-int MoE needs dynamic per-token activation quantization."
            )

        # Weight can be channel-wise (group_size=None) or group-wise
        self.group_size = (
            weight_quant.group_size if (weight_quant.group_size is not None) else -1
        )
        if weight_quant.num_bits != 4:
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

    @property
    def is_monolithic(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        assert not layer.enable_eplb, "EPLB not supported for W4A8-int MoE yet."
        assert layer.activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
        ), "Only SiLU/SwiGLUGU/SwiGLUUG are supported."
        assert layer.expert_map is None, """expert_map/EP not implemented
        for CPU dyn-4bit MoE."""

        def _act_kind(s: MoEActivation) -> int:
            # 0 = SwiGLU_Gu (SiLU(g)*u), 1 = SwiGLU_Ug (SiLU(u)*g), 2 = SiLU
            if s == MoEActivation.SWIGLUSTEP:
                return 0
            if s == MoEActivation.SWIGLUOAI:
                return 1
            if s == MoEActivation.SILU:
                return 2
            raise ValueError(f"Unknown activation '{s}'")

        # Apply topk softmax on router output
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=layer.top_k,
            use_grouped_topk=layer.use_grouped_topk,
            renormalize=layer.renormalize,
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
            layer.apply_router_weight_on_input,
            int(_act_kind(layer.activation)),
        )


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
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
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
    ) -> mk.FusedMoEPrepareAndFinalize | None:
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
            per_act_token_quant=True,  # always use dynamc per-token
            per_out_ch_quant=True,  # always use per-channel
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        assert (
            prepare_finalize.activation_format == FusedMoEActivationFormat.Standard
        ), "BatchedExperts not supported"

        from vllm.model_executor.layers.fused_moe import CutlassExpertsW4A8Fp8

        experts: FusedMoEPermuteExpertsUnpermute

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        )

    @property
    def supports_eplb(self) -> bool:
        return False
