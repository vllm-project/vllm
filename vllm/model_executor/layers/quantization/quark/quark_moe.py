# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    int8_w8a8_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
    ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    Int8MoeBackend,
    convert_to_int8_moe_kernel_format,
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    backend_to_kernel_cls,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.quark.utils import QuarkQTensorHint
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    _ACTIVATION_QUANT_KEY_MAP,
    _WEIGHT_QUANT_KEY_MAP,
    OCP_MX_BLOCK_SIZE,
    OCP_MX_Scheme,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4W4A8StaticChannelSym,
    kInt8DynamicTensorAsym,
    kInt8DynamicTensorSym,
    kInt8DynamicTokenAsym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kInt8StaticTensorAsym,
    kInt8StaticTensorSym,
    kMxfp4Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

logger = init_logger(__name__)

__all__ = [
    "QuarkMoEMethod",
    "QuarkW8A8Fp8MoEMethod",
    "QuarkOCP_MX_MoEMethod",
    "QuarkNvfp4MoEMethod",
]


class QuarkMoEMethod(FusedMoEMethodBase):
    supported_activation_quant_keys: list[QuantKey | None] = []
    supported_weight_quant_keys: list[QuantKey] = []

    def __init__(
        self,
        moe: FusedMoEConfig,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
    ):
        super().__init__(moe)
        if activation_quant_key not in self.supported_activation_quant_keys:
            raise ValueError(
                f"Unsupported activation quant key: {activation_quant_key}"
            )
        if weight_quant_key not in self.supported_weight_quant_keys:
            raise ValueError(f"Unsupported weight quant key: {weight_quant_key}")
        self.activation_quant_key = activation_quant_key
        self.weight_quant_key = weight_quant_key
        self.has_bias = self.moe.has_bias

    @staticmethod
    def get_moe_method_target(
        quant_config: "QuarkConfig",
        layer_type: type[torch.nn.Module],
        layer_name: str,
    ) -> tuple[
        QuantKey | None,
        QuantKey | None,
        type["QuarkMoEMethod"] | None,
    ]:
        """Return the quantization target for a routed-experts layer."""
        layer_quant_config = quant_config._find_matched_config(layer_name, layer_type)
        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with "
                "output_tensors and bias "
                "quantized are not supported"
            )
        weight_config: QuarkQTensorHint = layer_quant_config.get("weight")
        input_config: QuarkQTensorHint = layer_quant_config.get("input_tensors")

        if (
            isinstance(weight_config, list)
            and len(weight_config) > 1
            or isinstance(input_config, list)
            and len(input_config) > 1
        ):
            if match := quant_config._is_fp8_w4a8(weight_config, input_config):
                return (
                    match.weight_quant_key,
                    match.activation_quant_key,
                    QuarkW4A8Fp8MoEMethod,
                )
            if match := quant_config._is_nvfp4(weight_config, input_config):
                return (
                    match.weight_quant_key,
                    match.activation_quant_key,
                    QuarkNvfp4MoEMethod,
                )
            raise RuntimeError("Unsupported FusedMoe scheme")

        weight_config = quant_config._unwrap_single_quant_config(weight_config)
        input_config = quant_config._unwrap_single_quant_config(input_config)

        if match := quant_config._is_fp8_w8a8(weight_config, input_config):
            return (
                match.weight_quant_key,
                match.activation_quant_key,
                QuarkW8A8Fp8MoEMethod,
            )
        if match := quant_config._is_w_ocp_mx_a_x(
            weight_config, input_config, allow_static_fp8=True
        ):
            return (
                match.weight_quant_key,
                match.activation_quant_key,
                QuarkOCP_MX_MoEMethod,
            )
        if match := quant_config._is_w8a8_int8(weight_config, input_config):
            return (
                match.weight_quant_key,
                match.activation_quant_key,
                QuarkW8A8Int8MoEMethod,
            )
        raise RuntimeError("Unsupported FusedMoe scheme")

    @staticmethod
    def get_moe_method(
        quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
        module: RoutedExperts,
        method_cls: type["QuarkMoEMethod"],
        weight_quant_key: QuantKey | None,
        activation_quant_key: QuantKey | None,
    ) -> "QuarkMoEMethod":
        if weight_quant_key is None:
            raise RuntimeError("Unsupported FusedMoe scheme")

        if method_cls is QuarkW4A8Fp8MoEMethod:
            return QuarkW4A8Fp8MoEMethod(module.moe_config, activation_quant_key)
        elif method_cls is QuarkNvfp4MoEMethod:
            return QuarkNvfp4MoEMethod(
                module.moe_config, quant_config, activation_quant_key
            )
        elif method_cls is QuarkW8A8Fp8MoEMethod:
            return QuarkW8A8Fp8MoEMethod(
                module.moe_config, weight_quant_key, activation_quant_key
            )
        elif method_cls is QuarkOCP_MX_MoEMethod:
            # All OCP MX schemes (W4A16, W4A8, etc.) handled by QuarkOCP_MX_MoEMethod
            # Backend selection happens inside via oracle
            return QuarkOCP_MX_MoEMethod(
                module.moe_config, weight_quant_key, activation_quant_key
            )
        elif method_cls is QuarkW8A8Int8MoEMethod:
            return QuarkW8A8Int8MoEMethod(
                module.moe_config, weight_quant_key, activation_quant_key
            )
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")


class QuarkW8A8Fp8MoEMethod(QuarkMoEMethod):
    supported_activation_quant_keys = [
        kFp8StaticTensorSym,
        kFp8DynamicTensorSym,
        kFp8DynamicTokenSym,
    ]
    supported_weight_quant_keys = [
        kFp8StaticChannelSym,
        kFp8StaticTensorSym,
    ]

    def __init__(
        self,
        moe: FusedMoEConfig,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
    ):
        super().__init__(moe, weight_quant_key, activation_quant_key)
        self.weight_qscheme = (
            "per_channel" if weight_quant_key == kFp8StaticChannelSym else "per_tensor"
        )
        self.input_qscheme = (
            "per_channel"
            if activation_quant_key == kFp8DynamicTokenSym
            else "per_tensor"
        )
        self.static_input_scales = activation_quant_key == kFp8StaticTensorSym
        per_tensor = (
            self.weight_qscheme == "per_tensor" and self.input_qscheme == "per_tensor"
        )
        per_channel = (
            self.weight_qscheme == "per_channel" and self.input_qscheme == "per_channel"
        )
        self.act_quant_group_shape = (
            GroupShape.PER_TOKEN if per_channel else GroupShape.PER_TENSOR
        )
        if not (per_tensor or per_channel):
            raise ValueError(
                "For FP8 Fused MoE layers, only per-tensor and per-channel "
                "scales for weights and activations are supported. Found "
                f"{self.weight_qscheme}, {self.input_qscheme}"
            )  # noqa E501

        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=moe,
            weight_key=self.weight_quant_key,
            activation_key=self.activation_quant_key,
        )

        self.model_type = getattr(
            get_current_vllm_config().model_config.hf_config, "model_type", None
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.zeros(
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
        if self.weight_qscheme == "per_tensor":
            # Allocate 2 scales for w1 and w3 respectively.
            # They are combined to a single scale after weight loading.
            if self.model_type != "gpt_oss":
                w13_weight_scale = torch.nn.Parameter(
                    torch.ones(
                        num_experts, self.moe.w13_num_shards, dtype=torch.float32
                    ),
                    requires_grad=False,
                )
            else:
                # For gpt_oss, the w1(gate) & w3(up) are fused as one.
                # Therefore, only one weight scale for each expert.
                w13_weight_scale = torch.nn.Parameter(
                    torch.ones(num_experts, 1, dtype=torch.float32), requires_grad=False
                )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSOR quantization for RoutedExperts.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        elif self.weight_qscheme == "per_channel":
            # quark's scale is 1 dim.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for RoutedExperts.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
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

        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.w13_bias, layer.w2_bias = None, None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer. "
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        if current_platform.is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = (
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )

        # For per-tensor case, Fp8 moe kernel needs single weight scale
        # for w13 per expert. Use max then dequant and requant each expert.
        if self.weight_qscheme == "per_tensor":
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values

            # For gpt_oss, w1 and w3 are fused into a single combined
            # gate_up_proj tensor with size 2*intermediate_size_per_partition
            # and only one scale per expert.
            # Process the entire weight tensor as one shard.
            if self.model_type == "gpt_oss":
                for expert_id in range(layer.local_num_experts):
                    # Process all 2*intermediate_size_per_partition rows at once
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id],
                        layer.w13_weight_scale[expert_id][0],
                    )
                    layer.w13_weight[expert_id], _ = ops.scaled_fp8_quant(
                        dq_weight, max_w13_scales[expert_id]
                    )
            else:
                # For non-gpt_oss, process w1 and w3 shards separately
                for expert_id in range(layer.local_num_experts):
                    start = 0
                    for shard_id in range(self.moe.w13_num_shards):
                        dq_weight = per_tensor_dequantize(
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            layer.w13_weight_scale[expert_id][shard_id],
                        )
                        (
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            _,
                        ) = ops.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                        start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

        # quark's scale is 1 dim.
        elif self.weight_qscheme == "per_channel":
            if self.act_quant_group_shape == GroupShape.PER_TOKEN:
                w13_weight_scale = layer.w13_weight_scale.unsqueeze(-1)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                w2_weight_scale = layer.w2_weight_scale.unsqueeze(-1)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_weight_scale, requires_grad=False
                )
        self._setup_kernel(layer)

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_input_scale=layer.w13_input_scale,
            w2_input_scale=layer.w2_input_scale,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_fp8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            fp8_backend=self.fp8_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        return make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            per_act_token_quant=self.input_qscheme == "per_channel",
            per_out_ch_quant=self.weight_qscheme == "per_channel",
            swiglu_limit=getattr(layer, "swiglu_limit", None),
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


class QuarkW8A8Int8MoEMethod(QuarkMoEMethod):
    """Quark W8A8 INT8 MoE method."""

    supported_activation_quant_keys = [
        kInt8StaticTensorSym,
        kInt8StaticTensorAsym,
        kInt8DynamicTensorSym,
        kInt8DynamicTensorAsym,
        kInt8DynamicTokenSym,
        kInt8DynamicTokenAsym,
    ]
    supported_weight_quant_keys = [
        kInt8StaticChannelSym,
        kInt8StaticTensorSym,
    ]

    def __init__(
        self,
        moe: FusedMoEConfig,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
    ):
        super().__init__(moe, weight_quant_key, activation_quant_key)

        # TODO: oracle needs to support asym int8.
        if self.activation_quant_key == kInt8StaticTensorAsym:
            self.activation_quant_key = kInt8StaticTensorSym
        if self.activation_quant_key == kInt8DynamicTensorAsym:
            self.activation_quant_key = kInt8DynamicTensorSym
        if self.activation_quant_key == kInt8DynamicTokenAsym:
            self.activation_quant_key = kInt8DynamicTokenSym

        self.weight_qscheme = (
            "per_channel" if weight_quant_key == kInt8StaticChannelSym else "per_tensor"
        )

        assert self.activation_quant_key is not None
        self.static_input_scales = self.activation_quant_key.scale.static

        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.moe_kernel: mk.FusedMoEKernel | None = None
        self.int8_backend: Int8MoeBackend | None = None
        self.experts_cls: type[mk.FusedMoEExperts] | None = None

        # Dynamic-activation INT8 MoE goes through the oracle + modular kernel.
        # The modular TritonExperts kernel consumes float activations and
        # quantizes them to int8 itself, so it cannot apply a loaded static
        # activation scale (this matches CompressedTensorsW8A8Int8MoEMethod).
        # TODO: Static-activation INT8 therefore stays on the legacy fused_experts
        # path (see apply()) for now, preserving pre-refactor behavior.
        # Needs to be migrated to expert backend.
        if not self.static_input_scales:
            # Map the Quark weight scheme to oracle quant keys. Per-channel
            # weights pair with dynamic per-token activations; per-tensor
            # weights with dynamic per-tensor activations.
            self.int8_backend, self.experts_cls = select_int8_moe_backend(
                config=moe,
                weight_key=self.weight_quant_key,
                activation_key=self.activation_quant_key,
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
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
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
        if self.weight_qscheme == "per_channel":
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        else:
            # per-tensor: one scalar per expert (two for the fused w1/w3)
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, self.moe.w13_num_shards, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            # Static activations: the per-expert scales are loaded from the
            # checkpoint (used by the legacy fused_experts path).
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            # Dynamic activations are quantized in-kernel (no stored scale).
            layer.w13_input_scale = None
            layer.w2_input_scale = None

        # ZERO POINTS (loaded but discarded after loading; kernel uses symmetric)
        w13_input_zero_point = torch.nn.Parameter(
            torch.zeros(num_experts, self.moe.w13_num_shards, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_zero_point", w13_input_zero_point)
        set_weight_attrs(w13_input_zero_point, extra_weight_attrs)

        w2_input_zero_point = torch.nn.Parameter(
            torch.zeros(num_experts, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_zero_point", w2_input_zero_point)
        set_weight_attrs(w2_input_zero_point, extra_weight_attrs)

        if self.weight_qscheme == "per_channel":
            w13_weight_zero_point = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            w2_weight_zero_point = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.int8),
                requires_grad=False,
            )
        else:
            w13_weight_zero_point = torch.nn.Parameter(
                torch.zeros(num_experts, self.moe.w13_num_shards, dtype=torch.int8),
                requires_grad=False,
            )
            w2_weight_zero_point = torch.nn.Parameter(
                torch.zeros(num_experts, dtype=torch.int8),
                requires_grad=False,
            )
        layer.register_parameter("w13_weight_zero_point", w13_weight_zero_point)
        set_weight_attrs(w13_weight_zero_point, extra_weight_attrs)
        layer.register_parameter("w2_weight_zero_point", w2_weight_zero_point)
        set_weight_attrs(w2_weight_zero_point, extra_weight_attrs)

        # BIAS
        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.w13_bias, layer.w2_bias = None, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Discard zero points (INT8 fused MoE kernel uses symmetric quant)
        for attr in (
            "w13_input_zero_point",
            "w2_input_zero_point",
            "w13_weight_zero_point",
            "w2_weight_zero_point",
        ):
            if hasattr(layer, attr):
                delattr(layer, attr)

        # For static input scales, collapse the per-expert scales to a single
        # value (the legacy fused_experts path expects one scale per layer).
        if self.static_input_scales:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "INT8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        # Per-channel scales: 2D [E, N] -> 3D [E, N, 1] for the int8 MoE kernel.
        if self.weight_qscheme == "per_channel":
            for attr in ("w13_weight_scale", "w2_weight_scale"):
                param = getattr(layer, attr, None)
                if param is not None and param.dim() == 2:
                    replace_parameter(
                        layer,
                        attr,
                        torch.nn.Parameter(
                            param.data.unsqueeze(-1).contiguous(),
                            requires_grad=False,
                        ),
                    )

        # For per-tensor weights, merge the w1/w3 scales into a single
        # per-expert scale (dequant -> requant at the max scale).
        if self.weight_qscheme == "per_tensor":
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values

            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(self.moe.w13_num_shards):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    layer.w13_weight[expert_id][start : start + shard_size, :], _, _ = (
                        ops.scaled_int8_quant(
                            dq_weight,
                            scale=max_w13_scales[expert_id],
                        )
                    )
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

        # Dynamic activations run through the oracle's modular kernel; static
        # activations use the legacy fused_experts path in apply().
        if not self.static_input_scales:
            assert self.int8_backend is not None
            assert self.experts_cls is not None
            w13, w2 = convert_to_int8_moe_kernel_format(
                int8_backend=self.int8_backend,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
                layer=layer,
                w13_scale=layer.w13_weight_scale,
            )
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        if not self.static_input_scales:
            assert self.int8_backend is not None
            assert self.experts_cls is not None
            self.moe_kernel = make_int8_moe_kernel(
                int8_backend=self.int8_backend,
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # Static-activation INT8 has no oracle backend (it uses the legacy
        # fused_experts path); build its config directly.
        if self.int8_backend is None:
            return int8_w8a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                per_act_token_quant=False,
            )
        return make_int8_moe_quant_config(
            int8_backend=self.int8_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            per_act_token_quant=(self.weight_qscheme == "per_channel"),
            layer=layer,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.moe_kernel is not None:
            return self.moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                expert_map=layer.expert_map,
                shared_experts_input=shared_experts_input,
            )

        # Static-activation INT8 MoE: legacy monolithic path (the modular kernel
        # quantizes activations dynamically and cannot apply a loaded scale).
        from vllm.model_executor.layers.fused_moe import fused_experts

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )


class QuarkW4A8Fp8MoEMethod(QuarkMoEMethod):
    supported_activation_quant_keys = [
        kFp8DynamicTokenSym,
        kFp8StaticTensorSym,
    ]
    supported_weight_quant_keys = [kInt4W4A8StaticChannelSym]

    def __init__(
        self,
        moe: FusedMoEConfig,
        activation_quant_key: QuantKey | None,
    ):
        super().__init__(moe, kInt4W4A8StaticChannelSym, activation_quant_key)

        assert rocm_aiter_ops.is_fused_moe_enabled(), (
            "W4A8 FP8 MoE requires ROCm AITER fused MoE support."
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        params_dtype = torch.uint32
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // 8,  # INT32 packing for W4
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 8,  # INT32 packing for W4
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Per-tensor fp8 weight scales
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, self.moe.w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Per-channel int4 weight scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
            layer.w13_weight.data, layer.w2_weight.data
        )
        layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)

        # INT4-FP8 : offset INT4 w13_weight_scale1 to single w13_weight_scale
        # Fp8 moe kernel needs single fp8 w13_weight_scale for w13 per expert.
        # We won't do requant each expert's fp8 weight (not direct available),
        # instead we adjust half of INT4 w13_weight_scale1 numbers
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        assert torch.all(max_w13_scales != 0), "fp8 weight scale cannot be zero."
        for expert_id in range(layer.local_num_experts):
            start = 0
            max_w13_scale_fp8 = max_w13_scales[expert_id]
            for shard_id in range(self.moe.w13_num_shards):
                if layer.w13_weight_scale[expert_id][shard_id] != max_w13_scale_fp8:
                    int4_rescale = (
                        layer.w13_weight_scale[expert_id][shard_id] / max_w13_scale_fp8
                    )
                    layer.w13_weight_scale_2[expert_id][start : start + shard_size] *= (
                        int4_rescale
                    )
                start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)

        # special hack to asm_moe, which takes (weight_scale1 * weight_scale) as post
        # GEMM scaling optimal design - shall apply per-column weight_scale1 before
        # GEMM, and weight_scale post
        for expert_id in range(layer.local_num_experts):
            layer.w13_weight_scale_2[expert_id] *= max_w13_scales[expert_id]
            layer.w2_weight_scale_2[expert_id] *= layer.w2_weight_scale[expert_id]

    def get_fused_moe_quant_config(self, layer):
        return fp8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale_2,
            w2_scale=layer.w2_weight_scale_2,
            per_out_ch_quant=True,
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            rocm_aiter_fused_experts,
        )

        return rocm_aiter_fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            quant_config=self.moe_quant_config,
            moe_config=layer.moe_config,
            expert_mask=layer.expert_mask,
        )


class QuarkOCP_MX_MoEMethod(QuarkMoEMethod):
    supported_activation_quant_keys = [
        *_ACTIVATION_QUANT_KEY_MAP.values(),
        kFp8DynamicTensorSym,
        kFp8StaticTensorSym,
        None,
    ]
    supported_weight_quant_keys = [*_WEIGHT_QUANT_KEY_MAP.values()]

    def __init__(
        self,
        moe: FusedMoEConfig,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
    ):
        super().__init__(moe, weight_quant_key, activation_quant_key)
        self.weight_dtype = next(
            dtype
            for dtype, quant_key in _WEIGHT_QUANT_KEY_MAP.items()
            if quant_key == weight_quant_key
        )
        if activation_quant_key in {kFp8DynamicTensorSym, kFp8StaticTensorSym}:
            self.input_dtype: str | None = "fp8"
        elif activation_quant_key is None:
            self.input_dtype = None
        else:
            self.input_dtype = next(
                dtype
                for dtype, quant_key in _ACTIVATION_QUANT_KEY_MAP.items()
                if quant_key == activation_quant_key
            )

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.ocp_mx_scheme is None:
            raise ValueError(
                f"Unsupported OCP MX dtype combination for MoE: "
                f"input_dtype={self.input_dtype}, weight_dtype={self.weight_dtype}. "
                f"Please check that the combination is supported in OCP_MX_Scheme."
            )

        # TODO(bowenbao): refactor and introduce backends for other OCP MX schemes,
        # use kernel abstraction for all OCP MX MOE implementations.
        self.mxfp4_backend: Mxfp4MoeBackend = Mxfp4MoeBackend.NONE
        self.experts_cls: type[mk.FusedMoEExperts] | None = None
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs (W4A8, TRITON backends)
        self.w13_precision_config = None
        self.w2_precision_config = None

        self.static_input_scales = activation_quant_key == kFp8StaticTensorSym

        # Select backend based on OCP MX scheme
        if self.ocp_mx_scheme == "w_mxfp4":
            # W4A16: weight-only MXFP4
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)
        elif self.ocp_mx_scheme == "w_mxfp4_a_fp8" and self.static_input_scales:
            # W4A8: MXFP4 weights + static FP8 activations
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
                moe, activation_key=kFp8StaticTensorSym
            )
        elif self.ocp_mx_scheme == "w_mxfp4_a_mxfp4":
            # W4A4: MXFP4 weights + MXFP4 activations
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
                moe, activation_key=kMxfp4Dynamic
            )

        # Validation for unsupported schemes
        if any(
            self.ocp_mx_scheme.endswith(a_scheme)
            for a_scheme in ["a_mxfp4", "a_mxfp6_e3m2", "a_mxfp6_e2m3"]
        ):
            if self.static_input_scales:
                raise NotImplementedError(
                    "QuarkOCP_MX_MoEMethod with static input scales is currently "
                    f"not implemented for OCP MX scheme {self.ocp_mx_scheme}. "
                    "Please open an issue."
                )
        elif self.ocp_mx_scheme.endswith("a_fp8") and not self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX_MoEMethod with dynamic input scales is currently "
                f"not implemented for OCP MX scheme {self.ocp_mx_scheme}. "
                "Please open an issue."
            )

        self.model_type = getattr(
            get_current_vllm_config().model_config.hf_config, "model_type", None
        )

        # If no native backend available, use emulation.
        if self.mxfp4_backend is Mxfp4MoeBackend.NONE:
            self.mxfp4_backend = Mxfp4MoeBackend.EMULATION

        self.experts_cls = backend_to_kernel_cls(self.mxfp4_backend)[0]

        logger.info_once(
            f"Using {self.mxfp4_backend.value} backend for {self.ocp_mx_scheme}"
        )

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        # Round per-partition sizes up to each backend's requirement. Emulation is
        # handled inside the helper too (OCP MX block alignment), so no special-case.
        if self.mxfp4_backend is not None:
            hidden_size, intermediate_size_per_partition = (
                mxfp4_round_up_hidden_size_and_intermediate_size(
                    self.mxfp4_backend, hidden_size, intermediate_size_per_partition
                )
            )
        return hidden_size, intermediate_size_per_partition

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        else:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        params_dtype = torch.uint8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                self.get_packed_dim(hidden_size, self.weight_dtype),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                self.get_packed_dim(intermediate_size_per_partition, self.weight_dtype),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)

        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.w13_bias, layer.w2_bias = None, None

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

    def process_weights_after_loading(self, layer):
        self._setup_kernel(layer)

    def _setup_kernel(self, layer: RoutedExperts):
        """Setup kernel using oracle functions for MXFP4 schemes (W4A16, W4A8)."""
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        # Convert weights to kernel format (handles all backend-specific logic)
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_weight_scale=layer.w13_weight_scale,
                w2_weight_scale=layer.w2_weight_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                w13_input_scale=layer.w13_input_scale,
                w2_input_scale=layer.w2_input_scale,
            )
        )

        # Handle weight/scale assignment based on backend type
        if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (
            Mxfp4MoeBackend.AITER_MXFP4_FP8,
        ):
            # Triton-based backends: w13/w2 are triton_kernels.tensor.Tensor
            # Store on layer for apply(), scales are PrecisionConfig
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale
        else:
            # Standard backends: replace parameters
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        torch.accelerator.empty_cache()

        # Build quant config and kernel
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        # For oracle-based backends (W4A16, W4A8), use make_mxfp4_moe_quant_config
        if self.mxfp4_backend not in (Mxfp4MoeBackend.NONE, Mxfp4MoeBackend.EMULATION):
            # Determine scale source based on backend type
            if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (
                Mxfp4MoeBackend.AITER_MXFP4_FP8,
            ):
                w1_scale = self.w13_precision_config
                w2_scale = self.w2_precision_config
            else:
                w1_scale = layer.w13_weight_scale
                w2_scale = layer.w2_weight_scale

            return make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                a1_scale=getattr(layer, "w13_input_scale", None),
                a2_scale=getattr(layer, "w2_input_scale", None),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                layer=layer,
            )

        # Emulation and other schemes
        if self.ocp_mx_scheme == "w_mxfp4":
            return mxfp4_w4a16_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
            )
        elif self.ocp_mx_scheme == "w_mxfp4_a_fp8":
            return mxfp4_w4a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                block_shape=None,
            )
        elif self.ocp_mx_scheme in ["w_mxfp6_e3m2_a_fp8", "w_mxfp6_e2m3_a_fp8"]:
            raise NotImplementedError(
                "Currently there is no corresponding fused moe quant config configured "
                f"in vLLM for OCP MX scheme {self.ocp_mx_scheme}. Please open an issue."
            )
        else:
            assert self.input_dtype is not None
            return ocp_mx_moe_quant_config(
                quant_dtype=self.input_dtype,
                weight_dtype=self.weight_dtype,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )

    @property
    def supports_eplb(self) -> bool:
        # AITER shuffle keeps expert dim outermost, so EPLB row moves are layout-safe.
        return True

    @property
    def is_monolithic(self) -> bool:
        if self.moe_kernel is not None:
            return self.moe_kernel.is_monolithic
        return False

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )


class QuarkNvfp4MoEMethod(QuarkMoEMethod):
    supported_activation_quant_keys = [kNvfp4Dynamic]
    supported_weight_quant_keys = [kNvfp4Static]

    def __init__(
        self,
        moe: FusedMoEConfig,
        quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
        activation_quant_key: QuantKey | None,
    ):
        super().__init__(moe, kNvfp4Static, activation_quant_key)
        self.quant_config = quant_config
        self.group_size = 16

        # Select experts implementation.
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=self.weight_quant_key,
            activation_key=self.activation_quant_key,
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
        layer.quant_config = self.quant_config
        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn

        # GEMM 1 - w13 weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # GEMM 2 - w2 weight
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight scales (per-group FP8 scales)
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
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
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Global weight scales (per-tensor FP32 scales)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, self.moe.w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input global scales (per-tensor FP32 scales)
        w13_input_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, self.moe.w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale_2", w13_input_scale_2)
        set_weight_attrs(w13_input_scale_2, extra_weight_attrs)

        w2_input_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale_2", w2_input_scale_2)
        set_weight_attrs(w2_input_scale_2, extra_weight_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """
        Convert NVFP4 MoE weights into kernel format and setup the kernel.
        """

        # Match existing NVFP4 MoE paths: fused w13 uses the w1 global scale.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )

        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()

        w2_weight_scale_2 = layer.w2_weight_scale_2

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
            w13_scale_2=w13_weight_scale_2,
            a13_scale=layer.w13_input_scale_2,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=w2_weight_scale_2,
            a2_scale=layer.w2_input_scale_2,
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w13_input_scale_2", a13_scale)

        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
        replace_parameter(layer, "w2_input_scale_2", a2_scale)

        # Setup modular kernel.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
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
            a13_scale=layer.w13_input_scale_2,
            a2_scale=layer.w2_input_scale_2,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            swiglu_alpha=getattr(layer, "swiglu_alpha", None),
            swiglu_beta=getattr(layer, "swiglu_beta", None),
            layer=layer,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
