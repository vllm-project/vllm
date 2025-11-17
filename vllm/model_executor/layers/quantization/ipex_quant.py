# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch
from packaging import version
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm._ipex_ops import ipex_ops as ops
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)
from vllm.model_executor.layers.quantization.awq import AWQLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_linear_quant_method,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata
from vllm.utils.collection_utils import is_list_of
from vllm.utils.math_utils import round_up

MIN_IPEX_VERSION = "2.6.0"


class IPEXConfig(QuantizationConfig):
    """INT8 quantization config class using IPEX for the CPU/XPU backend,
    including AWQ, GPTQ.
    """

    IPEX_QUANT_METHOD_MAP = {
        "awq": 1,
        "gptq": 0,
    }

    def __init__(
        self,
        method: str,
        weight_bits: int,
        group_size: int,
        is_qweight_sym: bool,
        full_config: dict[str, Any],
        dynamic: dict[str, dict[str, int | bool]],
        modules_to_not_convert: list[str] | None = None,
        desc_act: bool | None = None,
        lm_head_quantized: bool | None = None,
        modules_in_block_to_quantize: list[str] | None = None,
        checkpoint_format: str = "",
    ) -> None:
        super().__init__()
        self.dynamic = dynamic
        self.method = method
        self.linear_quant_method = method
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.modules_to_not_convert = modules_to_not_convert or []
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.modules_in_block_to_quantize = modules_in_block_to_quantize or []
        self.full_config = full_config
        self.pack_factor = 32 // self.weight_bits
        self.bit8_pack_factor = 8 // self.weight_bits

        if self.weight_bits not in [4]:
            raise ValueError(
                f"IPEX quantization supports weight bits [4], "
                f"but got {self.weight_bits}."
            )

        if self.method not in ["awq", "gptq"]:
            raise ValueError(
                f"IPEX quantization supports [awq, gptq], but got {self.method}."
            )
        self.is_qweight_sym = is_qweight_sym
        # used to identify GPTQ model quantized by autoround
        self.autoround_version = (
            full_config.get("autoround_version", "") if full_config is not None else ""
        )
        # GPTQ v1 and v2 format deals with zero points differently.
        # Currently GPTQModel stores v1 format checkpoints by default,
        # but provides the option to set `format="gptq_v2"` in `QuantizeConfig`.
        self.checkpoint_format = checkpoint_format

    def __repr__(self) -> str:
        return (
            f"IPEXConfig(method={self.method},"
            f"weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}),"
            f"dynamic={self.dynamic}, "
            f"modules_in_block_to_quantize={self.modules_in_block_to_quantize}),"
            f"checkpoint_format={self.checkpoint_format})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "ipex"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IPEXConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic
        method = cls.get_from_keys(config, ["quant_method"]).lower()
        if method == "awq":
            weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
            group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
            is_qweight_sym = not cls.get_from_keys_or(
                config, ["zero_point"], default=False
            )
            return cls(
                method,
                weight_bits,
                group_size,
                is_qweight_sym,
                config,
                dynamic,
                modules_to_not_convert,
                False,
                False,
            )
        # otherwise for gptq
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        desc_act = cls.get_from_keys_or(config, ["desc_act"], default=False)
        is_qweight_sym = cls.get_from_keys_or(config, ["sym"], default=True)
        modules_in_block_to_quantize = cls.get_from_keys_or(
            config, ["modules_in_block_to_quantize"], default=None
        )
        checkpoint_format = cls.get_from_keys_or(
            config, ["checkpoint_format"], default=""
        )
        return cls(
            method,
            weight_bits,
            group_size,
            is_qweight_sym,
            config,
            dynamic,
            [],
            desc_act,
            lm_head_quantized,
            modules_in_block_to_quantize,
            checkpoint_format,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
        if not current_platform.is_cpu() and not current_platform.is_xpu():
            return None

        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        if quant_method in ["awq", "gptq"]:
            return cls.get_name()

        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            if self.method == "awq":
                if is_layer_skipped(
                    prefix, self.modules_to_not_convert, self.packed_modules_mapping
                ):
                    return UnquantizedLinearMethod()
                return IPEXAWQLinearMethod(self)
            if self.method == "gptq":
                return get_linear_quant_method(
                    self, layer, prefix, IPEXGPTQLinearMethod
                )
        if isinstance(layer, FusedMoE) and self.method == "gptq":
            return XPUGPTQMarlinMoEMethod(self, layer.moe_config)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.modules_in_block_to_quantize is not None:
            self.modules_in_block_to_quantize = hf_to_vllm_mapper.apply_list(
                self.modules_in_block_to_quantize
            )

    def maybe_update_config(self, model_name: str, revision: str | None = None):
        if self.modules_in_block_to_quantize:
            if is_list_of(self.modules_in_block_to_quantize, list):
                # original modules_in_block_to_quantize: list[list[str]]
                # flatten original modules_in_block_to_quantize
                self.modules_in_block_to_quantize = [
                    item
                    for sublist in self.modules_in_block_to_quantize
                    for item in sublist
                ]
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_in_block_to_quantize = list(quant_layers)


class IPEXGPTQLinearMethod(GPTQLinearMethod):
    """GPTQ linear method using IPEX for the CPU/XPU backend."""

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        bias = layer.bias if not layer.skip_bias_add else None

        try:
            import intel_extension_for_pytorch as ipex

            if version.parse(ipex.__version__) < version.parse(MIN_IPEX_VERSION):
                raise ImportError(
                    "intel_extension_for_pytorch version is "
                    "wrong. Please install "
                    f"intel_extension_for_pytorch>={MIN_IPEX_VERSION}."
                )
        except ImportError as err:
            raise ImportError(
                "Please install "
                f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
                f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
                " to use IPEX-AWQ linear method."
            ) from err
        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK

        assert isinstance(self.quant_config, IPEXConfig)
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=self.quant_config.group_size,
        )
        layer.ipex_output_size = layer.qweight.shape[-1]
        g_idx = layer.g_idx if self.quant_config.desc_act else None
        layer.ipex_qlinear = (
            ipex.llm.quantization.woq_linear.IPEXWeightOnlyQuantizedLinear.from_weight(
                layer.qweight,
                layer.scales,
                layer.qzeros,
                layer.qweight.size(0),
                layer.ipex_output_size,
                qconfig=qconfig,
                g_idx=g_idx,
                bias=bias,
                group_size=self.quant_config.group_size,
                quant_method=IPEXConfig.IPEX_QUANT_METHOD_MAP["gptq"],
                weight_qscheme="sym" if self.quant_config.is_qweight_sym else "asym",
            )
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size,))


class IPEXAWQLinearMethod(AWQLinearMethod):
    """AWQ linear method using IPEX for the CPU/XPU backend."""

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer=layer)

        bias = layer.bias if not layer.skip_bias_add else None

        try:
            import intel_extension_for_pytorch as ipex

            if version.parse(ipex.__version__) < version.parse(MIN_IPEX_VERSION):
                raise ImportError(
                    "intel_extension_for_pytorch version is "
                    "wrong. Please install "
                    f"intel_extension_for_pytorch>={MIN_IPEX_VERSION}."
                )
        except ImportError as err:
            raise ImportError(
                "Please install "
                f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
                f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
                " to use IPEX-AWQ linear method."
            ) from err

        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH

        assert isinstance(self.quant_config, IPEXConfig)
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=self.quant_config.group_size,
        )

        layer.ipex_output_size = layer.qweight.size(1) * self.quant_config.pack_factor
        layer.ipex_qlinear = (
            ipex.llm.quantization.woq_linear.IPEXWeightOnlyQuantizedLinear.from_weight(
                layer.qweight,
                layer.scales,
                layer.qzeros,
                layer.qweight.size(0),
                layer.ipex_output_size,
                qconfig=qconfig,
                bias=bias,
                group_size=self.quant_config.group_size,
                quant_method=IPEXConfig.IPEX_QUANT_METHOD_MAP["awq"],  # type: ignore
                weight_qscheme="sym" if self.quant_config.is_qweight_sym else "asym",
            )
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size,))


class XPUFp8LinearMethod(Fp8LinearMethod):
    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)

    def process_weights_after_loading(self, layer: Module) -> None:
        # If checkpoint not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)
            # Update the layer with the new values.
            layer.weight = Parameter(qweight, requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        output = torch.ops.torch_ipex.fp8_gemm_w8a16(
            x, weight, True, weight_scale, bias
        )
        return output


class XPUFp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
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
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
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

        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # INPUT_SCALES
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        if not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = current_platform.fp8_dtype()
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.local_num_experts,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )
            for expert in range(layer.local_num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        import intel_extension_for_pytorch as ipex

        ep_rank_start = self.moe.ep_rank * self.moe.num_local_experts
        layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
            layer.w13_weight,
            layer.w2_weight,
            w1_scale_inv=layer.w13_weight_scale,
            w2_scale_inv=layer.w2_weight_scale,
            a1_scale_inv=layer.w13_input_scale,
            a2_scale_inv=layer.w2_input_scale,
            use_prepack=True,
            experts_start_id=ep_rank_start,
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
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
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            custom_routing_function=custom_routing_function,
        )


class XPUGPTQMarlinMoEMethod(FusedMoEMethodBase):
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    def __init__(
        self,
        quant_config: IPEXConfig,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config

        weight_bits = quant_config.weight_bits
        is_qweight_sym = quant_config.is_qweight_sym
        self.quant_type = self.TYPE_MAP[(weight_bits, is_qweight_sym)]

        if self.quant_type.size_bits != 4:
            raise ValueError("XPUGPTQMarlinMoEMethod only supports int4 now.")

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

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

        self.is_k_full = (not self.quant_config.desc_act) or (
            intermediate_size_per_partition == intermediate_size_full
        )
        intermediate_size_per_partition = round_up(
            intermediate_size_per_partition, self.quant_config.group_size
        )
        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            w2_scales_size = (
                intermediate_size_full
                if self.quant_config.desc_act
                else intermediate_size_per_partition
            )
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})
        # up_proj scales
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)
        # down_proj scales
        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import intel_extension_for_pytorch as ipex

        if self.quant_config.linear_quant_method == "gptq":
            layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                layer.w13_qweight.permute(0, 2, 1),
                layer.w2_qweight.permute(0, 2, 1),
                w1_scale_inv=layer.w13_scales.permute(0, 2, 1),
                w2_scale_inv=layer.w2_scales.permute(0, 2, 1),
                is_int4=True,
            )
        else:
            raise NotImplementedError(
                f"Unsupported quant method {self.quant_config.linear_quant_method} "
                "for XPU MOE."
            )

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        res = layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
        )
        return res
