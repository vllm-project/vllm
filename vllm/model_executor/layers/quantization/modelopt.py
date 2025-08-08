# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import Any, Callable, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    build_flashinfer_fp4_cutlass_moe_kernel,
    flashinfer_fp4_cutlass_moe_forward, reorder_w1w3_to_w3w1)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    apply_flashinfer_per_tensor_scale_fp8, rotate_flashinfer_fp8_moe_weights,
    swap_w13_to_w31)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear, is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin, prepare_moe_fp4_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape, cutlass_fp4_supported, is_layer_skipped, swizzle_blockscale)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, requantize_with_max_scale)
from vllm.model_executor.parameter import (ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.scalar_type import scalar_types
from vllm.utils import next_power_of_2
from vllm.utils.flashinfer import has_flashinfer_moe

logger = init_logger(__name__)

QUANT_ALGOS = ["FP8", "NVFP4"]
KV_CACHE_QUANT_ALGOS = ["FP8"]


class FlashinferMoeBackend(Enum):
    TENSORRT_LLM = "TensorRT-LLM"
    CUTLASS = "CUTLASS"


class ModelOptFp8Config(QuantizationConfig):
    """Config class for ModelOpt FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        kv_cache_quant_method: Optional[str] = None,
        exclude_modules: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.kv_cache_quant_method = kv_cache_quant_method
        self.exclude_modules = exclude_modules
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected ModelOpt fp8 checkpoint. Please note that"
                           " the format is experimental and could change.")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "modelopt"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        """Detect if this ModelOpt config should be used based on
        quantization config."""

        if hf_quant_cfg is None:
            return None

        # Use the community standard 'quant_method'
        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        # Only proceed if the method is explicitly "modelopt"
        if quant_method != "modelopt":
            return None

        # Look for ModelOpt-specific config structure
        if "quantization" in hf_quant_cfg:
            quant_config = hf_quant_cfg["quantization"]
            if isinstance(quant_config, dict):
                quant_algo = quant_config.get("quant_algo", "")
                if "FP8" in quant_algo:
                    return "modelopt"
        else:
            # Check for compressed-tensors style config with specific quant_algo
            quant_algo = hf_quant_cfg.get("quant_algo", "")
            if isinstance(quant_algo, str) and "FP8" in quant_algo:
                return "modelopt"

        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelOptFp8Config":
        # Handle both ModelOpt format and compressed-tensors style format
        if "quantization" in config:
            # ModelOpt format: {"quantization": {"quant_algo": "..."}}
            quant_config = cls.get_from_keys(config, ["quantization"])
            if not isinstance(quant_config, dict):
                raise ValueError(
                    "Expected 'quantization' to be a dictionary in config")
            quant_method = quant_config.get("quant_algo", "")
            if not quant_method:
                raise ValueError("Missing 'quant_algo' in quantization config")
            kv_cache_quant_method = quant_config.get("kv_cache_quant_algo")
            exclude_modules = quant_config.get("exclude_modules")
        else:
            # Compressed-tensors style format:
            # {"quant_algo": "...", "quant_method": "modelopt"}
            quant_method = config.get("quant_algo", "")
            kv_cache_quant_method = config.get("kv_cache_quant_algo")
            exclude_modules = config.get("exclude_modules")

        if quant_method not in QUANT_ALGOS:
            raise ValueError(
                f"ModelOpt currently only supports: {QUANT_ALGOS} "
                "quantizations in vLLM. Please check the "
                "`hf_quant_config.json` file for your model's "
                "quant configuration.")
        is_checkpoint_fp8_serialized = ("FP8" in quant_method)

        return cls(is_checkpoint_fp8_serialized, kv_cache_quant_method,
                   exclude_modules)

    def is_layer_excluded(self, prefix: str) -> bool:
        """
        Check if a layer should be excluded from quantization.

        This method handles both regular models and multimodal models that use
        the language_model prefix. For multimodal models, it checks if the
        module name (without the language_model prefix) is in the exclude list.
        """
        if self.exclude_modules is None:
            return False

        # Check if any excluded module matches the prefix
        for module in self.exclude_modules:
            if (module in prefix
                    or (prefix.startswith("language_model.")
                        and module in prefix.removeprefix("language_model."))):
                return True
        return False

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import
        if isinstance(layer, LinearBase):
            if self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            return ModelOptFp8LinearMethod(self)
        elif isinstance(layer, Attention):
            return ModelOptFp8KVCacheMethod(self)
        elif isinstance(layer, FusedMoE):
            return ModelOptFp8MoEMethod(self)
        return None


class ModelOptFp8LinearMethod(LinearMethodBase):
    """Linear method for Model Optimizer static quantization.
    Supports loading FP8 checkpoints with static weight scale and
    activation scale. Future support might be added for dynamic
    scales.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn datatype
        Args: quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config
        self.fp8_linear = Fp8LinearOp(
            act_quant_static=True, act_quant_group_shape=GroupShape.PER_TENSOR)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=weight_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)
            weight_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("weight_scale", weight_scale)
            # INPUT SCALE
            scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                            weight_loader=weight_loader)

            scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight
        max_w_scale = layer.weight_scale.max()
        if not (layer.weight_scale == layer.weight_scale[0]).all():
            max_w_scale, weight = requantize_with_max_scale(
                layer.weight, layer.weight_scale, layer.logical_widths)
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(),
                                      requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     input_scale=layer.input_scale,
                                     bias=bias)


class ModelOptFp8MoEMethod(FusedMoEMethodBase):
    """MoE method for ModelOpt FP8.
    Supports loading FP8 checkpoints with static weight scale and
    activation scale.
    Args:
        quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            cutlass_fp8_supported)
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.flashinfer_moe_enabled = False
        if envs.VLLM_USE_FLASHINFER_MOE_FP8 and has_flashinfer_moe():
            logger.info_once(
                "Using FlashInfer MoE FP8 kernels for ModelOptFp8MoEMethod.")
            self.flashinfer_moe_enabled = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        # Use FP8 dtype if checkpoint is serialized
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)
        weight_loader = extra_weight_attrs.get("weight_loader")

        w13_weight = ModelWeightParameter(
            data=torch.empty(num_experts,
                             2 * intermediate_size_per_partition,
                             hidden_size,
                             dtype=weight_dtype),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        w2_weight = ModelWeightParameter(
            data=torch.empty(num_experts,
                             hidden_size,
                             intermediate_size_per_partition,
                             dtype=weight_dtype),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALES - Per-tensor scaling for ModelOpts
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = PerTensorScaleParameter(
                data=torch.full(
                    (num_experts, 2),
                    1.0,
                    dtype=torch.float32,
                ),
                weight_loader=weight_loader,
            )
            w2_weight_scale = PerTensorScaleParameter(
                data=torch.full((num_experts, ), 1.0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            # Set weight loader attributes for scales
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})

            # INPUT SCALES - Per-tensor scaling for ModelOpt
            w13_input_scale = PerTensorScaleParameter(
                data=torch.full((num_experts, ), 1.0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            w2_input_scale = PerTensorScaleParameter(
                data=torch.full((num_experts, ), 1.0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process FP8 MoE weights after loading from serialized checkpoint.
        Only supports pre-quantized checkpoints with FP8 weights and scales.
        """

        layer.w13_weight = Parameter(layer.w13_weight.data,
                                     requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)

        from vllm._custom_ops import scaled_fp8_quant
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            per_tensor_dequantize)

        # Handle scale parameters
        if hasattr(layer,
                   "w13_weight_scale") and layer.w13_weight_scale is not None:
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max of the w1 and w3 scales
            # then dequant and requant each expert.
            if layer.w13_weight_scale.dim() == 2:

                # Get the maximum scale across w1 and w3 for each expert
                max_w13_scales = layer.w13_weight_scale.max(dim=1).values

                # Requantize each expert's weights using the combined scale
                # w13_weight (num_experts, 2 * intermediate_size, hidden_size)
                # where the first intermediate_size rows are w1, the next are w3
                intermediate_size = layer.w13_weight.shape[1] // 2
                for expert_id in range(layer.w13_weight.shape[0]):
                    start = 0
                    for shard_id in range(2):  # w1 and w3
                        # Dequantize using the original scale for this shard
                        dq_weight = per_tensor_dequantize(
                            layer.w13_weight[expert_id][start:start +
                                                        intermediate_size, :],
                            layer.w13_weight_scale[expert_id][shard_id],
                        )
                        # Requantize using the combined max scale

                        (
                            layer.w13_weight[expert_id][start:start +
                                                        intermediate_size, :],
                            _,
                        ) = scaled_fp8_quant(dq_weight,
                                             max_w13_scales[expert_id])

                        start += intermediate_size

                # Update the scale parameter to be per-expert
                layer.w13_weight_scale = Parameter(max_w13_scales,
                                                   requires_grad=False)
            else:
                layer.w13_weight_scale = Parameter(layer.w13_weight_scale.data,
                                                   requires_grad=False)

        if hasattr(layer,
                   "w2_weight_scale") and layer.w2_weight_scale is not None:
            layer.w2_weight_scale = Parameter(layer.w2_weight_scale.data,
                                              requires_grad=False)
        # Input scales must be equal for each expert in fp8 MoE layers.
        if hasattr(layer,
                   "w13_input_scale") and layer.w13_input_scale is not None:
            layer.w13_input_scale = Parameter(layer.w13_input_scale.max(),
                                              requires_grad=False)
        if hasattr(layer,
                   "w2_input_scale") and layer.w2_input_scale is not None:
            layer.w2_input_scale = Parameter(layer.w2_input_scale.max(),
                                             requires_grad=False)

        if self.flashinfer_moe_enabled:
            layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
            rotate_flashinfer_fp8_moe_weights(layer.w13_weight,
                                              layer.w2_weight)

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
            raise NotImplementedError(
                "EPLB not supported for `ModelOptFp8MoEMethod` yet.")

        if self.flashinfer_moe_enabled:
            assert activation == 'silu'
            assert not renormalize
            return apply_flashinfer_per_tensor_scale_fp8(
                layer=layer,
                hidden_states=x,
                router_logits=router_logits,
                routing_bias=e_score_correction_bias,
                global_num_experts=global_num_experts,
                top_k=top_k,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                apply_router_weight_on_input=apply_router_weight_on_input)

        # Expert selection
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts)
        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            use_fp8_w8a8=True,
            per_channel_quant=False,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


class ModelOptNvFp4Config(QuantizationConfig):
    """Config class for ModelOpt FP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool,
        kv_cache_quant_algo: Optional[str],
        exclude_modules: list[str],
        group_size: int = 16,
    ) -> None:
        super().__init__()
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected ModelOpt NVFP4 checkpoint. Please note that"
                " the format is experimental and could change in future.")

            self.group_size = group_size
            self.kv_cache_quant_algo = kv_cache_quant_algo
            self.exclude_modules = exclude_modules

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "modelopt_fp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        """Detect if this ModelOpt FP4 config should be used based on
        quantization config."""
        if hf_quant_cfg is None:
            return None

        # Use the community standard 'quant_method'
        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        # Only proceed if the method is explicitly "modelopt"
        if quant_method != "modelopt":
            return None

        # Look for ModelOpt-specific config structure
        if "quantization" in hf_quant_cfg:
            quant_config = hf_quant_cfg["quantization"]
            if isinstance(quant_config, dict):
                quant_algo = quant_config.get("quant_algo", "")
                if "NVFP4" in quant_algo:
                    return "modelopt_fp4"
        else:
            # Check for compressed-tensors style config with specific
            # quant_algo field
            quant_algo = hf_quant_cfg.get("quant_algo", "")
            if isinstance(quant_algo, str) and "FP4" in quant_algo.upper():
                return "modelopt_fp4"

        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelOptNvFp4Config":
        # Handle both traditional ModelOpt format and compressed-tensors
        # style format
        if "quantization" in config:
            # Traditional ModelOpt format:
            # {"quantization": {"quant_algo": "..."}}
            quant_config = cls.get_from_keys(config, ["quantization"])
            if not isinstance(quant_config, dict):
                raise ValueError(
                    "Expected 'quantization' to be a dictionary in config")

            quant_method = quant_config.get("quant_algo", "")
            if not quant_method:
                raise ValueError("Missing 'quant_algo' in quantization config")

            # Handle kv_cache_quant_algo with proper type validation
            kv_cache_quant_algo_raw = quant_config.get("kv_cache_quant_algo")
            if kv_cache_quant_algo_raw is None:
                # No KV cache quantization by default
                kv_cache_quant_algo = None
            elif isinstance(kv_cache_quant_algo_raw, str):
                kv_cache_quant_algo = kv_cache_quant_algo_raw
            else:
                raise ValueError(f"kv_cache_quant_algo must be a string, got "
                                 f"{type(kv_cache_quant_algo_raw)}")

            # Handle group_size with proper type validation
            group_size_raw = quant_config.get("group_size")
            if group_size_raw is None:
                group_size = 16  # Default value
            elif isinstance(group_size_raw, int):
                group_size = group_size_raw
            else:
                try:
                    group_size = int(group_size_raw)
                except (ValueError, TypeError):
                    raise ValueError(f"group_size must be an integer, got "
                                     f"{type(group_size_raw)}") from None

            exclude_modules = quant_config.get("exclude_modules", [])
            if not isinstance(exclude_modules, list):
                raise ValueError(f"exclude_modules must be a list, got "
                                 f"{type(exclude_modules)}")
        else:
            # Compressed-tensors style format:
            # {"quant_algo": "...", "quant_method": "modelopt"}
            quant_method = config.get("quant_algo", "")

            # Handle kv_cache_quant_algo with proper type validation
            kv_cache_quant_algo_raw = config.get("kv_cache_quant_algo")
            if kv_cache_quant_algo_raw is None:
                # No KV cache quantization by default
                kv_cache_quant_algo = None
            elif isinstance(kv_cache_quant_algo_raw, str):
                kv_cache_quant_algo = kv_cache_quant_algo_raw
            else:
                raise ValueError(f"kv_cache_quant_algo must be a string, got "
                                 f"{type(kv_cache_quant_algo_raw)}")

            # Handle group_size with proper type validation
            group_size_raw = config.get("group_size")
            if group_size_raw is None:
                group_size = 16  # Default value
            elif isinstance(group_size_raw, int):
                group_size = group_size_raw
            else:
                try:
                    group_size = int(group_size_raw)
                except (ValueError, TypeError):
                    raise ValueError(f"group_size must be an integer, got "
                                     f"{type(group_size_raw)}") from None

            exclude_modules = config.get("exclude_modules", [])
            if not isinstance(exclude_modules, list):
                raise ValueError(f"exclude_modules must be a list, got "
                                 f"{type(exclude_modules)}")

        if quant_method not in QUANT_ALGOS:
            raise ValueError(
                f"ModelOpt currently only supports: {QUANT_ALGOS} "
                "quantizations in vLLM. Please check the "
                "`hf_quant_config.json` file for your model's "
                "quant configuration.")
        is_checkpoint_nvfp4_serialized = ("NVFP4" in quant_method)

        # For FP4, these fields are required
        if is_checkpoint_nvfp4_serialized and "quantization" in config:
            # Check if required fields are present in the quantization config
            quant_config = config["quantization"]
            required_fields = [
                "group_size", "kv_cache_quant_algo", "exclude_modules"
            ]
            missing_fields = [
                field for field in required_fields if field not in quant_config
            ]
            if missing_fields:
                raise ValueError(
                    f"NVFP4 quantization requires the following fields in "
                    f"hf_quant_config.json: {missing_fields}")

        return cls(is_checkpoint_nvfp4_serialized, kv_cache_quant_algo,
                   exclude_modules, group_size)

    def is_layer_excluded(self, prefix: str,
                          exclude_modules: list[str]) -> bool:
        import regex as re
        for pattern in exclude_modules:
            regex_str = pattern.replace('.', r'\.').replace('*', r'.*')
            if re.fullmatch(regex_str, prefix):
                return True
        return False

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import
        if isinstance(layer, LinearBase):
            if (is_layer_skipped(prefix, self.exclude_modules)
                    or self.is_layer_excluded(prefix, self.exclude_modules)):
                return UnquantizedLinearMethod()
            return ModelOptNvFp4LinearMethod(self)
        elif isinstance(layer, Attention):
            return ModelOptFp8KVCacheMethod(self)
        elif isinstance(layer, FusedMoE):
            return ModelOptNvFp4FusedMoE(self)
        return None


class ModelOptFp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Union[ModelOptFp8Config,
                                           ModelOptNvFp4Config]):
        super().__init__(quant_config)


class ModelOptNvFp4LinearMethod(LinearMethodBase):
    """Linear method for Model Optimizer NVFP4.
    Supports loading NVFP4 checkpoints with the following structure:

    input_scale: torch.float32, scalar ,
    weight: NVFP4(represented as byte) Shape: [1, X, y/2]
    weight_scale: FP8-E4M3, Shape: [X, Y], aka per block scale,
    weight_scale_2: torch.float32, scalar,
    Args: quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptNvFp4Config) -> None:
        self.quant_config = quant_config
        self.cutlass_nvfp4_supported = cutlass_fp4_supported()
        self.use_marlin = False

        if not self.cutlass_nvfp4_supported:
            if is_fp4_marlin_supported():
                self.use_marlin = True
            else:
                raise ValueError("Current platform does not support NVFP4"
                                 " quantization. Please use Blackwell and"
                                 " above.")

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError("NVFP4 quantization was selected, "
                             " dynamic quantization is not supported.")
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        if (input_size_per_partition % 16 != 0):
            raise ValueError("Unsupported model when in features size is "
                             "not multiple of 16")
        # The nvfp4 weight is still represented as
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_nvfp4_serialized
                        else params_dtype)
        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                # 2 fp4 items are packed in the input dimension
                layer.output_size_per_partition,
                layer.input_size_per_partition // 2,
                dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # Input Weight Scale
        input_scale = PerTensorScaleParameter(data=torch.empty(
            len(output_partition_sizes), dtype=torch.float32),
                                              weight_loader=weight_loader)
        layer.register_parameter("input_scale", input_scale)

        # Global Weight Scale
        weight_scale_2 = PerTensorScaleParameter(data=torch.empty(
            len(output_partition_sizes), dtype=torch.float32),
                                                 weight_loader=weight_loader)
        layer.register_parameter("weight_scale_2", weight_scale_2)

        # Per Block Weight Scale
        weight_scale = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition // self.quant_config.group_size,
            dtype=weight_dtype,
        ),
                                            input_dim=1,
                                            output_dim=0,
                                            weight_loader=weight_loader)

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:

        # global scales:
        input_scale_2 = layer.input_scale.max().to(torch.float32)
        layer.input_scale = Parameter(input_scale_2, requires_grad=False)

        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)
        layer.weight_scale_2 = Parameter(weight_scale_2, requires_grad=False)

        layer.alpha = Parameter(layer.input_scale * layer.weight_scale_2,
                                requires_grad=False)

        # Swizzle the weight blockscale.
        # contracting dimension is input dimension
        # block_size = 16;
        assert (layer.weight_scale.dtype == torch.float8_e4m3fn), (
            "Weight Block scale must be represented as FP8-E4M3")
        swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)

        layer.weight_scale_swizzled = Parameter(swizzled_weight_scale,
                                                requires_grad=False)
        layer.weight = Parameter(layer.weight.data, requires_grad=False)

        if self.use_marlin:
            prepare_fp4_layer_for_marlin(layer)
            del layer.alpha
            del layer.input_scale
            del layer.weight_scale_swizzled

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_marlin:
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_scale_2=layer.weight_scale_2,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias)

        output_dtype = x.dtype
        output_shape = [x.shape[0], layer.weight.shape[0]]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        s_quant = 1 / layer.input_scale
        x_fp4, x_blockscale = scaled_fp4_quant(x, s_quant)

        # validate dtypes of quantized input, input block scale,
        # weight and weight_blockscale
        assert (x_fp4.dtype == torch.uint8)
        assert (layer.weight.dtype == torch.uint8)
        assert (x_blockscale.dtype == torch.float8_e4m3fn)
        assert (layer.weight_scale_swizzled.dtype == torch.float8_e4m3fn)
        assert (layer.alpha.dtype == torch.float32)

        out = cutlass_scaled_fp4_mm(x_fp4, layer.weight, x_blockscale,
                                    layer.weight_scale_swizzled, layer.alpha,
                                    output_dtype)
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


def _get_tile_tokens_dim(num_tokens: int, top_k: int, num_experts: int) -> int:
    # Guess tokens per expert assuming perfect expert distribution first.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    return tile_tokens_dim


class ModelOptNvFp4FusedMoE(FusedMoEMethodBase):
    """
    MoE Method for FP4 Quantization.
    Args:
        quant_config: NVFP4 Quant Config
    """

    def __init__(self, quant_config: ModelOptNvFp4Config) -> None:
        self.quant_config = quant_config
        from vllm.model_executor.layers.quantization.utils.nvfp4_moe_support import (  # noqa: E501
            detect_nvfp4_moe_support)
        _nvfp4 = detect_nvfp4_moe_support(self.__class__.__name__)
        self.cutlass_nvfp4_supported = _nvfp4.cutlass_supported
        self.allow_flashinfer = _nvfp4.allow_flashinfer
        self.use_marlin = _nvfp4.use_marlin
        self.flashinfer_moe_backend = None

        if self.allow_flashinfer:
            flashinfer_moe_backend = envs.VLLM_FLASHINFER_MOE_BACKEND
            if flashinfer_moe_backend == "throughput":
                self.flashinfer_moe_backend = FlashinferMoeBackend.CUTLASS
                logger.info_once("Using FlashInfer CUTLASS kernels for "
                                 "ModelOptNvFp4FusedMoE.")
            elif flashinfer_moe_backend == "latency":
                self.flashinfer_moe_backend = FlashinferMoeBackend.TENSORRT_LLM
                logger.info_once("Using FlashInfer TensorRT-LLM kernels for "
                                 "ModelOptNvFp4FusedMoE.")
            else:
                allowed_backends = ["throughput", "latency"]
                raise ValueError(
                    f"Unknown flashinfer moe backend: {flashinfer_moe_backend}"
                    f" expected one of {allowed_backends}")

        self.fused_experts: Optional[
            mk.FusedMoEModularKernel] = None  # type: ignore[assignment]

    def maybe_swap_experts_impl(
        self,
        moe_parallel_config: FusedMoEParallelConfig,
    ):
        if not self.allow_flashinfer:
            return
        self.fused_experts = build_flashinfer_fp4_cutlass_moe_kernel(
            moe_parallel_config)

    # This method update self.fused_experts
    # only prepare_finalize is not None call select_gemm_impl
    # so when native cutlass fp4, fused_expert is in fuse_moe.py fused_expert
    # when it's not called(TP case), we still have 2 kernels to use.
    def select_gemm_impl(self, prepare_finalize,
                         moe) -> mk.FusedMoEPermuteExpertsUnpermute:

        assert moe is not None and prepare_finalize is not None
        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (  # noqa: E501
            select_nvfp4_gemm_impl)

        return select_nvfp4_gemm_impl(self.allow_flashinfer, moe, logger)

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        FP4 variants use 'weight_scale_2' pattern for per-tensor weight scales.
        """
        return True

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError("NVFP4 quantization was selected, "
                             " dynamic quantization is not supported.")

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config
        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn
        weight_loader = extra_weight_attrs.get("weight_loader")
        # GEMM 1
        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                dtype=weight_dtype),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader)
        layer.register_parameter("w13_weight", w13_weight)

        # GEMM 2
        w2_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=weight_dtype),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader)
        layer.register_parameter("w2_weight", w2_weight)

        w13_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.quant_config.group_size,
                dtype=weight_scale_dtype),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition //
                self.quant_config.group_size,
                dtype=weight_scale_dtype),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})

        w13_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(num_experts, 2, dtype=torch.float32),
            weight_loader=weight_loader)
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)

        w2_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(num_experts, dtype=torch.float32),
            weight_loader=weight_loader)
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})

        w13_input_scale = PerTensorScaleParameter(data=torch.empty(
            num_experts, 2, dtype=torch.float32),
                                                  weight_loader=weight_loader)
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = PerTensorScaleParameter(data=torch.empty(
            num_experts, dtype=torch.float32),
                                                 weight_loader=weight_loader)
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def prepare_static_weight_layouts_for_trtllm_moe(
        self,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm1_scales_linear_fp4_bytes: torch.Tensor,
        gemm2_scales_linear_fp4_bytes: torch.Tensor,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare quantized weights for kernel (done offline with weights)."""
        from flashinfer import (reorder_rows_for_gated_act_gemm,
                                shuffle_matrix_a, shuffle_matrix_sf_a)
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        # Convert quantized weights to proper formats
        gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2)  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn).reshape(num_experts, 2 * intermediate_size,
                                         hidden_size //
                                         16)  # fp8 scaling factors

        gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2)  # packed fp4
        gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn).reshape(num_experts, hidden_size,
                                         intermediate_size //
                                         16)  # fp8 scaling factors

        # Reorder rows of W1 and scales for fused gated activation
        gemm1_weights_fp4_interleaved = []
        gemm1_scales_fp4_interleaved = []
        for i in range(num_experts):
            gemm1_weights_fp4_interleaved.append(
                reorder_rows_for_gated_act_gemm(gemm1_weights_fp4[i].clone()))
            gemm1_scales_fp4_interleaved.append(
                reorder_rows_for_gated_act_gemm(
                    gemm1_scales_linear_fp4[i].clone()))

        # Stack weights and scales for all experts
        gemm1_weights_fp4_interleaved = torch.stack(
            gemm1_weights_fp4_interleaved).reshape(num_experts,
                                                   2 * intermediate_size,
                                                   hidden_size // 2)
        gemm1_scales_fp4_interleaved = torch.stack(
            gemm1_scales_fp4_interleaved).reshape(num_experts,
                                                  2 * intermediate_size,
                                                  hidden_size // 16)

        # Shuffle weights and scaling factors for transposed mma output
        gemm1_weights_fp4_shuffled = []
        gemm1_scales_fp4_shuffled = []
        gemm2_weights_fp4_shuffled = []
        gemm2_scales_fp4_shuffled = []
        for i in range(num_experts):
            gemm1_weights_fp4_shuffled.append(
                shuffle_matrix_a(
                    gemm1_weights_fp4_interleaved[i].view(torch.uint8),
                    epilogue_tile_m))
            gemm1_scales_fp4_shuffled.append(
                shuffle_matrix_sf_a(
                    gemm1_scales_fp4_interleaved[i].view(torch.uint8),
                    epilogue_tile_m))

            gemm2_weights_fp4_shuffled.append(
                shuffle_matrix_a(gemm2_weights_fp4[i].view(torch.uint8),
                                 epilogue_tile_m))
            gemm2_scales_fp4_shuffled.append(
                shuffle_matrix_sf_a(
                    gemm2_scales_linear_fp4[i].view(torch.uint8),
                    epilogue_tile_m))

        # Stack weights for all experts
        gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
        gemm1_scales_fp4_shuffled = (
            torch.stack(gemm1_scales_fp4_shuffled).view(
                torch.float8_e4m3fn).reshape(num_experts,
                                             2 * intermediate_size,
                                             hidden_size // 16))

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled).view(
                torch.float8_e4m3fn).reshape(num_experts, hidden_size,
                                             intermediate_size // 16))
        return (gemm1_weights_fp4_shuffled, gemm1_scales_fp4_shuffled,
                gemm2_weights_fp4_shuffled, gemm2_scales_fp4_shuffled)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # GEMM 1 processing
        gemm1_weight = layer.w13_weight.data
        gemm1_weight_scale = layer.w13_weight_scale.data

        if self.allow_flashinfer:
            gemm1_weight, gemm1_weight_scale = reorder_w1w3_to_w3w1(
                gemm1_weight, gemm1_weight_scale, dim=-2)

        layer.w13_weight = Parameter(gemm1_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(gemm1_weight_scale,
                                           requires_grad=False)

        # Common processing for w13_weight_scale_2
        if not torch.allclose(layer.w13_weight_scale_2[:, 0],
                              layer.w13_weight_scale_2[:, 1]):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected.")

        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
        layer.w13_weight_scale_2 = Parameter(w13_weight_scale_2,
                                             requires_grad=False)

        # Common processing for input scales and alphas
        w13_input_scale = layer.w13_input_scale.max(dim=1).values.to(
            torch.float32)
        layer.g1_alphas = Parameter(
            (w13_input_scale * w13_weight_scale_2).to(torch.float32),
            requires_grad=False)

        # This is for quantization, so we need to invert it.
        layer.w13_input_scale_quant = Parameter(
            (1 / w13_input_scale).to(torch.float32), requires_grad=False)

        # GEMM 2 processing
        layer.g2_alphas = Parameter(
            (layer.w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False)

        # This is for quantization, so we need to invert it.
        layer.w2_input_scale_quant = Parameter(
            (1 / layer.w2_input_scale).to(torch.float32), requires_grad=False)

        # TensorRT-LLM specific processing
        if self.allow_flashinfer and \
            self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            # Prepare static weights for TRT-LLM kernel
            (gemm1_weights_fp4_shuffled, gemm1_scales_fp4_shuffled,
             gemm2_weights_fp4_shuffled, gemm2_scales_fp4_shuffled
             ) = self.prepare_static_weight_layouts_for_trtllm_moe(
                 layer.w13_weight,
                 layer.w2_weight,
                 layer.w13_weight_scale,
                 layer.w2_weight_scale,
                 layer.w2_weight.size(-2),  # hidden_size
                 layer.w13_weight.size(-2) // 2,  # intermediate_size
                 layer.w13_weight.size(0),  # num_experts
             )

            layer.gemm1_weights_fp4_shuffled = Parameter(
                gemm1_weights_fp4_shuffled, requires_grad=False)
            layer.gemm2_weights_fp4_shuffled = Parameter(
                gemm2_weights_fp4_shuffled, requires_grad=False)
            layer.gemm1_scales_fp4_shuffled = Parameter(
                gemm1_scales_fp4_shuffled, requires_grad=False)
            layer.gemm2_scales_fp4_shuffled = Parameter(
                gemm2_scales_fp4_shuffled, requires_grad=False)

            # Additional parameter needed for TRT-LLM
            layer.g1_scale_c = Parameter(
                (layer.w2_input_scale_quant * layer.g1_alphas).to(
                    torch.float32),
                requires_grad=False,
            )

            # Clean up weights that won't be used by TRT-LLM
            del layer.w2_weight
            del layer.w2_weight_scale
            del layer.w13_weight
            del layer.w13_weight_scale
        else:
            # Non-TRT-LLM processing (Cutlass or non-flashinfer)
            assert (layer.w13_weight_scale.shape[2] % 16 == 0), (
                "Expected weight_scale.dim(1) to be divisible by 16")
            assert (layer.w13_weight_scale.dtype == torch.float8_e4m3fn), (
                "Weight Blockscale must be represented as FP8-E4M3")
            w13_blockscale_swizzled = swizzle_blockscale(
                layer.w13_weight_scale)
            layer.w13_blockscale_swizzled = Parameter(w13_blockscale_swizzled,
                                                      requires_grad=False)

            assert (layer.w2_weight_scale.shape[2] % 16 == 0), (
                "Expected weight_scale.dim(1) to be divisible by 16")
            assert (layer.w2_weight_scale.dtype == torch.float8_e4m3fn), (
                "Weight Blockscale must be represented as FP8-E4M3")
            w2_blockscale_swizzled = swizzle_blockscale(layer.w2_weight_scale)
            layer.w2_blockscale_swizzled = Parameter(w2_blockscale_swizzled,
                                                     requires_grad=False)
            layer.w2_weight = Parameter(layer.w2_weight.data,
                                        requires_grad=False)

        if self.use_marlin:
            prepare_moe_fp4_layer_for_marlin(layer)
            del layer.g1_alphas
            del layer.g2_alphas
            del layer.w13_input_scale_quant
            del layer.w2_input_scale_quant
            del layer.w13_blockscale_swizzled
            del layer.w2_blockscale_swizzled

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
    ):
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `ModelOptNvFp4FusedMoE` yet.")
        assert activation == "silu", "Only SiLU activation is supported."

        if self.allow_flashinfer and \
            self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            import flashinfer

            from vllm.model_executor.models.llama4 import Llama4MoE

            a1_gscale = layer.w13_input_scale_quant
            (hidden_states_fp4,
             hidden_states_scale_linear_fp4) = flashinfer.fp4_quantize(
                 x,
                 a1_gscale,
                 is_sf_swizzled_layout=False,
             )
            use_llama4_routing = \
                custom_routing_function is Llama4MoE.custom_routing_function
            routing_method_type = flashinfer.RoutingMethodType.DeepSeekV3
            if use_llama4_routing:
                routing_method_type = flashinfer.RoutingMethodType.Llama4
            out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
                routing_logits=router_logits
                if use_llama4_routing else router_logits.to(torch.float32),
                routing_bias=e_score_correction_bias,
                hidden_states=hidden_states_fp4,
                hidden_states_scale=hidden_states_scale_linear_fp4.view(
                    torch.float8_e4m3fn).flatten(),
                gemm1_weights=layer.gemm1_weights_fp4_shuffled.data,
                gemm1_weights_scale=layer.gemm1_scales_fp4_shuffled.data.view(
                    torch.float8_e4m3fn),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=layer.gemm2_weights_fp4_shuffled.data,
                gemm2_weights_scale=layer.gemm2_scales_fp4_shuffled.data.view(
                    torch.float8_e4m3fn),
                gemm2_bias=None,
                output1_scale_scalar=layer.g1_scale_c.data,
                output1_scale_gate_scalar=layer.g1_alphas.data,
                output2_scale_scalar=layer.g2_alphas.data,
                num_experts=global_num_experts,
                top_k=top_k,
                n_group=num_expert_group,
                topk_group=topk_group,
                intermediate_size=layer.intermediate_size_per_partition,
                local_expert_offset=layer.ep_rank * layer.local_num_experts,
                local_num_experts=layer.local_num_experts,
                routed_scaling_factor=None,
                tile_tokens_dim=_get_tile_tokens_dim(x.shape[0], top_k,
                                                     layer.local_num_experts),
                routing_method_type=routing_method_type,
                do_finalize=True,
            )[0]
            return out

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)

        if self.use_marlin:
            return torch.ops.vllm.fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                global_scale1=layer.w13_weight_scale_2,
                global_scale2=layer.w2_weight_scale_2,
                quant_type_id=scalar_types.float4_e2m1f.id,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map)

        if self.fused_experts is None:
            # If no modular kernel is provided, use cutlass_moe_fp4 for TP case
            # only (no EP).
            from vllm.model_executor.layers.fused_moe.cutlass_moe import (
                cutlass_moe_fp4)
            out = cutlass_moe_fp4(
                a=x,
                w1_fp4=layer.w13_weight,
                w2_fp4=layer.w2_weight,
                w1_blockscale=layer.w13_blockscale_swizzled,
                w2_blockscale=layer.w2_blockscale_swizzled,
                g1_alphas=layer.g1_alphas,
                g2_alphas=layer.g2_alphas,
                a1_gscale=layer.w13_input_scale_quant,
                a2_gscale=layer.w2_input_scale_quant,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                m=x.shape[0],
                n=layer.w2_weight.shape[2] * 2,
                k=x.shape[1],
                e=layer.w13_weight.shape[0],
                device=x.device,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input)
        else:
            assert self.allow_flashinfer and \
               self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS
            out = flashinfer_fp4_cutlass_moe_forward(
                self.fused_experts,
                layer,
                x,
                topk_weights,
                topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        return out
