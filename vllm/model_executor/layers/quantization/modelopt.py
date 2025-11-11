# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    build_flashinfer_fp4_cutlass_moe_prepare_finalize,
    reorder_w1w3_to_w3w1,
    select_nvfp4_gemm_impl,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    apply_flashinfer_per_tensor_scale_fp8,
    build_flashinfer_fp8_cutlass_moe_prepare_finalize,
    flashinfer_cutlass_moe_fp8,
    get_flashinfer_moe_backend,
    is_flashinfer_supporting_global_sf,
    register_moe_scaling_factors,
    rotate_flashinfer_fp8_moe_weights,
    select_cutlass_fp8_gemm_impl,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    cutlass_fp4_supported,
    is_layer_skipped,
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    requantize_with_max_scale,
)
from vllm.model_executor.parameter import ModelWeightParameter, PerTensorScaleParameter
from vllm.scalar_type import scalar_types
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer,
    has_flashinfer_moe,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

QUANT_ALGOS = ["FP8", "NVFP4"]
KV_CACHE_QUANT_ALGOS = ["FP8"]


class ModelOptFp8Config(QuantizationConfig):
    """Config class for ModelOpt FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        kv_cache_quant_method: str | None = None,
        exclude_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.kv_cache_quant_method = kv_cache_quant_method
        self.exclude_modules = exclude_modules or []
        if is_checkpoint_fp8_serialized:
            logger.warning(
                "Detected ModelOpt fp8 checkpoint. Please note that"
                " the format is experimental and could change."
            )

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

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.exclude_modules is not None:
            self.exclude_modules = hf_to_vllm_mapper.apply_list(self.exclude_modules)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
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
                raise ValueError("Expected 'quantization' to be a dictionary in config")
            quant_method = quant_config.get("quant_algo", "")
            if not quant_method:
                raise ValueError("Missing 'quant_algo' in quantization config")
            kv_cache_quant_method = quant_config.get("kv_cache_quant_algo")
            # "exclude_modules" is the key in the legacy hf_quant_config.json
            exclude_modules = quant_config.get("exclude_modules")
        else:
            # Compressed-tensors style format:
            # {"quant_algo": "...", "quant_method": "modelopt"}
            quant_method = config.get("quant_algo", "")
            kv_cache_quant_method = config.get("kv_cache_quant_algo")
            # "ignore" is the key in config.json
            exclude_modules = config.get("ignore")

        if quant_method not in QUANT_ALGOS:
            raise ValueError(
                f"ModelOpt currently only supports: {QUANT_ALGOS} "
                "quantizations in vLLM. Please check the "
                "`hf_quant_config.json` file for your model's "
                "quant configuration."
            )
        is_checkpoint_fp8_serialized = "FP8" in quant_method

        return cls(is_checkpoint_fp8_serialized, kv_cache_quant_method, exclude_modules)

    def is_layer_excluded(self, prefix: str) -> bool:
        """
        Check if a layer should be excluded from quantization.
        Handles both exact matching (for fused layers) and substring matching.

        This method handles both regular models and multimodal models that use
        the language_model prefix. For multimodal models, it checks if the
        module name (without the language_model prefix) is in the exclude list.
        """
        if self.exclude_modules is None:
            return False

        # First check exact matching with fused layer support
        if is_layer_skipped(prefix, self.exclude_modules, self.packed_modules_mapping):
            return True

        # Then check substring matching for patterns not caught by exact match
        for module in self.exclude_modules:
            # Skip exact matches already handled above
            if module != prefix and (
                module in prefix
                or (
                    prefix.startswith("language_model.")
                    and module in prefix.removeprefix("language_model.")
                )
            ):
                return True
        return False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import (  # Avoid circular import
            Attention,
            MLAAttention,
        )

        if isinstance(layer, LinearBase):
            if self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            # Check if this is a vision model layer that should not be quantized
            if "vision_tower" in prefix or "vision_model" in prefix:
                return UnquantizedLinearMethod()
            return ModelOptFp8LinearMethod(self)
        elif isinstance(layer, (Attention, MLAAttention)):
            return ModelOptFp8KVCacheMethod(self)
        elif isinstance(layer, FusedMoE):
            return ModelOptFp8MoEMethod(self, layer)
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
            act_quant_static=True, act_quant_group_shape=GroupShape.PER_TENSOR
        )

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
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            weight_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("weight_scale", weight_scale)
            # INPUT SCALE
            scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )

            scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight
        max_w_scale = layer.weight_scale.max()
        if not (layer.weight_scale == layer.weight_scale[0]).all():
            max_w_scale, weight = requantize_with_max_scale(
                layer.weight, layer.weight_scale, layer.logical_widths
            )
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
        )


class ModelOptFp8MoEMethod(FusedMoEMethodBase):
    """MoE method for ModelOpt FP8.
    Supports loading FP8 checkpoints with static weight scale and
    activation scale.
    Args:
        quant_config: The ModelOpt quantization config.
    """

    def __init__(
        self,
        quant_config: ModelOptFp8Config,
        layer: torch.nn.Module,
    ) -> None:
        super().__init__(layer.moe_config)
        self.layer = layer
        self.quant_config = quant_config
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            cutlass_fp8_supported,
        )

        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.flashinfer_moe_backend: FlashinferMoeBackend | None = None
        if (
            envs.VLLM_USE_FLASHINFER_MOE_FP8
            and has_flashinfer_moe()
            and self.moe.is_act_and_mul
        ):
            self.flashinfer_moe_backend = get_flashinfer_moe_backend()
            logger.info_once(
                f"Using FlashInfer {self.flashinfer_moe_backend.value} kernels"
            )

    def maybe_make_prepare_finalize(
        self,
    ) -> mk.FusedMoEPrepareAndFinalize | None:
        # TRT LLM not supported with all2all yet.
        if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            return None
        elif self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            prepare_finalize = build_flashinfer_fp8_cutlass_moe_prepare_finalize(
                self.moe
            )
            logger.debug_once("%s", prepare_finalize.__class__.__name__)
            return prepare_finalize
        else:
            return super().maybe_make_prepare_finalize()

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        experts = select_cutlass_fp8_gemm_impl(
            self.moe,
            self.moe_quant_config,
        )
        logger.debug_once("Using %s", experts.__class__.__name__)
        return experts

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
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition

        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=weight_dtype,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        w2_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALES - Per-tensor scaling for ModelOpts
            # For gated MoE, allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            # For non-gated MoE, allocate 1 scale for w13.
            if self.moe.is_act_and_mul:
                w13_weight_scale_shape = (num_experts, 2)
            else:
                w13_weight_scale_shape = (num_experts, 1)
            w13_weight_scale = PerTensorScaleParameter(
                data=torch.full(
                    w13_weight_scale_shape,
                    1.0,
                    dtype=torch.float32,
                ),
                weight_loader=weight_loader,
            )
            w2_weight_scale = PerTensorScaleParameter(
                data=torch.full((num_experts,), 1.0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            # Set weight loader attributes for scales
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )

            # INPUT SCALES - Per-tensor scaling for ModelOpt
            w13_input_scale = PerTensorScaleParameter(
                data=torch.full((num_experts,), 1.0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            w2_input_scale = PerTensorScaleParameter(
                data=torch.full((num_experts,), 1.0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process FP8 MoE weights after loading from serialized checkpoint.
        Only supports pre-quantized checkpoints with FP8 weights and scales.
        """

        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)

        from vllm._custom_ops import scaled_fp8_quant
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            per_tensor_dequantize,
        )

        # Handle scale parameters
        if hasattr(layer, "w13_weight_scale") and layer.w13_weight_scale is not None:
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max of the w1 and w3 scales
            # then dequant and requant each expert.
            if (
                layer.w13_weight_scale.dim() == 2
                and layer.w13_weight_scale.shape[1] == 2
            ):
                assert self.moe.is_act_and_mul, (
                    "w13_weight_scale should have 2 elements per expert "
                    "only for gated MoE"
                )
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
                            layer.w13_weight[expert_id][
                                start : start + intermediate_size, :
                            ],
                            layer.w13_weight_scale[expert_id][shard_id],
                        )
                        # Requantize using the combined max scale

                        (
                            layer.w13_weight[expert_id][
                                start : start + intermediate_size, :
                            ],
                            _,
                        ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])

                        start += intermediate_size

                # Update the scale parameter to be per-expert
                layer.w13_weight_scale = Parameter(max_w13_scales, requires_grad=False)
            else:
                layer.w13_weight_scale = Parameter(
                    layer.w13_weight_scale.data, requires_grad=False
                )

        if hasattr(layer, "w2_weight_scale") and layer.w2_weight_scale is not None:
            layer.w2_weight_scale = Parameter(
                layer.w2_weight_scale.data, requires_grad=False
            )
        # Input scales must be equal for each expert in fp8 MoE layers.
        if hasattr(layer, "w13_input_scale") and layer.w13_input_scale is not None:
            layer.w13_input_scale = Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
        if hasattr(layer, "w2_input_scale") and layer.w2_input_scale is not None:
            layer.w2_input_scale = Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        if self.flashinfer_moe_backend is not None:
            layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
            register_moe_scaling_factors(layer)
            if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
                rotate_flashinfer_fp8_moe_weights(layer.w13_weight, layer.w2_weight)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            return None

        return fp8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            g1_alphas=(layer.w13_weight_scale * layer.w13_input_scale).squeeze(),
            w2_scale=layer.w2_weight_scale,
            g2_alphas=(layer.w2_weight_scale * layer.w2_input_scale).squeeze(),
            a1_scale=layer.w13_input_scale,
            a1_gscale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            a2_gscale=1.0 / layer.w2_input_scale,
            per_act_token_quant=False,
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
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `ModelOptFp8MoEMethod` yet."
            )

        if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            assert activation == "silu", (
                f"Expected 'silu' activation but got {activation}"
            )
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
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # Expert selection
        topk_weights, topk_ids, _ = FusedMoE.select_experts(
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
            indices_type=self.topk_indices_dtype,
        )

        if self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            assert not renormalize
            assert activation == "silu", (
                f"Expected 'silu' activation but got {activation}"
            )
            return flashinfer_cutlass_moe_fp8(
                x,
                layer,
                topk_weights,
                topk_ids,
                inplace=False,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        else:
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

            assert self.moe_quant_config is not None

            return fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                quant_config=self.moe_quant_config,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )


class ModelOptNvFp4Config(QuantizationConfig):
    """Config class for ModelOpt FP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool,
        kv_cache_quant_algo: str | None,
        exclude_modules: list[str],
        group_size: int = 16,
    ) -> None:
        super().__init__()
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected ModelOpt NVFP4 checkpoint. Please note that"
                " the format is experimental and could change in future."
            )

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

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.exclude_modules is not None:
            self.exclude_modules = hf_to_vllm_mapper.apply_list(self.exclude_modules)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
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
                raise ValueError("Expected 'quantization' to be a dictionary in config")

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
                raise ValueError(
                    f"kv_cache_quant_algo must be a string, got "
                    f"{type(kv_cache_quant_algo_raw)}"
                )

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
                    raise ValueError(
                        f"group_size must be an integer, got {type(group_size_raw)}"
                    ) from None

            # "exclude_modules" is the key in the legacy hf_quant_config.json
            exclude_modules = quant_config.get("exclude_modules", [])
            if not isinstance(exclude_modules, list):
                raise ValueError(
                    f"exclude_modules must be a list, got {type(exclude_modules)}"
                )
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
                raise ValueError(
                    f"kv_cache_quant_algo must be a string, got "
                    f"{type(kv_cache_quant_algo_raw)}"
                )

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
                    raise ValueError(
                        f"group_size must be an integer, got {type(group_size_raw)}"
                    ) from None

            # "ignore" is the key in config.json
            exclude_modules = config.get("ignore", [])
            if not isinstance(exclude_modules, list):
                raise ValueError(
                    f"exclude_modules must be a list, got {type(exclude_modules)}"
                )

        if quant_method not in QUANT_ALGOS:
            raise ValueError(
                f"ModelOpt currently only supports: {QUANT_ALGOS} "
                "quantizations in vLLM. Please check the "
                "`hf_quant_config.json` file for your model's "
                "quant configuration."
            )
        is_checkpoint_nvfp4_serialized = "NVFP4" in quant_method

        # For FP4, these fields are required
        if is_checkpoint_nvfp4_serialized and "quantization" in config:
            # Check if required fields are present in the quantization config
            quant_config = config["quantization"]
            required_fields = ["group_size", "kv_cache_quant_algo", "exclude_modules"]
            missing_fields = [
                field for field in required_fields if field not in quant_config
            ]
            if missing_fields:
                raise ValueError(
                    f"NVFP4 quantization requires the following fields in "
                    f"hf_quant_config.json: {missing_fields}"
                )

        return cls(
            is_checkpoint_nvfp4_serialized,
            kv_cache_quant_algo,
            exclude_modules,
            group_size,
        )

    def is_layer_excluded(self, prefix: str) -> bool:
        """
        Check if a layer should be excluded from quantization.
        Handles both exact matching (for fused layers) and pattern matching.
        """
        # First check exact matching with fused layer support
        if is_layer_skipped(prefix, self.exclude_modules, self.packed_modules_mapping):
            return True

        # Check regex pattern matching for patterns not caught by exact match
        import regex as re

        for pattern in self.exclude_modules:
            # Skip patterns that would be caught by exact matching
            if "*" in pattern or "." in pattern:
                regex_str = pattern.replace(".", r"\.").replace("*", r".*")
                if re.fullmatch(regex_str, prefix):
                    return True
        return False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import (  # Avoid circular import
            Attention,
            MLAAttention,
        )

        skip_layer = self.is_layer_excluded(prefix)
        if isinstance(layer, LinearBase):
            if skip_layer:
                return UnquantizedLinearMethod()
            # Check if this is a vision model layer that should not be quantized
            if "vision_tower" in prefix or "vision_model" in prefix:
                return UnquantizedLinearMethod()
            return ModelOptNvFp4LinearMethod(self)
        elif isinstance(layer, (Attention, MLAAttention)):
            return ModelOptFp8KVCacheMethod(self)
        elif isinstance(layer, FusedMoE):
            if skip_layer:
                return None
            return ModelOptNvFp4FusedMoE(self, layer.moe_config, layer)
        return None


class ModelOptFp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: ModelOptFp8Config | ModelOptNvFp4Config):
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

        self.backend = "none"
        if envs.VLLM_NVFP4_GEMM_BACKEND is None:
            if has_flashinfer():
                self.backend = "flashinfer-cutlass"
            elif cutlass_fp4_supported():
                self.backend = "cutlass"
            elif is_fp4_marlin_supported():
                self.backend = "marlin"
        elif envs.VLLM_NVFP4_GEMM_BACKEND.startswith("flashinfer-"):
            self.backend = envs.VLLM_NVFP4_GEMM_BACKEND
            assert has_flashinfer(), f"FlashInfer is required for {self.backend}"
        elif envs.VLLM_NVFP4_GEMM_BACKEND == "cutlass":
            self.backend = "cutlass"
            assert cutlass_fp4_supported(), f"Cutlass is required for {self.backend}"

        if self.backend == "none":
            raise ValueError(
                "No valid NVFP4 GEMM backend found. "
                "Please check your platform capability."
            )

        logger.info_once(f"Using {self.backend} for NVFP4 GEMM")

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
            raise ValueError(
                "NVFP4 quantization was selected, "
                " dynamic quantization is not supported."
            )
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        if input_size_per_partition % 16 != 0:
            raise ValueError(
                "Unsupported model when in features size is not multiple of 16"
            )
        # The nvfp4 weight is still represented as
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_nvfp4_serialized
            else params_dtype
        )
        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                # 2 fp4 items are packed in the input dimension
                layer.output_size_per_partition,
                layer.input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Input Weight Scale
        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_scale", input_scale)

        # Global Weight Scale
        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        # Per Block Weight Scale
        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        # global scales:
        input_scale_2 = layer.input_scale.max().to(torch.float32)
        layer.input_scale = Parameter(input_scale_2, requires_grad=False)

        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)
        layer.weight_scale_2 = Parameter(weight_scale_2, requires_grad=False)

        layer.alpha = Parameter(
            layer.input_scale * layer.weight_scale_2, requires_grad=False
        )

        # Calculate `1 / input_scale` so that we don't need to do so at runtime
        layer.input_scale_inv = Parameter(
            (1 / layer.input_scale).to(torch.float32), requires_grad=False
        )

        # Swizzle the weight blockscale.
        # contracting dimension is input dimension
        # block_size = 16;
        assert layer.weight_scale.dtype == torch.float8_e4m3fn, (
            "Weight Block scale must be represented as FP8-E4M3"
        )

        if self.backend == "marlin":
            prepare_fp4_layer_for_marlin(layer)
            del layer.alpha
            del layer.input_scale
        elif self.backend == "flashinfer-trtllm":
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight.data
            weight_scale = layer.weight_scale.data

            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            weight_scale = (
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.weight = Parameter(weight, requires_grad=False)
        else:
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backend == "marlin":
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_scale_2=layer.weight_scale_2,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        output_dtype = x.dtype
        output_shape = [x.shape[0], layer.weight.shape[0]]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_blockscale = scaled_fp4_quant(x, layer.input_scale_inv)

        # validate dtypes of quantized input, input block scale,
        # weight and weight_blockscale
        assert x_fp4.dtype == torch.uint8
        assert layer.weight.dtype == torch.uint8
        assert x_blockscale.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        mm_args = (
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
        )
        if self.backend.startswith("flashinfer-"):
            backend_name = self.backend[len("flashinfer-") :]
            out = flashinfer_scaled_fp4_mm(*mm_args, backend=backend_name)
        else:
            assert self.backend == "cutlass"
            out = cutlass_scaled_fp4_mm(*mm_args)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class ModelOptNvFp4FusedMoE(FusedMoEMethodBase):
    """
    MoE Method for FP4 Quantization.
    Args:
        quant_config: NVFP4 Quant Config
    """

    def __init__(
        self,
        quant_config: ModelOptNvFp4Config,
        moe: FusedMoEConfig,
        layer: torch.nn.Module,
    ) -> None:
        from vllm.model_executor.layers.quantization.utils.nvfp4_moe_support import (
            detect_nvfp4_moe_support,  # noqa: E501
        )

        super().__init__(moe)
        self.quant_config = quant_config
        self.layer = layer
        _nvfp4 = detect_nvfp4_moe_support(self.__class__.__name__)
        self.cutlass_nvfp4_supported = _nvfp4.cutlass_supported
        self.allow_flashinfer = _nvfp4.allow_flashinfer
        self.use_marlin = _nvfp4.use_marlin
        self.flashinfer_moe_backend = None
        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        if self.allow_flashinfer:
            self.flashinfer_moe_backend = get_flashinfer_moe_backend()
            logger.info_once(
                f"Using FlashInfer {self.flashinfer_moe_backend.value} kernels"
                " for ModelOptNvFp4FusedMoE."
            )

    def maybe_make_prepare_finalize(self) -> mk.FusedMoEPrepareAndFinalize | None:
        if self.use_marlin or (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            return None
        elif (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS
        ):
            # For now, fp4 moe only works with the flashinfer dispatcher.
            prepare_finalize = build_flashinfer_fp4_cutlass_moe_prepare_finalize(
                self.moe
            )
            logger.debug_once("%s", prepare_finalize.__class__.__name__)
            return prepare_finalize
        else:
            return super().maybe_make_prepare_finalize()

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        experts = select_nvfp4_gemm_impl(
            self.moe,
            self.moe_quant_config,
            allow_flashinfer=self.allow_flashinfer,
        )
        logger.debug_once("Using %s", experts.__class__.__name__)
        return experts

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        FP4 variants use 'weight_scale_2' pattern for per-tensor weight scales.
        """
        return True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError(
                "NVFP4 quantization was selected, "
                " dynamic quantization is not supported."
            )

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config
        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn
        weight_loader = extra_weight_attrs.get("weight_loader")
        global_num_experts = extra_weight_attrs.get("global_num_experts")
        # GEMM 1
        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        # GEMM 2
        w2_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        w13_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.quant_config.group_size,
                dtype=weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.quant_config.group_size,
                dtype=weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        w13_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(num_experts, 2, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)

        w2_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        use_global_sf = self.allow_flashinfer and is_flashinfer_supporting_global_sf(
            self.flashinfer_moe_backend
        )
        global_scale_num_experts = global_num_experts if use_global_sf else num_experts

        w13_input_scale = PerTensorScaleParameter(
            data=torch.empty(global_scale_num_experts, 2, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = PerTensorScaleParameter(
            data=torch.empty(global_scale_num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def prepare_static_weights_for_trtllm_fp4_moe(
        self,
        # args_dequant,
        # args,
        gemm1_weights,
        gemm2_weights,
        gemm1_scales_linear_fp4_bytes,
        gemm2_scales_linear_fp4_bytes,
        hidden_size,
        intermediate_size,
        num_experts,
    ):
        from flashinfer import nvfp4_block_scale_interleave
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        """Prepare quantized weights for kernel (done offline with weights)."""
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        # Convert quantized weights to proper formats
        gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2
        )  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 16
        )  # fp8 scaling factors

        gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )  # packed fp4
        gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, hidden_size, intermediate_size // 16
        )  # fp8 scaling factors

        gemm1_weights_fp4_shuffled = []
        gemm1_scales_fp4_shuffled = []
        gemm2_weights_fp4_shuffled = []
        gemm2_scales_fp4_shuffled = []
        for i in range(num_experts):
            # Calculate the permute indices for the following:
            # 1. Reorder rows of W1 and scales for fused gated activation
            # 2. Shuffle weights and scaling factors for transposed mma output
            # for both w3_w1 and w2 weights and scale factors
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_fp4_shuffled.append(
                gemm1_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm1_scales_fp4_shuffled.append(
                nvfp4_block_scale_interleave(
                    gemm1_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

            permute_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_fp4_shuffled.append(
                gemm2_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_fp4_shuffled.append(
                nvfp4_block_scale_interleave(
                    gemm2_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

        # Stack weights for all experts
        gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
        gemm1_scales_fp4_shuffled = (
            torch.stack(gemm1_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // 16)
        )
        return (
            gemm1_weights_fp4_shuffled,
            gemm1_scales_fp4_shuffled,
            gemm2_weights_fp4_shuffled,
            gemm2_scales_fp4_shuffled,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # GEMM 1 processing
        gemm1_weight = layer.w13_weight.data
        gemm1_weight_scale = layer.w13_weight_scale.data

        if self.allow_flashinfer:
            gemm1_weight, gemm1_weight_scale = reorder_w1w3_to_w3w1(
                gemm1_weight, gemm1_weight_scale, dim=-2
            )

        layer.w13_weight = Parameter(gemm1_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(gemm1_weight_scale, requires_grad=False)

        # Common processing for w13_weight_scale_2
        if not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )

        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
        layer.w13_weight_scale_2 = Parameter(w13_weight_scale_2, requires_grad=False)

        # Common processing for input scales and alphas
        use_global_sf = self.allow_flashinfer and is_flashinfer_supporting_global_sf(
            self.flashinfer_moe_backend
        )
        if use_global_sf:
            # For backends provide by Flashinfer, the input global scales are
            # shared across all experts.
            w13_input_scale = (
                layer.w13_input_scale.max().to(torch.float32).expand(layer.num_experts)
            )
        else:
            w13_input_scale = layer.w13_input_scale.max(dim=1).values.to(torch.float32)
        layer.g1_alphas = Parameter(
            (w13_input_scale * w13_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        # This is for quantization, so we need to invert it.
        layer.w13_input_scale_quant = Parameter(
            (1 / w13_input_scale).to(torch.float32), requires_grad=False
        )

        # GEMM 2 processing
        if use_global_sf:
            # For backends provide by Flashinfer, the input global scales are
            # shared across all experts.
            w2_input_scale = (
                layer.w2_input_scale.max().to(torch.float32).expand(layer.num_experts)
            )
        else:
            w2_input_scale = layer.w2_input_scale
        layer.g2_alphas = Parameter(
            (w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        # This is for quantization, so we need to invert it.
        layer.w2_input_scale_quant = Parameter(
            (1 / w2_input_scale).to(torch.float32), requires_grad=False
        )

        # TensorRT-LLM specific processing
        if (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            # Prepare static weights for TRT-LLM kernel
            # alternate: prepare_static_weight_layouts_for_trtllm_moe
            (
                gemm1_weights_fp4_shuffled,
                gemm1_scales_fp4_shuffled,
                gemm2_weights_fp4_shuffled,
                gemm2_scales_fp4_shuffled,
            ) = self.prepare_static_weights_for_trtllm_fp4_moe(
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                layer.w2_weight.size(-2),  # hidden_size
                layer.w13_weight.size(-2) // 2,  # intermediate_size
                layer.w13_weight.size(0),  # num_experts
            )
            logger.debug_once("Finished shuffling weights for TRT-LLM MOE")

            layer.gemm1_weights_fp4_shuffled = Parameter(
                gemm1_weights_fp4_shuffled, requires_grad=False
            )
            layer.gemm2_weights_fp4_shuffled = Parameter(
                gemm2_weights_fp4_shuffled, requires_grad=False
            )
            layer.gemm1_scales_fp4_shuffled = Parameter(
                gemm1_scales_fp4_shuffled, requires_grad=False
            )
            layer.gemm2_scales_fp4_shuffled = Parameter(
                gemm2_scales_fp4_shuffled, requires_grad=False
            )

            # Additional parameter needed for TRT-LLM
            layer.g1_scale_c = Parameter(
                (layer.w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
                requires_grad=False,
            )

            # Clean up weights that won't be used by TRT-LLM
            del layer.w2_weight
            del layer.w2_weight_scale
            del layer.w13_weight
            del layer.w13_weight_scale
        elif self.use_marlin:
            # Marlin processing
            prepare_moe_fp4_layer_for_marlin(layer)
            del layer.g1_alphas
            del layer.g2_alphas
            del layer.w13_input_scale_quant
            del layer.w2_input_scale_quant
        else:
            # Non-TRT-LLM processing (Cutlass or non-flashinfer)
            w13_blockscale_swizzled = swizzle_blockscale(layer.w13_weight_scale)
            layer.w13_weight_scale = Parameter(
                w13_blockscale_swizzled, requires_grad=False
            )

            w2_blockscale_swizzled = swizzle_blockscale(layer.w2_weight_scale)
            layer.w2_weight_scale = Parameter(
                w2_blockscale_swizzled, requires_grad=False
            )
            layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if (
            self.use_marlin
            or self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            return None

        return nvfp4_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            g1_alphas=layer.g1_alphas,
            g2_alphas=layer.g2_alphas,
            a1_gscale=layer.w13_input_scale_quant,
            a2_gscale=layer.w2_input_scale_quant,
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
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `ModelOptNvFp4FusedMoE` yet."
            )
        assert activation == "silu", "Only SiLU activation is supported."

        if (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            import flashinfer

            from vllm.model_executor.models.llama4 import Llama4MoE

            a1_gscale = layer.w13_input_scale_quant
            (hidden_states_fp4, hidden_states_scale_linear_fp4) = (
                flashinfer.fp4_quantize(
                    x,
                    a1_gscale,
                    is_sf_swizzled_layout=False,
                )
            )
            use_llama4_routing = (
                custom_routing_function is Llama4MoE.custom_routing_function
            )
            routing_method_type = flashinfer.RoutingMethodType.DeepSeekV3
            if use_llama4_routing:
                routing_method_type = flashinfer.RoutingMethodType.Llama4
            routing_bias = e_score_correction_bias
            if routing_bias is not None:
                routing_bias = routing_bias.to(torch.bfloat16)
            out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
                routing_logits=router_logits
                if use_llama4_routing
                else router_logits.to(torch.float32),
                routing_bias=routing_bias,
                hidden_states=hidden_states_fp4,
                hidden_states_scale=hidden_states_scale_linear_fp4.view(
                    torch.float8_e4m3fn
                ).flatten(),
                gemm1_weights=layer.gemm1_weights_fp4_shuffled.data,
                gemm1_weights_scale=layer.gemm1_scales_fp4_shuffled.data.view(
                    torch.float8_e4m3fn
                ),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=layer.gemm2_weights_fp4_shuffled.data,
                gemm2_weights_scale=layer.gemm2_scales_fp4_shuffled.data.view(
                    torch.float8_e4m3fn
                ),
                gemm2_bias=None,
                output1_scale_scalar=layer.g1_scale_c.data,
                output1_scale_gate_scalar=layer.g1_alphas.data,
                output2_scale_scalar=layer.g2_alphas.data,
                num_experts=global_num_experts,
                top_k=top_k,
                n_group=num_expert_group if num_expert_group is not None else 0,
                topk_group=topk_group if topk_group is not None else 0,
                intermediate_size=layer.intermediate_size_per_partition,
                local_expert_offset=layer.ep_rank * layer.local_num_experts,
                local_num_experts=layer.local_num_experts,
                routed_scaling_factor=None,
                tile_tokens_dim=None,
                routing_method_type=routing_method_type,
                do_finalize=True,
            )[0]
            return out

        topk_weights, topk_ids, _ = FusedMoE.select_experts(
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
            indices_type=self.topk_indices_dtype,
        )

        if self.use_marlin:
            return fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                None,
                None,
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
                expert_map=expert_map,
                workspace=layer.workspace,
            )

        elif (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS
        ):
            from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (  # noqa: E501
                flashinfer_cutlass_moe_fp4,
            )

            assert self.moe_quant_config is not None

            return flashinfer_cutlass_moe_fp4(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                quant_config=self.moe_quant_config,
                inplace=False,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        else:
            # If no modular kernel is provided, use cutlass_moe_fp4 for TP case
            # only (no EP).
            from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4

            assert self.moe_quant_config is not None
            return cutlass_moe_fp4(
                a=x,
                w1_fp4=layer.w13_weight,
                w2_fp4=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                quant_config=self.moe_quant_config,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
                # TODO: derive from arguments
                m=x.shape[0],
                n=layer.w2_weight.shape[2] * 2,
                k=x.shape[1],
                e=layer.w13_weight.shape[0],
            )
