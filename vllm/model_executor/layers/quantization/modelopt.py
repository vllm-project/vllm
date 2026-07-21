# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config_or_none
from vllm.config.quantization import QuantSpec
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MarlinNvFp4LinearKernel,
    NvFp4LinearLayerConfig,
    init_fp8_linear_kernel,
    init_mxfp8_linear_kernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    select_mxfp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format,
    is_global_sf_supported_for_nvfp4_backend,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    expose_input_quant_key,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_channel_strategy,
    process_fp8_weight_tensor_strategy_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FP4_DTYPE,
    QuantKey,
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    requantize_with_max_scale,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

QUANT_ALGOS = [
    # FP8 (per-tensor weight + optional static activation scale).
    "FP8",
    # FP8 per-channel weight scale + per-token activation scale.
    "FP8_PER_CHANNEL_PER_TOKEN",
    # FP8 per-block weight-only (ModelOpt may emit this as lowercase).
    "FP8_PB_WO",
    # NVFP4 W4A4 (4-bit float weights AND 4-bit float activations).
    "NVFP4",
    # W4A16 NVFP4 (4-bit float weights, fp16/bf16 activations).
    "W4A16_NVFP4",
    # MXFP8
    "MXFP8",
    # MIXED_PRECISION,
    "MIXED_PRECISION",
]
KV_CACHE_QUANT_ALGOS = ["FP8", "NVFP4"]


class ModelOptKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 or NVFP4 checkpoints.
    """

    def __init__(self, quant_config: "ModelOptQuantConfigBase"):
        super().__init__(quant_config)


class ModelOptQuantConfigBase(QuantizationConfig):
    FusedMoEMethodCls: type = FusedMoEMethodBase
    KVCacheMethodCls: type = BaseKVCacheMethod

    def __init__(
        self,
        exclude_modules: list[str],
    ):
        super().__init__()
        self.exclude_modules: list[str] = exclude_modules

    def linear_algo(self) -> str:
        """ModelOpt quant-algo string used to resolve linear layers.

        Fed to ``resolve()`` to build the ``QuantSpec`` for the generic
        ``ModelOptLinearMethod``. Overridden per homogeneous config; the mixed
        config resolves the algo per-prefix and does not use this.
        """
        raise NotImplementedError

    def is_layer_excluded(self, prefix: str) -> bool:
        """
        Check if a layer should be excluded from quantization.

        Handles both exact matching (for fused layers) and ModelOpt wildcard matching.

        The ModelOpt exclude_modules list is a list of wildcards.
        """
        if len(self.exclude_modules) == 0:
            return False

        # First check exact matching with fused layer support
        if is_layer_skipped(prefix, self.exclude_modules, self.packed_modules_mapping):
            return True

        # TODO: This special hard coded logic is not needed for quantized checkpoints
        # generated by ModelOpt >= 0.39.0 where they are handled natually by the
        # exclude_modules config. But need to keep them for loading quantized
        # checkpoints generated by older versions. Then check substring matching
        # for patterns not caught by exact match
        for exclude_module in self.exclude_modules:
            # Skip exact matches already handled above
            if exclude_module != prefix and (
                exclude_module in prefix
                or (
                    prefix.startswith("language_model.")
                    and exclude_module in prefix.removeprefix("language_model.")
                )
            ):
                return True

        # modelopt exclude modules are not simple strings, they are wildcards
        for wildcard_pattern in self.exclude_modules:
            if fnmatch(prefix, wildcard_pattern):
                return True

        return False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        # handle kv-cache first so we can focus only on weight quantization thereafter
        if isinstance(layer, (Attention, MLAAttention)):
            return self.KVCacheMethodCls(self)

        # handle exclusion
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None

        # TODO: This special hard coded logic is not needed for quantized checkpoints
        # generated by ModelOpt >= 0.39.0 where they are handled natually by the
        # exclude_modules config. But need to keep them for loading quantized
        # checkpoints generated by older versions. Then check substring matching
        # for patterns not caught by exact match
        if (
            "vision_tower" in prefix
            or "vision_model" in prefix
            or "vit_large_projector" in prefix
        ):
            return UnquantizedLinearMethod()

        # now, the layer is quantized, handle it here
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return build_linear_method(self, self.linear_algo(), prefix)
        elif isinstance(layer, RoutedExperts):
            quant_method = self.FusedMoEMethodCls(
                quant_config=self, moe_config=layer.moe_config
            )
            if getattr(quant_method, "backend", "") == "marlin":
                quant_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
            return quant_method

        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if len(self.exclude_modules) > 0:
            # This is a workaround for the weights remapping issue:
            # https://github.com/vllm-project/vllm/issues/28072
            # Right now, the Nvidia ModelOpt library use just one wildcard pattern:
            #        module_path*
            # It gets applied if the whole tree of modules rooted at module_path
            # is not quantized. Here we replace such pattern by 2 patterns that are
            # collectively equivalent to the original pattern:
            #        module_path
            #        module_path.*
            new_exclude_modules = []
            for exclude in self.exclude_modules:
                if len(exclude) >= 2 and exclude[-1] == "*" and exclude[-2] != ".":
                    new_exclude_modules.append(exclude[:-1])
                    new_exclude_modules.append(exclude[:-1] + ".*")
                else:
                    new_exclude_modules.append(exclude)

            self.exclude_modules = hf_to_vllm_mapper.apply_list(new_exclude_modules)

    @staticmethod
    def _extract_modelopt_quant_algo(
        hf_quant_cfg: dict[str, Any] | None,
    ) -> str | None:
        """Extract upper-cased quant_algo from a modelopt config.

        Returns the quant_algo string (upper-cased), or None if the config
        is not a modelopt config.
        """
        if hf_quant_cfg is None:
            return None
        if not hf_quant_cfg.get("quant_method", "").lower().startswith("modelopt"):
            return None
        if "quantization" in hf_quant_cfg:
            quant_config = hf_quant_cfg["quantization"]
            if isinstance(quant_config, dict):
                return str(quant_config.get("quant_algo", "")).upper()
            return None
        return str(hf_quant_cfg.get("quant_algo", "")).upper()

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        group_size: int | None,
    ) -> "ModelOptQuantConfigBase":
        raise NotImplementedError("Please implement this function in sub classes")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelOptQuantConfigBase":
        # Handle both ModelOpt format and compressed-tensors style format
        if "quantization" in config:
            # Traditional ModelOpt format:
            # {"quantization": {"quant_algo": "..."}}
            quant_config = cls.get_from_keys(config, ["quantization"])
            if not isinstance(quant_config, dict):
                raise ValueError("Expected 'quantization' to be a dictionary in config")

            quant_method = quant_config.get("quant_algo")

            # Handle kv_cache_quant_algo with proper type validation
            kv_cache_quant_method = quant_config.get("kv_cache_quant_algo")

            # Handle group_size with proper type validation
            group_size_raw = quant_config.get("group_size")

            # "exclude_modules" is the key in the legacy hf_quant_config.json
            exclude_modules = quant_config.get("exclude_modules", [])
        else:
            # Compressed-tensors style format (config.json quantization_config):
            # {"quant_algo": "...", "quant_method": "modelopt"}
            quant_method = config.get("quant_algo")

            # "kv_cache_scheme" (a dict) instead of "kv_cache_quant_algo" (a string).
            kv_cache_scheme = config.get("kv_cache_scheme")
            if isinstance(kv_cache_scheme, dict) and (
                kv_cache_scheme.get("type") == "float"
                and kv_cache_scheme.get("num_bits") == 8
            ):
                kv_cache_quant_method = "FP8"
            else:
                kv_cache_quant_method = None

            # "ignore" is the key in config.json
            exclude_modules = config.get("ignore", [])
            group_size_raw = config.get("group_size")

        if not quant_method:
            raise ValueError("Missing 'quant_algo' in quantization config")

        # Normalize quant_algo for robust matching (ModelOpt may emit lowercase).
        quant_method = str(quant_method).upper()

        if kv_cache_quant_method is None:
            # No KV cache quantization, keep this branch just to have this comment
            pass
        elif not isinstance(kv_cache_quant_method, str):
            raise ValueError(
                f"kv_cache_quant_algo must be a string, got "
                f"{type(kv_cache_quant_method)}"
            )
        else:
            kv_cache_quant_method = kv_cache_quant_method.upper()

        if not isinstance(exclude_modules, list):
            raise ValueError(
                f"exclude_modules must be a list, got {type(exclude_modules)}"
            )

        if group_size_raw is None:
            group_size = None
        elif isinstance(group_size_raw, int):
            group_size = group_size_raw
        else:
            try:
                group_size = int(group_size_raw)
            except (ValueError, TypeError):
                raise ValueError(
                    f"group_size must be an integer, got {type(group_size_raw)}"
                ) from None

        if quant_method not in QUANT_ALGOS:
            raise ValueError(
                f"ModelOpt currently only supports: {QUANT_ALGOS} "
                "quantizations in vLLM. Please check the "
                "`hf_quant_config.json` file for your model's "
                "quant configuration."
            )
        return cls._from_config(
            quant_method=quant_method,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            group_size=group_size,
            original_config=config,
        )


class ModelOptFp8Config(ModelOptQuantConfigBase):
    """Config class for ModelOpt FP8."""

    def __init__(
        self,
        quant_method: str,
        is_checkpoint_fp8_serialized: bool,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
    ) -> None:
        super().__init__(exclude_modules)
        self.quant_method = quant_method
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.kv_cache_quant_method = kv_cache_quant_method
        if is_checkpoint_fp8_serialized:
            logger.warning(
                "Detected ModelOpt fp8 checkpoint (quant_algo=%s). Please note "
                "that the format is experimental and could change.",
                quant_method,
            )

        # Validate the quant_algo; resolve() maps it to a QuantSpec at dispatch.
        if self.quant_method not in (
            "FP8",
            "FP8_PER_CHANNEL_PER_TOKEN",
            "FP8_PB_WO",
        ):
            raise ValueError(
                "Unsupported ModelOpt FP8 quant_algo for vLLM: "
                f"{self.quant_method}. Supported: FP8 / "
                "FP8_PER_CHANNEL_PER_TOKEN / FP8_PB_WO."
            )

    def linear_algo(self) -> str:
        return self.quant_method

    def get_name(self) -> QuantizationMethods:
        return "modelopt"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
        if algo is not None and algo == "FP8":
            return "modelopt"
        return None

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        **kwargs: Any,
    ) -> "ModelOptFp8Config":
        is_checkpoint_fp8_serialized = "FP8" in quant_method

        return cls(
            quant_method,
            is_checkpoint_fp8_serialized,
            kv_cache_quant_method,
            exclude_modules,
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
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(moe_config)
        self.quant_config = quant_config
        assert self.quant_config.is_checkpoint_fp8_serialized

        # Select Fp8 MoE backend
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=kFp8StaticTensorSym,
            activation_key=kFp8StaticTensorSym,
        )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: RoutedExperts,
    ) -> mk.FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
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
        layer.orig_dtype = params_dtype
        layer.num_experts = num_experts

        # Use FP8 dtype if checkpoint is serialized
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )
        weight_loader = extra_weight_attrs.get("weight_loader")

        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
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

        # WEIGHT SCALES - Per-tensor scaling for ModelOpts
        # For gated MoE, allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        # For non-gated MoE, allocate 1 scale for w13.
        w13_weight_scale = PerTensorScaleParameter(
            data=torch.full(
                (num_experts, w13_num_shards),
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

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor,
        w2_input_scale: torch.Tensor,
    ):
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

        # Setup modular kernel.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_fp8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            fp8_backend=self.fp8_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_input_scale = layer.w13_input_scale
        w2_input_scale = layer.w2_input_scale

        # Per tensor kernels require single activation scale. Use the max.
        w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
            w13_input_scale, w2_input_scale
        )
        replace_parameter(layer, "w13_input_scale", w13_input_scale)
        replace_parameter(layer, "w2_input_scale", w2_input_scale)

        # Per tensor kernels require single weight scale for w13 per expert, but
        # on disk there is a scale for w1 and w3. Use the max to requantize.
        shard_size = layer.intermediate_size_per_partition
        w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
            w13,
            w13_scale,
            shard_size,
            num_experts=layer.w13_weight.shape[0],
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale
        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
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
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            layer=layer,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
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
        assert not self.is_monolithic
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


ModelOptFp8Config.FusedMoEMethodCls = ModelOptFp8MoEMethod
ModelOptFp8Config.KVCacheMethodCls = ModelOptKVCacheMethod


class ModelOptNvFp4Config(ModelOptQuantConfigBase):
    """Config class for ModelOpt FP4."""

    def __init__(
        self,
        quant_method: str = "NVFP4",
        is_checkpoint_nvfp4_serialized: bool = False,
        kv_cache_quant_algo: str | None = None,
        exclude_modules: list[str] | None = None,
        group_size: int = 16,
    ) -> None:
        if exclude_modules is None:
            exclude_modules = []
        super().__init__(exclude_modules)
        self.quant_method = quant_method
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected ModelOpt NVFP4 checkpoint (quant_algo=%s). Please "
                "note that the format is experimental and could change in "
                "future.",
                quant_method,
            )

            self.group_size = group_size
            self.kv_cache_quant_algo = kv_cache_quant_algo

        # Validate the quant_algo; resolve() maps it to a QuantSpec at dispatch.
        # NVFP4       -> W4A4: cutlass NVFP4 GEMM with input quantization
        # W4A16_NVFP4 -> W4A16: FP4 Marlin GEMM with bf16/fp16 activations
        if quant_method not in ("NVFP4", "W4A16_NVFP4"):
            raise ValueError(
                f"Unsupported ModelOpt NVFP4 quant_algo: {quant_method}. "
                "Supported: NVFP4 / W4A16_NVFP4."
            )

    def linear_algo(self) -> str:
        return self.quant_method

    def get_name(self) -> QuantizationMethods:
        return "modelopt_fp4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
        if algo is not None and ("NVFP4" in algo or "FP4" in algo):
            return "modelopt_fp4"
        return None

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        group_size: int | None,
        **kwargs: Any,
    ) -> "ModelOptNvFp4Config":
        is_checkpoint_nvfp4_serialized = "NVFP4" in quant_method

        if group_size is None:
            group_size = 16  # Default value

        # For FP4, these fields are required
        if is_checkpoint_nvfp4_serialized and "quantization" in original_config:
            # Check if required fields are present in the quantization config
            quant_config = original_config["quantization"]
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
            quant_method,
            is_checkpoint_nvfp4_serialized,
            kv_cache_quant_method,
            exclude_modules,
            group_size,
        )


class ModelOptNvFp4FusedMoE(FusedMoEMethodBase):
    """
    MoE Method for FP4 Quantization.
    Args:
        quant_config: NVFP4 Quant Config
    """

    def __init__(
        self,
        quant_config: ModelOptNvFp4Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(moe_config)
        self.quant_config = quant_config
        # W4A16 mode fires for W4A16_NVFP4 on-disk checkpoints. With
        # activation_key=None every W4A4 backend's _supports_quant_scheme
        # rejects itself (they all require (kNvfp4Static, kNvfp4Dynamic)
        # exactly); only Marlin survives. Marlin's MoE path drops
        # activation scales in convert_to_nvfp4_moe_kernel_format, so no
        # other change is needed.
        self.use_a16 = quant_config.quant_method == "W4A16_NVFP4"
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=None if self.use_a16 else kNvfp4Dynamic,
        )

        self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
            self.nvfp4_backend
        )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        FP4 variants use 'weight_scale_2' pattern for per-tensor weight scales.
        """
        return True

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert self.quant_config.is_checkpoint_nvfp4_serialized

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config
        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn
        weight_loader = extra_weight_attrs.get("weight_loader")
        global_num_experts = extra_weight_attrs.get("global_num_experts")
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        # GEMM 1
        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
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
                w13_num_shards * intermediate_size_per_partition,
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
            data=torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
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

        global_sf_num_experts = (
            global_num_experts if self.use_global_sf else num_experts
        )
        w13_input_scale = PerTensorScaleParameter(
            data=torch.empty(
                global_sf_num_experts,
                w13_num_shards,
                dtype=torch.float32,
            ),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = PerTensorScaleParameter(
            data=torch.empty(global_sf_num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """
        Convert NVFP4 MoE weights into kernel format and setup the kernel.
        """

        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )
        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()

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
            a13_scale=layer.w13_input_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=layer.w2_weight_scale_2,
            a2_scale=layer.w2_input_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w13_input_scale", a13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
        replace_parameter(layer, "w2_input_scale", a2_scale)

        # Setup modular kernel.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            layer=layer,
        )

    @property
    def supports_eplb(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
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
        assert not self.is_monolithic
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


ModelOptNvFp4Config.FusedMoEMethodCls = ModelOptNvFp4FusedMoE
ModelOptNvFp4Config.KVCacheMethodCls = ModelOptKVCacheMethod


class ModelOptMxFp8Config(ModelOptQuantConfigBase):
    """Config class for ModelOpt MXFP8."""

    def __init__(
        self,
        is_checkpoint_mxfp8_serialized: bool,
        kv_cache_quant_algo: str | None,
        exclude_modules: list[str],
    ) -> None:
        super().__init__(exclude_modules)
        self.is_checkpoint_mxfp8_serialized = is_checkpoint_mxfp8_serialized

        if not is_checkpoint_mxfp8_serialized:
            raise ValueError(
                "MXFP8 quantization requires a serialized checkpoint. "
                "Dynamic quantization is not supported."
            )

        logger.warning(
            "Detected ModelOpt MXFP8 checkpoint. Please note that "
            "the format is experimental and could change in future."
        )

        self.kv_cache_quant_algo = kv_cache_quant_algo

    def linear_algo(self) -> str:
        return "MXFP8"

    def get_name(self) -> QuantizationMethods:
        return "modelopt_mxfp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Marlin kernel supports MXFP8 on SM80+
        return 80

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
        if algo is not None and "MXFP8" in algo:
            return "modelopt_mxfp8"
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelOptMxFp8Config":
        # MiniMax-style checkpoints tag `quant_method: "mxfp8"` + `ignored_layers`
        # (same on-disk format as ModelOpt MXFP8); normalize to the ModelOpt
        # schema and reuse the shared parser.
        if "quantization" not in config and not config.get("quant_algo"):
            config = {
                "quant_method": "modelopt",
                "quantization": {
                    "quant_algo": "MXFP8",
                    "kv_cache_quant_algo": config.get("kv_cache_quant_algo"),
                    "exclude_modules": config.get("ignored_layers", []) or [],
                },
            }
        return cast("ModelOptMxFp8Config", super().from_config(config))

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        **kwargs: Any,
    ) -> "ModelOptMxFp8Config":
        is_checkpoint_mxfp8_serialized = "MXFP8" in quant_method.upper()

        # For MXFP8, validate required fields in the config
        if is_checkpoint_mxfp8_serialized and "quantization" in original_config:
            quant_config = original_config["quantization"]
            required_fields = ["kv_cache_quant_algo", "exclude_modules"]
            missing_fields = [
                field for field in required_fields if field not in quant_config
            ]
            if missing_fields:
                raise ValueError(
                    f"MXFP8 quantization requires the following fields in "
                    f"hf_quant_config.json: {missing_fields}"
                )

        return cls(
            is_checkpoint_mxfp8_serialized,
            kv_cache_quant_method,
            exclude_modules,
        )


class ModelOptMxFp8FusedMoE(FusedMoEMethodBase):
    """FlashInfer TRTLLM MXFP8 block-scale MoE for ModelOpt checkpoints."""

    def __init__(
        self,
        quant_config: ModelOptMxFp8Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(moe_config)
        self.weight_block_size = [1, MXFP8_BLOCK_SIZE]
        self.quant_config = quant_config
        assert self.quant_config.is_checkpoint_mxfp8_serialized

        self.mxfp8_backend, self.experts_cls = select_mxfp8_moe_backend(config=self.moe)

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert layer.intermediate_size_per_partition == intermediate_size_per_partition
        assert layer.hidden_size == hidden_size
        layer.orig_dtype = params_dtype

        if hidden_size % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 MoE requires hidden_size divisible by {MXFP8_BLOCK_SIZE}, "
                f"got {hidden_size}."
            )
        if intermediate_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                "MXFP8 MoE requires intermediate_size_per_partition divisible by "
                f"{MXFP8_BLOCK_SIZE}, got {intermediate_size_per_partition}."
            )

        layer.num_experts = num_experts
        weight_loader = extra_weight_attrs.get("weight_loader")
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        # GEMM 1 weights: [E, (2I or I), H]
        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        # GEMM 2 weights: [E, H, I]
        w2_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        # Per-block (K=32) E8M0 scales.
        w13_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # Ensure the generic MoE weight-loader treats these as block scales.
        set_weight_attrs(
            layer.w13_weight_scale,
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
        )
        set_weight_attrs(
            layer.w2_weight_scale,
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
        )

    @staticmethod
    def _check_weight_dtypes(layer: torch.nn.Module) -> None:
        """Validate weight and scale dtypes before processing."""
        expected = {
            "w13_weight": MXFP8_VALUE_DTYPE,
            "w2_weight": MXFP8_VALUE_DTYPE,
            "w13_weight_scale": MXFP8_SCALE_DTYPE,
            "w2_weight_scale": MXFP8_SCALE_DTYPE,
        }
        for name, expected_dtype in expected.items():
            actual = getattr(layer, name).dtype
            if actual != expected_dtype:
                raise ValueError(
                    f"Expected {name} dtype {expected_dtype}, got {actual}."
                )

    def _shuffle_weights_for_trtllm(self, layer: torch.nn.Module) -> None:
        """Shuffle weights and scales into FlashInfer TRTLLM MXFP8 layout."""
        from flashinfer import (
            reorder_rows_for_gated_act_gemm,
            shuffle_matrix_a,
            shuffle_matrix_sf_a,
        )

        epilogue_tile_m = 128
        num_experts = layer.w13_weight.shape[0]
        is_gated = self.moe.is_act_and_mul
        intermediate_size_factor = 2 if is_gated else 1

        w13_weight = layer.w13_weight.data
        w13_scale = layer.w13_weight_scale.data
        if is_gated:
            # FI TRTLLM gated kernels use W31 ordering. Model checkpoints store
            # gated projection as W13, so convert once before shuffling.
            w13_weight = swap_w13_to_w31(w13_weight)
            w13_scale = swap_w13_to_w31(w13_scale)

        w13_weight_shuffled = []
        w2_weight_shuffled = []
        w13_scale_shuffled = []
        w2_scale_shuffled = []
        for i in range(num_experts):
            w13_i = w13_weight[i].reshape(
                intermediate_size_factor * layer.intermediate_size_per_partition, -1
            )
            w13_sf_i = w13_scale[i].reshape(
                intermediate_size_factor * layer.intermediate_size_per_partition, -1
            )
            if is_gated:
                # Reorder rows for gated activation layout expected by TRTLLM.
                w13_i = reorder_rows_for_gated_act_gemm(w13_i.clone())
                w13_sf_i = reorder_rows_for_gated_act_gemm(w13_sf_i.clone())

            w13_shuffled_i = shuffle_matrix_a(w13_i.view(torch.uint8), epilogue_tile_m)
            w2_shuffled_i = shuffle_matrix_a(
                layer.w2_weight.data[i].view(torch.uint8), epilogue_tile_m
            )
            w13_weight_shuffled.append(
                w13_shuffled_i.contiguous().view(MXFP8_VALUE_DTYPE)
            )
            w2_weight_shuffled.append(
                w2_shuffled_i.contiguous().view(MXFP8_VALUE_DTYPE)
            )
            w13_sf_shuffled_i = shuffle_matrix_sf_a(
                w13_sf_i.view(torch.uint8).reshape(
                    intermediate_size_factor * layer.intermediate_size_per_partition,
                    -1,
                ),
                epilogue_tile_m,
            )
            w2_sf_shuffled_i = shuffle_matrix_sf_a(
                layer.w2_weight_scale.data[i]
                .view(torch.uint8)
                .reshape(layer.hidden_size, -1),
                epilogue_tile_m,
            )
            w13_scale_shuffled.append(
                w13_sf_shuffled_i.contiguous().view(MXFP8_SCALE_DTYPE)
            )
            w2_scale_shuffled.append(
                w2_sf_shuffled_i.contiguous().view(MXFP8_SCALE_DTYPE)
            )

        replace_parameter(
            layer, "w13_weight", torch.stack(w13_weight_shuffled).contiguous()
        )
        replace_parameter(
            layer, "w2_weight", torch.stack(w2_weight_shuffled).contiguous()
        )
        replace_parameter(
            layer,
            "w13_weight_scale",
            torch.stack(w13_scale_shuffled).contiguous(),
        )
        replace_parameter(
            layer,
            "w2_weight_scale",
            torch.stack(w2_scale_shuffled).contiguous(),
        )

    def _dequant_mxfp8_weights_to_bf16(self, layer: RoutedExperts) -> None:
        """One-time MXFP8->BF16 weight dequant for the emulation path.

        On devices without a native MXFP8 MoE kernel (e.g. gfx942 / MI300),
        ``Mxfp8EmulationTritonExperts`` otherwise dequantizes every expert
        weight to BF16 on *every* forward step -- the dominant cost (conc1
        ~1.3 tok/s). Doing the dequant once here and replacing the MXFP8
        parameters with BF16 makes the MoE run exactly like a plain BF16
        checkpoint (full precision, no per-step dequant); SwiGLU-OAI is still
        applied by the experts' ``activation()`` override. The MXFP8 weights
        are freed by ``replace_parameter`` (BF16 is 2x their size; the small
        E8M0 scale tensors are left in place, unused).
        """
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            dequant_mxfp8_to_bf16,
        )

        target_dtype = getattr(layer, "orig_dtype", torch.bfloat16)
        num_experts = layer.w13_weight.shape[0]

        # dequant_mxfp8_to_bf16 handles arbitrary leading dims (*x.shape[:-1]),
        # so dequant the whole [E, N, K] weight in one vectorized call.
        w13_bf16 = dequant_mxfp8_to_bf16(layer.w13_weight, layer.w13_weight_scale).to(
            target_dtype
        )
        w2_bf16 = dequant_mxfp8_to_bf16(layer.w2_weight, layer.w2_weight_scale).to(
            target_dtype
        )

        replace_parameter(layer, "w13_weight", w13_bf16)
        replace_parameter(layer, "w2_weight", w2_bf16)

        logger.info_once(
            "MXFP8->BF16 load-time dequant complete (%d experts/layer); MoE "
            "now runs in BF16 with no per-step dequant.",
            num_experts,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # TODO(bnell): why is this required only for mxfp8?
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        layer._already_called_process_weights_after_loading = True

        self._check_weight_dtypes(layer)

        layer.weight_block_size = self.weight_block_size

        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.mxfp8_backend,
            layer=layer,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_input_scale=None,
            w2_input_scale=None,
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
            fp8_backend=self.mxfp8_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
        )

        # No native MXFP8 MoE kernel on this device (e.g. gfx942): the emulation
        # experts would dequant MXFP8->BF16 every forward step. Convert the
        # weights to BF16 once, here, so the MoE runs like a BF16 checkpoint.
        # Opt out (VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=0) to keep the 1-byte
        # MXFP8 weights and dequant per-step (~half the memory, much slower).
        if (
            self.mxfp8_backend == Fp8MoeBackend.EMULATION
            and envs.VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD
        ):
            self._dequant_mxfp8_weights_to_bf16(layer)

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: RoutedExperts,
    ) -> mk.FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return make_fp8_moe_quant_config(
            fp8_backend=self.mxfp8_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=self.weight_block_size,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            layer=layer,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
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
        assert not self.is_monolithic
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


# Register the method classes for ModelOptMxFp8Config
ModelOptMxFp8Config.FusedMoEMethodCls = ModelOptMxFp8FusedMoE
ModelOptMxFp8Config.KVCacheMethodCls = ModelOptKVCacheMethod


class ModelOptMixedPrecisionConfig(ModelOptQuantConfigBase):
    """Config class for ModelOpt MIXED_PRECISION.

    Supports checkpoints where different layers use different quantization
    algorithms (e.g., FP8 for dense layers and NVFP4 for MoE experts).
    The per-layer algorithm is specified in the ``quantized_layers`` dict
    inside ``config.json``'s ``quantization_config`` (preferred) or the
    legacy ``hf_quant_config.json``.
    """

    def __init__(
        self,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        quantized_layers: dict[str, dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: ModelOptNvFp4Config,
        w4a16_nvfp4_config: ModelOptNvFp4Config,
        mxfp8_config: ModelOptMxFp8Config,
    ) -> None:
        super().__init__(exclude_modules)
        self.kv_cache_quant_method = kv_cache_quant_method
        self.quantized_layers = quantized_layers
        self.fp8_config = fp8_config
        self.nvfp4_config = nvfp4_config
        self.w4a16_nvfp4_config = w4a16_nvfp4_config
        self.mxfp8_config = mxfp8_config

    def get_name(self) -> QuantizationMethods:
        return "modelopt_mixed"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Turing and up (SM75+): NVFP4 routed experts run via Marlin W4A16
        # (SM75+), FP8 weight-only dense via MarlinFP8 (cc>=7.5), and FP8 MoE,
        # if present, via Marlin (TritonExperts gates its FP8 schemes behind
        # supports_fp8(), cc>=89). None of these paths require native FP8 tensor
        # cores, so SM75 is sufficient. Validated end-to-end on a Tesla T4
        # (SM75) and A100 (SM80). Pairs with the FlashInfer attention SM80
        # lower bound so SM75 auto-selects a supported attention backend.
        return 75

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
        if algo is not None and algo == "MIXED_PRECISION":
            return "modelopt_mixed"
        return None

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        group_size: int | None,
        **kwargs: Any,
    ) -> "ModelOptMixedPrecisionConfig":
        if "quantization" in original_config:
            quantized_layers = original_config["quantization"].get(
                "quantized_layers", {}
            )
        else:
            quantized_layers = original_config.get("quantized_layers", {})

        if not quantized_layers:
            raise ValueError(
                "MIXED_PRECISION quant_algo requires a non-empty "
                "'quantized_layers' mapping in the quantization config."
            )

        # Determine group_size from the first NVFP4-family entry if not
        # provided. Both NVFP4 (W4A4) and W4A16_NVFP4 share the same packing
        # + group-size convention; either entry resolves the value.
        if group_size is None:
            for layer_info in quantized_layers.values():
                if layer_info.get("quant_algo", "").upper() in (
                    "NVFP4",
                    "W4A16_NVFP4",
                ):
                    group_size = layer_info.get("group_size", 16)
                    break
        if group_size is None:
            group_size = 16

        fp8_config = ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=[],
        )
        nvfp4_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
            group_size=group_size,
        )
        # Sibling config for layers that declare quant_algo: "W4A16_NVFP4".
        # get_quant_method resolves this sub-config to the (kNvfp4Static, None)
        # QuantSpec (W4A16) for the generic ModelOptLinearMethod. The MoE side
        # reads quant_config.quant_method == "W4A16_NVFP4" to set use_a16 →
        # Marlin backend in ModelOptNvFp4FusedMoE.__init__.
        w4a16_nvfp4_config = ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
            group_size=group_size,
        )

        mxfp8_config = ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
        )

        return cls(
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            nvfp4_config=nvfp4_config,
            w4a16_nvfp4_config=w4a16_nvfp4_config,
            mxfp8_config=mxfp8_config,
        )

    def _resolve_quant_algo(self, prefix: str) -> str | None:
        """Look up the quant_algo for a vLLM-side layer prefix.

        Tries three strategies in order:
        1. Direct lookup in ``quantized_layers``.
        2. Packed/fused-layer lookup (unfuse via ``packed_modules_mapping``).
        3. Prefix-based lookup for RoutedExperts (any child key starts with
           ``prefix + "."``).

        Returns the upper-cased quant_algo string, or *None* if the prefix
        is not found.
        """
        # 1. Direct lookup
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            if candidate in self.quantized_layers:
                return self.quantized_layers[candidate]["quant_algo"].upper()

        # 2. Packed / fused layer lookup
        proj_name = prefix.rsplit(".", 1)[-1]
        if self.packed_modules_mapping and proj_name in self.packed_modules_mapping:
            algos: set[str] = set()
            base = prefix.rsplit(".", 1)[0]
            for base_candidate in self._quantized_layer_prefix_candidates(base):
                for shard_name in self.packed_modules_mapping[proj_name]:
                    shard_prefix = f"{base_candidate}.{shard_name}"
                    if shard_prefix in self.quantized_layers:
                        algos.add(
                            self.quantized_layers[shard_prefix]["quant_algo"].upper()
                        )
            if len(algos) == 1:
                return algos.pop()
            if len(algos) > 1:
                raise ValueError(
                    f"Mixed quant_algo within fused layer {prefix}: "
                    f"{algos}. All shards must use the same quantization."
                )

        # 3. Prefix-based lookup (for RoutedExperts / parent modules)
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            prefix_dot = candidate + "."
            for key, info in self.quantized_layers.items():
                if key.startswith(prefix_dot):
                    return info["quant_algo"].upper()

        # FusedMoE expert prefix is e.g. "...moe.experts", while ModelOpt's
        # quantized_layers entries use "...moe.gate_proj" / "...moe.up_proj".
        if prefix.endswith(".experts"):
            parent_dot = prefix.rsplit(".experts", 1)[0] + "."
            for key, info in self.quantized_layers.items():
                if key.startswith(parent_dot):
                    return info["quant_algo"].upper()

        # 4. Parent-prefix fallback for fused projections whose config lists
        # shard names instead of vLLM's packed module name.
        fused_projection_shards = {
            "qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "gate_up_proj": ("gate_proj", "up_proj"),
        }
        shard_names = fused_projection_shards.get(proj_name)
        if shard_names is not None:
            for candidate in self._quantized_layer_prefix_candidates(prefix):
                parent_dot = candidate.rsplit(".", 1)[0] + "."
                shard_algos: set[str] = set()
                for shard_name in shard_names:
                    shard_prefix = f"{parent_dot}{shard_name}"
                    if shard_prefix in self.quantized_layers:
                        algo = self.quantized_layers[shard_prefix]["quant_algo"].upper()
                        shard_algos.add(algo)
                if len(shard_algos) == 1:
                    return shard_algos.pop()
                if len(shard_algos) > 1:
                    raise ValueError(
                        f"Mixed quant_algo within fused layer {prefix}: "
                        f"{shard_algos}. All shards must use the same quantization."
                    )

        return None

    @staticmethod
    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:
        candidates = [prefix]

        if prefix.endswith(".lm_head"):
            candidates.append("lm_head")

        if prefix.startswith("language_model.model."):
            candidates.append(
                "model.language_model." + prefix[len("language_model.model.") :]
            )
        elif prefix.startswith("model.language_model."):
            candidates.append(
                "language_model.model." + prefix[len("model.language_model.") :]
            )

        return tuple(dict.fromkeys(candidates))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        """Return quantize-method based on layer."""
        # KV-cache quantization
        if isinstance(layer, Attention):
            if self.kv_cache_quant_method:
                return ModelOptKVCacheMethod(self)
            return None

        # Excluded layers
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None

        quant_algo = self._resolve_quant_algo(prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            # Per-prefix algo -> its sub-config, then the generic linear method.
            # resolve() reads is_checkpoint_*_serialized/group_size off whichever
            # sub-config it is handed (read-only — never write back, or a
            # linear-only change leaks into the shared imported MoE method).
            subcfg = {
                "FP8": self.fp8_config,
                "NVFP4": self.nvfp4_config,
                "W4A16_NVFP4": self.w4a16_nvfp4_config,
                "MXFP8": self.mxfp8_config,
            }.get(quant_algo)
            if subcfg is None:
                # Layer not in quantized_layers — leave unquantized
                return UnquantizedLinearMethod()
            return build_linear_method(subcfg, quant_algo, prefix)

        if isinstance(layer, RoutedExperts):
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(
                    quant_config=self.fp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.nvfp4_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.w4a16_nvfp4_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            return None

        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        if self.quantized_layers:
            self.quantized_layers = hf_to_vllm_mapper.apply_dict(self.quantized_layers)


# ===========================================================================
# Generic QuantKey-driven linear method
#
# One ``ModelOptLinearMethod`` replaces the six per-format linear method
# classes. It composes a per-QuantKey weight scheme + activation scheme (from
# the ``QuantSpec`` pair produced by ``resolve``), runs a fixed create/process
# lifecycle, selects the kernel from the pair, and applies. See
# ``linear_design_concrete.md`` for the design and caveats C1-C13.
#
# Adding a format (developer guide):
#   * Composes as a (weight, activation) key pair -> add a QuantKeyScheme per
#     new key to SCHEME_FOR, plus a resolve() row returning the QuantSpec. No
#     new method class. (This is how all six existing formats are built.)
#   * Needs format-wide residue but the same lifecycle -> also return a
#     FormatScheme subclass from that resolve() row (extra_weights / pre_process
#     / post_process hooks).
#   * Genuinely cannot be a key pair (different lifecycle) -> write a bespoke
#     LinearMethodBase and register it in LINEAR_METHOD_BUILDERS by algo.
#   In all cases add the algo to the owning config's linear_algo()/validation.
# ===========================================================================


class Role(Enum):
    WEIGHT = "weight"
    ACT = "activation"


WEIGHT = Role.WEIGHT
ACT = Role.ACT

# Weight-loader "unloaded shard" marker — FP8 family fills scales with it; the
# NVFP4/MXFP8 families deliberately do not (C3, load-bearing asymmetry).
SENTINEL = torch.finfo(torch.float32).min


@dataclass(frozen=True)
class CkptCtx:
    """Per-checkpoint facts a QuantKey cannot carry."""

    serialized: bool
    group_size: int | None = None


@dataclass(frozen=True)
class RuntimeDtypes:
    """Runtime/model dtypes kernels need — format-agnostic."""

    input_dtype: torch.dtype
    out_dtype: torch.dtype
    marlin_input_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class Shapes:
    """Layer geometry from create_weights args."""

    out_parts: list[int]
    in_: int
    params_dtype: torch.dtype

    @property
    def out(self) -> int:
        return sum(self.out_parts)

    @property
    def nparts(self) -> int:
        return len(self.out_parts)


class QuantKeyScheme:
    """One scheme per QuantKey. Selected by key *content*; the base supplies the
    ``role`` from the slot it fills the key into (wkey->WEIGHT, akey->ACT).

    Every scheme branches on ``role`` explicitly and ``reject``s any role it has
    not validated — never falling through to the wrong role's registration
    (silent-garbage trap, C13).
    """

    key: QuantKey
    requires_serialized: bool = True
    # Whether the base advertises the kernel's input_quant_key on the layer
    # (enables upstream activation-quant fusion). Behavior-preserving per-format:
    # the old NVFP4 W4A4 method exposed it, the old FP8 method did NOT — and the
    # FP8 kernel *does* return a static key, so exposing there flips activation
    # quant into a fused path and diverges (C2). Read off the weight scheme;
    # default True (NVFP4), False on the FP8 schemes to preserve today's
    # behavior. Adopting FP8 fusion is a separate deliberate change.
    exposes_input_quant_key: bool = True

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        raise NotImplementedError

    def process(self, layer, role) -> None:
        pass

    @staticmethod
    def reject(role) -> None:
        raise NotImplementedError(
            f"role {role!r} not validated for this QuantKey scheme"
        )

    @staticmethod
    def register_params(layer, name, shape, dtype, cls, wl, *, init=None, **dims):
        p = cls(data=torch.empty(shape, dtype=dtype), weight_loader=wl, **dims)
        if init is not None:
            p.data.fill_(init)
        layer.register_parameter(name, p)


class KNvfp4Static(QuantKeyScheme):
    """NVFP4 weight scheme (W4A4 and W4A16 share it). Weight-role only today."""

    key = kNvfp4Static

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not WEIGHT:
            self.reject(role)
        if shapes.in_ % 16 != 0:
            raise ValueError(
                "Unsupported model when in features size is not multiple of 16"
            )
        weight_dtype = (
            torch.float8_e4m3fn if ctx.serialized else shapes.params_dtype
        )
        # Packed NVFP4 weight: 2 fp4 items per byte along the input dim.
        self.register_params(
            layer,
            "weight",
            (shapes.out, shapes.in_ // 2),
            torch.uint8,
            ModelWeightParameter,
            wl,
            input_dim=1,
            output_dim=0,
        )
        # Per-tensor global weight scale.
        self.register_params(
            layer,
            "weight_scale_2",
            (shapes.nparts,),
            torch.float32,
            PerTensorScaleParameter,
            wl,
        )
        # Per-block (group_size) weight scale.
        self.register_params(
            layer,
            "weight_scale",
            (shapes.out, shapes.in_ // ctx.group_size),
            weight_dtype,
            ModelWeightParameter,
            wl,
            input_dim=1,
            output_dim=0,
        )

    def process(self, layer, role) -> None:
        if role is not WEIGHT:
            self.reject(role)
        if torch.unique(layer.weight_scale_2).numel() != 1:
            logger.warning_once(
                "In NVFP4 linear, the global weight scale differs across "
                "parallel layers (e.g. q_proj, k_proj, v_proj). This will "
                "likely reduce accuracy. Consider a checkpoint with a shared "
                "global NVFP4 scale for parallel layers."
            )
        # Raw max, no reciprocation — Marlin/cutlass want ModelOpt's amax/2688.
        weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
        layer.weight_global_scale = Parameter(
            weight_global_scale, requires_grad=False
        )
        del layer.weight_scale_2


class KNvfp4Dynamic(QuantKeyScheme):
    """NVFP4 activation scheme (W4A4). Has a static global input scale on disk;
    the per-group scale is computed at runtime inside the kernel."""

    key = kNvfp4Dynamic

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not ACT:
            self.reject(role)
        self.register_params(
            layer,
            "input_scale",
            (shapes.nparts,),
            torch.float32,
            PerTensorScaleParameter,
            wl,
        )

    def process(self, layer, role) -> None:
        if role is not ACT:
            self.reject(role)
        if torch.unique(layer.input_scale).numel() != 1:
            logger.warning_once(
                "In NVFP4 linear, the global input scale differs across "
                "parallel layers (e.g. q_proj, k_proj, v_proj). This will "
                "likely reduce accuracy. Consider a checkpoint with a shared "
                "global NVFP4 scale for parallel layers."
            )
        input_global_scale = layer.input_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(
            input_global_scale, requires_grad=False
        )
        layer.input_global_scale_inv = Parameter(
            (1.0 / layer.input_global_scale).to(torch.float32),
            requires_grad=False,
        )
        del layer.input_scale


class KFp8StaticTensor(QuantKeyScheme):
    """Plain per-tensor static FP8 — bivalent: serves BOTH the weight slot and
    the activation slot (W8A8). One key in both QuantSpec slots."""

    key = kFp8StaticTensorSym
    requires_serialized = False  # FP8 alone allows a non-serialized checkpoint
    exposes_input_quant_key = False  # old FP8 method did not expose it (C2)

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is WEIGHT:
            weight_dtype = (
                torch.float8_e4m3fn if ctx.serialized else shapes.params_dtype
            )
            self.register_params(
                layer, "weight", (shapes.out, shapes.in_), weight_dtype,
                ModelWeightParameter, wl, input_dim=1, output_dim=0,
            )
            layer.orig_dtype = shapes.params_dtype
            if ctx.serialized:
                self.register_params(
                    layer, "weight_scale", (shapes.nparts,), torch.float32,
                    PerTensorScaleParameter, wl, init=SENTINEL,
                )
        elif role is ACT:
            if ctx.serialized:
                self.register_params(
                    layer, "input_scale", (shapes.nparts,), torch.float32,
                    PerTensorScaleParameter, wl, init=SENTINEL,
                )
        else:
            self.reject(role)

    def process(self, layer, role) -> None:
        if role is WEIGHT:
            weight = layer.weight
            max_w_scale = layer.weight_scale.max()
            if not (layer.weight_scale == layer.weight_scale[0]).all():
                max_w_scale, weight = requantize_with_max_scale(
                    layer.weight, layer.weight_scale, layer.logical_widths
                )
            # Transpose lives here (Scope A; belongs to the kernel — C1).
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        elif role is ACT:
            layer.input_scale = Parameter(
                layer.input_scale.max(), requires_grad=False
            )
        else:
            self.reject(role)


class KFp8StaticChannel(QuantKeyScheme):
    """Per-channel static FP8 weight (the 'PcPt' weight). Weight-role only —
    there is no static per-channel *activation* today."""

    key = kFp8StaticTokenSym
    exposes_input_quant_key = False  # old PcPt method did not expose it

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not WEIGHT:
            self.reject(role)
        self.register_params(
            layer, "weight", (shapes.out, shapes.in_), torch.float8_e4m3fn,
            ModelWeightParameter, wl, input_dim=1, output_dim=0,
        )
        self.register_params(
            layer, "weight_scale", (shapes.out,), torch.float32,
            ChannelQuantScaleParameter, wl, output_dim=0, init=SENTINEL,
        )

    def process(self, layer, role) -> None:
        if role is not WEIGHT:
            self.reject(role)
        weight, weight_scale, _ = process_fp8_weight_channel_strategy(
            layer.weight, layer.weight_scale.data
        )
        layer.weight = Parameter(weight.t(), requires_grad=False)  # C1 (Scope A)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)


class KFp8Block128(QuantKeyScheme):
    """128x128 block-static FP8 weight ('PbWo'). Weight-role only. ModelOpt
    exports the scale 4-D [out_blk,1,in_blk,1] (C6); process squeezes to 2-D.
    No transpose (block kernel keeps [out,in])."""

    key = kFp8Static128BlockSym
    exposes_input_quant_key = False

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not WEIGHT:
            self.reject(role)
        if shapes.out % 128 != 0 or shapes.in_ % 128 != 0:
            raise ValueError(
                f"FP8_PB_WO requires out/in divisible by 128, got "
                f"{shapes.out}x{shapes.in_}"
            )
        self.register_params(
            layer, "weight", (shapes.out, shapes.in_), torch.float8_e4m3fn,
            ModelWeightParameter, wl, input_dim=1, output_dim=0,
        )
        ob, ib = shapes.out // 128, shapes.in_ // 128
        self.register_params(
            layer, "weight_scale", (ob, 1, ib, 1), torch.float32,
            BlockQuantScaleParameter, wl, input_dim=2, output_dim=0,
            init=SENTINEL,
        )
        layer.weight_block_size = [128, 128]

    def process(self, layer, role) -> None:
        if role is not WEIGHT:
            self.reject(role)
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        s = layer.weight_scale
        if s.dim() == 4:
            s = s.squeeze(1).squeeze(-1)  # [ob,1,ib,1] -> [ob,ib]
        elif s.dim() != 2:
            raise ValueError(
                f"Unexpected FP8_PB_WO weight_scale shape {tuple(s.shape)}"
            )
        layer.weight_scale = Parameter(s.contiguous(), requires_grad=False)


class KMxfp8Static(QuantKeyScheme):
    """MXFP8 weight: fp8-e4m3 values + per-32-block e8m0 (uint8) scale.
    Weight-role only. process is validate-only + idempotency guard (C13)."""

    key = kMxfp8Static
    exposes_input_quant_key = False

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not WEIGHT:
            self.reject(role)
        if shapes.in_ % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires in divisible by {MXFP8_BLOCK_SIZE}, "
                f"got {shapes.in_}"
            )
        self.register_params(
            layer, "weight", (shapes.out, shapes.in_), MXFP8_VALUE_DTYPE,
            ModelWeightParameter, wl, input_dim=1, output_dim=0,
        )
        self.register_params(
            layer, "weight_scale",
            (shapes.out, shapes.in_ // MXFP8_BLOCK_SIZE), MXFP8_SCALE_DTYPE,
            ModelWeightParameter, wl, input_dim=1, output_dim=0,
        )

    def process(self, layer, role) -> None:
        if role is not WEIGHT:
            self.reject(role)
        # Idempotency: emulation kernel may dequant weight to >=2-byte at load.
        if layer.weight.element_size() >= 2:
            return
        assert layer.weight.ndim == 2 and layer.weight.dtype == MXFP8_VALUE_DTYPE
        assert layer.weight_scale.ndim == 2
        assert layer.weight_scale.dtype == MXFP8_SCALE_DTYPE


class KDynamicNoParam(QuantKeyScheme):
    """Dynamic activation with no stored scale (W8A8): quantized at runtime in
    the kernel. NOT the same as activation=None (weight-only) — init_fp8 needs a
    non-None activation key. Activation-role only. Serves the fp8 per-token, fp8
    per-block, and mxfp8 dynamic activation keys."""

    requires_serialized = False

    def create_weights(self, layer, role, ctx, shapes, wl) -> None:
        if role is not ACT:
            self.reject(role)
        # dynamic -> nothing stored

    def process(self, layer, role) -> None:
        if role is not ACT:
            self.reject(role)


SCHEME_FOR: dict[QuantKey | None, QuantKeyScheme] = {
    kNvfp4Static: KNvfp4Static(),
    kNvfp4Dynamic: KNvfp4Dynamic(),
    kFp8StaticTensorSym: KFp8StaticTensor(),
    kFp8StaticTokenSym: KFp8StaticChannel(),
    kFp8Static128BlockSym: KFp8Block128(),
    kMxfp8Static: KMxfp8Static(),
    kFp8DynamicTokenSym: KDynamicNoParam(),
    kFp8Dynamic128Sym: KDynamicNoParam(),
    kMxfp8Dynamic: KDynamicNoParam(),
}


def maybe_fuse_global_scales(layer) -> None:
    """alpha = input_global_scale * weight_global_scale, presence-gated.

    W4A4 has both -> computed; W4A16 has no input_global_scale -> skipped.
    """
    if hasattr(layer, "weight_global_scale") and hasattr(
        layer, "input_global_scale"
    ):
        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale,
            requires_grad=False,
        )


def select_linear_kernel(spec: QuantSpec, layer, rt: RuntimeDtypes):
    """Thin family dispatcher on the weight key: nvfp4 / mxfp8 / fp8."""
    w = spec.weight
    if w.dtype == FP4_DTYPE:
        if spec.activation is None:
            # W4A16: pin Marlin exactly like the old NVFP4 W4A16 method. We
            # can't route through init_nvfp4_linear_kernel(use_a16=True): under
            # VLLM_BATCH_INVARIANT its first branch force-selects Cutlass (W4A4),
            # whose apply reads layer.input_global_scale_inv — absent for W4A16,
            # so it AttributeErrors. Pinning matches old and is BI-safe.
            return MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())
        return init_nvfp4_linear_kernel(use_a16=False)  # W4A4 (Cutlass/etc.)
    if w.scale.dtype == MXFP8_SCALE_DTYPE:
        return init_mxfp8_linear_kernel()
    # fp8 family: init_fp8 routes block-vs-plain itself off the activation key.
    return init_fp8_linear_kernel(
        activation_quant_key=spec.activation,
        weight_quant_key=w,
        input_dtype=rt.input_dtype,
        out_dtype=rt.out_dtype,
        weight_shape=layer.weight.shape,
        module_name=type(layer).__name__,
    )


class FormatScheme:
    """Optional per-format hooks that compose *around* the QuantKey schemes.

    Extension seam for residue that belongs to a format as a whole rather than a
    single QuantKey — an extra parameter the weight/activation schemes don't
    cover, or a tweak before/after their ``process``. All hooks default to
    no-ops; most formats need none (they are a pure ``(weight, activation)`` key
    pair). To add one: subclass, override the hooks you need, and return an
    instance from the format's ``resolve()`` branch — no change to
    ``ModelOptLinearMethod`` itself.
    """

    def extra_weights(self, layer, shapes: "Shapes", ctx: CkptCtx, wl) -> None:
        """Register format-level params (after the key schemes' weights)."""

    def pre_process(self, layer) -> None:
        """Run before the key schemes' ``process``."""

    def post_process(self, layer) -> None:
        """Run after the key schemes' ``process``, before the kernel's."""


@register_weight_loader_v2_supported_method
class ModelOptLinearMethod(LinearMethodBase):
    """Generic, format-agnostic ModelOpt linear method. Holds a weight scheme +
    an activation scheme (from the QuantSpec pair) and an optional
    ``FormatScheme``, runs a fixed lifecycle, selects the kernel from the pair,
    and applies.

    Registered for weight_loader_v2 (like every ModelOpt linear method) so the
    ``BasevLLMParameter`` params route through the v2 fused loader, not the
    legacy shape-assert path.
    """

    def __init__(
        self,
        spec: QuantSpec,
        ctx: CkptCtx,
        format_scheme=None,
    ) -> None:
        self.spec = spec
        self.ctx = ctx
        self.fmt = format_scheme or FormatScheme()
        self.wkey = SCHEME_FOR[spec.weight]
        self.akey = None if spec.activation is None else SCHEME_FOR[spec.activation]
        self.out_dtype = torch.get_default_dtype()
        # Only the fp8/mxfp8 kernels read input_dtype; nvfp4 ignores it. During
        # real serving model_config is always set; fall back defensively when
        # there is no config context (bare unit-test dispatch).
        model_config = getattr(
            get_current_vllm_config_or_none(), "model_config", None
        )
        self.input_dtype = (
            model_config.dtype
            if model_config is not None
            else torch.get_default_dtype()
        )
        self.marlin_input_dtype = None
        # Kernel/backend are chosen in create_weights (after get_quant_method),
        # so the front-end marlin poke stays dormant here — same as the old
        # NVFP4 methods.
        self.kernel = None

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        if self.wkey.requires_serialized and not self.ctx.serialized:
            raise ValueError(
                f"{self.spec.weight} requires a serialized checkpoint"
            )
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        shapes = Shapes(output_partition_sizes, input_size_per_partition, params_dtype)

        self.wkey.create_weights(layer, WEIGHT, self.ctx, shapes, weight_loader)
        if self.akey:
            self.akey.create_weights(layer, ACT, self.ctx, shapes, weight_loader)
        self.fmt.extra_weights(layer, shapes, self.ctx, weight_loader)

        rt = RuntimeDtypes(self.input_dtype, self.out_dtype, self.marlin_input_dtype)
        self.kernel = select_linear_kernel(self.spec, layer, rt)
        if self.wkey.exposes_input_quant_key:
            expose_input_quant_key(layer, self.kernel)

    def process_weights_after_loading(self, layer) -> None:
        self.fmt.pre_process(layer)
        self.wkey.process(layer, WEIGHT)
        if self.akey:
            self.akey.process(layer, ACT)
        maybe_fuse_global_scales(layer)
        self.fmt.post_process(layer)
        self.kernel.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)


def resolve(algo: str, subcfg, prefix: str):
    """Turn an existing (untouched) sub-config into (QuantSpec, CkptCtx,
    format_scheme). Strictly read-only over ``subcfg`` — the one real hazard in
    mixed mode is writing back to a config shared with the imported MoE method.
    """
    if algo == "FP8":
        # Plain per-tensor static FP8 (W8A8): same key in both slots (bivalent).
        ctx = CkptCtx(
            serialized=subcfg.is_checkpoint_fp8_serialized, group_size=None
        )
        return (
            QuantSpec(weight=kFp8StaticTensorSym, activation=kFp8StaticTensorSym),
            ctx,
            None,
        )
    if algo == "FP8_PER_CHANNEL_PER_TOKEN":
        # PcPt: per-channel static weight, dynamic per-token activation (W8A8).
        ctx = CkptCtx(
            serialized=subcfg.is_checkpoint_fp8_serialized, group_size=None
        )
        return (
            QuantSpec(
                weight=kFp8StaticTokenSym, activation=kFp8DynamicTokenSym
            ),
            ctx,
            None,
        )
    if algo == "FP8_PB_WO":
        # PbWo: 128x128 block-static weight, dynamic per-block activation (W8A8).
        # C12: the generic base runs the block kernel's post-load, which the old
        # method skipped via a misnamed guard — validate vs CT block-FP8.
        ctx = CkptCtx(
            serialized=subcfg.is_checkpoint_fp8_serialized, group_size=None
        )
        return (
            QuantSpec(
                weight=kFp8Static128BlockSym, activation=kFp8Dynamic128Sym
            ),
            ctx,
            None,
        )
    if algo == "MXFP8":
        # MXFP8: block(32) e4m3 weight + e8m0 scale, dynamic activation.
        ctx = CkptCtx(
            serialized=subcfg.is_checkpoint_mxfp8_serialized, group_size=None
        )
        return QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic), ctx, None

    # NVFP4 family (W4A4 / W4A16).
    ctx = CkptCtx(
        serialized=subcfg.is_checkpoint_nvfp4_serialized,
        group_size=subcfg.group_size,
    )
    if algo == "NVFP4":
        # W4A4: static fp4 weight + dynamic fp4 activation (has a static global
        # input scale). alpha = weight_gs * input_gs.
        return QuantSpec(weight=kNvfp4Static, activation=kNvfp4Dynamic), ctx, None
    if algo == "W4A16_NVFP4":
        # W4A16: same fp4 weight, no activation quant. activation=None drives
        # use_a16=True in select_linear_kernel (-> Marlin) and skips alpha. The
        # old method's placeholder input_scale is intentionally dropped (C4).
        return QuantSpec(weight=kNvfp4Static, activation=None), ctx, None
    raise NotImplementedError(f"resolve: unsupported ModelOpt linear algo {algo!r}")


# Bespoke-method escape hatch. Almost every ModelOpt linear format is a
# (weight, activation) QuantKey pair handled by the generic ModelOptLinearMethod
# (optionally + a FormatScheme), so this stays empty. A format that genuinely
# cannot be expressed that way — a fundamentally different create/process/apply
# lifecycle — registers its own LinearMethodBase here, keyed by algo:
#
#     LINEAR_METHOD_BUILDERS["FOO"] = lambda cfg, prefix: ModelOptFooLinearMethod(cfg)
#
# Keep the ModelOpt* class-name prefix and decorate the class with
# @register_weight_loader_v2_supported_method if it uses BasevLLMParameter
# params. Also add the algo to the owning config's linear_algo()/validation.
LINEAR_METHOD_BUILDERS: dict[str, Callable[..., LinearMethodBase]] = {}


def build_linear_method(config, algo: str, prefix: str) -> LinearMethodBase:
    """Construct the linear method for ``algo``.

    Returns a bespoke method if one is registered in ``LINEAR_METHOD_BUILDERS``,
    else the generic ``ModelOptLinearMethod`` built from ``resolve(algo, config,
    prefix)``. Single indirection point shared by the homogeneous and
    mixed-precision dispatch, so a new format plugs in without editing either
    ``get_quant_method``.
    """
    builder = LINEAR_METHOD_BUILDERS.get(algo)
    if builder is not None:
        return builder(config, prefix)
    spec, ctx, format_scheme = resolve(algo, config, prefix)
    return ModelOptLinearMethod(spec, ctx, format_scheme)
