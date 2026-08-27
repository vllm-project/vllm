# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel,
    init_mxfp8_linear_kernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
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
from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
    prepare_nvfp4_megamoe_inputs,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    expose_input_quant_key,
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
    GroupShape,
    create_fp8_quant_key,
    is_layer_skipped,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
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
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform

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


class ModelOptKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 or NVFP4 checkpoints.
    """

    def __init__(self, quant_config: "ModelOptQuantConfigBase"):
        super().__init__(quant_config)


class ModelOptQuantConfigBase(QuantizationConfig):
    LinearMethodCls: type = LinearMethodBase
    FusedMoEMethodCls: type = FusedMoEMethodBase
    KVCacheMethodCls: type = BaseKVCacheMethod

    def __init__(
        self,
        exclude_modules: list[str],
    ):
        super().__init__()
        self.exclude_modules: list[str] = exclude_modules

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
            quant_method = self.LinearMethodCls(self)
            if getattr(quant_method, "backend", "") == "marlin":
                quant_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
            return quant_method
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

        # Select LinearMethod implementation based on quant_algo.
        if self.quant_method == "FP8":
            self.LinearMethodCls = ModelOptFp8LinearMethod
        elif self.quant_method == "FP8_PER_CHANNEL_PER_TOKEN":
            self.LinearMethodCls = ModelOptFp8PcPtLinearMethod
        elif self.quant_method == "FP8_PB_WO":
            self.LinearMethodCls = ModelOptFp8PbWoLinearMethod
        else:
            raise ValueError(
                "Unsupported ModelOpt FP8 quant_algo for vLLM: "
                f"{self.quant_method}. Supported: FP8 / "
                "FP8_PER_CHANNEL_PER_TOKEN / FP8_PB_WO."
            )

    def get_name(self) -> QuantizationMethods:
        return "modelopt"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

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
        self.out_dtype = get_current_vllm_config().model_config.dtype
        self.input_dtype = get_current_vllm_config().model_config.dtype

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
        layer.orig_dtype = params_dtype
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

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        max_w_scale = layer.weight_scale.max()
        if not (layer.weight_scale == layer.weight_scale[0]).all():
            max_w_scale, weight = requantize_with_max_scale(
                layer.weight, layer.weight_scale, layer.logical_widths
            )
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight.input_dim = 0
        layer.weight.output_dim = 1
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
        self.fp8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fp8_linear.apply_weights(layer, x, bias)


class ModelOptFp8PcPtLinearMethod(LinearMethodBase):
    """Linear method for ModelOpt FP8_PER_CHANNEL_PER_TOKEN checkpoints.

    Expected checkpoint structure (per Linear):
    - weight: fp8-e4m3fn, shape [out, in]
    - weight_scale: fp32, shape [out] (per-output-channel)
    - no input_scale (activations are dynamically quantized per-token)
    """

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = get_current_vllm_config().model_config.dtype
        self.input_dtype = get_current_vllm_config().model_config.dtype

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

        if not self.quant_config.is_checkpoint_fp8_serialized:
            raise ValueError(
                "FP8_PER_CHANNEL_PER_TOKEN currently only supports "
                "FP8-serialized checkpoints."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticTokenSym,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, weight_scale, _ = process_fp8_weight_channel_strategy(
            layer.weight, layer.weight_scale.data
        )
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        self.fp8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fp8_linear.apply_weights(layer, x, bias)


class ModelOptFp8PbWoLinearMethod(LinearMethodBase):
    """Linear method for ModelOpt FP8_PB_WO checkpoints.

    ModelOpt exports `weight_scale` as a 4D tensor:
      [out_blk, 1, in_blk, 1]
    where block size is typically 128 for both dims.

    vLLM executes it as FP8 GEMM with *dynamic per-token* activation quant.
    Output widths that are not block-aligned are padded and restored
    to their logical width before returning to the model.
    """

    _WEIGHT_BLOCK_SIZE: tuple[int, int] = (128, 128)

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config
        block_n, block_k = self._WEIGHT_BLOCK_SIZE
        self.weight_block_size = list(self._WEIGHT_BLOCK_SIZE)

        self.activation_quant_key = create_fp8_quant_key(
            static=False, group_shape=GroupShape(1, block_k)
        )
        self.weight_quant_key = create_fp8_quant_key(
            static=True, group_shape=GroupShape(block_n, block_k)
        )

        self.out_dtype = get_current_vllm_config().model_config.dtype
        self.input_dtype = get_current_vllm_config().model_config.dtype

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

        if not self.quant_config.is_checkpoint_fp8_serialized:
            raise ValueError(
                "FP8_PB_WO currently only supports FP8-serialized checkpoints."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Expose block size so the v2 weight loaders can translate offsets from
        # element-space -> block-space for BlockQuantScaleParameter.
        layer.weight_block_size = self.weight_block_size

        block_n, block_k = self._WEIGHT_BLOCK_SIZE
        remainder = output_size_per_partition % block_n
        self.output_padding = 0 if remainder == 0 else block_n - remainder
        self.logical_output_size = output_size_per_partition
        physical_output_size = output_size_per_partition + self.output_padding

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if input_size_per_partition % block_k != 0:
            raise ValueError(
                "ModelOpt FP8_PB_WO requires in_features divisible by "
                f"{block_k}, got {input_size_per_partition}."
            )

        out_blks = physical_output_size // block_n
        in_blks = input_size_per_partition // block_k

        # Match ModelOpt's exported shape so weight loading works without a
        # custom loader: [out_blk, 1, in_blk, 1]
        weight_scale = BlockQuantScaleParameter(
            data=torch.empty((out_blks, 1, in_blks, 1), dtype=torch.float32),
            input_dim=2,
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        self.w8a8_block_fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=(physical_output_size, input_size_per_partition),
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep weight in [out, in] layout for Fp8BlockScaledMMLinearKernel.
        weight = layer.weight.data
        if self.output_padding:
            padded_weight = weight.new_zeros(
                self.logical_output_size + self.output_padding,
                weight.shape[1],
            )
            padded_weight[: self.logical_output_size].copy_(weight)
            weight = padded_weight
        layer.weight = Parameter(weight, requires_grad=False)

        scale = layer.weight_scale
        if scale.dim() == 4:
            # [out_blk, 1, in_blk, 1] -> [out_blk, in_blk]
            scale = scale.squeeze(1).squeeze(-1)
        elif scale.dim() != 2:
            raise ValueError(
                "Unexpected ModelOpt FP8_PB_WO weight_scale shape: "
                f"{tuple(scale.shape)}."
            )

        layer.weight_scale = Parameter(scale.contiguous(), requires_grad=False)

        self.w8a8_block_fp8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kernel_bias = None if self.output_padding else bias
        output = self.w8a8_block_fp8_linear.apply_weights(layer, x, kernel_bias)
        if not self.output_padding:
            return output

        output = output[..., : self.logical_output_size].contiguous()
        if bias is not None:
            output.add_(bias)
        return output


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
            w13_input_scale,
            w2_input_scale,
            layer.moe_config.moe_parallel_config.enable_eplb,
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


ModelOptFp8Config.LinearMethodCls = ModelOptFp8LinearMethod
ModelOptFp8Config.FusedMoEMethodCls = ModelOptFp8MoEMethod
ModelOptFp8Config.KVCacheMethodCls = ModelOptKVCacheMethod


def _make_modelopt_nvfp4_moe_method(
    quant_config: "ModelOptNvFp4Config",
    layer: RoutedExperts,
) -> FusedMoEMethodBase:
    if layer.moe_config.moe_backend == "deep_gemm_mega_moe":
        # DeepSeek V4's model-level MegaMoE path constructs its own experts
        # directly, so it never reaches this RoutedExperts quant-method path.
        if quant_config.quant_method != "NVFP4":
            raise ValueError("deep_gemm_mega_moe requires NVFP4 W4A4, not W4A16.")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "NVFP4 deep_gemm_mega_moe does not support "
                "apply_router_weight_on_input=True."
            )
        return ModelOptNvFp4MegaMoE(
            quant_config=quant_config,
            moe_config=layer.moe_config,
        )
    return ModelOptNvFp4FusedMoE(
        quant_config=quant_config,
        moe_config=layer.moe_config,
    )


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

        # Select LinearMethod implementation based on quant_algo (FP8 pattern).
        # NVFP4         -> W4A4: cutlass NVFP4 GEMM with input quantization
        # W4A16_NVFP4   -> W4A16: FP4 Marlin GEMM with bf16/fp16 activations
        if quant_method == "NVFP4":
            self.LinearMethodCls = ModelOptNvFp4LinearMethod
        elif quant_method == "W4A16_NVFP4":
            self.LinearMethodCls = ModelOptNvFp4W4A16LinearMethod
        else:
            raise ValueError(
                f"Unsupported ModelOpt NVFP4 quant_algo: {quant_method}. "
                "Supported: NVFP4 / W4A16_NVFP4."
            )

    def get_name(self) -> QuantizationMethods:
        return "modelopt_fp4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        # Pure-NVFP4 checkpoints use this config directly rather than
        # ModelOptMixedPrecisionConfig.
        if isinstance(layer, RoutedExperts):
            if self.is_layer_excluded(prefix):
                return None
            return _make_modelopt_nvfp4_moe_method(self, layer)
        return super().get_quant_method(layer, prefix)

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
        self.marlin_input_dtype = None
        self.kernel = init_nvfp4_linear_kernel()

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

        # Input Global Scale
        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_scale", input_global_scale)

        # Weight Global Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_global_scale)

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

        expose_input_quant_key(layer, self.kernel)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (
            torch.unique(layer.input_scale).numel() != 1
            or torch.unique(layer.weight_scale_2).numel() != 1
        ):
            logger.warning_once(
                "In NVFP4 linear, the global scale for input or weight are different"
                " for parallel layers (e.g. q_proj, k_proj, v_proj). This "
                " will likely results in reduce accuracy. Please verify the model"
                " accuracy. Consider using a checkpoint with a shared global NVFP4"
                " scale for parallel layers."
            )

        # Rename ModelOpt checkpoint names to standardized names
        input_global_scale = layer.input_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
        del layer.input_scale

        weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
        layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
        del layer.weight_scale_2

        # Pre-compute alpha and inverse for runtime quantization
        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale, requires_grad=False
        )
        layer.input_global_scale_inv = Parameter(
            (1.0 / layer.input_global_scale).to(torch.float32), requires_grad=False
        )

        # Convert layer to NVFP4 linear kernel format
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)


class ModelOptNvFp4W4A16LinearMethod(LinearMethodBase):
    """Linear method for ModelOpt NVFP4 W4A16.

    4-bit NVFP4 weights, fp16/bf16 activations. Loads ModelOpt-style names
    directly (no on-disk conversion) and dispatches to a W4A16 GEMM:

        weight          uint8     packed NVFP4 (2 nibbles/byte along input dim)
        weight_scale    fp8-e4m3  per 16-elem group along input dim
        weight_scale_2  fp32      per-tensor global scale = amax / (6.0 * 448.0)

    No activation quantization. ModelOpt stores the global scale as
    amax/2688, so we rename weight_scale_2 -> weight_global_scale without
    reciprocation. The selected kernel converts it to its runtime format.
    The CT W4A16 path reciprocates because CT stores the inverse on disk.

    We also register a placeholder input_scale parameter so that W4A4-shaped
    checkpoints (which contain *_proj.input_scale tensors) can be loaded
    under this method without the per-shard loader hitting a KeyError on
    the merged-name lookup. The placeholder is discarded in
    process_weights_after_loading -- its value is never used.
    """

    def __init__(self, quant_config: ModelOptNvFp4Config) -> None:
        self.quant_config = quant_config
        self.marlin_input_dtype = None
        # `init_nvfp4_linear_kernel(use_a16=True)` is best of both worlds:
        # 1. `use_a16=True` forces  `Marlin`: https://github.com/vllm-project/vllm/commit/e68988a#diff-7135ab92aa94dfacb1ad3c77fc13f9c4ffe0b977f8eac5d86c2afe243e5f92a6R842-R889
        # for `--linear-backend=auto`, avoiding a W4A4 kernel that requires input_scale.
        # 2. Specifying e.g. `--linear-backend=humming` will override.
        self.kernel = init_nvfp4_linear_kernel(use_a16=True)

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
                "W4A16_NVFP4 quantization was selected; "
                "dynamic quantization is not supported."
            )
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.output_partition_sizes = output_partition_sizes

        if input_size_per_partition % 16 != 0:
            raise ValueError(
                "Unsupported model: input feature size is not a multiple of 16."
            )

        # Packed NVFP4 weights: uint8, 2 nibbles per byte along the input dim.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Per-tensor global weight scale (fp32). ModelOpt stores
        # amax / (NVFP4_max * fp8_e4m3_max) = amax / 2688. PerTensorScaleParameter
        # holds one entry per fused output partition (e.g. q/k/v in a fused QKV).
        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        # Per-group fp8 weight scale.
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # Placeholder input_scale param so W4A4-shaped checkpoints can be
        # loaded under this method without KeyError on the merged-name
        # lookup (qwen2-style stacked-loader path renames *_proj.input_scale
        # to e.g. qkv_proj.input_scale and looks it up unconditionally).
        # Discarded in process_weights_after_loading; never read by the kernel.
        # For native W4A16 checkpoints (no input_scale on disk) the param
        # stays uninitialized and is simply deleted.
        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "has_bias"):
            layer.has_bias = getattr(layer, "bias", None) is not None

        # Discard the input_scale placeholder. Whether it carries values
        # (W4A4 ckpt loaded as W4A16) or is uninitialized (native W4A16
        # ckpt), W4A16 mode does not quantize activations, so this is unused.
        if hasattr(layer, "input_scale"):
            del layer.input_scale

        if torch.unique(layer.weight_scale_2).numel() != 1:
            logger.warning_once(
                "In W4A16_NVFP4 linear, the global weight scale "
                "(weight_scale_2) differs across fused parallel layers "
                "(e.g. q/k/v_proj). This will likely reduce accuracy. "
                "Consider a checkpoint with a shared global scale."
            )

        # Rename weight_scale_2 -> weight_global_scale. NO reciprocation:
        # ModelOpt already stores amax/2688, which is exactly what Marlin
        # consumes via nvfp4_marlin_process_global_scale (called inside the
        # Marlin adapter's process_weights_after_loading).
        layer.weight_global_scale = Parameter(
            layer.weight_scale_2.max().to(torch.float32), requires_grad=False
        )
        del layer.weight_scale_2

        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)


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
            use_a16=self.use_a16,
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
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            swiglu_alpha=getattr(layer, "swiglu_alpha", None),
            swiglu_beta=getattr(layer, "swiglu_beta", None),
            layer=layer,
            use_a16=self.use_a16,
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
    ) -> torch.Tensor | UnfinalizedMoEOutput:
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


class ModelOptNvFp4MegaMoE(ModelOptNvFp4FusedMoE):
    """ModelOpt NVFP4 adapter for DeepGEMM's cooperative MegaMoE kernel.

    Unlike regular modular experts, MegaMoE owns EP dispatch, both expert
    GEMMs, combine, and (optionally) the BF16 shared MLP.  Keeping this as a
    quant method lets vLLM reuse RoutedExperts' existing ModelOpt checkpoint
    loader while preventing the runner from dispatching the same tokens twice.
    """

    def __init__(
        self,
        quant_config: ModelOptNvFp4Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        # Deliberately bypass ModelOptNvFp4FusedMoE.__init__: the regular NVFP4
        # oracle selects a GEMM-only Experts implementation, while MegaMoE is a
        # cooperative communication + compute kernel.  The inherited weight
        # loader uses only quant_config, use_a16, and use_global_sf; experts_cls
        # must remain unset because its presence marks the method monolithic.
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = quant_config
        self.use_a16 = False
        self.use_global_sf = True
        self._shared_experts_layer: torch.nn.Module | None = None
        self._shared_l1_weights: torch.Tensor | None = None
        self._shared_l2_weights: torch.Tensor | None = None
        self._num_shared_experts = 0
        self._routed_scaling_factor = 1.0
        self._deep_gemm: Any = None
        self._symm_buffer: Any = None

        parallel = moe_config.moe_parallel_config
        if not parallel.use_ep or parallel.ep_size <= 1:
            raise ValueError(
                "deep_gemm_mega_moe requires expert parallelism across at "
                "least two ranks."
            )
        if parallel.tp_size != 1:
            raise ValueError("deep_gemm_mega_moe requires TP=1 inside each EP rank.")
        if parallel.enable_eplb:
            raise NotImplementedError(
                "NVFP4 deep_gemm_mega_moe does not support EPLB yet."
            )
        if moe_config.activation.value != "silu":
            raise ValueError("NVFP4 deep_gemm_mega_moe currently requires SiLU.")
        if moe_config.in_dtype != torch.bfloat16:
            raise ValueError("NVFP4 deep_gemm_mega_moe requires BF16 activations.")
        if moe_config.has_bias:
            raise NotImplementedError(
                "NVFP4 deep_gemm_mega_moe does not support expert bias."
            )
        if moe_config.is_lora_enabled:
            raise NotImplementedError("NVFP4 deep_gemm_mega_moe does not support LoRA.")
        if moe_config.hidden_dim % 256 or moe_config.intermediate_size % 256:
            raise ValueError(
                "NVFP4 deep_gemm_mega_moe requires hidden and intermediate "
                "dimensions to be multiples of 256."
            )
        if moe_config.num_experts % parallel.ep_size != 0:
            raise ValueError(
                f"num_experts={moe_config.num_experts} must be divisible by "
                f"EP size {parallel.ep_size}."
            )

    @property
    def supports_internal_mk(self) -> bool:
        return True

    @property
    def supports_dbo(self) -> bool:
        return False

    @property
    def requires_moe_quant_config(self) -> bool:
        return False

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return self._shared_experts_layer is not None

    @property
    def mk_fuses_shared_experts(self) -> bool:
        return self._shared_experts_layer is not None

    @property
    def output_is_reduced(self) -> bool:
        # MegaMoE combine returns the completed output to each token's origin
        # rank, so the generic runner must not apply another cross-rank reduce.
        return True

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    @property
    def supports_eplb(self) -> bool:
        return False

    def bind_shared_experts(
        self,
        shared_experts: torch.nn.Module | None,
        *,
        routed_output_transform: torch.nn.Module | None = None,
    ) -> None:
        self._shared_experts_layer = None
        if shared_experts is None or routed_output_transform is not None:
            return

        gate_up = getattr(shared_experts, "gate_up_proj", None)
        down = getattr(shared_experts, "down_proj", None)
        act_fn = getattr(shared_experts, "act_fn", None)
        compatible = (
            not getattr(shared_experts, "shard_sequence_parallel", False)
            and getattr(shared_experts, "expert_gate", None) is None
            and gate_up is not None
            and down is not None
            and getattr(gate_up, "tp_size", None) == 1
            and getattr(down, "tp_size", None) == 1
            and getattr(gate_up, "bias", None) is None
            and getattr(down, "bias", None) is None
            and isinstance(
                getattr(gate_up, "quant_method", None), UnquantizedLinearMethod
            )
            and isinstance(getattr(down, "quant_method", None), UnquantizedLinearMethod)
            and isinstance(act_fn, SiluAndMul)
        )
        if not compatible:
            logger.info_once(
                "Shared expert is not compatible with DeepGEMM fusion; "
                "using the regular shared-expert path.",
                scope="local",
            )
            return

        gate_up_weight = getattr(gate_up, "weight", None)
        down_weight = getattr(down, "weight", None)
        hidden = self.moe.hidden_dim
        if (
            gate_up_weight is None
            or down_weight is None
            or gate_up_weight.dtype != torch.bfloat16
            or down_weight.dtype != torch.bfloat16
            or gate_up_weight.ndim != 2
            or down_weight.ndim != 2
            or gate_up_weight.shape[0] % 2
        ):
            return
        shared_intermediate = gate_up_weight.shape[0] // 2
        if (
            tuple(gate_up_weight.shape) != (2 * shared_intermediate, hidden)
            or tuple(down_weight.shape) != (hidden, shared_intermediate)
            or shared_intermediate % self.moe.intermediate_size
        ):
            return
        self._shared_experts_layer = shared_experts

    def bind_routed_scaling_factor(self, routed_scaling_factor: float) -> None:
        self._routed_scaling_factor = routed_scaling_factor

    @staticmethod
    def _pack_e4m3_scales(scale: torch.Tensor) -> torch.Tensor:
        if scale.dtype != torch.float8_e4m3fn:
            raise TypeError(f"Expected E4M3 block scales, got {scale.dtype}.")
        if scale.shape[-1] % 4:
            raise ValueError(
                "DeepGEMM packs four E4M3 scales per int32; got trailing "
                f"dimension {scale.shape[-1]}."
            )
        packed = scale.contiguous().view(torch.uint8).view(torch.int32)
        # Preserve the public [E, N, K/64] shape with the N-major TMA stride
        # expected by the NVFP4 MegaMoE kernel.
        return packed.transpose(-1, -2).contiguous().transpose(-1, -2)

    def _transform_shared_weights(self, deep_gemm) -> None:
        if self._shared_experts_layer is None or self._shared_l1_weights is not None:
            return

        cached_l1 = getattr(
            self._shared_experts_layer,
            "_deep_gemm_mega_shared_l1_weights",
            None,
        )
        if cached_l1 is not None:
            shared_layer = cast(Any, self._shared_experts_layer)
            self._shared_l1_weights = cached_l1
            self._shared_l2_weights = shared_layer._deep_gemm_mega_shared_l2_weights
            self._num_shared_experts = (
                self._shared_l1_weights.shape[0] // 2 // self.moe.intermediate_size
            )
            return

        gate_up = getattr(self._shared_experts_layer, "gate_up_proj", None)
        down = getattr(self._shared_experts_layer, "down_proj", None)
        if gate_up is None or down is None:
            raise TypeError(
                "DeepGEMM fused shared experts require gate_up_proj and down_proj."
            )

        shared_l1 = gate_up.weight.data
        shared_l2 = down.weight.data
        if shared_l1.dtype != torch.bfloat16 or shared_l2.dtype != torch.bfloat16:
            raise TypeError(
                "DeepGEMM fused shared experts require BF16 weights; got "
                f"{shared_l1.dtype} and {shared_l2.dtype}."
            )
        hidden = self.moe.hidden_dim
        if shared_l1.ndim != 2 or shared_l1.shape[0] % 2:
            raise ValueError(f"Unexpected shared L1 shape {tuple(shared_l1.shape)}.")
        shared_intermediate = shared_l1.shape[0] // 2
        if tuple(shared_l1.shape) != (2 * shared_intermediate, hidden) or tuple(
            shared_l2.shape
        ) != (hidden, shared_intermediate):
            raise ValueError(
                "DeepGEMM fused shared expert needs unsharded BF16 weights; got "
                f"L1={tuple(shared_l1.shape)}, L2={tuple(shared_l2.shape)}."
            )
        if shared_intermediate % self.moe.intermediate_size:
            raise ValueError(
                f"shared intermediate {shared_intermediate} must be a multiple "
                f"of routed intermediate {self.moe.intermediate_size}."
            )
        self._num_shared_experts = shared_intermediate // self.moe.intermediate_size
        self._shared_l1_weights, self._shared_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                shared_l1.contiguous(), shared_l2.contiguous()
            )
        )
        shared_layer = cast(Any, self._shared_experts_layer)
        shared_layer._deep_gemm_mega_shared_l1_weights = self._shared_l1_weights
        shared_layer._deep_gemm_mega_shared_l2_weights = self._shared_l2_weights
        gate_up.weight = None
        down.weight = None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        required_apis = (
            "transform_weights_for_mega_moe",
            "get_symm_buffer_for_mega_moe",
            "fp4_fp4_mega_moe",
        )
        if not envs.VLLM_USE_DEEP_GEMM:
            raise RuntimeError(
                "deep_gemm_mega_moe was selected while VLLM_USE_DEEP_GEMM=0."
            )
        if not current_platform.is_device_capability_family(100):
            raise RuntimeError("NVFP4 deep_gemm_mega_moe requires SM100-family GPUs.")
        if deep_gemm is None or any(
            not hasattr(deep_gemm, api) for api in required_apis
        ):
            raise RuntimeError(
                "Installed DeepGEMM does not provide the NVFP4 MegaMoE APIs."
            )
        if not hasattr(layer, "_deep_gemm_mega_l1_weights"):
            w13_scale = self._pack_e4m3_scales(layer.w13_weight_scale.data)
            w2_scale = self._pack_e4m3_scales(layer.w2_weight_scale.data)
            l1_weights, l2_weights = deep_gemm.transform_weights_for_mega_moe(
                (layer.w13_weight.data.view(torch.int8).contiguous(), w13_scale),
                (layer.w2_weight.data.view(torch.int8).contiguous(), w2_scale),
            )
            layer._deep_gemm_mega_l1_weights = l1_weights
            layer._deep_gemm_mega_l2_weights = l2_weights

            a1_scale = layer.w13_input_scale.data.max().float().reshape(())
            if not bool(torch.isfinite(a1_scale).item()) or a1_scale.item() <= 0:
                raise ValueError(
                    "NVFP4 MegaMoE activation scale must be positive and finite."
                )
            layer._deep_gemm_mega_a1_scale = a1_scale
            layer._deep_gemm_mega_a1_gscale = a1_scale.reciprocal()
            layer._deep_gemm_mega_l1_alphas = (
                layer.w13_weight_scale_2.data.float().reshape(-1, 2).contiguous()
                * a1_scale
            )
            layer._deep_gemm_mega_l2_alphas = (
                layer.w2_weight_scale_2.data.float().reshape(-1).contiguous()
            )
            a2_scale = layer.w2_input_scale.data.max().float().reshape(())
            layer._deep_gemm_mega_a2_scales = layer._deep_gemm_mega_l2_alphas.new_full(
                layer._deep_gemm_mega_l2_alphas.shape,
                a2_scale,
            )

            # The transformed L2 weight aliases its loader Parameter. The other
            # transformed tensors own fresh storage, so dropping the loader-side
            # Parameters releases only redundant memory.
            layer.w13_weight = None
            layer.w13_weight_scale = None
            layer.w13_weight_scale_2 = None
            layer.w13_input_scale = None
            layer.w2_weight = None
            layer.w2_weight_scale = None
            layer.w2_weight_scale_2 = None
            layer.w2_input_scale = None

        if not hasattr(layer, "_deep_gemm_mega_a1_gscale"):
            layer._deep_gemm_mega_a1_gscale = (
                layer._deep_gemm_mega_a1_scale.reciprocal()
            )

        self._transform_shared_weights(deep_gemm)
        self._initialize_runtime(deep_gemm)

        logger.info_once(
            "Using DeepGEMM NVFP4 MegaMoE (fused_shared=%s).",
            self._shared_experts_layer is not None,
            scope="global",
        )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None

    def _initialize_runtime(self, deep_gemm) -> None:
        from vllm.distributed import get_ep_group

        ep_group = get_ep_group()
        device = torch.accelerator.current_device_index()
        key = (
            device,
            self.moe.num_experts,
            self.moe.max_num_tokens,
            self.moe.experts_per_token,
            self.moe.hidden_dim,
            self.moe.intermediate_size,
            self._num_shared_experts,
        )
        cache = getattr(ep_group, "_modelopt_nvfp4_mega_moe_buffers", None)
        if cache is None:
            cache = {}
            cast(Any, ep_group)._modelopt_nvfp4_mega_moe_buffers = cache
        symm_buffer = cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                ep_group.device_group,
                self.moe.num_experts,
                self.moe.max_num_tokens,
                self.moe.experts_per_token,
                self.moe.hidden_dim,
                self.moe.intermediate_size,
                num_shared_experts=self._num_shared_experts,
                mma_type="fp4xfp4",
            )
            cache[key] = symm_buffer
        self._deep_gemm = deep_gemm
        self._symm_buffer = symm_buffer

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise TypeError(f"NVFP4 MegaMoE expects BF16 input, got {x.dtype}.")
        num_tokens = x.shape[0]
        if num_tokens > self.moe.max_num_tokens:
            raise ValueError(
                f"NVFP4 MegaMoE got M={num_tokens}, capacity={self.moe.max_num_tokens}."
            )
        if self._symm_buffer is None or self._deep_gemm is None:
            raise RuntimeError(
                "NVFP4 MegaMoE runtime was not initialized after weight loading."
            )
        symm_buffer = self._symm_buffer
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        prepare_nvfp4_megamoe_inputs(
            x,
            layer._deep_gemm_mega_a1_gscale,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            is_padding=is_padding,
        )

        y = torch.empty_like(x)
        shared_kwargs = {}
        if self._shared_experts_layer is not None:
            if shared_experts_input is None:
                raise RuntimeError("Fused shared experts require their BF16 input.")
            shared_kwargs = {
                "shared_l1_weights": self._shared_l1_weights,
                "shared_l2_weights": self._shared_l2_weights,
                "x_bf16": shared_experts_input,
            }

        self._deep_gemm.fp4_fp4_mega_moe(
            y,
            layer._deep_gemm_mega_l1_weights,
            layer._deep_gemm_mega_l2_weights,
            symm_buffer,
            activation_clamp=getattr(layer, "swiglu_limit", None),
            fast_math=True,
            l1_alphas=layer._deep_gemm_mega_l1_alphas,
            l2_alphas=layer._deep_gemm_mega_l2_alphas,
            a2_scales=layer._deep_gemm_mega_a2_scales,
            routed_scaling_factor=(
                self._routed_scaling_factor
                if self._shared_experts_layer is not None
                else 1.0
            ),
            **shared_kwargs,
        )
        return y


ModelOptNvFp4Config.LinearMethodCls = ModelOptNvFp4LinearMethod
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


class ModelOptMxFp8LinearMethod(LinearMethodBase):
    """Linear method for ModelOpt MXFP8 quantization."""

    def __init__(self, quant_config: ModelOptMxFp8Config) -> None:
        self.quant_config = quant_config

        if not self.quant_config.is_checkpoint_mxfp8_serialized:
            raise ValueError(
                "MXFP8 currently only supports serialized checkpoints. "
                "Dynamic quantization is not supported."
            )

        self.kernel = init_mxfp8_linear_kernel()

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

        if not self.quant_config.is_checkpoint_mxfp8_serialized:
            raise ValueError(
                "MXFP8 quantization was selected, but checkpoint is not "
                "MXFP8 serialized. Dynamic quantization is not supported."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires input dimension to be divisible by "
                f"{MXFP8_BLOCK_SIZE}, got {input_size_per_partition}"
            )

        # Weight tensor: FP8 E4M3 format
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Weight scale tensor (E8M0 encoded as uint8), one scale per block of 32 along K
        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Idempotent: the emulation kernel may dequant the weight to BF16 at load
        # time (>=2-byte). If already converted, there is nothing left to do --
        # avoid re-running the MXFP8-only validation/conversion below.
        if layer.weight.element_size() >= 2:
            return

        # Validate weight tensor
        if layer.weight.ndim != 2:
            raise ValueError(
                f"MXFP8 weight must be 2D tensor [N, K], got {layer.weight.ndim}D "
                f"with shape {tuple(layer.weight.shape)}"
            )

        if layer.weight.dtype != MXFP8_VALUE_DTYPE:
            raise ValueError(
                f"MXFP8 weight must be {MXFP8_VALUE_DTYPE} (FP8 E4M3), "
                f"got {layer.weight.dtype}. The checkpoint may not be properly "
                f"quantized with MXFP8."
            )

        # Validate weight scale tensor (should be 2D, not swizzled)
        assert layer.weight_scale.ndim == 2, (
            f"MXFP8 weight scale must be 2D, got {layer.weight_scale.ndim}D"
        )
        assert layer.weight_scale.dtype == MXFP8_SCALE_DTYPE, (
            f"MXFP8 weight scale must be {MXFP8_SCALE_DTYPE},"
            f" got {layer.weight_scale.dtype}"
        )

        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


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
ModelOptMxFp8Config.LinearMethodCls = ModelOptMxFp8LinearMethod
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
        # ModelOptNvFp4Config.__init__ keys LinearMethodCls off quant_method,
        # so this instance auto-selects ModelOptNvFp4W4A16LinearMethod. The
        # MoE side reads quant_config.quant_method == "W4A16_NVFP4" to set
        # use_a16 → Marlin backend in ModelOptNvFp4FusedMoE.__init__.
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

        # RoutedExperts expert prefix is e.g. "...moe.experts", while ModelOpt's
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
            if quant_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)
            if quant_algo == "FP8_PB_WO":
                return ModelOptFp8PbWoLinearMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4W4A16LinearMethod(self.w4a16_nvfp4_config)
            if quant_algo == "MXFP8":
                return ModelOptMxFp8LinearMethod(self.mxfp8_config)
            # Layer not in quantized_layers — leave unquantized
            return UnquantizedLinearMethod()

        if isinstance(layer, RoutedExperts):
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(
                    quant_config=self.fp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "NVFP4":
                return _make_modelopt_nvfp4_moe_method(
                    self.nvfp4_config,
                    layer,
                )
            if quant_algo == "W4A16_NVFP4":
                return _make_modelopt_nvfp4_moe_method(
                    self.w4a16_nvfp4_config,
                    layer,
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
