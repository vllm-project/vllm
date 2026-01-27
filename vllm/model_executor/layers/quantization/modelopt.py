# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.attention.layer import Attention
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_kernel_for_mkm,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    is_global_sf_supported_for_nvfp4_backend,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_kernel_for_mkm,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
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
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    build_flashinfer_fp4_cutlass_moe_prepare_finalize,
    flashinfer_trtllm_fp4_moe,
    flashinfer_trtllm_fp4_routed_moe,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    apply_fi_trtllm_fp8_per_tensor_moe,
    build_flashinfer_fp8_cutlass_moe_prepare_finalize,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_tensor_strategy_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    cutlass_fp4_supported,
    is_layer_skipped,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
    kNvfp4Dynamic,
    kNvfp4Static,
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
    requantize_with_max_scale,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer,
)

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
    # FP4
    "NVFP4",
]
KV_CACHE_QUANT_ALGOS = ["FP8"]


class ModelOptFp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
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
    ) -> Optional["QuantizeMethodBase"]:
        # handle kv-cache first so we can focus only on weight quantization thereafter
        if isinstance(layer, Attention):
            return self.KVCacheMethodCls(self)

        # handle exclusion
        if self.is_layer_excluded(prefix):
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None

        # TODO: This special hard coded logic is not needed for quantized checkpoints
        # generated by ModelOpt >= 0.39.0 where they are handled natually by the
        # exclude_modules config. But need to keep them for loading quantized
        # checkpoints generated by older versions. Then check substring matching
        # for patterns not caught by exact match
        if "vision_tower" in prefix or "vision_model" in prefix:
            return UnquantizedLinearMethod()

        # now, the layer is quantized, handle it here
        if isinstance(layer, LinearBase):
            quant_method = self.LinearMethodCls(self)
            if getattr(quant_method, "backend", "") == "marlin":
                quant_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
            return quant_method
        elif isinstance(layer, FusedMoE):
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
            # Compressed-tensors style format:
            # {"quant_algo": "...", "quant_method": "modelopt"}
            quant_method = config.get("quant_algo")
            kv_cache_quant_method = config.get("kv_cache_quant_algo")
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
        return 89

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
                quant_algo = str(quant_config.get("quant_algo", ""))
                if "FP8" in quant_algo.upper():
                    return "modelopt"
        else:
            # Check for compressed-tensors style config with specific quant_algo
            quant_algo = str(hf_quant_cfg.get("quant_algo", ""))
            if "FP8" in quant_algo.upper():
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
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            out_dtype=torch.get_default_dtype(),
            module_name=self.__class__.__name__,
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
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticTokenSym,
            out_dtype=torch.get_default_dtype(),
            module_name=self.__class__.__name__,
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

    def process_weights_after_loading(self, layer: Module) -> None:
        layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

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
    """

    _WEIGHT_BLOCK_SIZE: tuple[int, int] = (128, 128)

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config
        block_n, block_k = self._WEIGHT_BLOCK_SIZE
        self.weight_block_size = list(self._WEIGHT_BLOCK_SIZE)
        self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
            weight_group_shape=GroupShape(block_n, block_k),
            act_quant_group_shape=GroupShape(1, block_k),
            cutlass_block_fp8_supported=cutlass_block_fp8_supported(),
            use_aiter_and_is_supported=False,
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

        block_n, block_k = self._WEIGHT_BLOCK_SIZE
        if output_size_per_partition % block_n != 0:
            raise ValueError(
                "ModelOpt FP8_PB_WO requires out_features divisible by "
                f"{block_n}, got {output_size_per_partition}."
            )
        if input_size_per_partition % block_k != 0:
            raise ValueError(
                "ModelOpt FP8_PB_WO requires in_features divisible by "
                f"{block_k}, got {input_size_per_partition}."
            )

        out_blks = output_size_per_partition // block_n
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

    def process_weights_after_loading(self, layer: Module) -> None:
        # Keep weight in [out, in] layout for W8A8BlockFp8LinearOp.
        layer.weight = Parameter(layer.weight.data, requires_grad=False)

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

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.w8a8_block_fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=None,
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

        # Delay creation of the kernel until after process-weights.
        self.kernel: mk.FusedMoEModularKernel | None = None

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        if self.kernel is not None:
            return self.kernel.prepare_finalize.topk_indices_dtype()
        return None

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalize | None:
        # TRT LLM not supported with all2all yet.
        if self.fp8_backend == Fp8MoeBackend.FLASHINFER_TRTLLM:
            return None
        elif self.fp8_backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
            # For no-EP case, don't use the MKM framework.
            if not self.moe.moe_parallel_config.use_all2all_kernels:
                return None

            prepare_finalize = build_flashinfer_fp8_cutlass_moe_prepare_finalize(
                self.moe,
                use_deepseek_fp8_block_scale=False,
            )
            logger.debug_once("%s", prepare_finalize.__class__.__name__)
            return prepare_finalize
        return super().maybe_make_prepare_finalize(routing_tables)

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        return make_fp8_moe_kernel_for_mkm(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            experts_cls=self.experts_cls,
            prepare_finalize=prepare_finalize,
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
        layer: torch.nn.Module,
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
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.kernel, self.use_inplace = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
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
        if layer.enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for FlashInfer TRTLLM FP8 MoE Backend."
            )
        # TODO(rob): this validation should happen at kernel selection
        # time in the oracle rather than here.
        assert layer.activation == "silu", (
            f"Expected 'silu' activation but got {layer.activation}"
        )
        assert not layer.renormalize
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

        # TODO(rob): this validation should happen at kernel selection
        # time in the oracle rather than here.
        if self.fp8_backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
            assert layer.activation in ("silu", "relu2_no_mul"), (
                "Expected activation to be in ('silu', 'relu2_no_mul'),"
                f"but got {layer.activation}"
            )

        assert self.kernel is not None
        return self.kernel(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            inplace=self.use_inplace,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )


ModelOptFp8Config.LinearMethodCls = ModelOptFp8LinearMethod
ModelOptFp8Config.FusedMoEMethodCls = ModelOptFp8MoEMethod
ModelOptFp8Config.KVCacheMethodCls = ModelOptFp8KVCacheMethod


class ModelOptNvFp4Config(ModelOptQuantConfigBase):
    """Config class for ModelOpt FP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool,
        kv_cache_quant_algo: str | None,
        exclude_modules: list[str],
        group_size: int = 16,
    ) -> None:
        super().__init__(exclude_modules)
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected ModelOpt NVFP4 checkpoint. Please note that"
                " the format is experimental and could change in future."
            )

            self.group_size = group_size
            self.kv_cache_quant_algo = kv_cache_quant_algo

    def get_name(self) -> QuantizationMethods:
        return "modelopt_fp4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

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
        elif envs.VLLM_NVFP4_GEMM_BACKEND == "marlin":
            self.backend = "marlin"
            assert is_fp4_marlin_supported(), f"Marlin is required for {self.backend}"

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
            # Swizzle block scales and pad the packed NVFP4 weights for kernel
            # alignment (CUTLASS/FlashInfer require K and N divisible by 32).
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)

            weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
                layer.weight.data
            )
            layer.weights_padding_cols = weights_padding_cols
            layer.weight = Parameter(weight, requires_grad=False)

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
                input_dtype=self.marlin_input_dtype,
            )

        output_dtype = x.dtype

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_blockscale = scaled_fp4_quant(
            x, layer.input_scale_inv, is_sf_swizzled_layout=True, backend=self.backend
        )

        # validate dtypes of quantized input, input block scale,
        # weight and weight_blockscale
        assert x_fp4.dtype == torch.uint8
        assert layer.weight.dtype == torch.uint8
        assert x_blockscale.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        # Pad activations to match weight K-dimension padding
        weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
        output_size = layer.output_size_per_partition
        output_shape = [x.shape[0], output_size]
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

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

        # Slice output to remove N-dimension padding
        out = slice_nvfp4_output(out, output_size)

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
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(moe_config)
        self.quant_config = quant_config
        # Select experts implementation.
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )

        # Delay creation of the kernel until after process-weights.
        self.kernel: mk.FusedMoEModularKernel | None = None

        self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
            self.nvfp4_backend
        )

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        if self.kernel is not None:
            return self.kernel.prepare_finalize.topk_indices_dtype()
        return None

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalize | None:
        if self.nvfp4_backend == NvFp4MoeBackend.FLASHINFER_TRTLLM:
            return None
        elif self.nvfp4_backend == NvFp4MoeBackend.FLASHINFER_CUTLASS:
            # For no-EP case, don't use the MKM framework.
            if not self.moe.moe_parallel_config.use_all2all_kernels:
                return None
            # For now, fp4 moe only works with the flashinfer dispatcher.
            prepare_finalize = build_flashinfer_fp4_cutlass_moe_prepare_finalize(
                self.moe
            )
            logger.debug_once("%s", prepare_finalize.__class__.__name__)
            return prepare_finalize
        else:
            return super().maybe_make_prepare_finalize(routing_tables)

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        return make_nvfp4_moe_kernel_for_mkm(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
            experts_cls=self.experts_cls,
            prepare_finalize=prepare_finalize,
        )

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
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

        # Setup modular kernel for TP case and naive DP/EP case.
        # In non-naive DP/EP case, we will create a ModularKernelMethod.
        # TODO(rob): unify these so FP8MoEMethod owns the ModularKernel
        # in both cases.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config and (
            (not self.moe.moe_parallel_config.use_all2all_kernels)
            or self.moe.moe_parallel_config.use_naive_all2all_kernels
        ):
            assert self.experts_cls is not None
            self.kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
            )

    @property
    def do_post_quant_allgather(self):
        return self.nvfp4_backend == NvFp4MoeBackend.FLASHINFER_TRTLLM

    def prepare_dp_allgather_tensor(
        self,
        layer: FusedMoE,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Optionally prepare extra tensors to carry through DP allgather/EP."""
        if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:
            raise RuntimeError(
                "prepare_dp_allgather_tensor is only supported for "
                "FlashInfer TRTLLM NVFP4 MoE backend."
            )

        import flashinfer

        hidden_states_fp4, hidden_states_sf = flashinfer.fp4_quantize(
            hidden_states,
            layer.a1_gscale,
            is_sf_swizzled_layout=False,
        )
        extra_tensors: list[torch.Tensor] = [hidden_states_sf]
        return hidden_states_fp4, extra_tensors

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
    def supports_eplb(self) -> bool:
        return True

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
            assert self.kernel is not None
            return self.kernel(
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


ModelOptNvFp4Config.LinearMethodCls = ModelOptNvFp4LinearMethod
ModelOptNvFp4Config.FusedMoEMethodCls = ModelOptNvFp4FusedMoE
ModelOptNvFp4Config.KVCacheMethodCls = ModelOptFp8KVCacheMethod
