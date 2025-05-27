# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm._custom_ops import (cutlass_scaled_fp4_mm,
                              cutlass_scaled_mm_supports_fp4, scaled_fp4_quant)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear, is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin, prepare_moe_fp4_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, requantize_with_max_scale)
from vllm.model_executor.parameter import (ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

QUANT_ALGOS = ["FP8", "NVFP4"]
KV_CACHE_QUANT_ALGOS = ["FP8"]


class ModelOptFp8Config(QuantizationConfig):
    """Config class for ModelOpt FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
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
    def from_config(cls, config: dict[str, Any]) -> "ModelOptFp8Config":
        quant_config = cls.get_from_keys(config, ["quantization"])
        quant_method = quant_config["quant_algo"]
        if quant_method not in QUANT_ALGOS:
            raise ValueError(f"ModelOpt currently only supports: {QUANT_ALGOS}"
                             " quantizations in vLLM. Please check the "
                             "`hf_quant_config.json` file for your model's "
                             "quant configuration.")
        is_checkpoint_fp8_serialized = ("FP8" in quant_method)

        return cls(is_checkpoint_fp8_serialized)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import
        if isinstance(layer, LinearBase):
            return ModelOptFp8LinearMethod(self)
        elif isinstance(layer, Attention):
            return ModelOptFp8KVCacheMethod(self)
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

    def __init__(self, quant_config: ModelOptFp8Config):
        self.quant_config = quant_config
        self.fp8_linear = Fp8LinearOp()

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


class ModelOptNvFp4Config(QuantizationConfig):
    """Config class for ModelOpt FP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool,
        kv_cache_quant_algo: str,
        exclude_modules: list[str],
        group_size: int = 16,
    ) -> None:
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
    def from_config(cls, config: dict[str, Any]) -> "ModelOptNvFp4Config":
        quant_config = cls.get_from_keys(config, ["quantization"])
        quant_method = quant_config["quant_algo"]
        if quant_method not in QUANT_ALGOS:
            raise ValueError(f"ModelOpt currently only supports: {QUANT_ALGOS}"
                             " quantizations in vLLM. Please check the "
                             "`hf_quant_config.json` file for your model's "
                             "quant configuration.")
        is_checkpoint_nvfp4_serialized = ("NVFP4" in quant_method)
        if ("group_size" and "kv_cache_quant_algo"
                and "exclude_modules") not in quant_config:
            raise ValueError("NVFP4 quantization requires group size and "
                             "kv_cache_quant_algo specified in "
                             "hf_quant_config.json")
        kv_cache_quant_algo = quant_config["kv_cache_quant_algo"]
        group_size = quant_config["group_size"]
        exclude_modules = quant_config["exclude_modules"]
        return cls(is_checkpoint_nvfp4_serialized, kv_cache_quant_algo,
                   exclude_modules, group_size)

    def is_layer_excluded(self, prefix: str, exclude_modules: list):
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


def cutlass_fp4_supported() -> bool:
    if not current_platform.is_cuda():
        return False
    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()
    return cutlass_scaled_mm_supports_fp4(capability)


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

    def __init__(self, quant_config: ModelOptNvFp4Config):
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

    def swizzle_blockscale(self, scale: torch.tensor):
        assert (scale.dtype == torch.float8_e4m3fn)
        # Pad and blockwise interleave weight_scale
        scale_ndim = scale.ndim
        if scale.ndim == 2:
            scale = scale.unsqueeze(0)
        assert scale.ndim == 3
        B, M, K = scale.shape
        round_up_multiple = lambda x, m: (x + m - 1) // m * m
        M_padded = round_up_multiple(M, 128)
        K_padded = round_up_multiple(K, 4)
        padded_scale = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype)
        padded_scale[:B, :M, :K] = scale
        batches, rows, cols = padded_scale.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32,
                                            cols // 4, 4)
        swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
        swizzled_scale = swizzled_scale.contiguous().cuda()
        return (swizzled_scale.reshape(M, K)
                if scale_ndim == 2 else swizzled_scale.reshape(B, M, K))

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
        assert (layer.weight_scale.shape[1] % 16 == 0), (
            "Expected weight_scale.dim(1) to be divisible by 16")
        assert (layer.weight_scale.dtype == torch.float8_e4m3fn), (
            "Weight Block scale must be represented as FP8-E4M3")
        swizzled_weight_scale = self.swizzle_blockscale(layer.weight_scale)

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


class ModelOptNvFp4FusedMoE(FusedMoEMethodBase):
    """
    MoE Method for FP4 Quantization.
    Args: 
        quant_config: NVFP4 Quant Config
    """

    def __init__(self, quant_config: ModelOptNvFp4Config):
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

    def swizzle_blockscale(self, scale: torch.tensor):
        assert (scale.dtype == torch.float8_e4m3fn)
        # Pad and blockwise interleave weight_scale
        scale_ndim = scale.ndim
        if scale.ndim == 2:
            scale = scale.unsqueeze(0)
        assert scale.ndim == 3
        B, M, K = scale.shape
        round_up_multiple = lambda x, m: (x + m - 1) // m * m
        M_padded = round_up_multiple(M, 128)
        K_padded = round_up_multiple(K, 4)
        padded_scale = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype)
        padded_scale[:B, :M, :K] = scale
        batches, rows, cols = padded_scale.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32,
                                            cols // 4, 4)
        swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
        swizzled_scale = swizzled_scale.contiguous().cuda()
        return (swizzled_scale.reshape(M, K)
                if scale_ndim == 2 else swizzled_scale.reshape(B, M, K))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # GEMM 1
        assert torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]), (
                "w1_weight_scale_2 must match w3_weight_scale_2")

        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
        layer.w13_weight_scale_2 = Parameter(w13_weight_scale_2,
                                             requires_grad=False)

        w13_input_scale = layer.w13_input_scale.max(dim=1).values.to(
            torch.float32)
        layer.g1_alphas = Parameter(
            (w13_input_scale * w13_weight_scale_2).to(torch.float32),
            requires_grad=False)

        assert (layer.w13_weight_scale.shape[2] % 16 == 0), (
            "Expected weight_scale.dim(1) to be divisible by 16")
        assert (layer.w13_weight_scale.dtype == torch.float8_e4m3fn), (
            "Weight Blockscale must be represented as FP8-E4M3")
        w13_blockscale_swizzled = self.swizzle_blockscale(
            layer.w13_weight_scale)

        layer.w13_blockscale_swizzled = Parameter(w13_blockscale_swizzled,
                                                  requires_grad=False)

        # This is for quantization, so we need to invert it.
        layer.w13_input_scale_quant = Parameter(
            (1 / w13_input_scale).to(torch.float32), requires_grad=False)

        layer.w13_weight = Parameter(layer.w13_weight.data,
                                     requires_grad=False)

        # GEMM 2
        layer.g2_alphas = Parameter(
            (layer.w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False)

        # This is for quantization, so we need to invert it.
        layer.w2_input_scale_quant = Parameter(
            (1 / layer.w2_input_scale).to(torch.float32), requires_grad=False)

        assert (layer.w2_weight_scale.shape[2] % 16 == 0), (
            "Expected weight_scale.dim(1) to be divisible by 16")
        assert (layer.w2_weight_scale.dtype == torch.float8_e4m3fn), (
            "Weight Blockscale must be represented as FP8-E4M3")
        w2_blockscale_swizzled = self.swizzle_blockscale(layer.w2_weight_scale)

        layer.w2_blockscale_swizzled = Parameter(w2_blockscale_swizzled,
                                                 requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)

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
    ):
        if self.use_marlin:
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
                global_num_experts=global_num_experts,
                expert_map=expert_map)

        assert activation == "silu", "Only SiLU activation is supported."
        assert not apply_router_weight_on_input, (
            "Router weight on input is not "
            "supported for ModelOptNvFp4FusedMoE.")
        assert expert_map is None, ("Expert Parallelism / expert_map "
                                    "is currently not supported for "
                                    "ModelOptNvFp4FusedMoE.")

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

        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            cutlass_moe_fp4)

        # Cutlass moe takes in activations in BF16/Half precision
        # and fp4 quantized weights loaded from the checkpoint
        return cutlass_moe_fp4(a=x,
                               w1_fp4=layer.w13_weight,
                               w1_blockscale=layer.w13_blockscale_swizzled,
                               w1_alphas=layer.g1_alphas,
                               w2_fp4=layer.w2_weight,
                               w2_blockscale=layer.w2_blockscale_swizzled,
                               w2_alphas=layer.g2_alphas,
                               topk_weights=topk_weights,
                               topk_ids=topk_ids,
                               m=x.shape[0],
                               n=layer.w2_weight.shape[2] * 2,
                               k=x.shape[1],
                               e=layer.w13_weight.shape[0],
                               a1_gscale=layer.w13_input_scale_quant,
                               a2_gscale=layer.w2_input_scale_quant,
                               device=x.device).to(x.dtype)
