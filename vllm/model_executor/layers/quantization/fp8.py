# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE, FusedMoEActivationFormat, FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize,
    FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig, fp8_w8a8_moe_quant_config)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend, apply_flashinfer_per_tensor_scale_fp8,
    build_flashinfer_fp8_cutlass_moe_prepare_finalize,
    flashinfer_cutlass_moe_fp8, get_flashinfer_moe_backend,
    register_moe_scaling_factors, rotate_flashinfer_fp8_moe_weights,
    select_cutlass_fp8_gemm_impl, swap_w13_to_w31)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    apply_fp8_block_linear, check_aiter_fp8_linear_support,
    create_fp8_input_scale, create_fp8_scale_parameter,
    create_fp8_weight_parameter, get_col_major_tma_aligned_tensor,
    maybe_post_process_fp8_weight_block, process_fp8_weight_block_strategy,
    process_fp8_weight_tensor_strategy, requant_weight_ue8m0_inplace,
    validate_fp8_block_shape)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin,
    prepare_moe_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape, is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, all_close_1d, cutlass_block_fp8_supported,
    cutlass_fp8_supported, maybe_create_device_identity,
    normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils import has_deep_gemm
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used, is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer_moe

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


def _is_col_major(x: torch.Tensor) -> bool:
    assert x.dim() == 3
    b, m, n = x.shape
    return x.stride(0) == m * n and x.stride(1) == 1 and x.stride(2) == m


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[list[str]] = None,
        weight_block_size: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now.")
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions")
            if activation_scheme != "dynamic":
                raise ValueError("The block-wise quantization only supports "
                                 "dynamic activation scheme for now, but got "
                                 f"{activation_scheme} activation scheme.")
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(
                self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"],
                                                 None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config,
                                                  ["modules_to_not_convert"],
                                                  None)
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme,
                   ignored_layers=ignored_layers,
                   weight_block_size=weight_block_size)

    def get_xpu_quant_method(self, layer: torch.nn.Module,
                             prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention
        from vllm.model_executor.layers.quantization.ipex_quant import (
            XPUFp8LinearMethod, XPUFp8MoEMethod)
        fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=self.is_checkpoint_fp8_serialized,
            activation_scheme=self.activation_scheme,
            ignored_layers=self.ignored_layers,
            weight_block_size=self.weight_block_size)

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix=prefix,
                                ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return XPUFp8LinearMethod(fp8_config)
        elif isinstance(layer, FusedMoE):
            return XPUFp8MoEMethod(fp8_config, layer)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if current_platform.is_xpu():
            return self.get_xpu_quant_method(layer, prefix)
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix=prefix,
                                ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self, layer)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> Optional[str]:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        # If no matches, return None
        return None


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (not current_platform.has_device_capability(89)
                           or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        self.use_aiter_and_is_supported = check_aiter_fp8_linear_support()

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"
        # Use per-token quantization for better perf if dynamic and cutlass
        if not self.act_q_static and cutlass_fp8_supported():
            self.act_q_group_shape = GroupShape.PER_TOKEN
        else:
            self.act_q_group_shape = GroupShape.PER_TENSOR

        self.fp8_linear = Fp8LinearOp(
            act_quant_static=self.act_q_static,
            act_quant_group_shape=self.act_q_group_shape)

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
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            validate_fp8_block_shape(layer, input_size, output_size,
                                     input_size_per_partition,
                                     output_partition_sizes,
                                     self.weight_block_size)

        # WEIGHT
        if self.quant_config.is_checkpoint_fp8_serialized:
            weight = create_fp8_weight_parameter(output_size_per_partition,
                                                 input_size_per_partition,
                                                 weight_loader)
        else:
            # For non-serialized checkpoints, use original dtype
            weight = ModelWeightParameter(data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype),
                                          input_dim=1,
                                          output_dim=0,
                                          weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if not self.block_quant:
                scale = create_fp8_scale_parameter(PerTensorScaleParameter,
                                                   output_partition_sizes,
                                                   input_size_per_partition,
                                                   None, weight_loader)
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                layer.register_parameter("weight_scale", scale)
            else:
                assert not self.act_q_static
                assert self.weight_block_size is not None
                scale = create_fp8_scale_parameter(BlockQuantScaleParameter,
                                                   output_partition_sizes,
                                                   input_size_per_partition,
                                                   self.weight_block_size,
                                                   weight_loader)
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                # The weight_scale_inv name is intentional for deepseekv3
                layer.register_parameter("weight_scale_inv", scale)

            # INPUT ACTIVATION SCALE
            if self.act_q_static:
                scale = create_fp8_input_scale(output_partition_sizes,
                                               weight_loader)
                set_weight_attrs(scale, {"scale_type": "input_scale"})
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        size_k_first = True
        input_scale = None
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert not self.act_q_static
            size_k_first = False

            weight, weight_scale = process_fp8_weight_block_strategy(
                layer.weight, layer.weight_scale_inv)
            # Delete the weight_scale_inv parameter to avoid confusion
            # with the weight_scale parameter
            del layer.weight_scale_inv

        # If checkpoint not serialized fp8, quantize the weights.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)
            weight = qweight.t()

        # If checkpoint is fp8 per-tensor, handle that there are N scales for N
        # shards in a fused module
        else:
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            if not self.use_marlin:
                weight, weight_scale, input_scale = (
                    process_fp8_weight_tensor_strategy(
                        weight, weight_scale, layer.logical_widths,
                        getattr(layer, 'input_scale', None)))
                if self.act_q_static:
                    assert input_scale is not None
                    input_scale = input_scale.max()
            weight = weight.t()

        # Update layer with new values.
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        layer.input_scale = Parameter(
            input_scale,
            requires_grad=False) if input_scale is not None else None

        if self.use_marlin:
            prepare_fp8_layer_for_marlin(layer, size_k_first)
            # Activations not quantized for marlin.
            del layer.input_scale
            return

        if self.block_quant:
            maybe_post_process_fp8_weight_block(
                layer, self.cutlass_block_fp8_supported)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias)

        if self.block_quant:
            return apply_fp8_block_linear(
                layer,
                input=x,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
                use_aiter_and_is_supported=self.use_aiter_and_is_supported)

        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=layer.input_scale,
                                     bias=bias)


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.layer = layer
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None

        self.flashinfer_moe_backend: Optional[FlashinferMoeBackend] = None
        self.fused_experts: Optional[
            mk.FusedMoEModularKernel] = None  # type: ignore
        if envs.VLLM_USE_FLASHINFER_MOE_FP8 and has_flashinfer_moe():
            self.flashinfer_moe_backend = get_flashinfer_moe_backend()
            logger.info_once(
                f"Using FlashInfer {self.flashinfer_moe_backend.value} kernels"
            )
        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (not current_platform.has_device_capability(89)
                           or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        # Check for DeepGemm support.
        self.allow_deep_gemm = False
        if envs.VLLM_USE_DEEP_GEMM:
            if not has_deep_gemm():
                logger.warning_once("Failed to import DeepGemm kernels.")
            elif not self.block_quant:
                logger.warning_once("Model is not block quantized. Not using "
                                    "DeepGemm kernels")
            elif (is_deep_gemm_supported()):
                logger.info_once("Using DeepGemm kernels for Fp8MoEMethod.")
                self.allow_deep_gemm = True
            else:
                logger.warning_once(
                    "DeepGemm not supported on the current platform.")

        # Check for CutlassBlockScaledGroupedGemm support.
        self.allow_cutlass_block_scaled_grouped_gemm = False
        if not self.block_quant:
            logger.debug_once("Model is not block quantized. Not using "
                              "CutlassBlockScaledGroupedGemm kernels")
        elif (current_platform.is_cuda()
              and current_platform.is_device_capability(100)):
            logger.info_once(
                "Using CutlassBlockScaledGroupedGemm kernels for Fp8MoEMethod."
            )
            self.allow_cutlass_block_scaled_grouped_gemm = True
        else:
            logger.warning_once(
                "CutlassBlockScaledGroupedGemm not supported on the current "
                "platform.")

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn
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
                    f"weight quantization block_n = {block_n}.")
            if (tp_size > 1
                    and intermediate_size_per_partition % block_k != 0):
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, 2, dtype=torch.float32),
                                                  requires_grad=False)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        else:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) //
                         block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.
             value} if self.block_quant else
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8.")

            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        # Lazy import to avoid importing triton too early.
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            is_rocm_aiter_moe_enabled, shuffle_weights)

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

        # TODO (rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            if current_platform.is_fp8_fnuz():
                w13_weight, w13_weight_scale_inv, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale_inv,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale_inv, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale_inv,
                        layer.w2_input_scale)
            elif self.flashinfer_moe_backend is not None:
                # NOTE: weights have to be swapped since the activation is
                # applied on different half for flashinfer vs vllm
                w13_weight = swap_w13_to_w31(layer.w13_weight.data)
                w13_weight_scale_inv = swap_w13_to_w31(
                    layer.w13_weight_scale_inv.data)
                w2_weight = layer.w2_weight.data
                w2_weight_scale_inv = layer.w2_weight_scale_inv.data
            else:
                w13_weight = layer.w13_weight.data
                w13_weight_scale_inv = layer.w13_weight_scale_inv.data
                w2_weight = layer.w2_weight
                w2_weight_scale_inv = layer.w2_weight_scale_inv

            # torch.compile() cannot use Parameter subclasses.
            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(w13_weight_scale_inv,
                                                   requires_grad=False)
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(w2_weight_scale_inv,
                                                  requires_grad=False)
            if self.rocm_aiter_moe_enabled:
                # reshaping weights is required for aiter moe kernel.
                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight.data, layer.w2_weight.data)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)

            # DeepGemm scales need to be transposed and aligned. We try to do
            # it ahead of time for performance reasons.
            if self.allow_deep_gemm and not is_deep_gemm_e8m0_used():
                if _is_col_major(layer.w13_weight_scale_inv):
                    layer.w13_weight_scale_inv = \
                        get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv)
                if _is_col_major(layer.w2_weight_scale_inv):
                    layer.w2_weight_scale_inv = \
                        get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv)

        # If checkpoint is fp16, quantize in place.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = current_platform.fp8_dtype()
            w13_weight = torch.empty_like(layer.w13_weight.data,
                                          dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
                layer.local_num_experts,
                dtype=torch.float32,
                device=w13_weight.device),
                                                        requires_grad=False)
            for expert in range(layer.local_num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], layer.w2_weight_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w2_weight.data[expert, :, :])
            layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                 requires_grad=False)
            if self.rocm_aiter_moe_enabled:
                # reshaping weights is required for aiter moe kernel.
                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight, layer.w2_weight)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)
        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if (layer.w13_input_scale is None
                        or layer.w2_input_scale is None):
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None.")
                if (not all_close_1d(layer.w13_input_scale)
                        or not all_close_1d(layer.w2_input_scale)):
                    logger.warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer.")
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False)
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False)
            if current_platform.is_fp8_fnuz():
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale,
                        layer.w2_input_scale)
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                      requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False)
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                     requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                           requires_grad=False)
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False)

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            if self.rocm_aiter_moe_enabled:
                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight, layer.w2_weight)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)

            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)

            if self.flashinfer_moe_backend is not None:
                # NOTE: weights have to be swapped since the activation is
                # applied on different half for flashinfer vs vllm
                assert not self.block_quant
                register_moe_scaling_factors(layer)
                w13_weight = swap_w13_to_w31(layer.w13_weight.data)
                if self.flashinfer_moe_backend == \
                    FlashinferMoeBackend.TENSORRT_LLM:
                    rotate_flashinfer_fp8_moe_weights(w13_weight, w2_weight)
                layer.w13_weight.data = w13_weight.data

        if self.use_marlin:
            prepare_moe_fp8_layer_for_marlin(layer, False)
            # Activations not quantized for marlin.
            del layer.w13_input_scale
            del layer.w2_input_scale

        if is_deep_gemm_e8m0_used() and self.block_quant:
            assert layer.weight_block_size is not None
            # Re-quantise the expert weights so their scales are UE8M0.
            block_sz = tuple(layer.weight_block_size)
            requant_weight_ue8m0_inplace(
                layer.w13_weight.data,
                layer.w13_weight_scale_inv.data,
                block_sz,
            )
            requant_weight_ue8m0_inplace(
                layer.w2_weight.data,
                layer.w2_weight_scale_inv.data,
                block_sz,
            )

            # Ensure column-major TMA alignment expected by DeepGEMM.
            if _is_col_major(layer.w13_weight_scale_inv):
                layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(
                    layer.w13_weight_scale_inv)
            if _is_col_major(layer.w2_weight_scale_inv):
                layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(
                    layer.w2_weight_scale_inv)

    def maybe_make_prepare_finalize(
            self) -> Optional[mk.FusedMoEPrepareAndFinalize]:
        if (self.rocm_aiter_moe_enabled or self.use_marlin
                or self.flashinfer_moe_backend
                == FlashinferMoeBackend.TENSORRT_LLM):
            return None
        elif self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            prepare_finalize = (
                build_flashinfer_fp8_cutlass_moe_prepare_finalize(self.moe))
            logger.debug_once("%s", prepare_finalize.__class__.__name__)
            return prepare_finalize
        else:
            return super().maybe_make_prepare_finalize()

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        from vllm.model_executor.layers.fused_moe import (
            BatchedTritonOrDeepGemmExperts, TritonOrDeepGemmExperts)

        assert not self.use_marlin and not self.rocm_aiter_moe_enabled, (
            "Marlin and ROCm AITER are not supported with all2all yet.")

        assert self.moe_quant_config is not None

        if (prepare_finalize.activation_format ==
                FusedMoEActivationFormat.BatchedExperts):
            max_num_tokens_per_rank = (
                prepare_finalize.max_num_tokens_per_rank())
            assert max_num_tokens_per_rank is not None
            logger.debug(
                "BatchedTritonOrDeepGemmExperts(%s): "
                "max_tokens_per_rank=%s, block_size=%s, per_act_token=%s",
                self.__class__.__name__, max_num_tokens_per_rank,
                self.weight_block_size, False)
            return BatchedTritonOrDeepGemmExperts(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
                allow_deep_gemm=self.allow_deep_gemm,
            )
        elif self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            experts = select_cutlass_fp8_gemm_impl(
                self.moe,
                self.moe_quant_config,
            )
            logger.debug_once("Using %s", experts.__class__.__name__)
            return experts
        else:
            logger.debug(
                "TritonOrDeepGemmExperts(%s): block_size=%s, per_act_token=%s",
                self.__class__.__name__, self.weight_block_size, False)
            return TritonOrDeepGemmExperts(
                quant_config=self.moe_quant_config,
                allow_deep_gemm=self.allow_deep_gemm,
            )

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        if self.use_marlin:
            return None

        return fp8_w8a8_moe_quant_config(
            w1_scale=(layer.w13_weight_scale_inv
                      if self.block_quant else layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale_inv
                      if self.block_quant else layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.weight_block_size,
        )

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
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)

        if (self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
                and self.fused_experts is None):
            assert activation == 'silu', (
                f"Expected 'silu' activation but got {activation}")
            assert scoring_func == 'sigmoid', (
                f"Expected 'sigmoid' scoring func but got {scoring_func}")
            if self.block_quant:
                import vllm.model_executor.layers.fused_moe.flashinfer_trtllm_moe  # noqa: E501, F401
                assert (renormalize and use_grouped_topk
                        and custom_routing_function is None)

                return torch.ops.vllm.flashinfer_fused_moe_blockscale_fp8(
                    routing_logits=router_logits.to(torch.float32),
                    routing_bias=e_score_correction_bias,
                    x=x,
                    w13_weight=layer.w13_weight,
                    w13_weight_scale_inv=layer.w13_weight_scale_inv,
                    w2_weight=layer.w2_weight,
                    w2_weight_scale_inv=layer.w2_weight_scale_inv,
                    global_num_experts=global_num_experts,
                    top_k=top_k,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                    intermediate_size=layer.intermediate_size_per_partition,
                    expert_offset=layer.ep_rank * layer.local_num_experts,
                    local_num_experts=layer.local_num_experts,
                    block_shape=self.weight_block_size,
                    routed_scaling=routed_scaling_factor,
                )
            else:
                assert (not renormalize
                        and custom_routing_function is not None)
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
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

        #
        # Note: the order of checks is important since self.fused_experts
        # can override fused_experts or cutlass but not rocm or marlin.
        #
        if self.rocm_aiter_moe_enabled:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
                rocm_aiter_fused_experts)
            assert self.fused_experts is None
            return rocm_aiter_fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                quant_config=self.moe_quant_config)
        elif self.use_marlin:
            assert activation == "silu", (
                f"{activation} not supported for Marlin MoE.")
            assert self.fused_experts is None
            return torch.ops.vllm.fused_marlin_moe(
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
                quant_type_id=scalar_types.float8_e4m3fn.id,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                workspace=layer.workspace)
        elif self.fused_experts:
            return self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
            )
        elif self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            assert self.block_quant is None
            assert (not renormalize and custom_routing_function is not None)
            assert activation == 'silu', (
                f"Expected 'silu' activation but got {activation}")
            assert scoring_func == 'sigmoid', (
                f"Expected 'sigmoid' scoring func but got {scoring_func}")

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
            from vllm.model_executor.layers.fused_moe import fused_experts
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                quant_config=self.moe_quant_config,
                allow_deep_gemm=self.allow_deep_gemm,
                allow_cutlass_block_scaled_grouped_gemm=(
                    self.allow_cutlass_block_scaled_grouped_gemm))


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
