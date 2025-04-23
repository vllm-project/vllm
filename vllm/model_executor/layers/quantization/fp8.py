# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
import os
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, apply_fp8_linear, convert_to_channelwise,
    cutlass_block_fp8_supported, cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize,
    requantize_with_max_scale)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    dynamic_quant,
    dequant_block_fp8_weight_naive,
    apply_block_fp8_linear_hpu_dynamic,
    apply_block_fp8_linear_hpu_dequant)

if current_platform.is_hpu():
    import habana_frameworks.torch as htorch
    from vllm_hpu_extension.ops import scaled_fp8_quant
    ops.scaled_fp8_quant = scaled_fp8_quant

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)

VLLM_REQUANT_FP8_INC = os.getenv("VLLM_REQUANT_FP8_INC", "0") in ["1", "true"]


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
    ) -> None:
        self.enable_runtime_dequant = os.environ.get("VLLM_ENABLE_RUNTIME_DEQUANT", "0") in ["1", "true"]
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
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
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"],
                                                 None)
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme,
                   ignored_layers=ignored_layers,
                   weight_block_size=weight_block_size)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
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
        self.cutlass_fp8_supported = False
        self.cutlass_block_fp8_supported = False
        if current_platform.is_cuda_alike():
            self.cutlass_fp8_supported = cutlass_fp8_supported()
            self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()

        self.use_marlin = False
        if not current_platform.is_hpu():
            # For GPUs that lack FP8 hardware support, we can leverage the
            # Marlin kernel for fast weight-only FP8 quantization
            self.use_marlin = (not current_platform.has_device_capability(89)
                               or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        self.block_quant = self.quant_config.weight_block_size is not None
        if self.block_quant:
            # Marlin doesn't support block-wise fp8
            self.use_marlin = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.block_quant:
            tp_size = get_tensor_model_parallel_world_size()
            assert self.quant_config.weight_block_size is not None
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if (tp_size > 1
                    and input_size // input_size_per_partition == tp_size
                    and input_size_per_partition % block_k != 0):
                raise ValueError(
                    f"Weight input_size_per_partition = "
                    f"{input_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")
            # Required by column parallel or enabling merged weights
            if (tp_size > 1 and output_size // output_size_per_partition
                    == tp_size) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}.")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
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

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if not self.block_quant and current_platform.is_hpu():
                scale = ChannelQuantScaleParameter(
                    data=torch.empty(output_size_per_partition,
                                        dtype=torch.float32),
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            elif not self.block_quant:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes),
                                     dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)
            else:
                assert self.quant_config.activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                # The weight_scale_inv name is intentional for deepseekv3
                layer.register_parameter("weight_scale_inv", scale)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                scale = PerTensorScaleParameter(data=torch.empty(
                    len(output_partition_sizes), dtype=torch.float32),
                                                weight_loader=weight_loader)

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def dequant_block_fp8_weight(self, layer) -> torch.Tensor:
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight
        dequant_weight = dequant_block_fp8_weight_naive(
            layer.weight,
            layer.weight_scale_inv.data,
            self.quant_config.weight_block_size,
            original_M=layer.orig_M,
            original_N=layer.orig_N,
            do_unpad=True,
        )
        return dequant_weight

    def process_weights_after_loading(self, layer: Module) -> None:
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            if current_platform.is_hpu():
                from vllm.model_executor.layers.quantization.utils.fp8_utils import pad_block_fp8_weight_naive
                weight, orig_M, orig_N = pad_block_fp8_weight_naive(
                    layer.weight.data,
                    layer.weight_scale_inv.data,
                    self.quant_config.weight_block_size)
                if self.quant_config.enable_runtime_dequant:
                    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                    orig_M = torch.nn.Parameter(torch.tensor(orig_M, dtype=torch.int32), requires_grad=False)
                    orig_N = torch.nn.Parameter(torch.tensor(orig_N, dtype=torch.int32), requires_grad=False)
                    layer.register_parameter("orig_M", orig_M)
                    layer.register_parameter("orig_N", orig_N)
                else:
                    weight, weight_scale_inv = dynamic_quant(dequant_block_fp8_weight_naive(
                        weight,
                        layer.weight_scale_inv.data,
                        self.quant_config.weight_block_size,
                        original_M=orig_M,
                        original_N=orig_N,
                        do_unpad=True))
                    weight_scale_inv = weight_scale_inv.squeeze(-1)
                    layer.weight.data.copy_(weight)
                    layer.weight_scale_inv = Parameter(weight_scale_inv,
                                                    requires_grad=False)
                return
            if current_platform.is_rocm():
                weight, weight_scale_inv, _ = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=layer.weight,
                        weight_scale=layer.weight_scale_inv)
            else:
                weight = layer.weight.data
                weight_scale_inv = layer.weight_scale_inv.data

            # Torch.compile cannot use Parameter subclasses.
            layer.weight = Parameter(weight, requires_grad=False)
            layer.weight_scale_inv = Parameter(weight_scale_inv,
                                               requires_grad=False)
            return

        if current_platform.is_hpu():
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(layer.input_scale.max(),
                                              requires_grad=False)
            return
        # If checkpoint not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)

            # If using marlin (w8a16), kernel uses channelwise weights,
            # so extend the weight scales to be channelwise.
            if self.use_marlin:
                assert weight_scale.numel() == 1
                weight_scale = convert_to_channelwise(
                    weight_scale.expand(len(layer.logical_widths)),
                    layer.logical_widths)

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

        # If checkpoint is fp8, handle that there are N scales for N
        # shards in a fused module
        else:
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                    requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = torch.nn.Parameter(layer.input_scale.data,
                                                       requires_grad=False)
            # If using marlin (w8a16), kernel uses channelwise weights,
            # so extend the weight scales to be channelwise.
            if self.use_marlin:
                weight = layer.weight
                weight_scale = convert_to_channelwise(layer.weight_scale,
                                                      layer.logical_widths)

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            else:
                # Dequant -> Quant with max scale so we can run per tensor.
                weight = layer.weight
                weight_scale = layer.weight_scale

                # If rocm, use float8_e4m3fnuz.
                if current_platform.is_rocm():
                    weight, weight_scale, input_scale = \
                        normalize_e4m3fn_to_e4m3fnuz(
                            weight=weight,
                            weight_scale=weight_scale,
                            input_scale=layer.input_scale)
                    if input_scale is not None:
                        layer.input_scale = Parameter(input_scale,
                                                      requires_grad=False)

                weight_scale, weight = requantize_with_max_scale(
                    weight=weight,
                    weight_scale=weight_scale,
                    logical_widths=layer.logical_widths,
                )

            # Update layer with new values.
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(layer.input_scale.max(),
                                              requires_grad=False)

        if self.use_marlin:
            prepare_fp8_layer_for_marlin(layer)
            # Activations not quantized for marlin.
            del layer.input_scale

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

        # Note: lazy import to avoid triton import error.
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            apply_w8a8_block_fp8_linear)
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            if current_platform.is_hpu():
                if self.quant_config.enable_runtime_dequant:
                    return apply_block_fp8_linear_hpu_dequant(
                        input=x,
                        weight=layer.weight,
                        block_size=self.quant_config.weight_block_size,
                        weight_scale=layer.weight_scale_inv,
                        input_scale=layer.input_scale,
                        bias=bias,
                        original_M=layer.orig_M,
                        original_N=layer.orig_N,
                        do_unpad=True,
                    )
                else:
                    return apply_block_fp8_linear_hpu_dynamic(
                        input=x,
                        weight=layer.weight,
                        weight_scale=layer.weight_scale_inv,
                        input_scale=layer.input_scale,
                        bias=bias,
                    )
            return apply_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            )
        if current_platform.is_hpu() and self.quant_config.activation_scheme == "dynamic":
            return apply_block_fp8_linear_hpu_dynamic(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
            )
        if current_platform.is_hpu() and self.quant_config.activation_scheme == "static":
            x_fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0/layer.input_scale, False, False, torch.float8_e4m3fn)[0]
            return torch.ops.hpu.fp8_gemm_v2(
                A=x_fp8,
                trans_A=False,
                B=layer.weight,
                trans_B=True,
                D=None,
                out_dtype=x.dtype,
                A_scale_inv=layer.input_scale,
                B_scale_inv=layer.weight_scale_inv,
                bias=bias,
                accumulate=False)
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            # Default to using per_token quantization if cutlass is supported
            use_per_token_if_dynamic=self.cutlass_fp8_supported)


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

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None
        self.moe_n_slice = int(os.environ.get("VLLM_MOE_N_SLICE", 8))
        self.enable_dmoe_dynamic_scale = os.environ.get("VLLM_DMOE_DYNAMIC_SCALE", False) in ["1", "true"]
        self.use_static_moe = os.environ.get("VLLM_USE_STATIC_MOE", "0") in ["1", "true"]
        self.optimize_with_partial_experts = os.environ.get("VLLM_OPTIMIZE_WITH_PARTIAL_EXPERTS", "0") in ["1", "true"]

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        layer.quant_config = self.quant_config
        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
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
        if current_platform.is_hpu() and not self.block_quant:
            w13_weight_scale = torch.nn.Parameter(data=torch.ones(
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32),
                                                    requires_grad=False)
            w2_weight_scale = torch.nn.Parameter(data=torch.ones(
                num_experts, hidden_size, dtype=torch.float32),
                                                    requires_grad=False)
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        elif not self.block_quant:
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
            
            if current_platform.is_hpu() and VLLM_REQUANT_FP8_INC:
                moe_op = layer.moe_op
                os.environ["INC_DYNAMIC_MOE_EXPERTS"] = str(moe_op.num_experts)
                for index in range(moe_op.num_experts):
                    moe_op.w13_list[index].set_weight(layer.w13_weight[index])
                    moe_op.w13_list[index].set_scale_inv_fp8(
                        layer.w13_weight_scale_inv[index]
                    )
                    moe_op.w13_list[index].set_weight_block_size(
                        layer.quant_config.weight_block_size
                    )

                    moe_op.w2_list[index].set_weight(layer.w2_weight[index])
                    moe_op.w2_list[index].set_scale_inv_fp8(
                        layer.w2_weight_scale_inv[index]
                    )
                    moe_op.w2_list[index].set_weight_block_size(
                        layer.quant_config.weight_block_size
                    )
                import habana_frameworks.torch as htorch
                htorch.core.mark_step()

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        if self.block_quant:
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            )
        else:
            if current_platform.is_hpu():
                extra_weight_attrs.update(
                    {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
                )
            else:
                extra_weight_attrs.update(
                    {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
                )
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
        # TODO (rob): refactor block quant into separate class.
        if self.block_quant:
            if current_platform.is_hpu():
                if self.quant_config.enable_runtime_dequant:
                    return
                w13_weight, w13_weight_scale_inv = dynamic_quant(dequant_block_fp8_weight_naive(
                    layer.w13_weight.data,
                    layer.w13_weight_scale_inv.data,
                    self.quant_config.weight_block_size))
                w2_weight, w2_weight_scale_inv = dynamic_quant(dequant_block_fp8_weight_naive(
                    layer.w2_weight.data,
                    layer.w2_weight_scale_inv.data,
                    self.quant_config.weight_block_size))
                w13_weight_scale_inv, w2_weight_scale_inv = w13_weight_scale_inv.squeeze(-1), w2_weight_scale_inv.squeeze(-1)
                layer.w13_weight.data.copy_(w13_weight)
                layer.w2_weight.data.copy_(w2_weight)
                layer.w13_weight_scale_inv = Parameter(w13_weight_scale_inv,
                                                       requires_grad=False)
                layer.w2_weight_scale_inv = Parameter(w2_weight_scale_inv,
                                                      requires_grad=False)
                return
            if current_platform.is_rocm():
                w13_weight, w13_weight_scale_inv, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale_inv,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale_inv, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale_inv,
                        layer.w2_input_scale)
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
            return
        if self.quant_config.activation_scheme == "dynamic" and current_platform.is_hpu():
            return
        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            # If rocm, use float8_e4m3fnuz as dtype
            fp8_dtype = torch.float8_e4m3fnuz \
                        if current_platform.is_rocm() else torch.float8_e4m3fn
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
            return

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
            # If rocm, normalize the weights and scales to e4m3fnuz
            if current_platform.is_rocm():
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
            if current_platform.is_hpu():
                if self.quant_config.activation_scheme == "static":
                    num_experts = layer.w13_weight.shape[0]
                    self.w13_weight_list = [layer.w13_weight.data[i,...] for i in range(num_experts)]
                    self.w2_weight_list = [layer.w2_weight.data[i,...] for i in range(num_experts)]
                    self.w13_weight_scale_list = [layer.w13_weight_scale_inv.data[i,...] for i in range(num_experts)]
                    self.w2_weight_scale_list = [layer.w2_weight_scale_inv.data[i,...] for i in range(num_experts)]
                    self.w2_input_scale_list = [layer.w2_input_scale.data.unsqueeze(0).repeat(num_experts)[i] for i in range(num_experts)]
                return
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

            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)
            return

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
        ep_rank=0,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        if current_platform.is_hpu():
            return self.forward_hpu(x=x,
                            layer=layer,
                            router_logits=router_logits,
                            top_k=top_k,
                            renormalize=renormalize,
                            use_grouped_topk=use_grouped_topk,
                            topk_group=topk_group,
                            num_expert_group=num_expert_group,
                            custom_routing_function=custom_routing_function,
                            scoring_func=scoring_func,
                            e_score_correction_bias=e_score_correction_bias,
                            ep_rank=ep_rank)

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

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=True,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=(layer.w13_weight_scale_inv
                      if self.block_quant else layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale_inv
                      if self.block_quant else layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.quant_config.weight_block_size,
        )

    def forward_hpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        ep_rank=0,
    ):
        if self.quant_config.activation_scheme == "static" and layer.dp_size > 1:
            x_scale = layer.w13_input_scale.data
            x = torch.ops.hpu.cast_to_fp8_v2(x, 1.0/x_scale, False, False, torch.float8_e4m3fn)[0]
            cu_tokens_across_dp_cpu = get_forward_context(
            ).dp_metadata.cu_tokens_across_dp_cpu
            hidden_states_across_dp = get_forward_context(
            ).dp_metadata.hidden_states_across_dp
            x = layer.multicast_fn(x, cu_tokens_across_dp_cpu,\
                hidden_states_across_dp)

        batch_size, seq_len, hidden_dim = x.shape
        num_experts = layer.local_num_experts
        n_expert_slice = num_experts // self.moe_n_slice
        # num_experts = layer.w13_weight.shape[0]
        # n_expert_slice = layer.w13_weight.shape[0] // self.moe_n_slice
        assert n_expert_slice * self.moe_n_slice == num_experts
        x = x.view(-1, hidden_dim)
        total_num_experts = router_logits.size(-1)
        if seq_len == 1 and (num_experts == total_num_experts) and (batch_size * top_k <= 64):
            # conditionining on 1. not pre_dequant, 2. decode phase, 3. not with EP>1 4. Batch_size < 8
            use_partial_experts = True if self.optimize_with_partial_experts else False
        else:
            use_partial_experts = False

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

        ep_shift = ep_rank * num_experts
        use_static_moe = self.use_static_moe

        def do_static_moe_with_dynamic_scaling(x, topk_ids, topk_weights, w13_weight_fp8, w2_weight_fp8, total_num_experts, num_experts, w13_weight_scale_inv_fp8=None, w2_weight_scale_inv_fp8=None):
            x_fp8, x_scale = dynamic_quant(x)
            # padded_weights shape is (total_num_experts, num_tokens)
            experts_mask = torch.zeros((x.size(0), total_num_experts), dtype=x.dtype, device=x.device)
            experts_mask.scatter_(-1, topk_ids, topk_weights)
            experts_mask = experts_mask.transpose(0, 1)

            if seq_len > 1:
                mask_weights = torch.zeros((x_fp8.size(0), total_num_experts), dtype=x.dtype, device=x.device)
                mask_weights.scatter_(-1, topk_ids, 1)
                mask_weights = mask_weights.transpose(0, 1)

            for i in range(num_experts):
                w13_weight_fp8_slice = w13_weight_fp8[i, ...]
                w2_weight_fp8_slice = w2_weight_fp8[i, ...]
                w13_scale_fp8_slice = w13_weight_scale_inv_fp8[i, ...]
                w2_scale_fp8_slice = w2_weight_scale_inv_fp8[i, ...]

                if seq_len > 1:
                    mask_weight = mask_weights[i + ep_shift].unsqueeze(1)
                    current_state_static = x_fp8 * mask_weight.to(torch.float8_e4m3fn)
                else:
                    current_state_static = x_fp8

                up_gate_states = torch.ops.hpu.fp8_gemm_v2(
                    A=current_state_static,
                    trans_A=False,
                    B=w13_weight_fp8_slice,
                    trans_B=True,
                    D=None,
                    out_dtype=torch.bfloat16,
                    A_scale_inv=x_scale,
                    B_scale_inv=w13_scale_fp8_slice,
                    bias=None,
                    accumulate=False)

                d = up_gate_states.shape[-1] // 2
                current_state_static = F.silu(up_gate_states[..., :d]) * up_gate_states[..., d:]

                current_state_static, current_state_static_scale = dynamic_quant(current_state_static)
                current_hidden_states = torch.ops.hpu.fp8_gemm_v2(
                    current_state_static,
                    False,
                    w2_weight_fp8_slice,
                    True,
                    None,
                    torch.bfloat16,
                    current_state_static_scale,
                    w2_scale_fp8_slice,
                    None,
                    False,
                )
                padded_weight = experts_mask[i + ep_shift].unsqueeze(1)
                if i == 0:
                    final_hidden_states = current_hidden_states * padded_weight
                else:
                    final_hidden_states.add_(current_hidden_states * padded_weight)

            return final_hidden_states

        def do_dynamic_moe_with_static_scaling(x, topk_ids, topk_weights, w13_weight_fp8, w2_weight_fp8, moe_n_slice, n_expert_slice, w13_weight_scale_inv_fp8, w2_weight_scale_inv_fp8):
            x_scale = layer.w13_input_scale.data
            if layer.dp_size == 1:
                x_fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0/x_scale, False, False, torch.float8_e4m3fn)[0]
            else:
                x_fp8 = x
            final_hidden_states = torch.ops.hpu.mixture_of_experts(
                hidden_states=x_fp8,
                expert_routing_table=(topk_ids.to(torch.int64) - ep_shift),
                router_weights=topk_weights.to(x.dtype),
                w12=self.w13_weight_list,
                w3=self.w2_weight_list,
                d_scale_hidden_states=x_scale,
                d_scale_intermediate_hidden_states=self.w2_input_scale_list,
                d_scale_w12=self.w13_weight_scale_list,
                d_scale_w3=self.w2_weight_scale_list,
                permuted_weights=True,
                activation="silu",
                experts_min=0,
                experts_max=(num_experts - 1),
            )
            return final_hidden_states.view(-1, x.shape[1])

        def do_dynamic_moe_with_dynamic_scaling(x, topk_ids, topk_weights, w13_weight_fp8, w2_weight_fp8, moe_n_slice, n_expert_slice, w13_weight_scale_inv_fp8=None, w2_weight_scale_inv_fp8=None):
            x_fp8, x_scale = dynamic_quant(x, single_scale=True)
            for i in range(moe_n_slice):
                min_expert = i * n_expert_slice
                max_expert = (i + 1) * n_expert_slice

                w13_list_slice = [w13_weight_fp8[j, ...] for j in range(min_expert, max_expert)]
                w2_list_slice = [w2_weight_fp8[j, ...] for j in range(min_expert, max_expert)]
                w13_weight_scale = [w13_weight_scale_inv_fp8[j, ...] for j in range(min_expert, max_expert)]
                w2_weight_scale = [w2_weight_scale_inv_fp8[j,...] for j in range(min_expert, max_expert)]

                current_hidden_states = torch.ops.hpu.mixture_of_experts(
                                            hidden_states=x_fp8,
                                            expert_routing_table=topk_ids.to(torch.int64),
                                            router_weights=topk_weights.to(x.dtype),
                                            w12=w13_list_slice,
                                            w3=w2_list_slice,
                                            d_scale_hidden_states=x_scale,
                                            d_scale_w12=w13_weight_scale,
                                            d_scale_w3=w2_weight_scale,
                                            permuted_weights=True,
                                            activation="silu",
                                            experts_min=min_expert + ep_shift,
                                            experts_max=max_expert - 1 + ep_shift)
                htorch.core.mark_step()
                if i == 0:
                    final_hidden_states = current_hidden_states
                else:
                    final_hidden_states.add_(current_hidden_states)
            return final_hidden_states

        def do_dynamic_moe_with_dequant(x, topk_ids, topk_weights, w13_weight_fp8, w2_weight_fp8, moe_n_slice, n_expert_slice, w13_weight_scale_inv_fp8=None, w2_weight_scale_inv_fp8=None):
            w13_weight = dequant_block_fp8_weight_naive(w13_weight_fp8,
                                                        w13_weight_scale_inv_fp8,
                                                        block_size=self.quant_config.weight_block_size,
                                                        dtype=x.dtype)
            w2_weight = dequant_block_fp8_weight_naive(w2_weight_fp8,
                                                    w2_weight_scale_inv_fp8,
                                                    block_size=self.quant_config.weight_block_size,
                                                    dtype=x.dtype)
            for i in range(moe_n_slice):
                min_expert = i * n_expert_slice
                max_expert = (i + 1) * n_expert_slice

                w13_list_slice = [w13_weight[j, ...] for j in range(min_expert, max_expert)]
                w2_list_slice = [w2_weight[j, ...] for j in range(min_expert, max_expert)]

                current_hidden_states = torch.ops.hpu.mixture_of_experts(
                                            hidden_states=x,
                                            expert_routing_table=topk_ids.to(torch.int64),
                                            router_weights=topk_weights.to(x.dtype),
                                            w12=w13_list_slice,
                                            w3=w2_list_slice,
                                            permuted_weights=True,
                                            activation="silu",
                                            experts_min=min_expert + ep_shift,
                                            experts_max=max_expert - 1 + ep_shift)
                htorch.core.mark_step()
                if i == 0:
                    final_hidden_states = current_hidden_states
                else:
                    final_hidden_states.add_(current_hidden_states)
            return final_hidden_states

        if use_partial_experts:
            w13_weight_fp8 = layer.w13_weight.index_select(0, topk_ids.view(-1))
            w13_weight_scale_inv_fp8 = layer.w13_weight_scale_inv.index_select(0, topk_ids.view(-1))
            w2_weight_fp8 = layer.w2_weight.index_select(0, topk_ids.view(-1))
            w2_weight_scale_inv_fp8 = layer.w2_weight_scale_inv.index_select(0, topk_ids.view(-1))
            actual_total_experts = w13_weight_fp8.size(0)
            topk_ids_dense = torch.arange(actual_total_experts, device=topk_ids.device).view(topk_ids.size(0), topk_ids.size(1))
            topk_ids = topk_ids_dense
            actual_num_experts = actual_total_experts
            moe_n_slice = 4 if actual_total_experts >= 64 else 1
            n_expert_slice = actual_total_experts // moe_n_slice
        else:
            actual_total_experts = total_num_experts
            actual_num_experts = num_experts
            moe_n_slice = self.moe_n_slice
            n_expert_slice = actual_num_experts // moe_n_slice
            if self.quant_config.enable_runtime_dequant and VLLM_REQUANT_FP8_INC:
                assert not use_partial_experts, "Partial experts not supported with VLLM_REQUANT_FP8_INC"
                final_hidden_states = layer.moe_op(
                    x,
                    topk_ids.to(torch.int64),
                    topk_weights.to(x.dtype),
                )
                return final_hidden_states.view(-1, x.shape[1])
            w13_weight_fp8 = layer.w13_weight.data
            w13_weight_scale_inv_fp8 = layer.w13_weight_scale_inv.data
            w2_weight_fp8 = layer.w2_weight.data
            w2_weight_scale_inv_fp8 = layer.w2_weight_scale_inv.data

        if self.quant_config.activation_scheme == "dynamic":
            if self.quant_config.enable_runtime_dequant:
                final_hidden_states = do_dynamic_moe_with_dequant(x, topk_ids, topk_weights, w13_weight_fp8, w2_weight_fp8, moe_n_slice, n_expert_slice, w13_weight_scale_inv_fp8, w2_weight_scale_inv_fp8)
            elif not use_static_moe and self.enable_dmoe_dynamic_scale:
                final_hidden_states = do_dynamic_moe_with_dynamic_scaling(x, topk_ids, topk_weights, w13_weight_fp8, w2_weight_fp8, moe_n_slice, n_expert_slice, w13_weight_scale_inv_fp8, w2_weight_scale_inv_fp8)
            else:
                final_hidden_states = do_static_moe_with_dynamic_scaling(x, topk_ids, topk_weights, w13_weight_fp8, w2_weight_fp8, actual_total_experts, actual_num_experts, w13_weight_scale_inv_fp8, w2_weight_scale_inv_fp8)
        elif self.quant_config.activation_scheme == "static":
            final_hidden_states = do_dynamic_moe_with_static_scaling(x, topk_ids, topk_weights, w13_weight_fp8, w2_weight_fp8, moe_n_slice, n_expert_slice, w13_weight_scale_inv_fp8, w2_weight_scale_inv_fp8)
        else:
            raise ValueError("Unknown activation scheme")

        return final_hidden_states.view(-1, x.shape[1])


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
