from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
import os
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    dynamic_quant,
    apply_w8a8_block_fp8_linear,
    apply_block_fp8_linear_hpu_dequant,
    apply_block_fp8_linear_hpu_dynamic)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, apply_fp8_linear, convert_to_channelwise,
    cutlass_fp8_supported, normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize,
    requantize_with_max_scale)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

if current_platform.is_hpu():
    import habana_frameworks.torch as htorch
    from vllm_hpu_extension.ops import scaled_fp8_quant
    ops.scaled_fp8_quant = scaled_fp8_quant

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
    ) -> None:
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
        if current_platform.is_cuda_alike():
            self.cutlass_fp8_supported = cutlass_fp8_supported()

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
            if not self.block_quant:
                if self.quant_config.activation_scheme == "dynamic":
                    scale = ChannelQuantScaleParameter(
                        data=torch.empty(output_size_per_partition,
                                         dtype=torch.float32),
                        output_dim=0,
                        weight_loader=weight_loader,
                    )
                    scale[:] = torch.finfo(torch.float32).min
                    layer.register_parameter("weight_scale_inv", scale)
                else:
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

        #print("fp8.py !!!!!!!!!!!!!!!!!!!!!!!! weight shape: " + str(layer.weight.shape) + "  weight_scale_inv shape: " + str(layer.weight_scale_inv.shape))

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        if self.block_quant:
            if current_platform.is_hpu():
                from vllm.model_executor.layers.quantization.utils.fp8_utils import pad_block_fp8_weight_naive
                layer.weight, orig_M, orig_N = pad_block_fp8_weight_naive(
                    layer.weight,
                    layer.weight_scale_inv,
                    self.quant_config.weight_block_size)
                orig_M = torch.nn.Parameter(torch.tensor(orig_M, dtype=torch.int32), requires_grad=False)
                orig_N = torch.nn.Parameter(torch.tensor(orig_N, dtype=torch.int32), requires_grad=False)
                layer.register_parameter("orig_M", orig_M)
                layer.register_parameter("orig_N", orig_N)
            if current_platform.is_rocm():
                weight, weight_scale, _ = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=layer.weight,
                        weight_scale=layer.weight_scale_inv,
                        input_scale=layer.input_scale)
                layer.weight = Parameter(weight, requires_grad=False)
                layer.weight_scale_inv = Parameter(weight_scale,
                                                   requires_grad=False)
            return
        if self.quant_config.activation_scheme == "dynamic" and current_platform.is_hpu():
            return
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                          requires_grad=False)
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
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            if current_platform.is_hpu():
                return apply_block_fp8_linear_hpu_dequant(
                    input=x,
                    weight=layer.weight,
                    block_size=self.quant_config.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=layer.input_scale,
                    bias=bias,
                    original_M=layer.orig_M,
                    original_N=layer.orig_N,
                )
            else:
                return apply_w8a8_block_fp8_linear(
                    input=x,
                    weight=layer.weight,
                    block_size=self.quant_config.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=layer.input_scale,
                    bias=bias,
                )
        if self.quant_config.activation_scheme == "dynamic" and current_platform.is_hpu():
            return apply_block_fp8_linear_hpu_dynamic(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
            )

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
        self.moe_n_slice = int(os.environ.get("VLLM_MOE_N_SLICE", 4))

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
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
        if not self.block_quant:
            if self.quant_config.activation_scheme == "dynamic":
                w13_weight_scale = torch.nn.Parameter(data=torch.ones(
                    num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32),
                                                      requires_grad=False)
                w2_weight_scale = torch.nn.Parameter(data=torch.ones(
                    num_experts, hidden_size, dtype=torch.float32),
                                                     requires_grad=False)
#                print("fp8.py ??????????????????  MoE weight scale init here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " + "  w13_weight_scale shape: " + str(w13_weight_scale.shape) + "  w2_weight_scale shape: " + str(w2_weight_scale.shape) + "  extra_weight_attrs: " + str(extra_weight_attrs))
                layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
                layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            else:
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
        if self.block_quant:
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            )
        else:
            if self.quant_config.activation_scheme == "dynamic":
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

#        print("fp8.py ??????????????????  MoE weight scale init here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " + "  w13_weight_scale_inv shape: " + str(layer.w13_weight_scale_inv.shape) + "  w2_weight_scale_inv shape: " + str(layer.w2_weight_scale_inv.shape))

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        if self.block_quant:
            if current_platform.is_hpu():
                from vllm.model_executor.layers.quantization.utils.fp8_utils import pad_block_fp8_weight_naive
                layer.w13_weight, orig_M_w13, orig_N_w13 = pad_block_fp8_weight_naive(
                    layer.w13_weight,
                    layer.w13_weight_scale_inv,
                    self.quant_config.weight_block_size)
                layer.w2_weight, orig_M_w2, orig_N_w2 = pad_block_fp8_weight_naive(
                    layer.w2_weight,
                    layer.w2_weight_scale_inv,
                    self.quant_config.weight_block_size)
                orig_M_w13 = torch.nn.Parameter(torch.tensor(orig_M_w13, dtype=torch.int32), requires_grad=False)
                orig_N_w13 = torch.nn.Parameter(torch.tensor(orig_N_w13, dtype=torch.int32), requires_grad=False)
                layer.register_parameter("orig_M_w13", orig_M_w13)
                layer.register_parameter("orig_N_w13", orig_N_w13)
                orig_M_w2 = torch.nn.Parameter(torch.tensor(orig_M_w2, dtype=torch.int32), requires_grad=False)
                orig_N_w2 = torch.nn.Parameter(torch.tensor(orig_N_w2, dtype=torch.int32), requires_grad=False)
                layer.register_parameter("orig_M_w2", orig_M_w2)
                layer.register_parameter("orig_N_w2", orig_N_w2)
            if current_platform.is_rocm():
                w13_weight, w13_weight_scale_inv, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale_inv,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale_inv, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale_inv,
                        layer.w2_input_scale)
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                      requires_grad=False)
                layer.w13_weight_scale_inv = torch.nn.Parameter(
                    w13_weight_scale_inv, requires_grad=False)
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                     requires_grad=False)
                layer.w2_weight_scale_inv = torch.nn.Parameter(
                    w2_weight_scale_inv, requires_grad=False)
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False)
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
                layer.num_experts,
                dtype=torch.float32,
                device=w13_weight.device),
                                                        requires_grad=False)
            for expert in range(layer.num_experts):
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

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.num_experts):
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
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        ep_rank=0,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        if current_platform.is_hpu():
            return self.forward_hpu(
                x=x,
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
                ep_rank=ep_rank,
            )

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
            w1_scale=(layer.w13_weight_scale_inv
                      if self.block_quant else layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale_inv
                      if self.block_quant else layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.quant_config.weight_block_size,
        )

    def dequant_weight(self, weight, weight_scale, block_size):
        weight_shape_len = len(weight.shape)
        if weight_shape_len == 2:
            weight_scale_m, weight_scale_n = weight_scale.shape
            weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
            weight = weight.view(weight_scale_m, block_size, weight_scale_n, block_size)
            dequant_weight = weight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16)
            dequant_weight = dequant_weight.view(weight_scale_m*block_size, weight_scale_n*block_size)
        elif weight_shape_len == 3:
            num_expert, weight_scale_m, weight_scale_n = weight_scale.shape
            weight_scale = weight_scale.view(num_expert, weight_scale_m, 1, weight_scale_n, 1)
            weight = weight.view(num_expert, weight_scale_m, block_size, weight_scale_n, block_size)
            dequant_weight = weight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16)
            dequant_weight = dequant_weight.view(num_expert, weight_scale_m*block_size, weight_scale_n*block_size)
        else:
            raise ValueError("Only support original weight shape is either 2 or 3")

        return dequant_weight

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
        batch_size, seq_len, hidden_dim = x.shape
        bt = batch_size * seq_len
        x = x.view(-1, hidden_dim)

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

        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.long()
        num_experts = layer.w13_weight.shape[0]
        total_num_experts = router_logits.size(1)
        ep_shift = ep_rank * num_experts

        if self.block_quant:
            orig_M_w13 = layer.orig_M_w13.data
            orig_N_w13 = layer.orig_N_w13.data
            orig_M_w2 = layer.orig_M_w2.data
            orig_N_w2 = layer.orig_N_w2.data

        if self.quant_config.activation_scheme == "dynamic" and not self.block_quant:
            x_fp8, x_scale = dynamic_quant(x)

        padded_weights = torch.zeros((bt, total_num_experts), dtype=x.dtype, device=x.device)
        padded_weights.scatter_(-1, topk_ids, topk_weights)
        padded_weights = padded_weights.transpose(0, 1)

        if seq_len > 1:
            mask_weights = torch.zeros((bt, total_num_experts), dtype=x.dtype, device=x.device)
            mask_weights.scatter_(-1, topk_ids, 1)
            mask_weights = mask_weights.transpose(0, 1)

        for idx in range(num_experts):
            w13_weight = layer.w13_weight[idx, ...]
            w2_weight = layer.w2_weight[idx, ...]
            w13_scale = layer.w13_weight_scale_inv[idx, ...]
            w2_scale = layer.w2_weight_scale_inv[idx, ...]

            if self.block_quant:
                w13_weight = self.dequant_weight(w13_weight, w13_scale, self.quant_config.weight_block_size[0])

            if seq_len > 1:
                mask_weight = mask_weights[idx + ep_shift].unsqueeze(1)
                if self.block_quant:
                    current_state_static = x * mask_weight
                else:
                    current_state_static = x_fp8 * mask_weight.to(torch.float8_e4m3fn)
            else:
                if self.block_quant:
                    current_state_static = x
                else:
                    current_state_static = x_fp8

            if self.block_quant:
                up_gate_states = torch.matmul(current_state_static, w13_weight.transpose(0, 1))
            else:
                up_gate_states = torch.ops.hpu.fp8_gemm_v2(
                    current_state_static,
                    False,
                    w13_weight,
                    True,
                    None,
                    torch.bfloat16,
                    x_scale,
                    w13_scale,
                    None,
                    False,
                )
            d = up_gate_states.shape[-1] // 2
            current_state_static = F.silu(up_gate_states[..., :d]) * up_gate_states[..., d:]

            if self.block_quant:
                w2_weight = self.dequant_weight(w2_weight, w2_scale, self.quant_config.weight_block_size[0])
                current_hidden_states = torch.matmul(current_state_static, w2_weight.transpose(0, 1))
            else:
                current_state_static, current_state_static_scale = dynamic_quant(current_state_static)
                current_hidden_states = torch.ops.hpu.fp8_gemm_v2(
                    current_state_static,
                    False,
                    w2_weight,
                    True,
                    None,
                    torch.bfloat16,
                    current_state_static_scale,
                    w2_scale,
                    None,
                    False,
                )
            padded_weight = padded_weights[idx + ep_shift].unsqueeze(1)
            if idx == 0:
                final_hidden_states = current_hidden_states * padded_weight
            else:
                final_hidden_states.add_(current_hidden_states * padded_weight)
        
        if seq_len > 1:
            htorch.core.mark_step()

        return final_hidden_states.view(-1, x.shape[1])




class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
