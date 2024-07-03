from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  fused_moe)
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import print_warning_once

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


def cutlass_fp8_supported() -> bool:
    capability = current_platform.get_device_capability()
    capability = capability[0] * 10 + capability[1]

    return ops.cutlass_scaled_mm_supports_fp8(capability)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme)

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


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
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def _create_scale_param(
        self,
        scale_name: str,
        layer: torch.nn.Module,
        output_partition_sizes: List[int],
        **extra_weight_attrs,
    ) -> None:
        scale = Parameter(torch.empty(len(output_partition_sizes),
                                      dtype=torch.float32),
                          requires_grad=False)
        scale[:] = torch.finfo(torch.float8_e4m3fn).min
        layer.register_parameter(scale_name, scale)
        set_weight_attrs(scale, {
            **extra_weight_attrs,
            "needs_scalar_to_array": True,
        })

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
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.process_after_load = True
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=weight_dtype),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs,
            "input_dim": 1,
            "output_dim": 0,
        })

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            self._create_scale_param(
                scale_name="weight_scale",
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                **extra_weight_attrs)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                self._create_scale_param(
                    scale_name="input_scale",
                    layer=layer,
                    output_partition_sizes=output_partition_sizes,
                    **extra_weight_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return

        # If checkpoint is fp/bf16 (not serialized fp8), quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.logical_widths = None
            layer.input_scale = None
            return

        # If checkpoint is fp8, requantize the separately quantized logical
        # weights into a single fp8 weight with a single weight scale.
        else:
            # WEIGHT_SCALE / WEIGHT
            #   Loop over logical weights, requantizing with single scale.
            max_w_scale = layer.weight_scale.max()

            # QKV / MLP is fused in the on disk checkpoint if any of the
            # weight scales are still set to the default since we initialize
            # N weight scales for N shards but we only load 1 weight scale
            # from disk in this case. As a result, we skip dequant -> requant
            # since we already have quantized QKV together.
            # Sample Model with fused checkpoint:
            #   * nm-testing/Phi-3-mini-128k-instruct-FP8
            unfused_module_in_checkpoint = (
                layer.weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min)

            if unfused_module_in_checkpoint:
                start = 0
                for idx, logical_width in enumerate(layer.logical_widths):
                    end = start + logical_width
                    weight_dq = per_tensor_dequantize(
                        layer.weight[start:end, :], layer.weight_scale[idx])

                    layer.weight[start:end, :] = per_tensor_quantize(
                        weight_dq, layer.weight_scale.max())
                    start = end
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

            # WEIGHT
            #   Transpose weight for passing to torch._scaled_mm
            weight = layer.weight
            layer.weight = Parameter(weight.t(), requires_grad=False)

            # INPUT ACTIVATION SCALE
            #   Dynamic: set to None (required input to ops.scaled_fp8_quant).
            #   Static:  set to max of the input_scales (since they are equal).
            if self.quant_config.activation_scheme == "dynamic":
                layer.input_scale = None
            elif self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(layer.input_scale.max(),
                                              requires_grad=False)
            else:
                raise ValueError(
                    f"Unknown scheme {self.quant_config.activation_scheme}")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_scale computed from x.
        #   If static, layer.input_scale is scalar and x_scale is input_scale.

        if bias is None and self.cutlass_fp8_supported:
            qinput, x_scale = ops.scaled_fp8_quant(x, layer.input_scale)

            # Fused GEMM_DQ
            output = ops.cutlass_scaled_mm(
                qinput,
                layer.weight,
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
            )

        else:
            qinput, x_scale = ops.scaled_fp8_quant(x,
                                                   layer.input_scale,
                                                   batch_dim_padding=17)

            # Fused GEMM_DQ -- note we padded the input above because
            # torch._scaled_mm is more performant for matrices with
            # batch dimension > 16. Note that this could change
            # in the future.
            output, _ = torch._scaled_mm(
                qinput,
                layer.weight,
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                bias=bias,
            )

        return torch.narrow(output, 0, 0, x.shape[0])


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

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):

        layer.process_after_load = True

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size,
                                                    hidden_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size,
                                                   intermediate_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                  2,
                                                  dtype=torch.float32),
                                       requires_grad=False)
        layer.register_parameter("w13_scale", w13_scale)

        w2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                 dtype=torch.float32),
                                      requires_grad=False)
        layer.register_parameter("w2_scale", w2_scale)

        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_scale, extra_weight_attrs)
            set_weight_attrs(w2_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8.")

            a13_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                      dtype=torch.float32),
                                           requires_grad=False)
            layer.register_parameter("a13_scale", a13_scale)
            set_weight_attrs(a13_scale, extra_weight_attrs)

            a2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                     dtype=torch.float32),
                                          requires_grad=False)
            layer.register_parameter("a2_scale", a2_scale)
            set_weight_attrs(a2_scale, extra_weight_attrs)
        else:
            layer.a13_scale = None
            layer.a2_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return

        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            w13_weight = torch.empty_like(layer.w13_weight.data,
                                          dtype=torch.float8_e4m3fn)
            w2_weight = torch.empty_like(layer.w2_weight.data,
                                         dtype=torch.float8_e4m3fn)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_scale = torch.nn.Parameter(torch.ones(
                layer.num_experts,
                dtype=torch.float32,
                device=w13_weight.device),
                                                 requires_grad=False)
            for expert in range(layer.num_experts):
                w13_weight[expert, :, :], layer.w13_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], layer.w2_scale[
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
                if layer.a13_scale is None or layer.a2_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None.")
                if (not all_close_1d(layer.a13_scale)
                        or not all_close_1d(layer.a2_scale)):
                    print_warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer. ")
                layer.a13_scale = torch.nn.Parameter(layer.a13_scale.max(),
                                                     requires_grad=False)
                layer.a2_scale = torch.nn.Parameter(layer.a2_scale.max(),
                                                    requires_grad=False)

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_scale.max(dim=1).values
            for expert_id in range(layer.num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :] = per_tensor_quantize(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            layer.w13_scale = torch.nn.Parameter(max_w13_scales,
                                                 requires_grad=False)
            return

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool = True) -> torch.Tensor:

        return fused_moe(x,
                         layer.w13_weight,
                         layer.w2_weight,
                         router_logits,
                         top_k,
                         renormalize=renormalize,
                         inplace=True,
                         use_fp8=True,
                         w1_scale=layer.w13_scale,
                         w2_scale=layer.w2_scale,
                         a1_scale=layer.a13_scale,
                         a2_scale=layer.a2_scale)


class Fp8KVCacheMethod(QuantizeMethodBase):
    """Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """Create "weight" (aka kv_scale) for an attention layer.

        Args:
            layer: The layer that is using the QuantizeMethodBase factory.
        """
        # Initialize the KV cache scale to 1.0 as the default value.
        # If the kv_scale appears in the checkpoint, it will be
        # overwritten when loading weights.
        layer.kv_scale = Parameter(torch.tensor(1.0), requires_grad=False)

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError("Fp8KVCacheMethod.apply should not be called.")

    def process_weights_after_loading(self, layer: Module) -> None:
        # If the kv-cache dtype is auto, we enforce the kv-scale to be 1.0
        # regardless whether the kv-scale is available in the checkpoint.
        if layer.kv_cache_dtype != "auto":
            kv_scale = layer.kv_scale.to("cpu").tolist()
            if not isinstance(kv_scale, float):
                raise ValueError("Only support per-tensor scaling factor "
                                 "for fp8 KV cache")
            layer._kv_scale = kv_scale
            if layer._kv_scale == 1.0 and "e5m2" not in layer.kv_cache_dtype:
                print_warning_once(
                    "Using KV cache scaling factor 1.0 for fp8_e4m3. This may "
                    "cause accuracy issues. Please make sure kv-cache scaling "
                    "factor is available in the fp8 checkpoint.")
        del layer.kv_scale


def per_tensor_quantize(tensor: torch.Tensor,
                        inv_scale: Union[float, torch.Tensor]) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def per_tensor_dequantize(
        tensor: torch.Tensor, inv_scale: Union[float,
                                               torch.Tensor]) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))
