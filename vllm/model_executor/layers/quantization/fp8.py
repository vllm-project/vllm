from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

ACTIVATION_SCHEMES = ["static", "dynamic"]


class FP8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_serialized: bool = False,
        activation_scheme: str = "dynamic",
    ) -> None:
        self.is_serialized = is_serialized
        assert activation_scheme in ACTIVATION_SCHEMES
        self.activation_scheme = activation_scheme

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # TODO: PyTorch 2.3.0+ is required to run FP8 on
        # SM 89 (e.g. Ada) GPUs. Specifically, this PR has to
        # be included: https://github.com/pytorch/pytorch/pull/118881
        return 89

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FP8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        return cls(is_serialized=is_serialized,
                   activation_scheme=activation_scheme)

    def get_linear_method(self) -> "Fp8LinearMethod":
        return Fp8LinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/scale activation scale.

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

    def __init__(self, quant_config: FP8Config):
        self.quant_config = quant_config

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

        layer.logical_widths = output_partition_sizes
        output_size_per_partition = sum(output_partition_sizes)

        # WEIGHT
        weight_dtype = torch.float8_e4m3fn if self.quant_config.is_serialized else params_dtype
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

        # WEIGHT SCALE
        weight_scale = Parameter(torch.empty(
            len(output_partition_sizes),
            dtype=torch.float32,
        ),
                                 requires_grad=False)
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(
            weight_scale, {
                **extra_weight_attrs,
                "shard_indexer": self.scales_shard_indexer,
            })

        # ACTIVATION SCALE
        if self.quant_config.activation_scheme == "static":
            act_scale = Parameter(torch.empty(len(output_partition_sizes),
                                              dtype=torch.float32),
                                  requires_grad=False)
            layer.register_parameter("act_scale", act_scale)
            set_weight_attrs(act_scale, {
                **extra_weight_attrs,
                "shard_indexer":
                self.scales_shard_indexer,
            })

    def shard_id_as_int(self, shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id
        assert isinstance(shard_id, str)
        qkv_idxs = {"q": 0, "k": 1, "v": 2}
        assert shard_id in qkv_idxs
        return qkv_idxs[shard_id]

    def scales_shard_indexer(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: Union[str, int],
        logical_widths: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del logical_widths
        return param[self.shard_id_as_int(shard_id)], loaded_weight

    def process_weights_after_loading(self, layer: Module) -> None:
        # Although the linear_method is propagated to all layers,
        # only linear layers invoke "create_weights". So we check
        # whether "weight_scale" is registered to determine
        # whether the layer is a linear layer that requires quantization.
        if not hasattr(layer, "weight_scale"):
            return

        # If we loaded in an FP8 checkpoint, we can skip weight quantization
        if self.quant_config.is_serialized:
            # torch._scaled_mm requires column-major in the second
            # input (weight), so we transpose the quantized weight.
            # TODO
            return

        qweight, weight_scale = per_tensor_quantize_dynamic(layer.weight)
        # torch._scaled_mm requires column-major in the second
        # input (weight), so we transpose the quantized weight.
        # TODO
        # layer.weight = Parameter(qweight.t(), requires_grad=False)
        layer.weight = Parameter(qweight, requires_grad=False)
        weight_scales = torch.tensor(
            [weight_scale for _ in layer.logical_widths], dtype=torch.float32)
        layer.weight_scale.data.copy_(weight_scales)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.quant_config.activation_scheme == "static":
            # Empirically, these are all the same
            x_scale = layer.act_scale.max()
            qinput = per_tensor_quantize_static(x, x_scale)
        else:
            qinput, x_scale = per_tensor_quantize_dynamic(x)

        # # TODO: Inefficient loop over each shard since there is a per-tensor
        # # scale for each shard.
        # # To be replaced by cutlass gemm with epilogue fusion for performance.
        # output = torch.zeros(x.shape[0],
        #                      layer.weight.shape[0],
        #                      dtype=x.dtype,
        #                      device="cuda")
        # start_offset = 0
        # for _, (logical_width, w_scale) in enumerate(
        #         zip(layer.logical_widths, layer.weight_scale)):
        #     end_offset = start_offset + logical_width

        #     cuda_compute_capability = torch.cuda.get_device_capability()
        #     if cuda_compute_capability >= (9, 0):
        #         out, _ = torch._scaled_mm(
        #             qinput,
        #             layer.weight[start_offset:end_offset, :].t(),
        #             out_dtype=x.dtype,
        #             scale_a=x_scale,
        #             scale_b=w_scale,
        #         )
        #     else:
        #         out = torch.nn.functional.linear(
        #             qinput.to(x.dtype) * x_scale.to(x.dtype),
        #             layer.weight[start_offset:end_offset, :].to(x.dtype) * w_scale.to(x.dtype),
        #         )

        #     output[:, start_offset:end_offset] = out
        #     start_offset = end_offset

        w_scale = layer.weight_scale.max()

        cuda_compute_capability = torch.cuda.get_device_capability()
        if cuda_compute_capability >= (9, 0):
            output, _ = torch._scaled_mm(
                qinput,
                layer.weight.t(),
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=w_scale,
            )
        else:
            output = torch.nn.functional.linear(
                qinput.to(x.dtype) * x_scale.to(x.dtype),
                layer.weight.to(x.dtype) * w_scale.to(x.dtype),
            )

        if bias is not None:
            output = output + bias

        return output


def per_tensor_quantize_static(tensor: torch.Tensor,
                               inv_scale: float) -> torch.Tensor:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
        inv_scale: The scale.
    """
    # Scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def per_tensor_quantize_dynamic(
        tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor dynamic scaling factor.
    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    min_val, max_val = tensor.aminmax()
    amax = min_val.abs().max(max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale
