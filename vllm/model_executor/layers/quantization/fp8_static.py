from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


class FP8StaticConfig(QuantizationConfig):
    """Config class for FP8."""

    @classmethod
    def get_name(cls) -> str:
        return "fp8_static"

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
    def from_config(cls, config: Dict[str, Any]) -> "FP8StaticConfig":
        return cls()

    def get_linear_method(self) -> "Fp8LinearMethod":
        return Fp8LinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for StaticFP8
    .
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: FP8StaticConfig):
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
        
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=torch.float8_e4m3fn),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)

        weight_scale = Parameter(
            torch.empty(
                len(output_partition_sizes), 
                 device='cuda', dtype=torch.float32,
            ), requires_grad=False
        )
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, extra_weight_attrs)
        set_weight_attrs(weight_scale, {
            "shard_indexer": self.scales_shard_indexer,
        })

        in_scale = Parameter(
            torch.empty(
                len(output_partition_sizes), 
                 device='cuda', dtype=torch.float32,
            ), requires_grad=False
        )
        layer.register_parameter("in_scale", in_scale)
        set_weight_attrs(in_scale, extra_weight_attrs)
        set_weight_attrs(in_scale, {
            "shard_indexer": self.scales_shard_indexer,
        })

        layer.logical_widths = output_partition_sizes

    def shard_id_as_int(
        self, 
        shard_id: Union[str, int]
    ) -> int:
        if isinstance(shard_id, int):
            return shard_id
        assert isinstance(shard_id, str)
        qkv_idxs = { "q": 0, "k": 1, "v": 2 }
        assert shard_id in qkv_idxs
        return qkv_idxs[shard_id]

    # def scales_shard_splitter_NKK(
    #     self,
    #     param: torch.Tensor,
    #     loaded_weight: torch.Tensor,
    #     shard_id: Union[str, int],
    #     logical_widths: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     shard_id = self.shard_id_as_int(shard_id)
    #     offset = sum(logical_widths[:shard_id]) 
    #     size = logical_widths[shard_id]
    #     # update loaded weight with copies for broadcast.
    #     loaded_weight = loaded_weight.repeat(size)
    #     return param[offset : offset + size], loaded_weight
    
    def scales_shard_indexer(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: Union[str, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(f"----- shard_id: {shard_id}")
        # print(f"----- loaded_weight: {loaded_weight}")
        return param[self.shard_id_as_int(shard_id)], loaded_weight

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:        
        logical_widths = layer.logical_widths
        q_weight = layer.weight
        w_scales = layer.weight_scale
        in_scales = layer.in_scale

        output = torch.zeros(x.shape[0], q_weight.shape[0], dtype=x.dtype, device="cuda")
        start_offset = 0
        for _, (logical_width, w_scale, in_scale) in enumerate(zip(logical_widths, w_scales, in_scales)):
            end_offset = start_offset + logical_width
            weight_dq = self._dequantize(q_weight[start_offset:end_offset, :], w_scale, x.dtype)
            x_dq = self._fake_quantize_static(x, in_scale)

            # print(f"x_dq[0,0]: {x_dq[0,0]} // weight_dq[0,0]: {weight_dq[0,0]}")
            output[:, start_offset:end_offset] = torch.nn.functional.linear(x_dq, weight_dq)
            start_offset = end_offset
        
        assert end_offset == output.shape[1]
        # print(output)
        # print(output.dtype)
        return output

    def _quantize_dynamic(self, x: torch.Tensor):
        finfo = torch.finfo(torch.float8_e4m3fn)
        min_val, max_val = x.aminmax()
        amax = min_val.abs().max(max_val.abs())
        scale = finfo.max / amax.clamp(min=1e-12)

        # print(finfo.max)
        # print(amax)
        # print(finfo.max / amax.clamp(min=1e-12))
        # assert False
        # scale and clamp the tensor to bring it to
        # the representative range of float8 data type
        # (as default cast is unsaturated)
        qweight = (x * scale).clamp(min=finfo.min, max=finfo.max)
        # Return both float8 data and the inverse scale (as float),
        # as both required as inputs to torch._scaled_mm
        # print(scale)
        return qweight, scale.float().reciprocal()
    
    def _quantize(self, x: torch.Tensor, inv_scale: torch.tensor):
        finfo = torch.finfo(torch.float8_e4m3fn)
        return (x / inv_scale).clamp(min=finfo.min, max=finfo.max)
        
    def _dequantize(self, xq: torch.Tensor, inv_scale: torch.tensor, dtype: torch.dtype):
        return (xq.to(dtype) * inv_scale)
    
    def _fake_quantize_static(self, x: torch.Tensor, inv_scale: torch.Tensor):
        xq = self._quantize(x, inv_scale)
        # xq, inv_scale = self._dynamic_quantize(x)
        # print(inv_scale)
        xdq = self._dequantize(xq, inv_scale, x.dtype)

        # print(f"----- inv_scale: {inv_scale} // x[0,0]: {x[0,0]} // xq[0,0]: {xq[0,0]} // xdq[0,0]: {xdq[0,0]}")

        return xdq


def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor static scaling factor.

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
