from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.nn import Parameter, Module

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsW8A8FP8"]


class CompressedTensorsFp8(CompressedTensorsScheme):        
    def create_weights(self,
                       layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype,
                       weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=torch.float8_e4m3fn),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": weight_loader,
        })

        weight_scale = Parameter(torch.empty(len(output_partition_sizes),
                                             dtype=torch.float32),
                                             requires_grad=False)

        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {
            "weight_loader": weight_loader,
            "ignore_warning": True,
            "fp8_scales_shard_indexer": self.scales_shard_indexer,
        })
        
        input_scale = Parameter(torch.empty(len(output_partition_sizes),
                                            dtype=torch.float32),
                                requires_grad=False)

        layer.register_parameter("input_scale", input_scale)
        set_weight_attrs(input_scale, {
            "weight_loader": weight_loader,
            "ignore_warning": True,
            "fp8_scales_shard_indexer": self.scales_shard_indexer,
        })


    def scales_shard_indexer(self, 
                             param: torch.Tensor, 
                             loaded_weight: torch.Tensor,
                             shard_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        qkv_idxs = {"q": 0, "k": 1, "v": 2}

        if isinstance(shard_id, int):
            pass
        elif isinstance(shard_id, str):
            if shard_id not in qkv_idxs:
                raise ValueError(f"Unknown shard_id: {shard_id}")
            shard_id = qkv_idxs[shard_id]
        else:
            ValueError(f"Shard id must be int or str but got {type(shard_id)}")

        return param[shard_id].reshape(-1), loaded_weight


    def process_weights_after_loading(self, layer) -> None:
        # WEIGHT_SCALE / WEIGHT
        #   Loop over logical weights, requantizing with single scale.
        max_w_scale = layer.weight_scale.max()
        start = 0
        for idx, logical_width in enumerate(layer.logical_widths):
            end = start + logical_width
            weight_dq = per_tensor_dequantize(layer.weight[start:end, :],
                                                layer.weight_scale[idx])

            layer.weight[start:end, :] = per_tensor_quantize(
                weight_dq, layer.weight_scale.max())
            start = end
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # WEIGHT
        #   Transpose weight for passing to torch._scaled_mm
        weight = layer.weight
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(),
                                      requires_grad=False)

    def apply_weights(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        assert bias is None

        qinput, x_scale = ops.scaled_fp8_quant(x, layer.input_scale)

        # Fused GEMM_DQ
        output = ops.cutlass_scaled_mm(
            qinput,
            layer.weight,
            out_dtype=x.dtype,
            scale_a=x_scale,
            scale_b=layer.weight_scale,
        )

        return torch.narrow(output, 0, 0, x.shape[0])


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
