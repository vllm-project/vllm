from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

from vllm.distributed import get_tensor_model_parallel_world_size
g_matmul_fp8 = None

class DeepSpeedFPConfig(QuantizationConfig):
    """Config for DeepSpeed FP quantizer. It supports fp6 and fp8.
    
    Args: 
        weight_bits: the target quantization bits, 6 or 8.
        group_size: group size for quantizaiton, default to 128.
    """

    def __init__(
        self,
        weight_bits: int = 8,
        group_size: int = 512,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.valid_types = [torch.bfloat16, torch.float16]

        if self.weight_bits not in (6, 8):
            raise ValueError(
                "Currently, only 6-bit or 8-bit weight quantization are "
                f"supported for DeepSpeed FP quantizaiton, but got "
                f"{self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"DeepSpeedFPConfig(weight_bits={self.weight_bits}), "
                f"group_size={self.group_size}")

    @classmethod
    def get_name(cls) -> str:
        return "DeepSpeedFP"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepSpeedFPConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits=weight_bits, group_size=group_size)

    def get_linear_method(self) -> "DeepSpeedFPLinearMethod":
        return DeepSpeedFPLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["DeepSpeedFPLinearMethod"]:
        if isinstance(layer, LinearBase):
            return DeepSpeedFPLinearMethod(self)
        return None


class DeepSpeedFPLinearMethod(LinearMethodBase):
    """Linear method for DeepSpeedFP quantizer.

    Args:
        quant_config: the DeepSpeedFP quantization config.
    """

    def __init__(self, quant_config: DeepSpeedFPConfig, enable_fused_kernel=True):
        self.quant_config = quant_config
        self.weight = None
        self.enable_fused_kernel = enable_fused_kernel

    def create_weights(self,
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int],
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       weight_loader=None,
                       **extra_weight_attrs):
        del output_size
        del input_size
        output_size_per_partition = sum(output_partition_sizes)
        weight = DeepSpeedFPParameter(
            torch.Size((output_size_per_partition, input_size_per_partition)),
            params_dtype=params_dtype,
            quant_config=self.quant_config,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })
        layer.register_parameter("weight", weight)

        def quant_weight_loader(param, loaded_weight, *args, **kwargs):
            if weight_loader is not None:
                if not hasattr(param, 'shadow_data'):
                    param.shadow_data = torch.empty(
                        param.orig_shape, 
                        dtype=loaded_weight.dtype, 
                        device=loaded_weight.device)
                    param.shadow_data.input_dim = param.input_dim
                    param.shadow_data.output_dim = param.output_dim
                    tp_size = get_tensor_model_parallel_world_size()
                weight_loader(param.shadow_data, loaded_weight, *args, **kwargs)
                loaded_weight = param.shadow_data

            param.ds_quantize_(
                loaded_weight.transpose(-1, -2).contiguous().cuda() if self.enable_fused_kernel else loaded_weight.cuda()
            )

        extra_weight_attrs["weight_loader"] = quant_weight_loader
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight
        if self.enable_fused_kernel:
            y = g_matmul_fp8(
                x, 
                weight, 
                weight.quantization_scales(), 
                weight.fp_quantizer.group_size
            )
            return y if bias is None else (y + bias)
        else:
            weight = layer.weight
            y = weight.ds_dequantize()
            return F.linear(x, y, bias)


class DeepSpeedFPParameter(nn.Parameter):
    """
    DeepSpeedFP quantized parameter class that implements fp8/fp6
    quantization deepspeed. Weights are stored in quantized form on
    GPUs, and can be dequantized on-the-fly when needed by the model.
    """

    def __new__(cls, orig_shape: torch.Size, params_dtype: torch.dtype,
                quant_config: DeepSpeedFPConfig, use_meta_tensor=True):
        try:
            import deepspeed
            if deepspeed.__version__ < "0.14.2":
                raise ImportError("deepspeed version is wrong. Please "
                                  "install deepspeed>=0.14.2.")
            global g_matmul_fp8
            from deepspeed.ops.fp_quantizer import FP_Quantize, matmul_fp8
            g_matmul_fp8 = matmul_fp8
        except ImportError as err:
            raise ImportError("Please install deepspeed>=0.14.2 via "
                              "`pip install deepspeed>=0.14.2` to use "
                              "deepspeedfp quantizer.") from err
        data = torch.empty((
            orig_shape.numel() // quant_config.group_size,
            quant_config.group_size * quant_config.weight_bits // 8 + 4,
        ),
                           dtype=torch.int8)
        self = torch.Tensor._make_subclass(cls, data, data.requires_grad)
        self.orig_shape = orig_shape
        self.quant_config = quant_config
        self.fp_quantizer = FP_Quantize(group_size=quant_config.group_size)
        self.fp_quantizer.orig_shape = orig_shape
        self.fp_quantizer.orig_dtype = params_dtype
        self.scale = None
        self.use_meta_tensor = use_meta_tensor
        return self

    def ds_quantize_(self, tensor: torch.Tensor):
        assert tensor.device.type == "cuda" and tensor.dtype != torch.int8
        q_data = self.fp_quantizer.quantize(
            tensor.data,
            q_bits=self.quant_config.weight_bits,
            return_meta_tensor=self.use_meta_tensor
        )
        if use_meta_tensor:
            q_data, self.scale = q_data
        self.data.copy_(q_data)
        del q_data
        return self.data

    def quantization_scales(self):
        return self.fp_quantizer.get_scales()

    def ds_dequantize(self, fp_out=None) -> torch.Tensor:
        """
        Return a tensor containing the dequantized weights of this parameter.
        """
        assert self.data.device.type == "cuda" and self.data.dtype == torch.int8
        return self.fp_quantizer.dequantize(
            self.data, 
            fp_out=fp_out, 
            q_bits=self.quant_config.weight_bits
            scale=self.scale,
        )

    def ds_selective_dequantize(self, indices, fp_out=None) -> torch.Tensor:
        """
        Return a tensor where only the weights at `indices` are dequantized
        (to save HBM -> SRAM bandwidth).
        """
        assert self.data.device.type == "cuda" and self.data.dtype == torch.int8
        return self.fp_quantizer.selective_dequantize(
            self.data,
            indices,
            fp_out=fp_out,
            q_bits=self.quant_config.weight_bits
            scale=self.scale,)
