from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn 

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


class DeepSpeedFPConfig(QuantizationConfig):
    """Config for DeepSpeed FP quantizer. It supports fp6 and fp8."""

    def __init__(
        self,
        weight_bits: int = 8,
        rounding: str = "nearest",
        mantissa_bits: int = 3,
        q_range = 480.0,
        group_size: int = 512,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.rounding = rounding
        self.mantissa_bits = mantissa_bits
        self.q_range = q_range
        self.valid_types = [torch.bfloat16, torch.float16]

        if self.weight_bits not in [6, 8]:
            raise ValueError(
                "Currently, only 6-bit or 8-bit weight quantization are "
                f"supported for DeepSpeed FP quantizaiton, but got {self.weight_bits} bits."
            )

    def __repr__(self) -> str:
        return (f"DeepSpeedFPConfig(weight_bits={self.weight_bits}), "
                f"group_size={self.group_size}, "
                f"rounding={self.rounding}, "
                f"mantissa_bits={self.mantissa_bits}, "
                f"")

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


class DeepSpeedFPLinearMethod(LinearMethodBase):
    """Linear method for DeepSpeedFP quantizer.

    Args:
        quant_config: the DeepSpeedFP quantization config.
    """

    def __init__(self, quant_config: DeepSpeedFPConfig):
        self.quant_config = quant_config
        self.weight = None
        self.q_weight = None
        # create the quantizer
        from deepspeed.ops.fp_quantizer import FP_Quantize
        self.quantizer = FP_Quantize(
            group_size=self.quant_config.group_size,
        )

    def create_weights(self,
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_size_per_partition: int,
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       **extra_weight_attrs):
        del output_size
        group_size = self.quant_config.group_size
        orig_numel = input_size_per_partition * output_size_per_partition
        weight = DeepSpeedFPQuantizedParameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype
            ).cpu(),
            requires_grad=False,
            quantization=self.quant_config,
        )
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
        })
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight
        y = weight.dequantized()
        return F.linear(x, y, bias)

    def dequantize(self, weight):
        assert weight.data.dtype == torch.int8 and weight.data.device.type == "cuda"
        with torch.cuda.stream(torch.cuda.current_stream(weight.data.device)):
            return self.quantizer.dequantize(weight.data, self.scales)

    def quantize(self, weight, return_meta_tensor=True):
        with torch.cuda.stream(torch.cuda.current_stream(weight.data.device)):
            return self.quantizer.quantize(weight.data, return_meta_tensor=return_meta_tensor)


class DeepSpeedFPQuantizedParameter(nn.Parameter):
    """
    DeepSpeedFP quantized parameter class that implements weight quantization via deepspeed. Weights
    are stored in quantized form on GPUs, and can be dequantized on-the-fly when
    needed by the model. The weights are actually quantized during any `.to(device)`
    if `device` is a cuda device.
    """
    def __new__(
        cls,
        data,
        requires_grad: bool = False,  # quantized weights should be frozen by default
        quantization: DeepSpeedFPConfig = None,
        quantizer=None,  # HF expects this argument.
    ):
        if quantization is None:
            quantization = DeepSpeedFPConfig()
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quant_config = quantization
        self.data = data
        from deepspeed.ops.fp_quantizer import FP_Quantize
        if quantizer is not None:
            self.quantizer = quantizer
        else:
            self.quantizer = FP_Quantize(
                group_size=quantization.group_size,
            )
        self._ensure_quantized(self)
        return self

    def _ensure_quantized(self, tensor: torch.Tensor):
        # If the tensor is on a cuda device and is not quantized, then quantize it in-place.
        if tensor.device.type == "cuda" and tensor.dtype != torch.int8:
            with torch.cuda.stream(torch.cuda.current_stream(tensor.device)):
                tensor.data = self.quantizer.quantize(tensor.data, q_bits=self.quant_config.weight_bits)
            assert tensor.dtype == torch.int8

    def dequantized(self, fp_out=None) -> torch.Tensor:
        """
        Return a tensor containing the dequantized weights of this parameter.
        """
        if self.data.device.type == "cuda" and self.data.dtype == torch.int8:
            with torch.cuda.stream(torch.cuda.current_stream(self.data.device)):
                return self.quantizer.dequantize(self.data, fp_out=fp_out, q_bits=self.quant_config.weight_bits)
        return self.data

    def selective_dequantized(self, indices, fp_out=None) -> torch.Tensor:
        """Return a tensor where only the weights at `indices` are dequantized (to save bandwidth)."""
        if self.data.device.type == "cuda" and self.data.dtype == torch.int8:
            with torch.cuda.stream(torch.cuda.current_stream(self.data.device)):
                return self.quantizer.selective_dequantize(self.data, 
                                                           indices, 
                                                           fp_out=fp_out, 
                                                           q_bits=self.quant_config.weight_bits)
        return self.data

    def __getstate__(self):
        state = self.__dict__
        state["data"] = self.data
        state["requires_grad"] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.quantizer = state["quantizer"]
        self.data = state["data"]
        self.requires_grad = state["requires_grad"]

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quantizer = copy.deepcopy(state["quantizer"])
        new_instance.data = copy.deepcopy(state["data"])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def cuda(self, device=None, non_blocking=False):
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

    def to(self, *args, **kwargs):
        """
        Move the parameter to the given device. Then, if the device is a cuda device,
        quantize it.
        """
        tensor = super().to(*args, **kwargs)
        self._ensure_quantized(tensor)
        return tensor

    @property
    def is_arctic(self):
        return True

    def cuda_parameter(self, device=None, non_blocking=False):
        a = self.to(device="cuda" if device is None else device, non_blocking=non_blocking)
        return torch.Tensor._make_subclass(DeepSpeedFPQuantizedParameter, a, self.requires_grad)
