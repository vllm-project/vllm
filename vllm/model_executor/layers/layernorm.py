"""Custom normalization layers."""
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from vllm.model_executor.custom_op import CustomOp


def _cast_if_autocast_enabled(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (_cast_if_autocast_enabled(self.weight)
                           if self.weight is not None else self.weight)
        downcast_bias = (_cast_if_autocast_enabled(self.bias)
                         if self.bias is not None else self.bias)
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from vllm import _custom_ops as ops

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


class LPRMSNorm(nn.Module):
    """Root mean square normalization with low precision
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation is required to use torch.autocast."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        # downcast variables
        x = _cast_if_autocast_enabled(x)
        self.weight = _cast_if_autocast_enabled(self.weight)
        if residual is not None:
            residual = _cast_if_autocast_enabled(residual)

        # do calculations with downcasted variables
        with torch.autocast(enabled=False, device_type=x.device.type):
            if residual is not None:
                x = x + residual.to(torch.float32)
                residual = x.to(orig_dtype)

            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


NORM_CLASS_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "layernorm": torch.nn.LayerNorm,
    "low_precision_layernorm": LPLayerNorm,
    "rmsnorm": RMSNorm,
    "low_precision_rmsnorm": LPRMSNorm,
}
