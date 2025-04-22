# SPDX-License-Identifier: Apache-2.0
"""Custom activation functions."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import LazyDict


@CustomOp.register("fatrelu_and_mul")
class FatreluAndMul(CustomOp):
    """An activation function for FATReLU.

    The function computes x -> FATReLU(x[:d]) * x[d:] where
    d = x.shape[-1] // 2.
    This is used in openbmb/MiniCPM-S-1B-sft.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self, threshold: float = 0.):
        super().__init__()
        self.threshold = threshold
        if current_platform.is_cuda_alike():
            self.op = torch.ops._C.fatrelu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        x1 = F.threshold(x1, self.threshold, 0.0)
        return x1 * x2

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x, self.threshold)
        return out


@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            self.op = torch.ops._C.silu_and_mul
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops
            self.op = ipex_ops.silu_and_mul

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward_neuron(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        x_reshaped = x.view(-1, x.shape[-1])
        s = x_reshaped[:, :d] * F.sigmoid(x_reshaped[:, :d])
        result = s * x_reshaped[:, d:]
        return result.view(*x.shape[:-1], d)


@CustomOp.register("mul_and_silu")
class MulAndSilu(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> x[:d] * silu(x[d:]) where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike():
            self.op = torch.ops._C.mul_and_silu
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops
            self.op = ipex_ops.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return x[..., :d] * F.silu(x[..., d:])

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    # TODO implement forward_xpu for MulAndSilu
    # def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:


@CustomOp.register("gelu_and_mul")
class GeluAndMul(CustomOp):
    """An activation function for GeGLU.

    The function computes x -> GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            if approximate == "none":
                self.op = torch.ops._C.gelu_and_mul
            elif approximate == "tanh":
                self.op = torch.ops._C.gelu_tanh_and_mul
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops
            if approximate == "none":
                self.op = ipex_ops.gelu_and_mul
            else:
                self.op = ipex_ops.gelu_tanh_and_mul

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def extra_repr(self) -> str:
        return f'approximate={repr(self.approximate)}'


@CustomOp.register("gelu_new")
class NewGELU(CustomOp):

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            self.op = torch.ops._C.gelu_new
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops
            self.op = ipex_ops.gelu_new

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c *
                                           (x + 0.044715 * torch.pow(x, 3.0))))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


@CustomOp.register("gelu_fast")
class FastGELU(CustomOp):

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            self.op = torch.ops._C.gelu_fast
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops
            self.op = ipex_ops.gelu_fast

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 *
                                           (1.0 + 0.044715 * x * x)))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


@CustomOp.register("quick_gelu")
class QuickGELU(CustomOp):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    def __init__(self):
        super().__init__()
        if current_platform.is_cuda_alike() or current_platform.is_cpu():
            self.op = torch.ops._C.gelu_quick
        elif current_platform.is_xpu():
            from vllm._ipex_ops import ipex_ops
            self.op = ipex_ops.gelu_quick

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    # TODO implement forward_xpu for QuickGELU
    # def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:


@CustomOp.register("relu2")
class ReLUSquaredActivation(CustomOp):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return torch.square(F.relu(x))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(
        self,
        act_module: nn.Module,
        intermediate_size: int,
        input_is_parallel: bool = True,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size_per_partition = divide(intermediate_size,
                                                     tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(
            torch.empty(intermediate_size_per_partition, dtype=params_dtype))
        set_weight_attrs(self.scales, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


_ACTIVATION_REGISTRY = LazyDict({
    "gelu":
    lambda: nn.GELU(),
    "gelu_fast":
    lambda: FastGELU(),
    "gelu_new":
    lambda: NewGELU(),
    "gelu_pytorch_tanh":
    lambda: nn.GELU(approximate="tanh"),
    "relu":
    lambda: nn.ReLU(),
    "relu2":
    lambda: ReLUSquaredActivation(),
    "silu":
    lambda: nn.SiLU(),
    "quick_gelu":
    lambda: QuickGELU(),
})


def get_act_fn(act_fn_name: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_REGISTRY[act_fn_name]


_ACTIVATION_AND_MUL_REGISTRY = LazyDict({
    "gelu": lambda: GeluAndMul(),
    "silu": lambda: SiluAndMul(),
    "gelu_and_mul": lambda: GeluAndMul(),
})


def get_act_and_mul_fn(act_fn_name: str) -> nn.Module:
    """Get an activation-and-mul (i.e. SiluAndMul) function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_AND_MUL_REGISTRY:
        raise ValueError(
            f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_AND_MUL_REGISTRY[act_fn_name]
