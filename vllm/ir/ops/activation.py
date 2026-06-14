# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import torch
from torch import Tensor

from ..op import register_op

c_gelu_new = math.sqrt(2.0 / math.pi)


@register_op
def gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """
    Standard GELU activation function.
    
    Formula: x * 0.5 * (1.0 + erf(x / sqrt(2)))
    
    Args:
        x: Input tensor
        approximate: If 'tanh', use tanh approximation for faster computation
    """
    from vllm.platforms import CpuArchEnum, current_platform

    # ARM NEON LUT optimization for BF16
    is_arm_bf16 = (
        current_platform.is_cpu()
        and current_platform.get_cpu_architecture() == CpuArchEnum.ARM
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and hasattr(torch.ops._C, "activation_lut_bf16")
    )
    if is_arm_bf16:
        out = torch.empty_like(x)
        torch.ops._C.activation_lut_bf16(out, x, "gelu")
        return out

    import torch.nn.functional as F
    return F.gelu(x, approximate=approximate)


@gelu.register_input_generator
def _gelu_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    return (x,)


@register_op
def gelu_and_mul(x: Tensor, approximate: str = "none") -> Tensor:
    """
    GeGLU activation function: GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.
    
    This is used in models with gated feed-forward networks.
    
    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """
    import torch.nn.functional as F
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate=approximate) * x[..., d:]


@gelu_and_mul.register_input_generator
def _gelu_and_mul_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> tuple:
    # hidden_size must be even for gelu_and_mul (it's 2*d)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    return (x,)


@register_op
def gelu_new(x: Tensor) -> Tensor:
    """
    New GELU activation function.
    
    Formula: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    This is the GELU approximation used in GPT-2 and other transformer models.
    """
    return 0.5 * x * (1.0 + torch.tanh(c_gelu_new * (x + 0.044715 * torch.pow(x, 3.0))))


@gelu_new.register_input_generator
def _gelu_new_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    return (x,)


@register_op
def gelu_fast(x: Tensor) -> Tensor:
    """
    Fast GELU activation function.
    
    Formula: 0.5 * x * (1.0 + tanh(x * 0.7978845608 * (1.0 + 0.044715 * x^2)))
    
    A computationally efficient approximation of the GELU function.
    """
    return 0.5 * x * (
        1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
    )


@gelu_fast.register_input_generator
def _gelu_fast_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    return (x,)


@register_op
def quick_gelu(x: Tensor) -> Tensor:
    """
    Quick GELU activation function.
    
    Formula: x * sigmoid(1.702 * x)
    
    A fast approximation of GELU used in various transformer models.
    Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    """
    return x * torch.sigmoid(1.702 * x)


@quick_gelu.register_input_generator
def _quick_gelu_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    return (x,)
