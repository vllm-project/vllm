"""Custom activation functions."""
import torch
import torch.nn as nn

from vllm import activation_ops


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        activation_ops.silu_and_mul(out, x)
        return out
    
class DequantSiluAndMulQuant(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[1] // 2.

    Shapes:
        x: (num_tokens, 2 * d)
        return: (num_tokens, d)
    """
    # TODO(Zhang Ying): use_per_token_quant
    def __init__(self, dequant_scale: float = 1.0, quant_scale: float = 1.0, use_per_token_quant: bool = True) -> None:
        super().__init__()
        self.register_buffer('dequant_scale', torch.tensor(dequant_scale, dtype=torch.float32, requires_grad=False))
        self.register_buffer('quant_scale', torch.tensor(quant_scale, dtype=torch.float32, requires_grad=False))
        self.use_per_token_quant =use_per_token_quant
        
    def _apply(self, fn):
        super()._apply(fn)
        self.dequant_scale = self.dequant_scale.cpu()
        self.quant_scale = self.quant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(torch.float32)
        self.quant_scale = self.quant_scale.to(*args, **kwargs)
        self.quant_scale = self.quant_scale.to(torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        d = x.shape[1] // 2
        out = torch.empty(num_tokens, d, dtype=torch.int8, device=x.device)
        if self.use_per_token_quant: 
            scale = torch.empty(num_tokens, dtype=torch.float32, device=x.device)
            # tmp is used in kernel func
            tmp = torch.empty(num_tokens, d, dtype=torch.float32, device=x.device)
            activation_ops.invoke_dequant_silu_and_mul_quant(out, x, self.dequant_scale.item(), self.dequant_scale.item(), scale, tmp)
            return out, scale
        else:
            activation_ops.invoke_dequant_silu_and_mul_quant(out, x, self.dequant_scale.item(), self.dequant_scale.item(), self.quant_scale.item())
            return (out,)
    
class NewGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        activation_ops.gelu_new(out, x)
        return out


class FastGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        activation_ops.gelu_fast(out, x)
        return out


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_fast": FastGELU(),
    "gelu_new": NewGELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
}


def get_act_fn(act_fn: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn = act_fn.lower()
    if act_fn in _ACTIVATION_REGISTRY:
        return _ACTIVATION_REGISTRY[act_fn]
    raise ValueError(f"Activation function {act_fn!r} is not supported.")
