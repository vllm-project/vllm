"""Custom normalization layers."""
import torch
import torch.nn as nn

from vllm import layernorm_ops


class RMSNorm(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        layernorm_ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

class I8RMSNorm(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x, dtype=torch.int8)
        layernorm_ops.invoke_rms_norm_quant(out, x, self.weight.data, self.variance_epsilon)
        return out
    

class DequantAddResidualI8RMSNormQuant(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        scale: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_buffer(
            "a", torch.tensor(scale, dtype=torch.float32, requires_grad=False)
        )
        self.variance_epsilon = eps
    
    def _apply(self, fn):
        super()._apply(fn)
        self.a = self.a.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.a = self.a.to(*args, **kwargs)
        self.a = self.a.to(torch.float32)
        return self

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x, dtype=torch.int8)
        layernorm_ops.invoke_dequant_add_residual_rms_norm_quant(out, x, residual, self.weight.data, self.variance_epsilon, self.a.item())
        return residual, out
