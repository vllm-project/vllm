"""Custom normalization layers."""
import torch
import torch.nn as nn

from vllm import layernorm_ops


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6,
                 use_quant: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_quant = use_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(
            x, dtype=torch.int8) if self.use_quant else torch.empty_like(x)
        layernorm_ops.rms_norm(out, x, self.weight.data, self.variance_epsilon,
                               self.use_quant)
        return out


class DequantAddResidualI8RMSNormQuant(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # TODO(Zhang Ying): use_per_token_quant
    def __init__(self,
                 hidden_size: int,
                 dequant_scale: float = 1.0,
                 use_per_token_dequant: bool = True,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.register_buffer(
            "dequant_scale",
            torch.tensor(dequant_scale,
                         dtype=torch.float32,
                         requires_grad=False))
        self.use_per_token_dequant = use_per_token_dequant

    def _apply(self, fn):
        super()._apply(fn)
        self.dequant_scale = self.dequant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(torch.float32)
        return self

    def forward(self,
                residual: torch.Tensor,
                x: torch.Tensor,
                scale: torch.Tensor = None) -> torch.Tensor:
        out = torch.empty_like(x, dtype=torch.int8)
        if self.use_per_token_dequant and scale is not None:
            scale = scale * self.dequant_scale.item()
            layernorm_ops.invoke_dequant_add_residual_rms_norm_quant(
                out, x, residual, self.weight.data, scale,
                self.variance_epsilon)
        else:
            layernorm_ops.invoke_dequant_add_residual_rms_norm_quant(
                out, x, residual, self.weight.data, self.dequant_scale.item(),
                self.variance_epsilon)
        return residual, out
