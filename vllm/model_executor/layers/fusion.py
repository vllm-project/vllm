import torch
import torch.nn as nn

from vllm import fused_kernels
    
class DequantAddResidual(nn.Module):
    # TODO(Zhang Ying): use_per_token_quant
    def __init__(self, dequant_scale: float = 1.0, use_per_token_dequant: bool = True) -> None:
        super().__init__()
        self.register_buffer(
            "dequant_scale", torch.tensor(dequant_scale, dtype=torch.float32, requires_grad=False)
        )
        self.use_per_token_dequant =use_per_token_dequant

    def _apply(self, fn):
        super()._apply(fn)
        self.dequant_scale = self.dequant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(torch.float32)
        return self

    def forward(self, residual: torch.Tensor, x: torch.Tensor, scale: torch.Tensor = None) -> torch.Tensor:
        out = torch.empty_like(residual)
        if self.use_per_token_dequant and scale is not None:
            scale = scale * self.dequant_scale.item()
            fused_kernels.invoke_dequant_add_residual(out, x, residual, scale)
        else:
            fused_kernels.invoke_dequant_add_residual(out, x, residual, self.dequant_scale.item())
        return out
