import torch
import torch.nn as nn

from vllm import fused_kernels


class DequantAddResidual(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "a", torch.tensor(scale, dtype=torch.float32, requires_grad=False)
        )

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
        out = torch.empty_like(residual)
        fused_kernels.invoke_dequant_add_residual(out, x, residual, self.a.item())
        return out