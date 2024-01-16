import torch
import torch.nn as nn

from vllm._C import ops


class DequantAddResidual(nn.Module):
    """ A fused function.
    It dequantizes x and adds the dequanted x with the residual in a fusion kernel function.
    """

    # TODO(Zhang Ying): use_per_token_quant
    def __init__(self, use_per_token_dequant: bool = True) -> None:
        super().__init__()
        self.use_per_token_dequant = use_per_token_dequant

    def forward(self,
                x: torch.Tensor,
                residual: torch.Tensor,
                weight_dequant_scale: float,
                scale: torch.Tensor = None) -> torch.Tensor:
        out = torch.empty_like(residual)
        if self.use_per_token_dequant and scale is not None:
            ops.dequant_add_residual(out, x, residual, scale,
                                     weight_dequant_scale)
        else:
            ops.dequant_add_residual(out, x, residual, weight_dequant_scale)
        return out, None
