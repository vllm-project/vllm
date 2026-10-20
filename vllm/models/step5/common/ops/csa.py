# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA helpers shared by the paged kernels.

Region selection depends on rounding to the OCP E4M3 grid, so it is done
explicitly, also on gfx942 whose native FP8 is E4M3FNUZ.
"""

import torch

_OCP_E4M3 = torch.float8_e4m3fn


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to the OCP E4M3 grid, saturating like Triton's ``float8e4nv`` cast."""
    x = x.to(torch.bfloat16).clamp(-448, 448)
    return x.to(_OCP_E4M3).to(torch.bfloat16)


def indexer_norm(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-centred RMSNorm on ``index_q``, LayerNorm on ``index_k``, in FP32."""
    dtype = index_q.dtype
    q = index_q.float()
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)
    q = q * (1.0 + q_norm_weight.float())
    k = torch.nn.functional.layer_norm(
        index_k.float(),
        (index_k.shape[-1],),
        k_norm_weight.float(),
        k_norm_bias.float(),
        eps,
    )
    return q.to(dtype), k.to(dtype)
