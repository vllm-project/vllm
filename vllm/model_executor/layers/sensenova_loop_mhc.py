# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class MHCRouter(nn.Module):
    def __init__(self, hidden_size: int, settings: dict[str, Any]) -> None:
        super().__init__()
        self.streams = settings["streams"]
        self.iterations = settings["iterations"]
        self.fixed_identity = settings["fixed_identity"]
        self.pre_scale = settings.get("pre_scale", 1.0)
        num_outputs = 2 * self.streams
        if not self.fixed_identity:
            num_outputs += self.streams**2
        self.weight = nn.Parameter(
            torch.empty(num_outputs, self.streams * hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_outputs, dtype=torch.float32), requires_grad=False
        )

    def forward(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual.is_cuda:
            from vllm.model_executor.kernels.mhc import sensenova_loop as mhc_kernels
            from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton

            normalized = mhc_kernels.mhc_rmsnorm_triton(residual)
        else:
            flat = residual.flatten(-2).float()
            normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)

        if residual.is_cuda and residual.shape[0] <= 32:
            logits = mhc_kernels.mhc_project_triton(normalized, self.weight, self.bias)
        else:
            logits = F.linear(normalized, self.weight, self.bias)

        streams = self.streams
        pre = logits[..., :streams].sigmoid()
        post = 2 * logits[..., streams : 2 * streams].sigmoid()
        if self.fixed_identity:
            mix = torch.eye(streams, device=residual.device, dtype=torch.float32)
        else:
            mix = logits[..., 2 * streams :].unflatten(-1, (streams, streams))
            if residual.is_cuda:
                mix = mhc_kernels.mhc_log_sinkhorn_triton(mix, self.iterations)
            else:
                for _ in range(self.iterations):
                    mix = mix - mix.logsumexp(-1, keepdim=True)
                    mix = mix - mix.logsumexp(-2, keepdim=True)
                mix = mix.exp()
        if residual.is_cuda and residual.dtype == torch.bfloat16:
            # The checkpoint rounds coefficients before stream reduction.
            branch = hc_collapse_triton(residual, pre.to(residual.dtype).float())
        else:
            branch = (pre.to(residual.dtype).unsqueeze(-2) @ residual).squeeze(-2)
        return branch * self.pre_scale, post, mix

    @staticmethod
    def merge(
        residual: torch.Tensor,
        branch: torch.Tensor,
        post: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        if residual.is_cuda and residual.dtype == torch.bfloat16:
            from vllm.model_executor.kernels.mhc.sensenova_loop import mhc_merge_triton

            return mhc_merge_triton(residual, branch, post, mix)
        mixed = mix.to(residual.dtype) @ residual
        update = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
        return mixed + update
