import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.utils import is_cpu

if not is_cpu():
    from xformers import ops as xops


class ViTAttention(nn.Module):
    def __init__(self, scale:float, dropout:float=0.0):
        super().__init__()
        self.scale = scale
        self.dropout = dropout

        if is_cpu():
            self.attn_fn = self._sdpa_forward
        else:
            self.attn_fn = self._xformers_forward

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Num_patches x Num_heads x Head_dim"""
        return self.attn_fn(q, k, v)

    def _sdpa_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.dropout)
        x = x.transpose(1, 2)
        return x

    def _xformers_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        x = xops.memory_efficient_attention_forward(q, k, v, scale=self.scale, p=self.dropout)
        return x
