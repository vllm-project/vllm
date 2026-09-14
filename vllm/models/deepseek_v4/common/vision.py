# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 vision tower (ViT + aligner) with TP-sharded linears.

Ported from the official reference implementation
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp). Weight names match the HF
checkpoint so no renaming is needed at load time. Attention and MLP weights
are tensor-parallel sharded (replicated when the vision head count is not
divisible by TP size, or under ``--mm-encoder-tp-mode data``); the patch
embed and norms are replicated, so the residual stream is full-width on
every rank.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel


@lru_cache(8)
def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Replicated: the residual stream is full-width on every rank.
        self.proj = ReplicatedLinear(
            3 * config.vision_patch_size**2,
            config.vision_dim,
            bias=True,
            quant_config=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.proj(x.flatten(1))
        return out


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        # Convention from minimax_m3/siglip: compute the TP/DP choice locally
        # instead of threading a flag through every constructor.
        use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.n_heads = config.vision_n_heads // self.tp_size
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.hidden_size = config.vision_dim
        self.wqkv = QKVParallelLinear(
            config.vision_dim,
            self.head_dim,
            config.vision_n_heads,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.wqkv",
            disable_tp=use_data_parallel,
        )
        self.wo = RowParallelLinear(
            config.vision_dim,
            config.vision_dim,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.wo",
            disable_tp=use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        n = x.size(0)
        qkv, _ = self.wqkv(x)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in qkv.chunk(3, -1))
        q = apply_rotary(q, cos, sin).unsqueeze(0)  # (b=1, n, h, d)
        k = apply_rotary(k, cos, sin).unsqueeze(0)
        # One image per call: a dense batch, no varlen packing metadata.
        o = self.attn(q, k, v.unsqueeze(0))
        out, _ = self.wo(o.reshape(n, -1))
        return out


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
        self.w1 = MergedColumnParallelLinear(
            config.vision_dim,
            [config.vision_inter_dim] * 2,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.w1",
            disable_tp=use_data_parallel,
        )
        self.w2 = RowParallelLinear(
            config.vision_inter_dim,
            config.vision_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.w2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w1(x)
        out, _ = self.w2(self.act_fn(gate_up))
        return out


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.norm1 = DeepseekV4RMSNorm(config.vision_dim)
        self.attn = DeepseekV4VisionAttention(config, prefix=f"{prefix}.attn")
        self.norm2 = DeepseekV4RMSNorm(config.vision_dim)
        self.mlp = DeepseekV4VisionMLP(config, prefix=f"{prefix}.mlp")

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4ViT(nn.Module):
    """DeepSeek-V4 ViT: full bidirectional attention per image, 2D RoPE."""

    def __init__(self, config):
        super().__init__()
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [
                DeepseekV4VisionBlock(config, prefix=f"blocks.{i}")
                for i in range(config.vision_n_layers)
            ]
        )
        self.norm = DeepseekV4RMSNorm(config.vision_dim)

    def forward(
        self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int
    ) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_vit_h, n_vit_w, self.rope_dim, self.rope_theta)
        cos = cos.to(device=x.device)
        sin = sin.to(device=x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4Aligner(nn.Module):
    """Spatial merge (downsample_ratio x downsample_ratio) + MLP projector."""

    def __init__(self, config):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = ColumnParallelLinear(
            in_dim,
            config.hidden_size,
            bias=True,
            quant_config=None,
            disable_tp=use_data_parallel,
        )
        self.w2 = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            quant_config=None,
            disable_tp=use_data_parallel,
        )

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_vit_h, n_vit_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_vit_w % r, 0, -n_vit_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        hidden, _ = self.w1(x)
        out, _ = self.w2(F.gelu(hidden))
        return out
