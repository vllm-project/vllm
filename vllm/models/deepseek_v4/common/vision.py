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

import itertools
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
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
from vllm.model_executor.models.vision import (
    get_load_balance_assignment,
    is_vit_use_data_parallel,
)


def _compute_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


@lru_cache(8)
def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return _compute_vision_cos_sin(n_h, n_w, dim, theta)


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
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = x.size(0)
        qkv, _ = self.wqkv(x)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in qkv.chunk(3, -1))
        q = apply_rotary(q, cos, sin).unsqueeze(0)  # (b=1, n, h, d)
        k = apply_rotary(k, cos, sin).unsqueeze(0)
        # Dense (cu_seqlens=None): one image per call. Varlen (cu_seqlens
        # set): packed multi-image batch for encoder CUDA graph replay.
        o = self.attn(
            q, k, v.unsqueeze(0), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
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
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens, max_seqlen)
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

    def forward_packed(
        self,
        patches: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Varlen path for encoder CUDA graphs: multiple images packed along
        rows, per-image attention via ``cu_seqlens``, RoPE tables precomputed
        on device. Pure-tensor (no host reads, no H2D), safe to capture."""
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, cos, sin, cu_seqlens, max_seqlen)
        return self.norm(x)


def build_packed_vit_metadata(
    grids: list[list[int]] | list[tuple[int, int]],
    *,
    rope_dim: int,
    rope_theta: float,
    device: torch.device,
    max_seqlen_override: int | None = None,
    cached: bool = True,
) -> dict[str, torch.Tensor]:
    """Precompute packed-batch ViT metadata for ``DeepseekV4ViT.forward_packed``.

    Args:
        grids: ``[n_vit_h, n_vit_w]`` per image, in packing order.
        max_seqlen_override: Worst-case value baked in at CUDA graph capture
            (the attention wrapper reads ``max_seqlen`` on the host, so the
            capture-time value becomes a graph constant).
        cached: Use the shared ``lru_cache`` for RoPE tables. Capture-time
            dummy grids pass ``False`` to avoid evicting real entries.
    """
    cos_sin = get_vision_cos_sin if cached else _compute_vision_cos_sin
    cos_list: list[torch.Tensor] = []
    sin_list: list[torch.Tensor] = []
    cu_seqlens = [0]
    max_seqlen = 0
    for n_h, n_w in grids:
        cos, sin = cos_sin(n_h, n_w, rope_dim, rope_theta)
        cos_list.append(cos)
        sin_list.append(sin)
        n = n_h * n_w
        cu_seqlens.append(cu_seqlens[-1] + n)
        max_seqlen = max(max_seqlen, n)
    if cos_list:
        cos = torch.cat(cos_list).to(device)
        sin = torch.cat(sin_list).to(device)
    else:
        cos = torch.zeros((0, 1, rope_dim), dtype=torch.float32, device=device)
        sin = torch.zeros((0, 1, rope_dim), dtype=torch.float32, device=device)
    if max_seqlen_override is not None:
        max_seqlen = max_seqlen_override
    return {
        "vit_cos": cos,
        "vit_sin": sin,
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        # Read on the host by the attention wrapper; keep on CPU.
        "max_seqlen": torch.tensor(max_seqlen, dtype=torch.int32),
    }


def build_packed_merge_metadata(
    grids: list[list[int]] | list[tuple[int, int]],
    downsample_ratio: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Gather indices/mask for ``DeepseekV4Aligner.forward_packed``.

    For each aligner output row, the ``r x r`` source patch rows in the
    packed ViT output; positions past the image edge (the eager path pads
    them with zeros) get mask 0 and a clamped in-range index.
    """
    r = downsample_ratio
    idx_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    offset = 0
    for n_h, n_w in grids:
        rows = -(-n_h // r)
        cols = -(-n_w // r)
        bi = torch.arange(rows).repeat_interleave(cols)
        bj = torch.arange(cols).repeat(rows)
        di = torch.arange(r).repeat_interleave(r)
        dj = torch.arange(r).repeat(r)
        hi = bi[:, None] * r + di[None, :]
        wj = bj[:, None] * r + dj[None, :]
        valid = (hi < n_h) & (wj < n_w)
        idx = offset + hi.clamp(max=n_h - 1) * n_w + wj.clamp(max=n_w - 1)
        idx_list.append(idx)
        mask_list.append(valid)
        offset += n_h * n_w
    if idx_list:
        merge_idx = torch.cat(idx_list).to(device)
        merge_mask = torch.cat(mask_list).unsqueeze(-1).to(device=device, dtype=dtype)
    else:
        merge_idx = torch.zeros((0, r * r), dtype=torch.int64, device=device)
        merge_mask = torch.zeros((0, r * r, 1), dtype=dtype, device=device)
    return {"merge_idx": merge_idx, "merge_mask": merge_mask}


class DeepseekV4Aligner(nn.Module):
    """Spatial merge (downsample_ratio x downsample_ratio) + MLP projector."""

    def __init__(self, config):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
        self.downsample_ratio = config.vision_downsample_ratio
        self.out_dim = config.hidden_size
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

    def forward_packed(
        self,
        x: torch.Tensor,
        merge_idx: torch.Tensor,
        merge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Packed multi-image path for encoder CUDA graphs.

        ``merge_idx``/``merge_mask`` come from ``build_packed_merge_metadata``;
        gather+mask reproduces the eager ``F.pad`` + ``F.unfold`` merge
        exactly (unfold rows are channel-major, so the gathered patch rows
        are transposed before flattening).
        """
        r2 = self.downsample_ratio**2
        m = merge_idx.shape[0]
        gathered = (x[merge_idx] * merge_mask).view(m, r2, -1)
        gathered = gathered.permute(0, 2, 1).reshape(m, -1)
        hidden, _ = self.w1(gathered)
        out, _ = self.w2(F.gelu(hidden))
        return out


def run_dp_sharded_vision_tower(
    vision_model: DeepseekV4ViT,
    aligner: DeepseekV4Aligner,
    patches: torch.Tensor,
    vit_grid: list[list[int]],
) -> list[torch.Tensor]:
    """Run the ViT + aligner with images sharded across TP ranks.

    Every rank holds the full tower weights (``--mm-encoder-tp-mode data``)
    and receives the full ``patches`` batch. Images are assigned to ranks by
    patch count (greedy load balancing), each rank encodes only its share,
    and per-image embeddings are exchanged with one padded all-gather and
    returned in the original image order.

    Args:
        vision_model: The (weight-replicated) ViT tower.
        aligner: The (weight-replicated) spatial-merge projector.
        patches: ``(sum(n_vit_h * n_vit_w), 3, p, p)`` patches of all images.
        vit_grid: ``[n_vit_h, n_vit_w]`` per image.

    Returns:
        One ``(n_aligner_rows, hidden_size)`` embedding tensor per image.
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    sizes = [h * w for h, w in vit_grid]
    cum_patches = [0, *itertools.accumulate(sizes)]
    image_to_tp_rank, gpu_sample_counts, _ = get_load_balance_assignment(sizes, tp_size)
    cum_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

    # Rows the aligner emits per image: each grid dim is padded up to a
    # multiple of the merge ratio before r x r patches fold into one row.
    r = aligner.downsample_ratio
    rows_per_image = [-(-h // r) * -(-w // r) for h, w in vit_grid]

    def rank_image_idxs(g: int) -> list[int]:
        return image_to_tp_rank[cum_sample_counts[g] : cum_sample_counts[g + 1]]

    # All-gather needs one shape on every rank; pad each rank's packed
    # output to the largest per-rank row count (computable locally).
    max_rows = max(
        sum(rows_per_image[i] for i in rank_image_idxs(g)) for g in range(tp_size)
    )

    local_embeds = [
        aligner(
            vision_model(
                patches[cum_patches[i] : cum_patches[i + 1]],
                vit_grid[i][0],
                vit_grid[i][1],
            ),
            vit_grid[i][0],
            vit_grid[i][1],
        )
        for i in rank_image_idxs(tp_rank)
    ]
    if local_embeds:
        embeds_local = torch.cat(local_embeds, dim=0)
    else:
        embeds_local = patches.new_zeros((0, aligner.out_dim))
    if embeds_local.shape[0] < max_rows:
        embeds_local = torch.cat(
            [
                embeds_local,
                embeds_local.new_zeros(
                    (max_rows - embeds_local.shape[0], embeds_local.shape[1])
                ),
            ],
            dim=0,
        )
    gathered = tensor_model_parallel_all_gather(embeds_local.contiguous(), dim=0)

    out: list[torch.Tensor] = [None] * len(vit_grid)  # type: ignore[list-item]
    for g in range(tp_size):
        offset = g * max_rows
        for i in rank_image_idxs(g):
            out[i] = gathered[offset : offset + rows_per_image[i]]
            offset += rows_per_image[i]
    return out
