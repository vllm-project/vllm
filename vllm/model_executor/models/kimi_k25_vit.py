# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Vision tower implementation for Kimi-K2.5 model.

This module provides the vision encoder components for Kimi-K2.5,
including 3D patch embedding, RoPE position embedding, and
temporal pooling for video chunks.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import GELUActivation

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model,
)
from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig

logger = init_logger(__name__)


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@get_rope_shape_decorate
@torch.compile(dynamic=True)
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sincos positional embedding from grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """Generate 1D sincos positional embedding."""
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    """2D learnable position embedding with temporal extension."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
                )

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out

    def compute_pos_embeds(
        self,
        grid_thw_list: list[list[int]] | list[tuple[int, int, int]],
        device: torch.device,
        dtype: torch.dtype | None = None,
        pad_to: int | None = None,
    ) -> torch.Tensor:
        """Compute the additive position embedding tensor without adding it
        to any input — used by ``prepare_encoder_metadata`` to precompute a
        fixed-shape buffer for CUDA graph capture/replay.

        Returns a tensor of shape ``(sum(t*h*w), dim)`` (or ``pad_to``), on
        ``device`` and optionally cast to ``dtype``.
        """
        pos_embs: list[torch.Tensor] = []
        # Cache interpolated 2D grids per (h, w) within a single call so
        # multiple items in the same batch with identical spatial shapes
        # (common: a batch of same-resolution images) share one
        # F.interpolate kernel launch.
        pos_emb_2d_cache: dict[tuple[int, int], torch.Tensor] = {}
        for t, h, w in grid_thw_list:
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            hw_key = (h, w)
            pos_emb_2d = pos_emb_2d_cache.get(hw_key)
            if pos_emb_2d is None:
                if (h, w) == self.weight.shape[:-1]:
                    pos_emb_2d = self.weight.flatten(end_dim=1)
                else:
                    pos_emb_2d = get_rope_shape(
                        self.weight,
                        interpolation_mode=self.interpolation_mode,
                        shape=(h, w),
                    )
                pos_emb_2d_cache[hw_key] = pos_emb_2d

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
                )
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        if pos_embs:
            out = torch.cat(pos_embs, dim=0)
        else:
            out = torch.zeros(
                (0, self.dim), dtype=self.weight.dtype, device=self.weight.device
            )

        if device is not None and out.device != device:
            out = out.to(device=device, non_blocking=True)
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype=dtype)
        if pad_to is not None and out.size(0) < pad_to:
            pad = torch.zeros(
                (pad_to - out.size(0), out.size(1)),
                dtype=out.dtype,
                device=out.device,
            )
            out = torch.cat([out, pad], dim=0)
        return out


class MoonVision3dPatchEmbed(nn.Module):
    """3D patch embedding for vision tower."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
    ):
        super().__init__()
        assert isinstance(patch_size, int | Sequence), (
            f"Invalid patch_size type: {type(patch_size)}"
        )
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, (
            f"Expected patch_size to be a tuple of 2, got {patch_size}"
        )
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(
        self,
        x: torch.Tensor,
        grid_thws: torch.Tensor,
        *,
        pos_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        # Precomputed path (CUDA graph replay): ``pos_embeds`` is padded
        # to the same worst-case length as ``x``, so we can add them
        # element-wise.  The padded tail adds zeros to padded pixel
        # positions, whose attention contribution is later masked out by
        # ``cu_seqlens`` and whose post-tpool contribution is routed to
        # the dead slot.  Eager path falls back to the per-item
        # interpolation loop in ``self.pos_emb``.
        x = x + pos_embeds if pos_embeds is not None else self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support."""

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return (
            f"dim={self.dim}, max_height={self.max_height}, "
            f"max_width={self.max_width}, theta_base={self.theta_base}"
        )

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid."""
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def _ensure_freqs_cis(self, device: torch.device) -> None:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False
            )

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        return self.get_freqs_cis_from_list(grid_thws.tolist(), device)

    def get_freqs_cis_from_list(
        self,
        shapes: list[list[int]] | list[tuple[int, int, int]],
        device: torch.device,
        pad_to: int | None = None,
    ) -> torch.Tensor:
        """List-based variant that avoids ``grid_thws.tolist()`` and
        supports right-padding to a fixed worst-case length (needed
        for CUDA graph capture/replay where buffer shapes are static).
        """
        self._ensure_freqs_cis(device)

        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        # Cache per-(h, w) slices within a single call — common batches
        # have repeated spatial shapes, and each cache hit replaces a
        # narrow ``[:h, :w]`` slice + reshape on the GPU with a Python
        # dict lookup.
        slice_cache: dict[tuple[int, int], torch.Tensor] = {}
        parts: list[torch.Tensor] = []
        for t, h, w in shapes:
            base = slice_cache.get((h, w))
            if base is None:
                base = self.freqs_cis[:h, :w].reshape(-1, self.dim // 2)
                slice_cache[(h, w)] = base
            parts.append(base.repeat(t, 1) if t != 1 else base)
        freqs_cis = (
            torch.cat(parts, dim=0)
            if parts
            else torch.empty((0, self.dim // 2), dtype=torch.complex64, device=device)
        )
        if pad_to is not None and freqs_cis.shape[0] < pad_to:
            pad_len = pad_to - freqs_cis.shape[0]
            pad = torch.zeros(
                (pad_len, self.dim // 2),
                dtype=freqs_cis.dtype,
                device=freqs_cis.device,
            )
            freqs_cis = torch.cat([freqs_cis, pad], dim=0)
        return freqs_cis


class MLP2(nn.Module):
    """Two-layer MLP with tensor parallel support."""

    def __init__(
        self,
        dims: list[int],
        activation,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        assert len(dims) == 3
        self.use_data_parallel = use_data_parallel
        self.fc0 = ColumnParallelLinear(
            dims[0],
            dims[1],
            bias=bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc0"),
            disable_tp=self.use_data_parallel,
        )
        self.fc1 = RowParallelLinear(
            dims[1],
            dims[2],
            bias=bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc1"),
            disable_tp=self.use_data_parallel,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc0(x)
        x = self.activation(x)
        x, _ = self.fc1(x)
        return x


class MoonViTEncoderLayer(nn.Module):
    """Single encoder layer for MoonViT with TP/DP support."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        activation=F.gelu,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.use_data_parallel = is_vit_use_data_parallel()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.tp_size = (
            1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.num_attention_heads_per_partition = divide(num_heads, self.tp_size)

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2(
            [hidden_dim, mlp_dim, hidden_dim],
            activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=self.use_data_parallel,
        )
        self.wqkv = QKVParallelLinear(
            hidden_size=hidden_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.wqkv",
            disable_tp=self.use_data_parallel,
        )
        self.wo = RowParallelLinear(
            hidden_dim,
            hidden_dim,
            bias=attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.wo",
            disable_tp=self.use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ):
        """Compute self-attention with packed QKV.

        Args:
            x (torch.Tensor): (seqlen, hidden_dim)
            cu_seqlens (torch.Tensor): cumulative sequence lengths
            rope_freqs_cis: precomputed 2D RoPE frequencies.
            max_seqlen: precomputed max sequence length (scalar tensor).
                If ``None`` it is computed on the fly from ``cu_seqlens``,
                which triggers a D2H sync and should be avoided inside
                CUDA graph capture.
            sequence_lengths: per-sequence lengths, only consumed by the
                FlashInfer CuDNN backend; ``None`` otherwise.
        """
        seq_length = x.size(0)
        xqkv, _ = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_out = self.attn(
            xq.unsqueeze(0),
            xk.unsqueeze(0),
            xv.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        attn_out = attn_out.reshape(
            seq_length,
            self.num_attention_heads_per_partition
            * self.hidden_size_per_attention_head,
        )
        attn_out, _ = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attention_qkvpacked(
            hidden_states,
            cu_seqlens,
            rope_freqs_cis,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonViT3dEncoder(nn.Module):
    """Full encoder stack for MoonViT 3D."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = "spatial_temporal",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert video_attn_type == "spatial_temporal", (
            f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
        )
        self.video_attn_type = video_attn_type
        # Cached encoder-shape fields so the (eager) forward can ask
        # ``MMEncoderAttention`` for backend-specific metadata rewrites
        # (FlashInfer ``cu_seqlens`` recomputation, ``sequence_lengths``
        # padding, bucketed ``max_seqlen``) without a round-trip to the
        # parent module.
        self.hidden_size = hidden_dim
        head_size = block_cfg["hidden_dim"] // block_cfg["num_heads"]
        self.tp_size = (
            1
            if is_vit_use_data_parallel()
            else get_tensor_model_parallel_world_size()
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=head_size,
            dtype=torch.get_default_dtype(),
        )
        self.rope_2d = Rope2DPosEmbRepeated(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(
                    **block_cfg,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
        *,
        encoder_metadata: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if encoder_metadata is None:
            rope_freqs_cis = self.rope_2d.get_freqs_cis(
                grid_thws=grid_thws, device=hidden_states.device
            )

            # Build cu_seqlens in numpy so we can feed it through the
            # backend-aware MMEncoderAttention helpers (FlashInfer needs
            # ``sequence_lengths`` + rewritten ``cu_seqlens`` + bucketed
            # ``max_seqlen`` — if we hand it a bare CUDA tensor those
            # helpers are never invoked and it asserts).
            shapes_np = (
                grid_thws.detach().cpu().numpy()
                if isinstance(grid_thws, torch.Tensor)
                else np.asarray(grid_thws, dtype=np.int32)
            )
            patches_per_frame = np.repeat(
                (shapes_np[:, 1] * shapes_np[:, 2]).astype(np.int32),
                shapes_np[:, 0].astype(np.int64),
            )
            if patches_per_frame.size > 0:
                cu_seqlens_np = np.concatenate(
                    [
                        np.zeros(1, dtype=np.int32),
                        patches_per_frame.cumsum(dtype=np.int32),
                    ]
                )
            else:
                cu_seqlens_np = np.zeros(1, dtype=np.int32)
            device = hidden_states.device
            sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
                self.attn_backend, cu_seqlens_np, device
            )
            max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
                self.attn_backend, cu_seqlens_np
            )
            max_seqlen = torch.tensor(max_seqlen_val, dtype=torch.int32)
            cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
                self.attn_backend,
                cu_seqlens_np,
                self.hidden_size,
                self.tp_size,
                device,
            )
        else:
            rope_freqs_cis = encoder_metadata["rope_freqs_cis"]
            cu_seqlens = encoder_metadata["cu_seqlens"]
            max_seqlen = encoder_metadata.get("max_seqlen")
            sequence_lengths = encoder_metadata.get("sequence_lengths")

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rope_freqs_cis=rope_freqs_cis,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    """Temporal pooling patch merger."""
    kh, kw = merge_kernel_size
    lengths = (grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]).tolist()
    seqs = x.split(lengths, dim=0)

    outputs = []
    for seq, (t, h, w) in zip(seqs, grid_thws.tolist()):
        nh, nw = h // kh, w // kw
        # Reshape: (t*h*w, d) -> (t, nh, kh, nw, kw, d)
        v = seq.view(t, nh, kh, nw, kw, -1)
        # Temporal pooling first (reduces tensor size before permute)
        v = v.mean(dim=0)  # (nh, kh, nw, kw, d)
        # Spatial rearrangement: (nh, kh, nw, kw, d) -> (nh, nw, kh, kw, d)
        out = v.permute(0, 2, 1, 3, 4).reshape(nh * nw, kh * kw, -1)
        outputs.append(out)

    return outputs


def _build_tpool_indices(
    grid_thw_list: list[list[int]] | list[tuple[int, int, int]],
    kh: int,
    kw: int,
    max_total_patches: int,
    max_post_tpool_patches: int,
    max_output_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute index buffers that turn :func:`tpool_patch_merger` into a
    pair of pure tensor ops (``scatter_reduce`` + ``index_select``) so the
    whole merger can live inside a CUDA graph.

    The dead slot is **pinned to index 0** of the post-tpool space so that
    the graph-captured buffers and smaller replay buffers agree regardless
    of how many real slots are populated.  Real slots occupy indices
    ``1 .. max_post_tpool_patches``; padded input patches and padded
    output positions both route to index 0.

    Returns three numpy arrays sized to the worst case:

    * ``temporal_gather_idx`` ``(max_total_patches,)``: for each input
      patch, the post-tpool slot (>= 1) it contributes to.  Padding
      positions map to slot 0 (the dead slot).
    * ``temporal_divisor`` ``(max_post_tpool_patches + 1,)``: per post-
      tpool slot, the count of contributing frames (``t_i``).  Slot 0
      has divisor 1.0 so the division is stable.
    * ``spatial_gather_idx`` ``(max_output_slots * kh * kw,)``: gather
      table mapping ``(item, nh, kh_off, nw, kw_off)`` to the
      ``(slot=nh*nw, kh*kw)`` layout.  Padded positions map to slot 0.
    """
    DEAD_SLOT = 0  # fixed index across capture and replay

    temporal_gather_idx = np.full(max_total_patches, DEAD_SLOT, dtype=np.int64)
    temporal_divisor = np.ones(max_post_tpool_patches + 1, dtype=np.float32)
    spatial_gather_idx = np.full(max_output_slots * kh * kw, DEAD_SLOT, dtype=np.int64)

    input_offset = 0  # offset into concatenated input patches
    post_tpool_offset = 1  # start at 1; slot 0 is the dead slot
    slot_offset = 0  # offset into flattened output slots (per-item nh*nw)

    # Precompute kh*kw/kw-sized ranges once; reused per item to
    # vectorize the inner fill and avoid Python-level loops (those
    # dominated replay cost before: ~18K iters per typical batch).
    kh_range = np.arange(kh, dtype=np.int64)
    kw_range = np.arange(kw, dtype=np.int64)

    for t, h, w in grid_thw_list:
        if h % kh != 0 or w % kw != 0:
            raise ValueError(
                f"grid ({h}, {w}) not divisible by merge kernel ({kh}, {kw})"
            )
        nh = h // kh
        nw = w // kw
        hw = h * w
        n_item_patches = t * hw
        n_item_slots = nh * nw

        # --- temporal scatter target ---
        # For every input patch (t_idx, r, c) the post-tpool target is
        # ``post_tpool_offset + r*w + c`` — independent of ``t_idx``.
        # Build one (h*w,) row and tile it ``t`` times.
        row_targets = post_tpool_offset + np.arange(hw, dtype=np.int64)
        # Equivalent to ``np.tile(row_targets, t)`` but written as a
        # broadcast to keep the allocation linear.
        temporal_gather_idx[
            input_offset : input_offset + n_item_patches
        ] = np.broadcast_to(row_targets, (t, hw)).reshape(-1)
        # Each post-tpool slot accumulates ``t`` frames.
        temporal_divisor[post_tpool_offset : post_tpool_offset + hw] = float(t)

        # --- spatial rearrangement gather table ---
        # For every (nh_i, nw_i, kh_i, kw_i):
        #   out_slot = slot_offset + nh_i*nw + nw_i
        #   out_idx  = out_slot*(kh*kw) + kh_i*kw + kw_i
        #   src      = post_tpool_offset + (nh_i*kh+kh_i)*w + (nw_i*kw+kw_i)
        # Build the 4D grid with broadcasting in one shot.
        nh_i = np.arange(nh, dtype=np.int64).reshape(nh, 1, 1, 1)
        nw_i = np.arange(nw, dtype=np.int64).reshape(1, nw, 1, 1)
        kh_i = kh_range.reshape(1, 1, kh, 1)
        kw_i = kw_range.reshape(1, 1, 1, kw)
        src = (
            post_tpool_offset
            + (nh_i * kh + kh_i) * w
            + (nw_i * kw + kw_i)
        )  # (nh, nw, kh, kw)
        # Output layout: ``(slot=nh*nw, kh*kw)`` flattened.  The 4D src
        # is already in ``(nh, nw, kh, kw)`` order, so a flat reshape
        # gives exactly the order expected by ``spatial_gather_idx``.
        spatial_gather_idx[
            slot_offset * (kh * kw) : (slot_offset + n_item_slots) * (kh * kw)
        ] = src.reshape(-1)

        input_offset += n_item_patches
        post_tpool_offset += hw
        slot_offset += n_item_slots

    # Dead-slot divisor stays 1.0 to keep the division finite.
    temporal_divisor[DEAD_SLOT] = 1.0
    return temporal_gather_idx, temporal_divisor, spatial_gather_idx


def tpool_patch_merger_indexed(
    x: torch.Tensor,
    metadata: dict[str, torch.Tensor | None],
    kh: int,
    kw: int,
) -> torch.Tensor:
    """Static-shape variant of :func:`tpool_patch_merger` suitable for
    CUDA graph capture/replay.

    Unlike the eager implementation (which iterates over a Python list and
    emits per-item tensors), this version does the whole merge with two
    tensor ops against precomputed indices:

    1. ``scatter_reduce_`` sums input patches into post-tpool slots (mean
       is applied explicitly by dividing by ``tpool_temporal_divisor``).
    2. ``index_select`` rearranges ``(item, nh, kh, nw, kw)`` into the
       ``(slot=nh*nw, kh*kw)`` layout expected by the projector.

    Slot 0 is reserved as a fixed "dead slot" so the index layout is
    identical whether called from capture (worst-case-sized inputs) or
    replay (partial-sized inputs copied into the captured buffers).
    Padded input patches and unused output positions both route to it.

    Args:
        x: encoded patches, shape ``(max_total_patches, hidden)``.  Values
            past the valid range are ignored because their corresponding
            index rows point to slot 0 (dead slot).
        metadata: dict containing ``tpool_temporal_gather_idx``,
            ``tpool_temporal_divisor``, and ``tpool_spatial_gather_idx``
            (see :func:`_build_tpool_indices`).
        kh, kw: temporal-pool merge kernel.

    Returns:
        ``(max_output_slots, kh*kw, hidden)`` tensor, consumable by the
        projector the same way the eager merger's per-item outputs are.
        The dead slot lives in the internal ``spatial_sum`` buffer and
        never appears in this output — ``spatial_gather_idx`` only
        addresses real slots (1..max_post_tpool_patches).  Output rows
        past the replay's valid range still live here but contain
        padded (zero) slot values and must be dropped by the caller
        using ``per_item_output_tokens``.
    """
    max_total_patches, hidden = x.shape
    temporal_gather_idx = metadata["tpool_temporal_gather_idx"]
    temporal_divisor = metadata["tpool_temporal_divisor"]
    spatial_gather_idx = metadata["tpool_spatial_gather_idx"]

    # temporal_divisor has shape (max_post_tpool_patches + 1,) where
    # index 0 is the dead slot and 1..max_post_tpool_patches are real
    # slots.  We allocate ``spatial_sum`` with the same leading dim.
    num_slots_incl_dead = temporal_divisor.shape[0]

    spatial_sum = torch.zeros(
        num_slots_incl_dead,
        hidden,
        dtype=x.dtype,
        device=x.device,
    )
    spatial_sum.scatter_reduce_(
        0,
        temporal_gather_idx.unsqueeze(-1).expand(max_total_patches, hidden),
        x,
        reduce="sum",
        include_self=True,
    )
    spatial_mean = spatial_sum / temporal_divisor.unsqueeze(-1).to(spatial_sum.dtype)

    gathered = spatial_mean.index_select(0, spatial_gather_idx)
    return gathered.view(-1, kh * kw, hidden)


class MoonViT3dPretrainedModel(nn.Module):
    """Main vision tower model.

    Uses KimiK25VisionConfig directly from transformers_utils/configs/kimi_k25.py.
    """

    def __init__(
        self,
        config: KimiK25VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        config = deepcopy(config)
        self.config = config  # Required for run_dp_sharded_mrope_vision_model
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type
        # Expose encoder-shape fields for CUDA graph capture sizing.
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.tp_size = (
            1 if is_vit_use_data_parallel() else get_tensor_model_parallel_world_size()
        )

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": get_act_fn("gelu_pytorch_tanh"),
                "attn_bias": True,
            },
            video_attn_type=config.video_attn_type,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "encoder"),
        )

        # Attention backend used by MMEncoderAttention inside each block.
        # Cached here so ``prepare_encoder_metadata`` can ask
        # ``MMEncoderAttention`` for backend-specific buffer transforms
        # (FlashInfer ``cu_seqlens`` rewrite, ``sequence_lengths`` padding,
        # bucketed ``max_seqlen``, ...).  Matches the cached state inside
        # each ``MMEncoderAttention`` layer.
        self.attn_backend = get_vit_attn_backend(
            head_size=self.head_size,
            dtype=torch.get_default_dtype(),
        )

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    # -- CUDA graph metadata --------------------------------------------
    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]] | list[tuple[int, int, int]],
        *,
        max_batch_size: int | None = None,
        max_frames_per_batch: int | None = None,
        max_total_patches: int | None = None,
        max_output_slots: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute encoder + tpool metadata from ``grid_thw_list``.

        Shared by the eager path, CUDA graph capture, and CUDA graph
        replay so all three paths read the same precomputed buffers.

        Args:
            grid_thw_list: per-item grids as lists of ``[t, h, w]``.
            max_batch_size: if set, pad ``cu_seqlens`` to this length
                (needed for graph replay where shapes are static).
            max_frames_per_batch: if set, overrides ``max_batch_size`` for
                ``cu_seqlens`` padding.  Each video item contributes ``t``
                attention sequences (one per frame), so total sequences
                can exceed ``max_batch_size``; this sizes the buffer to
                the total frame budget.  Falls back to ``max_batch_size``.
            max_total_patches: pad the input-patch-sized index buffers
                to this length.  Defaults to the sum of ``t*h*w`` across
                items (no padding).
            max_output_slots: pad the output-slot-sized index buffers to
                this length.  Defaults to the sum of ``(h//kh)*(w//kw)``.
            max_seqlen_override: if set, use this instead of computing
                ``max_seqlen`` from ``cu_seqlens`` (needed for capture:
                the scalar is baked into the graph, so it must cover the
                worst replay case).
            device: placement for output tensors.  Defaults to
                ``self.device``.
        """
        if device is None:
            device = self.device
        kh, kw = self.merge_kernel_size

        # Normalize to list of 3-tuples so downstream code doesn't see
        # torch tensors (we want pure-Python / numpy index construction
        # to avoid capturing device kernels here).
        shapes: list[tuple[int, int, int]] = [
            (int(t), int(h), int(w)) for t, h, w in grid_thw_list
        ]

        metadata: dict[str, torch.Tensor | None] = {}

        # Positional embeddings (learnable 2D interp + temporal sincos).
        # Precomputed so that the patch_embed can do a simple tensor-add
        # inside the graph instead of a Python loop + F.interpolate.
        total_patches = sum(t * h * w for t, h, w in shapes)
        if max_total_patches is None:
            max_total_patches = total_patches
        metadata["pos_embeds"] = self.patch_embed.pos_emb.compute_pos_embeds(
            shapes,
            device=device,
            dtype=self.dtype,
            pad_to=max_total_patches,
        )

        # 2D RoPE freqs — one entry per input patch (one-to-one with
        # ``pos_embeds`` / ``hidden_states`` after patch embed).
        rope_freqs_cis = self.encoder.rope_2d.get_freqs_cis_from_list(
            shapes, device=device, pad_to=max_total_patches
        )
        metadata["rope_freqs_cis"] = rope_freqs_cis

        # cu_seqlens — one sequence per frame (one per (item, t_idx))
        patches_per_frame = np.array(
            [h * w for t, h, w in shapes for _ in range(t)], dtype=np.int32
        )
        if patches_per_frame.size > 0:
            cu_seqlens_np = np.concatenate(
                [
                    np.zeros(1, dtype=np.int32),
                    patches_per_frame.cumsum(dtype=np.int32),
                ]
            )
        else:
            cu_seqlens_np = np.zeros(1, dtype=np.int32)

        pad_to = (
            max_frames_per_batch if max_frames_per_batch is not None else max_batch_size
        )
        if pad_to is not None:
            num_seqs = len(cu_seqlens_np) - 1
            if num_seqs < pad_to:
                cu_seqlens_np = np.concatenate(
                    [
                        cu_seqlens_np,
                        np.full(
                            pad_to - num_seqs,
                            cu_seqlens_np[-1],
                            dtype=np.int32,
                        ),
                    ]
                )

        # Backend-specific sequence_lengths (FlashInfer).
        metadata["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens_np, device
        )

        # max_seqlen — kept on CPU so attention wrappers that call
        # ``.item()`` don't induce a D2H sync in the captured graph
        # (the scalar is baked at capture time; capture must pick a
        # value >= any replay-time max seqlen).
        if max_seqlen_override is not None:
            max_seqlen_val = max_seqlen_override
        else:
            max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
                self.attn_backend, cu_seqlens_np
            )
        metadata["max_seqlen"] = torch.tensor(max_seqlen_val, dtype=torch.int32)

        # cu_seqlens may also get backend-specific rewrites (FlashInfer).
        metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens_np,
            self.hidden_size,
            self.tp_size,
            device,
        )

        # Indexed tpool buffers (sized to worst case).
        total_output_slots = sum((h // kh) * (w // kw) for t, h, w in shapes)
        max_post_tpool_patches = sum(h * w for t, h, w in shapes)
        if max_output_slots is None:
            max_output_slots = total_output_slots
        # post-tpool slots scale with h*w per item; align with output slots
        # upper bound so replays with smaller inputs still fit.
        max_post_tpool_patches = max(max_post_tpool_patches, max_output_slots * kh * kw)

        (
            temporal_gather_idx_np,
            temporal_divisor_np,
            spatial_gather_idx_np,
        ) = _build_tpool_indices(
            shapes,
            kh=kh,
            kw=kw,
            max_total_patches=max_total_patches,
            max_post_tpool_patches=max_post_tpool_patches,
            max_output_slots=max_output_slots,
        )

        metadata["tpool_temporal_gather_idx"] = torch.from_numpy(
            temporal_gather_idx_np
        ).to(device=device, non_blocking=True)
        metadata["tpool_temporal_divisor"] = torch.from_numpy(temporal_divisor_np).to(
            device=device, non_blocking=True
        )
        metadata["tpool_spatial_gather_idx"] = torch.from_numpy(
            spatial_gather_idx_np
        ).to(device=device, non_blocking=True)

        return metadata

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
        *,
        encoder_metadata: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): Temporal, height and width.
            encoder_metadata: precomputed buffers for CUDA graph replay.
                When provided, the encoder and the patch merger read
                from these buffers instead of computing per-batch state,
                and the merger returns a packed ``(max_slots, kh*kw, d)``
                tensor.  When ``None`` the eager per-item list output is
                returned (unchanged from before this feature).

        Returns:
            List of per-item tensors in eager mode; a single
            ``(max_slots, kh*kw, hidden)`` tensor in graph mode.
        """
        pos_embeds = None
        if encoder_metadata is not None:
            pos_embeds = encoder_metadata.get("pos_embeds")

        hidden_states = self.patch_embed(pixel_values, grid_thws, pos_embeds=pos_embeds)
        hidden_states = self.encoder(
            hidden_states, grid_thws, encoder_metadata=encoder_metadata
        )
        if self.merge_type != "sd2_tpool":
            raise NotImplementedError(f"Not support {self.merge_type}")

        kh, kw = self.merge_kernel_size
        if encoder_metadata is None:
            return tpool_patch_merger(
                hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size
            )
        return tpool_patch_merger_indexed(hidden_states, encoder_metadata, kh=kh, kw=kw)


@torch.inference_mode()
def mm_projector_forward(mm_projector: torch.nn.Module, vt_output: list[torch.Tensor]):
    """Apply MM projector to vision tower outputs."""
    num_embedding_list = [x.shape[0] for x in vt_output]
    batched = torch.cat(vt_output, dim=0)
    proj_out = mm_projector(batched)
    proj_out = proj_out.reshape(-1, proj_out.shape[-1])
    proj_out = torch.split(proj_out, num_embedding_list)
    return proj_out


@torch.inference_mode()
def vision_tower_forward(
    vision_tower: Any,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    mm_projector: Any,
    use_data_parallel: bool,
) -> list[torch.Tensor]:
    """DP-sharded vision tower forward with mrope.

    Uses vLLM's standard data parallelism utility to shard the batch
    across available GPUs, enabling parallel processing of vision features.
    """
    if use_data_parallel:
        grid_thw_list = grid_thw.tolist()
        vt_outputs = run_dp_sharded_mrope_vision_model(
            vision_model=vision_tower,
            pixel_values=pixel_values,
            grid_thw_list=grid_thw_list,
            rope_type="rope_2d",
        )
    else:
        vt_outputs = vision_tower(pixel_values, grid_thw)
    tensors = mm_projector_forward(mm_projector, list(vt_outputs))
    return list(tensors)


class KimiK25MultiModalProjector(nn.Module):
    """Multi-modal projector with patch merging for Kimi-K2.5."""

    def __init__(
        self,
        config: KimiK25VisionConfig,
        use_data_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.use_data_parallel = use_data_parallel

        # Hidden size after patch merging
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = config.hidden_size * merge_h * merge_w

        self.pre_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.linear_1 = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.linear_2 = ReplicatedLinear(
            self.hidden_size,
            config.mm_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )
        self.act = GELUActivation()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states
