# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import PretrainedConfig

from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)

# Initial sizes of the per-axis RoPE cos/sin tables. 512 covers the t bound
# (500-frame processor cap / temporal_patch_size=2) and 4096 covers h/w
# (pixel-count caps with MAX_RATIO=200 give h/w <= ~680); tables are grown
# on demand since max_pixels is request-overridable.
_ROPE_TABLE_T = 512
_ROPE_TABLE_HW = 4096


class MiniMaxVLPatchEmbed(nn.Module):
    """Conv3d-based patch embedding.

    Takes flat tokens of shape (N, C * temporal_patch_size * patch_size²)
    and projects each to a hidden-size embedding.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        compression = config.img_token_compression_config
        temporal_patch_size = compression.get("temporal_patch_size", 2)
        patch_size = config.patch_size
        num_channels = config.num_channels

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_channels = num_channels
        self.hidden_size = config.hidden_size

        self.patch_embedding = Conv3dLayer(
            in_channels=num_channels,
            out_channels=config.hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (N, C * temporal_patch_size * patch_size²)
        if self.patch_embedding.weight.dtype != pixel_values.dtype:
            self.patch_embedding = self.patch_embedding.to(pixel_values.dtype)
        x = pixel_values.reshape(
            pixel_values.shape[0],
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.patch_embedding(x).reshape(x.shape[0], -1)


class MiniMaxVLAttention(nn.Module):
    """Multi-head attention with MiniMax's partial 3D RoPE.

    Partial means only the first ``rot_dim`` (< head_dim) dimensions of
    Q and K are rotated; the remaining dims are passed through unchanged.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.head_dim = embed_dim // num_heads
        self.num_heads_per_partition = dist_utils.divide(num_heads, self.tp_size)

        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            prefix=f"{prefix}.attn",
        )
        # Partial 3D RoPE geometry: rot_dim is split evenly across t/h/w
        # (same formula as MiniMaxVLVisionTransformer; rot_dim may be <
        # head_dim). mrope_section is in half-dim units for triton_mrope.
        rope_dims = 2 * (self.head_dim // 2)
        axis_dim = 2 * ((rope_dims // 3) // 2)
        self.rotary_dim = 3 * axis_dim
        self.mrope_section = [axis_dim // 2] * 3

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (N, 1, embed_dim)  [seq=N, batch=1, chan=embed_dim]
        x_qkv, _ = self.qkv_proj(x)  # (N, 1, 3 * heads_per_part * head_dim)
        seq_len, batch_size, _ = x_qkv.shape

        qkv = x_qkv.view(seq_len, 3, self.num_heads_per_partition, self.head_dim)
        v = qkv[:, 2].unsqueeze(0)  # (b=1, N, heads, head_dim) strided view

        # In-place partial 3D RoPE on strided q/k views of the packed qkv
        # tensor: one kernel launch, no repack/upcast copies. fp32 cos/sin
        # make the kernel compute in fp32, matching the reference
        # ``_minimax_rope_applier`` precision.
        q = qkv[:, 0].reshape(seq_len, -1)
        k = qkv[:, 1].reshape(seq_len, -1)
        triton_mrope(
            q,
            k,
            rotary_cos,
            rotary_sin,
            self.mrope_section,
            self.head_dim,
            self.rotary_dim,
            mrope_interleaved=False,
            is_neox_style=True,
        )
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)

        # Flash attention → (b, N, heads, head_dim)
        context = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        # Back to (N, 1, embed_dim)
        context = rearrange(context, "b s h d -> s b (h d)", b=batch_size)
        output, _ = self.out_proj(context)
        return output


class MiniMaxVLEncoderLayer(nn.Module):
    """Single CLIP-style transformer block."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.self_attn = MiniMaxVLAttention(
            embed_dim=embed_dim,
            num_heads=config.num_attention_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        use_data_parallel = is_vit_use_data_parallel()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.act = get_act_fn(getattr(config, "hidden_act", "gelu"))
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (N, 1, hidden_size)
        x = x + self.self_attn(
            self.layer_norm1(x),
            cu_seqlens,
            rotary_cos,
            rotary_sin,
            max_seqlen,
            sequence_lengths,
        )
        residual = x
        x, _ = self.fc1(self.layer_norm2(x))
        x = self.act(x)
        x, _ = self.fc2(x)
        return residual + x


class MiniMaxVLEncoder(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        num_hidden_layers_override: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        n = (
            config.num_hidden_layers
            if num_hidden_layers_override is None
            else num_hidden_layers_override
        )
        self.layers = nn.ModuleList(
            [
                MiniMaxVLEncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(n)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                cu_seqlens,
                rotary_cos,
                rotary_sin,
                max_seqlen,
                sequence_lengths,
            )
        return x


class MiniMaxVLVisionTransformer(nn.Module):
    """CLIP-based ViT with 3D RoPE (t/h/w decomposed).

    Faithfully mirrors the reference ``MiniMaxVLVisionTransformer``.
    FLASHINFER backend is not supported; standard flash-attn is used.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        compression = config.img_token_compression_config
        self.spatial_merge_size: int = compression.get("spatial_merge_size", 2)
        self.temporal_patch_size: int = compression.get("temporal_patch_size", 2)
        self.vision_segment_max_frames: int | None = getattr(
            config, "vision_segment_max_frames", None
        )
        self.use_data_parallel = is_vit_use_data_parallel()

        embed_dim = config.hidden_size
        head_dim = embed_dim // config.num_attention_heads
        # Backend selection + sharding info for building encoder metadata.
        # Defaults to FLASH_ATTN on SM80+; --mm-encoder-attn-backend FLASHINFER
        # selects the cuDNN ViT prefill path.
        self.hidden_size = embed_dim
        self.tp_size = (
            1
            if self.use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype()
        )
        rope_dims = 2 * (head_dim // 2)

        # Split rope dims evenly across t/h/w (same formula as the reference)
        self.t_dim = int(2 * ((rope_dims // 3) // 2))
        self.h_dim = int(2 * ((rope_dims // 3) // 2))
        self.w_dim = int(2 * ((rope_dims // 3) // 2))
        # rot_dim = t_dim + h_dim + w_dim (may be < head_dim)

        rope_theta: float = getattr(config, "rope_theta", 10000.0)
        inv_freq_t = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.t_dim, 2, dtype=torch.float32) / self.t_dim)
        )
        inv_freq_h = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.h_dim, 2, dtype=torch.float32) / self.h_dim)
        )
        inv_freq_w = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.w_dim, 2, dtype=torch.float32) / self.w_dim)
        )
        self.register_buffer("inv_freq_t", inv_freq_t, persistent=False)
        self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)

        # Vision analog of the text model's cos_sin_cache: per-axis cos/sin
        # tables indexed by position, gathered in _get_3d_rope_embed.
        cos_t, sin_t = self._build_rope_table(_ROPE_TABLE_T, inv_freq_t)
        cos_h, sin_h = self._build_rope_table(_ROPE_TABLE_HW, inv_freq_h)
        cos_w, sin_w = self._build_rope_table(_ROPE_TABLE_HW, inv_freq_w)
        self.register_buffer("rope_cos_t", cos_t, persistent=False)
        self.register_buffer("rope_sin_t", sin_t, persistent=False)
        self.register_buffer("rope_cos_h", cos_h, persistent=False)
        self.register_buffer("rope_sin_h", sin_h, persistent=False)
        self.register_buffer("rope_cos_w", cos_w, persistent=False)
        self.register_buffer("rope_sin_w", sin_w, persistent=False)

        self.embeddings = MiniMaxVLPatchEmbed(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        n_layers = config.num_hidden_layers
        if num_hidden_layers_override is None:
            num_hidden_layers_override = n_layers
        self.encoder = MiniMaxVLEncoder(
            config=config,
            num_hidden_layers_override=num_hidden_layers_override,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
        )

        if require_post_norm is None:
            require_post_norm = num_hidden_layers_override == n_layers
        self.post_layernorm = (
            nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            if require_post_norm
            else None
        )

        # out_hidden_size needed by run_dp_sharded_mrope_vision_model
        self.out_hidden_size = embed_dim

    # ── RoPE helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _build_rope_table(
        n: int, inv_freq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(
            torch.arange(n, device=inv_freq.device, dtype=inv_freq.dtype), inv_freq
        )
        return freqs.cos(), freqs.sin()

    def _grow_rope_table(
        self, axis: str, inv_freq: torch.Tensor, min_positions: int
    ) -> None:
        n = getattr(self, f"rope_cos_{axis}").shape[0]
        if min_positions <= n:
            return
        cos, sin = self._build_rope_table(max(min_positions, 2 * n), inv_freq)
        setattr(self, f"rope_cos_{axis}", cos)
        setattr(self, f"rope_sin_{axis}", sin)

    def _get_3d_rope_embed(
        self, grid_t: int, grid_h: int, grid_w: int, spatial_merge_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 3D RoPE cos/sin for a single (T, H, W) grid.

        Returns (3, T*H*W, half_rot_dim) t/h/w cos and sin planes for
        triton_mrope, gathered from the cached per-axis tables: each axis's
        values sit at their section columns of their own plane (the kernel's
        per-axis masks read only those columns).
        """
        tokens_per_frame = grid_h * grid_w

        tpos_ids = (
            torch.arange(grid_t, device=self.inv_freq_t.device)
            .unsqueeze(1)
            .expand(-1, tokens_per_frame)
            .flatten()
        )

        hpos_ids = (
            torch.arange(grid_h, device=self.inv_freq_h.device)
            .unsqueeze(1)
            .expand(-1, grid_w)
            .reshape(
                grid_h // spatial_merge_size,
                spatial_merge_size,
                grid_w // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .unsqueeze(0)
            .expand(grid_t, -1, -1, -1, -1)
            .flatten()
        )
        wpos_ids = (
            torch.arange(grid_w, device=self.inv_freq_w.device)
            .unsqueeze(0)
            .expand(grid_h, -1)
            .reshape(
                grid_h // spatial_merge_size,
                spatial_merge_size,
                grid_w // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .unsqueeze(0)
            .expand(grid_t, -1, -1, -1, -1)
            .flatten()
        )

        self._grow_rope_table("t", self.inv_freq_t, grid_t)
        self._grow_rope_table("h", self.inv_freq_h, grid_h)
        self._grow_rope_table("w", self.inv_freq_w, grid_w)

        half_t, half_h, half_w = self.t_dim // 2, self.h_dim // 2, self.w_dim // 2
        n = tpos_ids.numel()
        cos = self.rope_cos_t.new_zeros(3, n, half_t + half_h + half_w)
        sin = self.rope_sin_t.new_zeros(3, n, half_t + half_h + half_w)
        cos[0, :, :half_t] = self.rope_cos_t[tpos_ids]
        cos[1, :, half_t : half_t + half_h] = self.rope_cos_h[hpos_ids]
        cos[2, :, half_t + half_h :] = self.rope_cos_w[wpos_ids]
        sin[0, :, :half_t] = self.rope_sin_t[tpos_ids]
        sin[1, :, half_t : half_t + half_h] = self.rope_sin_h[hpos_ids]
        sin[2, :, half_t + half_h :] = self.rope_sin_w[wpos_ids]
        return cos, sin

    def _get_rope_embed_3d(
        self, grid_thw: list[list[int]], spatial_merge_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = [
            self._get_3d_rope_embed(t, h, w, spatial_merge_size) for t, h, w in grid_thw
        ]
        cos, sin = zip(*embeds, strict=True)
        return torch.cat(cos, dim=1), torch.cat(sin, dim=1)

    # ── Frame-limit helper (mirrors the reference) ───────────────────────

    def _apply_max_frames_limit(self, grid_thw: list[list[int]]) -> list[list[int]]:
        if self.vision_segment_max_frames is None:
            return grid_thw
        max_f = self.vision_segment_max_frames
        out: list[list[int]] = []
        for t, h, w in grid_thw:
            if t <= max_f:
                out.append([t, h, w])
            else:
                for i in range(0, t, max_f):
                    out.append([min(max_f, t - i), h, w])
        return out

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        # pixel_values: (total_N, C * temporal_patch_size * patch_size²)
        # Output:       (total_N, hidden_size)

        hidden = self.embeddings(pixel_values)  # (total_N, hidden_size)
        hidden = self.pre_layrnorm(hidden)

        limited = self._apply_max_frames_limit(grid_thw)

        # Token-level cumulative sequence lengths (one segment per limited grid).
        lens = [t * h * w for t, h, w in limited]
        cu_seqlens_np = np.zeros(len(lens) + 1, dtype=np.int32)
        np.cumsum(np.array(lens, dtype=np.int32), out=cu_seqlens_np[1:])

        # Backend-specific encoder metadata. For FLASH_ATTN this returns the raw
        # token cu_seqlens, the max segment length, and sequence_lengths=None;
        # for FLASHINFER (cuDNN) it repacks cu_seqlens into element-offset
        # indptrs, buckets max_seqlen, and builds padded per-sequence lengths.
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens_np, hidden.device
        )
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens_np),
            dtype=torch.int32,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens_np,
            self.hidden_size,
            self.tp_size,
            hidden.device,
        )

        # 3D RoPE cos/sin: (3, total_N, half_rot_dim) fp32 t/h/w planes for
        # triton_mrope, gathered from the cached per-axis tables. Kept fp32
        # so the kernel computes the rotation in fp32, matching the reference.
        rotary_cos, rotary_sin = self._get_rope_embed_3d(
            limited, self.spatial_merge_size
        )
        rotary_cos = rotary_cos.to(device=hidden.device)
        rotary_sin = rotary_sin.to(device=hidden.device)

        # Encoder expects (N, 1, hidden_size) — add batch dim
        hidden = hidden.unsqueeze(1)

        hidden = self.encoder(
            hidden,
            cu_seqlens,
            rotary_cos,
            rotary_sin,
            max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        hidden = hidden.squeeze(1)  # back to (total_N, hidden_size)

        if self.post_layernorm is not None:
            hidden = self.post_layernorm(hidden)

        return hidden


class MiniMaxVLMultiModalProjector(nn.Module):
    """Two-layer MLP projector: vision_hidden → text_hidden."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_size: int | None,
        multimodal_projector_bias: bool,
        projector_hidden_act: str = "gelu",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        mid = projector_hidden_size if projector_hidden_size else text_hidden_size
        use_dp = is_vit_use_data_parallel()
        self.linear_1 = ColumnParallelLinear(
            vision_hidden_size,
            mid,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
            disable_tp=use_dp,
        )
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            mid,
            text_hidden_size,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
            disable_tp=use_dp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_1(x)
        x = self.act(x)
        x, _ = self.linear_2(x)
        return x


class MiniMaxVLPatchMerger(nn.Module):
    def __init__(
        self,
        spatial_merge_size: int,
        text_hidden_size: int,
        projector_hidden_size: int | None,
        patch_merge_bias: bool,
        projector_hidden_act: str = "gelu",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        mid = projector_hidden_size if projector_hidden_size else text_hidden_size
        merge_in = text_hidden_size * spatial_merge_size**2
        use_dp = is_vit_use_data_parallel()
        self.linear_1 = ColumnParallelLinear(
            merge_in,
            mid,
            bias=patch_merge_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
            disable_tp=use_dp,
        )
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            mid,
            text_hidden_size,
            bias=patch_merge_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
            disable_tp=use_dp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, text_hidden_size) → (N // merge_size², text_hidden_size)
        x = x.reshape(x.shape[0] // (self.spatial_merge_size**2), -1)
        x, _ = self.linear_1(x)
        x = self.act(x)
        x, _ = self.linear_2(x)
        return x


class MiniMaxVLVisionModel(nn.Module):
    """Full vision model: ViT → projector → patch merger."""

    def __init__(
        self,
        config: PretrainedConfig,
        text_hidden_size: int,
        projector_hidden_size: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        compression = config.img_token_compression_config
        spatial_merge_size: int = compression.get("spatial_merge_size", 2)
        self.spatial_merge_size = spatial_merge_size
        self.use_data_parallel = is_vit_use_data_parallel()

        # The released checkpoint ships no ``post_layernorm`` weights and
        # uses ``vision_feature_layer=-1`` with ``vision_feature_select_strategy
        # ="full"``, i.e. the raw last encoder hidden state (CLIP's
        # ``last_hidden_state`` is taken before the post layernorm). Applying an
        # untrained post layernorm here would corrupt the visual features.
        self.vision_model = MiniMaxVLVisionTransformer(
            config=config,
            require_post_norm=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )
        self.multi_modal_projector = MiniMaxVLMultiModalProjector(
            vision_hidden_size=config.hidden_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            multimodal_projector_bias=getattr(
                config, "multimodal_projector_bias", True
            ),
            projector_hidden_act=getattr(config, "projector_hidden_act", "gelu"),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )
        self.patch_merge_mlp = MiniMaxVLPatchMerger(
            spatial_merge_size=spatial_merge_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            patch_merge_bias=getattr(config, "patch_merge_bias", True),
            projector_hidden_act=getattr(config, "projector_hidden_act", "gelu"),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "patch_merge_mlp"),
        )

        self.dtype = self.vision_model.embeddings.patch_embedding.weight.dtype
        self.out_hidden_size = text_hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        hidden = self.vision_model(pixel_values=pixel_values, grid_thw=grid_thw)
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        hidden = self.multi_modal_projector(hidden)
        hidden = self.patch_merge_mlp(hidden)
        return hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj.", "q_proj.", "q"),
            ("qkv_proj.", "k_proj.", "k"),
            ("qkv_proj.", "v_proj.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
