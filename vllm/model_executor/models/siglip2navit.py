# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Implementation of SiglipVisionModel intended to be only used
within a vision language model."""

from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Siglip2VisionConfig
from transformers.configuration_utils import PreTrainedConfig

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import async_tensor_h2d

from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from .vision import is_vit_use_data_parallel


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_patches = config.num_patches
        self.preserve_original_pe = config.preserve_original_pe
        self.hidden_stride = config.hidden_stride

        # siglip2 naflex
        if self.num_patches > 0:
            self.patch_embedding = ReplicatedLinear(
                input_size=config.num_channels * self.patch_size * self.patch_size,
                output_size=self.embed_dim,
                return_bias=False,
            )
            if self.preserve_original_pe:
                self.position_embedding_size = int(self.num_patches**0.5)
                self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

        else:
            self.patch_embedding = Conv2dLayer(
                in_channels=config.num_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                padding="valid",
            )
            if self.preserve_original_pe:
                self.num_patches = (self.image_size // self.patch_size) ** 2
                self.position_embedding_size = self.image_size // self.patch_size
                self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def get_position_embeddings(
        self,
        grid_thws: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Interpolate the learned position embeddings to each item's grid.

        Returns a tensor of shape (sum(t * h * w), embed_dim), ordered the
        same way as the patchified pixel values.
        """
        positional_embeddings = (
            self.position_embedding.weight.reshape(
                self.position_embedding_size, self.position_embedding_size, -1
            )
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
        )
        pos_embeds = []
        for t, h, w in grid_thws:
            t, h, w = int(t), int(h), int(w)
            pe = F.interpolate(
                positional_embeddings,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
            pe = pe[0].repeat(t, 1)
            pe = pe.reshape(
                t,
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
                -1,
            )
            pos_embeds.append(pe.permute(0, 1, 3, 2, 4, 5).reshape(t * h * w, -1))
        return torch.cat(pos_embeds)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: Sequence[Sequence[int]] | None = None,
        pos_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
        pixel_values (`torch.FloatTensor`):
            Pixel values of shape (
                num_patches,
                num_channels * temporal_patch_size * patch_size * patch_size
            )
        grid_thws:
            grid shape (num_items, 3)
        pos_embeds:
            Precomputed position embeddings from `get_position_embeddings`.
            If None, they are computed from `grid_thws`.

        """
        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        if isinstance(self.patch_embedding, LinearBase):
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        elif isinstance(self.patch_embedding, Conv2dLayer):
            pixel_values = pixel_values.view(
                -1,
                self.config.num_channels * self.config.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            )
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            patch_embeds = patch_embeds.reshape(-1, self.embed_dim)

        if self.preserve_original_pe:
            if pos_embeds is None:
                assert grid_thws is not None
                pos_embeds = self.get_position_embeddings(grid_thws)
            patch_embeds = patch_embeds + pos_embeds

        return patch_embeds


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_flash_attn_backend: bool,
    apply_rotary_emb: ApplyRotaryEmb,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()

    if is_flash_attn_backend and current_platform.is_cuda():
        apply_rotary_emb_func = apply_rotary_emb.forward_cuda
    elif is_flash_attn_backend and current_platform.is_rocm():
        apply_rotary_emb_func = apply_rotary_emb.forward_hip
    else:
        apply_rotary_emb_func = apply_rotary_emb.forward_native

    q_embed = apply_rotary_emb_func(q, cos, sin)
    k_embed = apply_rotary_emb_func(k, cos, sin)

    return q_embed, k_embed


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        use_data_parallel = is_vit_use_data_parallel()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )
        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )

        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)
        self.use_rope = config.use_rope

        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            scale=self.scale,
            prefix=f"{prefix}.attn",
        )

        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            enable_fp32_compute=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel."""
        seq_length, embed_dim = hidden_states.shape

        qkv_states, _ = self.qkv_proj(hidden_states)
        queries, keys, values = qkv_states.chunk(3, dim=-1)

        queries = queries.view(seq_length, self.num_heads_per_partition, self.head_dim)
        keys = keys.view(seq_length, self.num_heads_per_partition, self.head_dim)
        values = values.view(seq_length, self.num_heads_per_partition, self.head_dim)

        if self.use_rope:
            assert position_embeddings is not None
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb(
                queries.unsqueeze(0),
                keys.unsqueeze(0),
                cos,
                sin,
                self.attn.is_flash_attn_backend,
                self.apply_rotary_emb,
            )
            queries = queries.squeeze(0)
            keys = keys.squeeze(0)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output = self.attn(
            query=queries.unsqueeze(0),
            key=keys.unsqueeze(0),
            value=values.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        attn_output = attn_output.reshape(
            seq_length, self.num_heads_per_partition * self.head_dim
        )

        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class Siglip2MLP(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        use_data_parallel = is_vit_use_data_parallel()
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.FloatTensor]:
        """Args:
        hidden_states: Input tensor of shape (batch, seq_len, embed_dim).
        cu_seqlens: Cumulative sequence lengths tensor.
        position_embeddings: Position embeddings tensor.
        max_seqlen: Maximum sequence length in `cu_seqlens`. Computed from
            `cu_seqlens` if None.

        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Siglip2Encoder(nn.Module):
    """Transformer encoder consisting of `config.num_hidden_layers`
    self attention layers. Each layer is a [`Siglip2EncoderLayer`].

    Args:
        config: PreTrainedConfig

    """

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

        self.rotary_pos_emb = VisionRotaryEmbedding(
            config.hidden_size // config.num_attention_heads // 2
        )
        self.patch_size = config.patch_size
        self.hidden_stride = config.hidden_stride
        self.window_size = config.window_size
        self.spatial_merge_unit = config.hidden_stride * config.hidden_stride
        if config.fullatt_block_indexes is None:
            self.fullatt_block_indexes = None
        else:
            self.fullatt_block_indexes = [
                int(i) for i in config.fullatt_block_indexes.split("|")
            ]

    @staticmethod
    def _grid_thws_to_list(
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> list[tuple[int, int, int]]:
        if isinstance(grid_thws, torch.Tensor):
            grid_thws = grid_thws.tolist()
        return [(int(t), int(h), int(w)) for t, h, w in grid_thws]

    # copied from qwen2.5_vl
    def rot_pos_emb(self, grid_thw: Sequence[Sequence[int]]) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        pos_ids = async_tensor_h2d(pos_ids, rotary_pos_emb_full.device)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(
        self, grid_thw: Sequence[Sequence[int]]
    ) -> tuple[torch.Tensor, list[int]]:
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        # patch (after merge) number in each window
        vit_merger_window_size = (
            self.window_size // self.hidden_stride // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.hidden_stride,  # number of patch after merge
                grid_w // self.hidden_stride,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += grid_t * llm_grid_h * llm_grid_w
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def get_encoder_cudagraph_max_window_seqs(
        self,
        token_budget: int,
        max_num_seqs: int,
    ) -> int:
        """Upper bound on the number of window-attention sequences for
        `token_budget` merged tokens split over at most `max_num_seqs`
        attention sequences (images or video frames).

        Mirrors the bound used by Qwen2.5-VL: the most non-empty windows for a
        fixed token budget come from a thin strip along one axis, plus one
        extra partial window per sequence.
        """
        vit_merger_window_size = (
            self.window_size // self.hidden_stride // self.patch_size
        )
        max_strip_windows = (
            token_budget + vit_merger_window_size - 1
        ) // vit_merger_window_size
        return min(token_budget, max_num_seqs + max_strip_windows)

    @staticmethod
    def _pad_cu_seqlens(cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
        """Pad `cu_seqlens` to `num_seqs + 1` entries by repeating the last
        offset, so the padded entries represent empty sequences."""
        num_pad = num_seqs + 1 - cu_seqlens.shape[0]
        if num_pad <= 0:
            return cu_seqlens
        return torch.cat([cu_seqlens, cu_seqlens[-1:].expand(num_pad)])

    def prepare_encoder_metadata(
        self,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
        *,
        max_num_seqs: int | None = None,
        max_num_window_seqs: int | None = None,
        max_seqlen_override: int | None = None,
        max_seqlen_window_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all grid-dependent encoder inputs.

        Shared by the eager forward path, CUDA graph capture and CUDA graph
        replay so that all of them run the exact same computation.

        Args:
            grid_thws: Grid configurations as a list of [t, h, w].
            max_num_seqs: If set, pad `cu_seqlens` to this many sequences
                (needed for a stable shape under CUDA graph capture/replay).
            max_num_window_seqs: If set, pad `cu_window_seqlens` to this many
                window sequences.
            max_seqlen_override: If set, use this value as the full-attention
                max_seqlen instead of computing it from `cu_seqlens`.
            max_seqlen_window_override: Same as `max_seqlen_override`, for
                window attention.
            device: Device to place the tensors on.

        """
        grid_thws = self._grid_thws_to_list(grid_thws)
        if device is None:
            device = self.rotary_pos_emb.inv_freq.device

        rotary_pos_emb = self.rot_pos_emb(grid_thws)
        window_index, cu_window_seqlens_list = self.get_window_index(grid_thws)
        cu_window_seqlens = torch.unique_consecutive(
            torch.tensor(cu_window_seqlens_list, dtype=torch.int32)
        )

        seq_len = rotary_pos_emb.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        window_index_dev = async_tensor_h2d(window_index, rotary_pos_emb.device)
        rotary_pos_emb = rotary_pos_emb[window_index_dev, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

        seqlens = [h * w for t, h, w in grid_thws for _ in range(t)]
        cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32).cumsum(0)

        reverse_indices = torch.argsort(window_index)

        if max_seqlen_override is None:
            max_seqlen_override = max(seqlens)
        if max_seqlen_window_override is None:
            max_seqlen_window_override = int(
                (cu_window_seqlens[1:] - cu_window_seqlens[:-1]).max()
            )

        if max_num_seqs is not None:
            cu_seqlens = self._pad_cu_seqlens(cu_seqlens, max_num_seqs)
        if max_num_window_seqs is not None:
            cu_window_seqlens = self._pad_cu_seqlens(
                cu_window_seqlens, max_num_window_seqs
            )

        return {
            "rotary_pos_emb_cos": emb.cos().to(device=device),
            "rotary_pos_emb_sin": emb.sin().to(device=device),
            "window_index": async_tensor_h2d(window_index, device),
            "reverse_indices": async_tensor_h2d(reverse_indices, device),
            "cu_seqlens": async_tensor_h2d(cu_seqlens, device),
            "cu_window_seqlens": async_tensor_h2d(cu_window_seqlens, device),
            # Kept on CPU: attention wrappers call `.item()` on these, and
            # under CUDA graph capture the value is baked into the graph.
            "max_seqlen_full": torch.tensor(max_seqlen_override, dtype=torch.int32),
            "max_seqlen_window": torch.tensor(
                max_seqlen_window_override, dtype=torch.int32
            ),
        }

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]] | None,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        r"""Args:
        inputs_embeds: Input tensor of shape
            (sequence_length, hidden_size).
            Embedded representation of the input tokens.
        grid_thws: Grid tensor of shape (num_items, 3)
            containing grid dimensions.
        encoder_metadata: Precomputed output of `prepare_encoder_metadata`.
            If None, it is computed from `grid_thws`.

        """
        if encoder_metadata is None:
            assert grid_thws is not None
            encoder_metadata = self.prepare_encoder_metadata(grid_thws)

        window_index = encoder_metadata["window_index"]
        reverse_indices = encoder_metadata["reverse_indices"]
        position_embeddings = (
            encoder_metadata["rotary_pos_emb_cos"],
            encoder_metadata["rotary_pos_emb_sin"],
        )

        seq_len, _ = inputs_embeds.size()
        inputs_embeds = inputs_embeds.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        inputs_embeds = inputs_embeds[window_index, :, :]
        inputs_embeds = inputs_embeds.reshape(seq_len, -1)

        hidden_states = inputs_embeds
        for index, block in enumerate(self.layers):
            if not self.fullatt_block_indexes or index in self.fullatt_block_indexes:
                cu_seqlens_tmp = encoder_metadata["cu_seqlens"]
                max_seqlen_tmp = encoder_metadata["max_seqlen_full"]
            else:
                cu_seqlens_tmp = encoder_metadata["cu_window_seqlens"]
                max_seqlen_tmp = encoder_metadata["max_seqlen_window"]
            hidden_states = block(
                hidden_states, cu_seqlens_tmp, position_embeddings, max_seqlen_tmp
            )

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[reverse_indices, :].reshape(seq_len, -1)

        return hidden_states


class Siglip2VisionTransformer(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def prepare_encoder_metadata(
        self,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """See `Siglip2Encoder.prepare_encoder_metadata`. Additionally
        precomputes the interpolated position embeddings, if used."""
        grid_thws = Siglip2Encoder._grid_thws_to_list(grid_thws)
        metadata = self.encoder.prepare_encoder_metadata(grid_thws, **kwargs)
        if self.embeddings.preserve_original_pe:
            metadata["pos_embeds"] = self.embeddings.get_position_embeddings(grid_thws)
        return metadata

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]] | None,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        r"""spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
        Tensor containing the spatial dimensions (height, width)
        of the input images.
        """
        if encoder_metadata is None:
            assert grid_thws is not None
            encoder_metadata = self.prepare_encoder_metadata(grid_thws)

        hidden_states = self.embeddings(
            pixel_values, pos_embeds=encoder_metadata.get("pos_embeds")
        )

        last_hidden_state = self.encoder(
            hidden_states, None, encoder_metadata=encoder_metadata
        )

        return last_hidden_state


class Siglip2NavitModel(torch.nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
        }
    )

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.vision_model = Siglip2VisionTransformer(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

    def prepare_encoder_metadata(
        self,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        return self.vision_model.prepare_encoder_metadata(grid_thws, **kwargs)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]] | None,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.vision_model(
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            encoder_metadata=encoder_metadata,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
