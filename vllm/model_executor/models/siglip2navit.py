# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Implementation of SiglipVisionModel intended to be only used
within a vision language model."""

from collections.abc import Iterable

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from transformers import Siglip2VisionConfig
from transformers.configuration_utils import PretrainedConfig

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
from vllm.config import MultiModalConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform


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
    def __init__(self, config: PretrainedConfig):
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

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (
                    num_patches,
                    num_channels * temporal_patch_size * patch_size * patch_size
                )
            grid_thws: (`torch.LongTensor`):
                grid shape (num_patches, 3)
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
            assert grid_thws is not None
            pos_embed_new = torch.zeros_like(patch_embeds)
            positional_embeddings = (
                self.position_embedding.weight.reshape(
                    self.position_embedding_size, self.position_embedding_size, -1
                )
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )
            cnt = 0
            for t, h, w in grid_thws:
                volume = t * h * w
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
                pe = pe.permute(0, 1, 3, 2, 4, 5).reshape(volume, -1)
                pos_embed_new[cnt : cnt + volume] = pe
                cnt += volume
            patch_embeds = patch_embeds + pos_embed_new

        return patch_embeds


# copy from flash_attn/layers/rotary.py
def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_flash_attn_backend: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    if is_flash_attn_backend and current_platform.is_cuda():
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

        apply_rotary_emb_func = apply_rotary_emb
    else:
        apply_rotary_emb_func = apply_rotary_emb_torch
    q_embed = apply_rotary_emb_func(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb_func(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
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
            prefix=f"{prefix}.attn",
            multimodal_config=multimodal_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        seq_length, embed_dim = hidden_states.shape

        qkv_states, _ = self.qkv_proj(hidden_states)
        queries, keys, values = qkv_states.chunk(3, dim=-1)

        queries = queries.view(seq_length, self.num_heads_per_partition, self.head_dim)
        keys = keys.view(seq_length, self.num_heads_per_partition, self.head_dim)
        values = values.view(seq_length, self.num_heads_per_partition, self.head_dim)

        if self.use_rope:
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb(
                queries.unsqueeze(0),
                keys.unsqueeze(0),
                cos,
                sin,
                self.attn.is_flash_attn_backend,
            )
            queries = queries.squeeze(0)
            keys = keys.squeeze(0)

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
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
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
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, embed_dim).
            cu_seqlens: Cumulative sequence lengths tensor.
            position_embeddings: Position embeddings tensor.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers`
    self attention layers. Each layer is a [`Siglip2EncoderLayer`].

    Args:
        config: PretrainedConfig
    """

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
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

    # copied from qwen2.5_vl
    def rot_pos_emb(self, grid_thw):
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
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
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
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            inputs_embeds: Input tensor of shape
                (batch_size, sequence_length, hidden_size).
                Embedded representation of the input tokens.
            grid_thws: Grid tensor of shape (num_patches, 3)
                containing grid dimensions.
                Whether or not to return a [`~utils.ModelOutput`] instead of
                a plain tuple.
        """
        rotary_pos_emb = self.rot_pos_emb(grid_thws)
        window_index, cu_window_seqlens = self.get_window_index(grid_thws)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=inputs_embeds.device,
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = inputs_embeds.size()
        inputs_embeds = inputs_embeds.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        inputs_embeds = inputs_embeds[window_index, :, :]
        inputs_embeds = inputs_embeds.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thws[:, 1] * grid_thws[:, 2], grid_thws[:, 0]
        ).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have
            #    same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852
            # for more information
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        reverse_indices = torch.argsort(window_index)

        hidden_states = inputs_embeds
        for index, block in enumerate(self.layers):
            if not self.fullatt_block_indexes or index in self.fullatt_block_indexes:
                cu_seqlens_tmp = cu_seqlens
            else:
                cu_seqlens_tmp = cu_window_seqlens
            hidden_states = block(hidden_states, cu_seqlens_tmp, position_embeddings)

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
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.encoder",
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.LongTensor,
    ) -> torch.Tensor:
        r"""
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width)
            of the input images.
        """
        hidden_states = self.embeddings(pixel_values, grid_thws)

        last_hidden_state = self.encoder(hidden_states, grid_thws)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class Siglip2NavitModel(torch.nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.vision_model = Siglip2VisionTransformer(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.vision_model",
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.LongTensor,
    ) -> torch.Tensor:
        return self.vision_model(
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
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
