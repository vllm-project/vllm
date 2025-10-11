# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Implementation of SiglipVisionModel intended to be only used
within a vision language model."""

from typing import Optional, Union

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithNoAttention

from vllm.platforms import _Backend, current_platform

from .vision import get_vit_attn_backend

is_hpu = current_platform.is_hpu()

if is_hpu:
    import habana_frameworks.torch.core as htcore
    from habana_frameworks.torch.hpex.kernels import FusedSDPA


class VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta
                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen,
                           device=self.inv_freq.device,
                           dtype=self.inv_freq.dtype)
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
            self.patch_embedding = nn.Linear(
                in_features=config.num_channels * self.patch_size *
                self.patch_size,
                out_features=self.embed_dim,
            )
            if self.preserve_original_pe:
                self.position_embedding_size = int(self.num_patches**0.5)
                self.position_embedding = nn.Embedding(self.num_patches,
                                                       self.embed_dim)

        else:
            self.patch_embedding = nn.Conv2d(
                in_channels=config.num_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                padding="valid",
            )
            if self.preserve_original_pe:
                self.num_patches = (self.image_size // self.patch_size)**2
                self.position_embedding_size = (self.image_size //
                                                self.patch_size)
                self.position_embedding = nn.Embedding(self.num_patches,
                                                       self.embed_dim)

    def forward(self,
                pixel_values: torch.FloatTensor,
                grid_thws: Optional[torch.LongTensor] = None) -> torch.Tensor:
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
        if isinstance(self.patch_embedding, nn.Linear):
            patch_embeds = self.patch_embedding(
                pixel_values.to(dtype=target_dtype))
        elif isinstance(self.patch_embedding, nn.Conv2d):
            pixel_values = pixel_values.view(
                -1, self.config.num_channels * self.config.temporal_patch_size,
                self.patch_size, self.patch_size)
            patch_embeds = self.patch_embedding(
                pixel_values.to(dtype=target_dtype))
            patch_embeds = patch_embeds.reshape(-1, self.embed_dim)

        if self.preserve_original_pe:
            assert grid_thws is not None
            pos_embed_new = torch.zeros_like(patch_embeds)
            positional_embeddings = self.position_embedding.weight.reshape(
                self.position_embedding_size, self.position_embedding_size,
                -1).unsqueeze(0).permute(0, 3, 1, 2)
            cnt = 0
            for t, h, w in grid_thws:
                volume = t * h * w
                pe = F.interpolate(positional_embeddings,
                                   size=(h, w),
                                   mode='bicubic',
                                   align_corners=False)
                pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
                pe = pe[0].repeat(t, 1)
                pe = pe.reshape(t, h // self.hidden_stride, self.hidden_stride,
                                w // self.hidden_stride, self.hidden_stride,
                                -1)
                pe = pe.permute(0, 1, 3, 2, 4, 5).reshape(volume, -1)
                pos_embed_new[cnt:cnt + volume] = pe
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
        return rearrange(torch.stack((-x2, x1), dim=-1),
                         "... d two -> ... (d two)",
                         two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(
        sin,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos +
            rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]
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
    if is_flash_attn_backend:
        from flash_attn.layers.rotary import apply_rotary_emb
        apply_rotary_emb_func = apply_rotary_emb
    else:
        apply_rotary_emb_func = apply_rotary_emb_torch
    q_embed = apply_rotary_emb_func(q.float(), cos.float(),
                                    sin.float()).type_as(q)
    k_embed = apply_rotary_emb_func(k.float(), cos.float(),
                                    sin.float()).type_as(k)
    return q_embed, k_embed


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.use_rope = config.use_rope

        # Detect attention implementation.
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
        if self.attn_backend not in {
                _Backend.FLASH_ATTN,
                _Backend.TORCH_SDPA,
        }:
            self.attn_backend = _Backend.TORCH_SDPA
        self.is_flash_attn_backend = self.attn_backend in {
            _Backend.FLASH_ATTN,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor,
                                            torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(seq_length, self.num_heads, self.head_dim)
        keys = keys.view(seq_length, self.num_heads, self.head_dim)
        values = values.view(seq_length, self.num_heads, self.head_dim)

        if self.use_rope:
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb(queries.unsqueeze(0),
                                                 keys.unsqueeze(0), cos, sin,
                                                 self.is_flash_attn_backend)
            queries = queries.squeeze(0)
            keys = keys.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        if self.is_flash_attn_backend:
            from flash_attn import flash_attn_varlen_func
            attn_output = flash_attn_varlen_func(
                queries, keys, values, cu_seqlens, cu_seqlens, max_seqlen,
                max_seqlen).reshape(seq_length, -1)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            # Execute attention entry by entry for speed & less VRAM.
            batch_size = cu_seqlens.shape[0] - 1
            outputs = []
            cu = cu_seqlens.tolist()
            for i in range(batch_size):
                start_idx = cu[i]
                end_idx = cu[i + 1]

                # Each sequence is processed independently.
                q_i = queries[start_idx:end_idx].unsqueeze(0)
                k_i = keys[start_idx:end_idx].unsqueeze(0)
                v_i = values[start_idx:end_idx].unsqueeze(0)

                # (1, seq_len, num_heads, head_dim) ->
                # (1, num_heads, seq_len, head_dim)
                q_i, k_i, v_i = [x.transpose(1, 2) for x in (q_i, k_i, v_i)]
                if is_hpu:
                    output_i = FusedSDPA.apply(q_i, k_i, v_i, None, 0.0, False,
                                               None)
                else:
                    output_i = F.scaled_dot_product_attention(q_i,
                                                              k_i,
                                                              v_i,
                                                              dropout_p=0.0)
                # (1, num_heads, seq_len, head_dim) -> (seq_len, embed_dim)
                output_i = output_i.transpose(1, 2).reshape(-1, self.embed_dim)
                outputs.append(output_i)

            attn_output = torch.cat(outputs, dim=0)
        attn_output = self.out_proj(attn_output)
        return attn_output


class Siglip2MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                position_embeddings: torch.Tensor) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all
                attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       cu_seqlens=cu_seqlens,
                                       position_embeddings=position_embeddings)
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

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Siglip2EncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

        self.rotary_pos_emb = VisionRotaryEmbedding(
            config.hidden_size // config.num_attention_heads // 2)
        self.patch_size = config.patch_size
        self.hidden_stride = config.hidden_stride
        self.window_size = config.window_size
        self.spatial_merge_unit = config.hidden_stride * config.hidden_stride
        if config.fullatt_block_indexes is None:
            self.fullatt_block_indexes = None
        else:
            self.fullatt_block_indexes = [
                int(i) for i in config.fullatt_block_indexes.split('|')
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
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
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
        vit_merger_window_size = (self.window_size // self.hidden_stride //
                                  self.patch_size)

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.hidden_stride,  # number of patch after merge
                grid_w // self.hidden_stride,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
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
            if self.fullatt_block_indexes:
                seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            if self.fullatt_block_indexes:
                cu_seqlens_tmp = seqlens.cumsum(
                    0) * self.spatial_merge_unit + cu_window_seqlens[-1]
                cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        grid_thws: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, ...]]]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape
                `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to
                directly pass an embedded representation. This is useful if
                you want more control over how to convert `input_ids` indices
                into associated vectors than the model's internal embedding
                lookup matrix.
            grid_thws (`torch.LongTensor`):
                grid shape (num_patches, 3)
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See
                `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of
                a plain tuple.
        """
        rotary_pos_emb = self.rot_pos_emb(grid_thws)
        window_index, cu_window_seqlens = self.get_window_index(grid_thws)

        # NOTE: unique_consecutive is a dynamic operation
        # we are using `remove_duplicates_cpu` instead
        def remove_duplicates_cpu(a):
            return [a[i] for i in range(len(a)) if i == 0 or a[i - 1] != a[i]]

        if self.fullatt_block_indexes:
            cu_window_seqlens = torch.tensor(
                cu_window_seqlens,
                device=inputs_embeds.device,
                dtype=grid_thws.dtype
                if torch.jit.is_tracing() else torch.int32,
            )

        seq_len, _ = inputs_embeds.size()
        inputs_embeds = inputs_embeds.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        inputs_embeds = inputs_embeds[window_index, :, :]
        inputs_embeds = inputs_embeds.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
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
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        reverse_indices = torch.argsort(window_index)
        encoder_states = () if output_hidden_states else None

        hidden_states = inputs_embeds

        htcore.mark_step()
        for index, block in enumerate(self.layers):
            if (not self.fullatt_block_indexes
                    or index in self.fullatt_block_indexes):
                cu_seqlens_tmp = cu_seqlens
            else:
                cu_seqlens_tmp = cu_window_seqlens
            hidden_states = block(hidden_states, cu_seqlens_tmp,
                                  position_embeddings)
            if output_hidden_states:
                hidden_states_ = hidden_states.reshape(
                    seq_len // self.spatial_merge_unit,
                    self.spatial_merge_unit, -1)
                encoder_states += (hidden_states_[reverse_indices, :].reshape(
                    seq_len, -1), )
        # tokens = self.post_trunk_norm(tokens)
        htcore.mark_step()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :].reshape(seq_len, -1)

        return hidden_states, encoder_states


class Siglip2VisionTransformer(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim,
                                           eps=config.layer_norm_eps)
        self._use_flash_attention_2 = \
            (config._attn_implementation == "flash_attention_2")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.LongTensor,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
    ) -> Union[
            tuple[torch.Tensor],
            tuple[torch.Tensor, tuple[torch.Tensor, ...]],
            BaseModelOutputWithNoAttention,
    ]:
        r"""
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width)
            of the input images.
        """
        hidden_states = self.embeddings(pixel_values, grid_thws)

        last_hidden_state, hidden_states = self.encoder(
            hidden_states, grid_thws, output_hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if not return_dict:
            output = (last_hidden_state, )
            output += (hidden_states, ) if output_hidden_states else ()
            return output

        return last_hidden_state


class Siglip2NavitModel(torch.nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.vision_model = Siglip2VisionTransformer(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.LongTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
            tuple[torch.Tensor],
            tuple[torch.Tensor, tuple[torch.Tensor, ...]],
            BaseModelOutputWithNoAttention,
    ]:

        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
