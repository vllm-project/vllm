# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Dots-path speech encoder for inference only (single GPU).

Ported from cybertron_alm ``dots_audio_encoder/modeling_whisper.py``.
Upstream ``WhisperEncoder`` is exposed as :class:`DotsSpeechEncoder`.
"""

import math
from typing import Any, ClassVar

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.utils import logging

from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported

_FLASH_ATTN_VERSION = 3 if is_fa_version_supported(3) else 2

logger = logging.get_logger(__name__)

__all__ = ["DotsSpeechEncoder"]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def swiglu(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(x1) * x2


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rope_parameters: dict,
        base_seq_len: int = 0,
    ):
        super().__init__()
        self.partial_rotary_factor = float(
            rope_parameters.get("partial_rotary_factor", 1.0)
        )
        self.rope_theta = float(rope_parameters.get("rope_theta", 10000.0))
        self.rope_type = rope_parameters.get("rope_type", "default")
        rotary_dim = int(head_dim * self.partial_rotary_factor)
        # keep even rotary dimension to align cos/sin pairs
        self.rotary_dim = (rotary_dim // 2) * 2
        self.attention_scaling = 1.0
        if self.rope_type != "default":
            logger.warning(
                "RoPE type %s is not implemented in WhisperLv3; using default",
                self.rope_type,
            )
        if self.rotary_dim > 0:
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                    / max(self.rotary_dim, 1)
                )
            )
        else:
            inv_freq = torch.tensor([])
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.base_seq_len = max(0, int(base_seq_len))
        self._cache: (
            tuple[int, torch.dtype, torch.device, torch.Tensor, torch.Tensor] | None
        ) = None

    @torch.no_grad()
    def get_cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.rotary_dim == 0:
            return None, None
        seq_len = position_ids.shape[-1]
        if position_ids.shape[0] == 1 and self._cache is not None:
            cached_seq_len, cached_dtype, cached_device, cached_cos, cached_sin = (
                self._cache
            )
            if (
                cached_seq_len >= seq_len
                and cached_dtype == dtype
                and cached_device == device
            ):
                return cached_cos[:, :seq_len, :], cached_sin[:, :seq_len, :]

        if position_ids.shape[0] == 1:
            cache_seq_len = max(seq_len, self.base_seq_len)
            position_ids = torch.arange(cache_seq_len, device=device)[None, :]
        # [1, D/2, 1]
        if self.inv_freq.dtype != torch.float32:
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(
                        0, self.rotary_dim, 2, dtype=torch.float, device=device
                    )
                    / max(self.rotary_dim, 1)
                )
            )
        else:
            inv_freq = self.inv_freq.to(device=device)
        inv_freq_expanded = inv_freq[None, :, None]
        # [B, 1, T]
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = device.type if isinstance(device, torch.device) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        cos = cos.to(dtype)
        sin = sin.to(dtype)
        if position_ids.shape[0] == 1:
            self._cache = (position_ids.shape[-1], dtype, device, cos, sin)
            return cos[:, :seq_len, :], sin[:, :seq_len, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if q.dim() == 4:
        # [B, T, H, D]
        rotary_dim = cos.shape[-1]
        cos = cos.unsqueeze(2)  # [B, T, 1, rotary_dim]
        sin = sin.unsqueeze(2)
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat(
            [k_embed, k_pass], dim=-1
        )

    if q.dim() == 3:
        # [tokens, H, D] (varlen flattened)
        rotary_dim = cos.shape[-1]
        cos = cos.unsqueeze(1)  # [tokens, 1, rotary_dim]
        sin = sin.unsqueeze(1)
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat(
            [k_embed, k_pass], dim=-1
        )

    return q, k


class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = None
    ):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0):
        return self.weight[
            past_key_values_length : past_key_values_length + input_ids.shape[1]
        ]


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward_flash_attn(
        self,
        hidden_states,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        output_attentions=False,
        rotary_cos=None,
        rotary_sin=None,
    ):
        """Dense eager attention with SGLang FA3 for packed variable-length input."""
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if cu_seqlens_q is None:
            bsz, tgt_len, _ = hidden_states.size()
            query_states = query_states.view(
                bsz, tgt_len, self.num_heads, self.head_dim
            )
            key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
            value_states = value_states.view(
                bsz, tgt_len, self.num_heads, self.head_dim
            )
            if rotary_cos is not None and rotary_sin is not None:
                cos = rotary_cos[:, :tgt_len, :]
                sin = rotary_sin[:, :tgt_len, :]
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
            attn_output, attn_probs = self._eager_attention(
                query_states, key_states, value_states, output_attentions
            )
            attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        else:
            query_states = query_states.view(-1, self.num_heads, self.head_dim)
            key_states = key_states.view(-1, self.num_heads, self.head_dim)
            value_states = value_states.view(-1, self.num_heads, self.head_dim)
            if rotary_cos is not None and rotary_sin is not None:
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, rotary_cos, rotary_sin
                )
            result = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                softmax_scale=self.scaling,
                causal=self.is_decoder,
                return_softmax_lse=output_attentions,
                fa_version=_FLASH_ATTN_VERSION,
            )
            if isinstance(result, tuple):
                attn_output = result[0]
                attn_probs = result[1] if output_attentions else None
            else:
                attn_output = result
                attn_probs = None
            attn_output = attn_output.view(-1, self.embed_dim)

        return self.out_proj(attn_output), attn_probs

    def _eager_attention(self, query, key, value, output_attentions):
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        if self.is_decoder:
            seq_len = scores.shape[-1]
            causal_mask = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=scores.device
            ).triu(1)
            scores.masked_fill_(causal_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(probabilities, value).transpose(1, 2)
        return output, probabilities if output_attentions else None


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        use_causal = getattr(config, "use_causal", False)
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=use_causal,  # enable causal attention when use_causal=True
        )
        self.use_causal = use_causal
        norm_cls = RMSNorm if getattr(config, "use_rms_norm", False) else nn.LayerNorm
        self.self_attn_layer_norm = norm_cls(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = (
            ACT2FN[config.activation_function]
            if config.activation_function != "swiglu"
            else swiglu
        )
        self.use_swiglu = config.activation_function == "swiglu"
        self.activation_dropout = config.activation_dropout
        ffn_dim = config.encoder_ffn_dim
        fc1_out = ffn_dim * 2 if self.use_swiglu else ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, fc1_out)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.final_layer_norm = norm_cls(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_kv: torch.Tensor = None,
        max_seqlen_q: int | None = None,
        max_seqlen_kv: int | None = None,
        output_attentions: bool = False,
        rotary_cos: torch.Tensor | None = None,
        rotary_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights = self.self_attn.forward_flash_attn(
            hidden_states=hidden_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            output_attentions=output_attentions,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.use_swiglu:
            hidden_states = swiglu(self.fc1(hidden_states))
        else:
            hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs: tuple[Any, ...] = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DotsSpeechPreTrainedModel(PreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = False
    _no_split_modules: ClassVar[list[str]] = ["WhisperEncoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class DotsSpeechEncoder(DotsSpeechPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.use_latent_input = getattr(config, "use_latent_input", False)
        self.use_causal = getattr(config, "use_causal", False)
        self.use_conv2d_stem = getattr(config, "use_conv2d_stem", False)
        latent_dim = getattr(config, "latent_dim", None)
        if self.use_latent_input and latent_dim is None:
            raise ValueError(
                "DotsSpeechEncoder: use_latent_input=True requires config.latent_dim"
            )
        if self.use_conv2d_stem and self.use_latent_input:
            raise ValueError(
                "DotsSpeechEncoder: use_conv2d_stem and use_latent_input are mutually exclusive"
            )
        self.num_mel_bins = (
            latent_dim if self.use_latent_input and latent_dim is not None else 128
        )
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if self.use_conv2d_stem:
            # Conv2D stem: 3 layers of stride-2 for 8x downsampling
            dhs = getattr(config, "downsample_hidden_size", 480)
            # Causal: keep freq padding, remove time padding (handled by pad(14,0) in forward)
            conv_padding = (1, 0) if self.use_causal else 1
            self.conv2d1 = nn.Conv2d(
                1, dhs, kernel_size=3, stride=2, padding=conv_padding
            )
            self.conv2d2 = nn.Conv2d(
                dhs, dhs, kernel_size=3, stride=2, padding=conv_padding
            )
            self.conv2d3 = nn.Conv2d(
                dhs, dhs, kernel_size=3, stride=2, padding=conv_padding
            )
            # After 3x stride-2 on freq=128: 128→64→32→16; linear projects dhs*16 → embed_dim
            freq_after = self.num_mel_bins
            for _ in range(3):
                freq_after = (freq_after + 1) // 2
            self.conv_out = nn.Linear(dhs * freq_after, embed_dim, bias=False)
            self.conv1 = None
            self.conv2 = None
        elif self.use_latent_input:
            # Latent path keeps length (stride=1) and applies GLU after conv1
            self.conv1 = nn.Conv1d(
                self.num_mel_bins, embed_dim * 2, kernel_size=3, stride=1, padding=1
            )
            self.conv2 = nn.Conv1d(
                embed_dim, embed_dim, kernel_size=3, stride=1, padding=1
            )
        else:
            self.conv1 = nn.Conv1d(
                self.num_mel_bins, embed_dim, kernel_size=3, padding=1
            )
            self.conv2 = nn.Conv1d(
                embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
            )

        self.use_rope = getattr(config, "use_rope", False)
        rope_parameters = getattr(config, "rope_parameters", {}) or {}
        self.rope_parameters = rope_parameters
        if self.use_rope:
            head_dim = embed_dim // config.encoder_attention_heads
            self.rotary_embedding = RotaryEmbedding(
                head_dim,
                rope_parameters,
                base_seq_len=self.max_source_positions,
            )
            self.embed_positions = None
        else:
            self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        norm_cls = RMSNorm if getattr(config, "use_rms_norm", False) else nn.LayerNorm
        self.layer_norm = norm_cls(config.d_model)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if self.use_conv2d_stem:
            return self.conv2d1
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    @staticmethod
    def _temporal_mask(feat, valid_lens):
        """Zero out temporal positions >= valid_lens. feat: [B,C,F,T], valid_lens: [B] tensor."""
        T = feat.shape[-1]
        mask = torch.arange(T, device=feat.device)[None, :] < valid_lens[:, None]
        return mask[:, None, None, :]

    def _conv2d_stem_one_chunk(self, chunk, chunk_valid_mel_lens=None):
        """Run 3x Conv2d(stride=2) + GELU with per-layer masking."""
        # Causal: left-pad time by 14 = 2 + 4 + 8 (receptive field of the 3-layer
        # stride-2 stack mapped back to input time). Combined with conv padding=(1,0),
        # this makes the whole conv2d stem strictly causal.
        if self.use_causal:
            chunk = nn.functional.pad(chunk, (14, 0))
            if chunk_valid_mel_lens is not None:
                chunk_valid_mel_lens = chunk_valid_mel_lens + 14
        # Step 0: mask mel — silence_mel(-1.5) → 0
        if chunk_valid_mel_lens is not None:
            chunk = chunk * self._temporal_mask(chunk, chunk_valid_mel_lens)
        chunk = nn.functional.gelu(self.conv2d1(chunk))
        # Step 1: mask conv1 output — gelu(conv(0)+bias) → 0
        if chunk_valid_mel_lens is not None:
            chunk_valid_mel_lens = (chunk_valid_mel_lens + 1) // 2
            chunk = chunk * self._temporal_mask(chunk, chunk_valid_mel_lens)
        chunk = nn.functional.gelu(self.conv2d2(chunk))
        # Step 2: mask conv2 output — gelu(bias) → 0
        if chunk_valid_mel_lens is not None:
            chunk_valid_mel_lens = (chunk_valid_mel_lens + 1) // 2
            chunk = chunk * self._temporal_mask(chunk, chunk_valid_mel_lens)
        chunk = nn.functional.gelu(self.conv2d3(chunk))
        # Step 3: mask conv3 output — gelu(bias) → 0
        if chunk_valid_mel_lens is not None:
            chunk_valid_mel_lens = (chunk_valid_mel_lens + 1) // 2
            chunk = chunk * self._temporal_mask(chunk, chunk_valid_mel_lens)
        return chunk

    def _forward_conv2d_stem(
        self, input_features, input_seq_lens=None, audio_sample_lens=None
    ):
        """Conv2D stem: 3x Conv2d(stride=2) → 8x downsample. Returns [B, T/8, embed_dim]."""
        x = input_features.unsqueeze(1)  # [B, 1, 128, T]

        if audio_sample_lens is not None:
            hop_length = 160
            if not isinstance(audio_sample_lens, torch.Tensor):
                audio_sample_lens = torch.tensor(audio_sample_lens, device=x.device)
            valid_mel_lens = audio_sample_lens.to(x.device) // hop_length
        else:
            valid_mel_lens = None
        x = self._conv2d_stem_one_chunk(x, valid_mel_lens)
        # [B, dhs, F_out, T/8] → [B, T/8, dhs*freq_after] → [B, T/8, embed_dim]
        batch, channels, frequency, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * frequency)
        inputs_embeds = self.conv_out(x)

        return inputs_embeds

    def _forward_conv1d_stem(self, input_features):
        """Standard Conv1d stem: conv1 + conv2(stride=2), 2x downsample. Returns [B, T/2, embed_dim]."""
        assert self.conv1 is not None and self.conv2 is not None
        if self.use_causal:
            x = nn.functional.pad(input_features, (2, 0))
            x = nn.functional.gelu(self.conv1(x))
            x = nn.functional.pad(x, (2, 0))
            x = nn.functional.gelu(self.conv2(x))
        else:
            x = nn.functional.gelu(self.conv1(input_features))
            x = nn.functional.gelu(self.conv2(x))
        return x.permute(0, 2, 1)

    def _forward_latent_stem(self, input_features):
        """Latent input stem: conv1+GLU + conv2, stride=1, no downsample. Returns [B, T, embed_dim]."""
        assert self.conv1 is not None and self.conv2 is not None
        if self.use_causal:
            x = nn.functional.pad(input_features, (2, 0))
            x = nn.functional.glu(self.conv1(x), dim=1)
            x = nn.functional.pad(x, (2, 0))
            x = nn.functional.gelu(self.conv2(x))
        else:
            x = nn.functional.glu(self.conv1(input_features), dim=1)
            x = nn.functional.gelu(self.conv2(x))
        return x.permute(0, 2, 1)

    def forward(
        self,
        input_features,
        input_seq_lens=None,
        audio_sample_lens=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Mel `[B, n_mels, T]` → encoder hidden states. Inference-only."""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if self.use_conv2d_stem:
            inputs_embeds = self._forward_conv2d_stem(
                input_features, input_seq_lens, audio_sample_lens
            )
        elif self.use_latent_input:
            inputs_embeds = self._forward_latent_stem(input_features)
        else:
            inputs_embeds = self._forward_conv1d_stem(input_features)
        rotary_cos = None
        rotary_sin = None
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )[None, :]
        if self.use_rope:
            rotary_cos, rotary_sin = self.rotary_embedding.get_cos_sin(
                position_ids, inputs_embeds.dtype, inputs_embeds.device
            )
            hidden_states = inputs_embeds
        else:
            assert self.embed_positions is not None
            embed_pos = self.embed_positions.weight[: inputs_embeds.shape[1]]
            hidden_states = inputs_embeds + embed_pos

        encoder_states: tuple[torch.Tensor, ...] | None = (
            () if output_hidden_states else None
        )
        all_attentions: tuple[torch.Tensor, ...] | None = (
            () if output_attentions else None
        )

        if input_seq_lens is not None:
            # Build varlen metadata and pack valid tokens without per-sample cat loops.
            # Read max on whatever device the caller provided: when it is a CPU
            # tensor this avoids a device->host sync (one per forward).
            max_seqlen_q = int(input_seq_lens.max().item())
            input_seq_lens = input_seq_lens.to(
                device=hidden_states.device, dtype=torch.long
            )
            B, S, D = hidden_states.shape
            max_seqlen_kv = max_seqlen_q
            cu_seqlens_q = torch.nn.functional.pad(
                input_seq_lens.cumsum(0, dtype=torch.int32), (1, 0)
            )
            cu_seqlens_kv = cu_seqlens_q

            token_positions = torch.arange(S, device=hidden_states.device)[None, :]
            valid_token_mask = token_positions < input_seq_lens[:, None]
            hidden_states = hidden_states[valid_token_mask]
            if rotary_cos is not None and rotary_sin is not None:
                packed_positions = token_positions.expand(B, S)[valid_token_mask]
                rotary_cos = rotary_cos.squeeze(0).index_select(0, packed_positions)
                rotary_sin = rotary_sin.squeeze(0).index_select(0, packed_positions)
        else:
            cu_seqlens_q = None
            cu_seqlens_kv = None
            max_seqlen_q = None
            max_seqlen_kv = None

        for encoder_layer in self.layers:
            if output_hidden_states:
                assert encoder_states is not None
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                output_attentions=output_attentions,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                assert all_attentions is not None
                all_attentions = all_attentions + (layer_outputs[1],)

        if input_seq_lens is not None:
            # Recover packed varlen output directly on device.
            recover_positions = torch.arange(max_seqlen_q, device=hidden_states.device)[
                None, :
            ]
            recover_mask = recover_positions < input_seq_lens[:, None]
            recovered = hidden_states.new_zeros(B, max_seqlen_q, D)
            recovered[recover_mask] = hidden_states
            hidden_states = recovered

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            assert encoder_states is not None
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )
