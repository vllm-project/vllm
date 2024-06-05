from typing import Any, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GemmaConfig, PreTrainedModel

from vllm.attention import Attention, AttentionMetadata


class Linear(nn.Module):
    """PyTorch Linear layer without parameter initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    # Reshape the output tensor to the original shape.
    return x_out.reshape(x_out.shape[0], x_out.shape[1], -1)


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = self._norm(x.float())
        output = x * (1 + self.weight.float())
        return output.to(orig_dtype)


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = Linear(self.hidden_size,
                             self.num_heads * self.head_dim,
                             bias=False)
        self.k_proj = Linear(self.hidden_size,
                             self.num_kv_heads * self.head_dim,
                             bias=False)
        self.v_proj = Linear(self.hidden_size,
                             self.num_kv_heads * self.head_dim,
                             bias=False)
        self.o_proj = Linear(self.num_heads * self.head_dim,
                             self.hidden_size,
                             bias=False)
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_emb(q, freqs_cis)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = apply_rotary_emb(k, freqs_cis)

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        o = self.o_proj(attn_output)
        return o


class GemmaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: GemmaConfig,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GemmaModel(nn.Module):

    def __init__(
        self,
        config: GemmaConfig,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(GemmaDecoderLayer(config))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        # FIXME(woosuk): Downcast the normalizer.
        hidden_states = hidden_states * (self.config.hidden_size**0.5)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(PreTrainedModel):

    def __init__(
        self,
        config: GemmaConfig,
    ):
        super().__init__(config)
        self.config = config

        self.model = GemmaModel(config)
        rope_theta = getattr(config, 'rope_theta', 10000)
        # [head_dim * 2, ] -> complex -> two dim (real, imaginary) implicitly
        freqs_cis = precompute_freqs_cis(config.head_dim,
                                         config.max_position_embeddings * 2,
                                         theta=rope_theta)
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        freqs_cis = self.freqs_cis.index_select(0, positions.flatten())
        freqs_cis = freqs_cis.view(batch_size, 1, seq_len, -1)
        hidden_states = self.model(
            input_ids=input_ids,
            freqs_cis=freqs_cis,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        hidden_states = hidden_states[indices]
        logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        return logits
