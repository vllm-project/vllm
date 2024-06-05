from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GemmaConfig, PreTrainedModel

from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up, _ = self.up_proj(x)
        fuse = gate * up
        outputs, _ = self.down_proj(fuse)
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

        self.q_proj = ColumnParallelLinear(self.hidden_size,
                                           self.num_heads * self.head_dim,
                                           bias=False)
        self.k_proj = ColumnParallelLinear(self.hidden_size,
                                           self.num_kv_heads * self.head_dim,
                                           bias=False)
        self.v_proj = ColumnParallelLinear(self.hidden_size,
                                           self.num_kv_heads * self.head_dim,
                                           bias=False)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim,
                                        self.hidden_size,
                                        bias=False)

        self.rope = get_rope(self.head_dim, self.head_dim, 8192, 10000)
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
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        q, k = self.rope(freqs_cis, q, k)
        q = q.reshape(batch_size, seq_len, -1)
        k = k.reshape(batch_size, seq_len, -1)

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        o, _ = self.o_proj(attn_output)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            freqs_cis=positions,
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
