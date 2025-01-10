from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.sequence import IntermediateTensors

from .utils import make_empty_intermediate_tensors_factory, make_layers


@dataclass(frozen=True)
class MoondreamTextConfig:
    dim: int = 2048
    n_layers: int = 24
    vocab_size: int = 51200
    max_context: int = 2048
    n_heads: int = 32
    prefix_attn: int = 730


@dataclass(frozen=True)
class MoondreamVisionConfig:
    enc_dim: int = 1152
    enc_patch_size: int = 14
    enc_n_layers: int = 27
    enc_ff_dim: int = 4304
    enc_n_heads: int = 16
    proj_out_dim: int = 2048
    crop_size: int = 378
    in_channels: int = 3
    max_crops: int = 12
    overlap_margin: int = 4
    proj_inner_dim: int = 8192


class MoondreamAttention(nn.Module):

    def __init__(self,
                 config: MoondreamTextConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.total_num_heads = config.n_heads
        self.hidden_size = config.dim
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)

        self.qkv = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_size,
            total_num_heads=self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

        scaling = self.head_size**-0.5
        rotary_dim = int(config.partial_rotary_factor *
                         (config.hidden_size // config.num_attention_heads))
        assert rotary_dim % 2 == 0

        rope_theta = getattr(config, "rope_theta", 10000.0)
        max_position_embeddings = getattr(config, "max_context", 2048)
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
        )
        self.attn = Attention(self.num_heads,
                              self.head_size,
                              scaling,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.proj(attn_output)
        return output


class MoondreamDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MoondreamTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = config.text
        self.ln = nn.LayerNorm(config.dim)
        self.attn = MoondreamAttention(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = MoondreamMLP(config, quant_config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        residual = hidden_states
        hidden_states = self.ln(hidden_states)
        attn_outputs = self.attn(
            position_ids=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states


class MoondreamMLP(nn.Module):

    def __init__(
        self,
        config: MoondreamTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=config.dim,
            output_size=config.dim * 4,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size=config.dim * 4,
            output_size=config.dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

        self.act_fn = get_act_fn("gelu_pytorch_tanh")

    def forward(self, x):
        x, _ = self.fc1(x)
        x = self.act_fn(x)
        x, _ = self.fc2(x)
        return x


class MoondreamTextModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: MoondreamTextConfig = vllm_config.model_config.hf_config.text
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.wte = VocabParallelEmbedding(config.vocab_size, config.dim)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.n_layers,
            lambda prefix: MoondreamDecoderLayer(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.post_ln = nn.LayerNorm(config.dim)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.dim))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.post_ln(hidden_states)
        return hidden_states
