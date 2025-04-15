# SPDX-License-Identifier: Apache-2.0
import math
import sys
from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import ModernBertConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import CrossEncodingPooler
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from .interfaces import SupportsCrossEncoding
from .utils import WeightsMapper, maybe_prefix


class ModernBertEmbeddings(nn.Module):

    def __init__(self, config: ModernBertConfig):

        super().__init__()
        self.config = config
        self.tok_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                     config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size,
                                 eps=config.layer_norm_eps,
                                 bias=config.norm_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds:
            return self.norm(inputs_embeds)
        else:
            inputs_embeds = self.tok_embeddings(input_ids)
            embeddings = self.norm(inputs_embeds)
            return embeddings


class ModernBertRotaryEmbedding(RotaryEmbedding):

    def __init__(self,
                 config: ModernBertConfig,
                 head_size: int,
                 dim: int,
                 base: float,
                 device: Optional[torch.device] = None):
        super().__init__(
            head_size=head_size,
            rotary_dim=dim,
            max_position_embeddings=config.max_position_embeddings,
            base=base,
            is_neox_style=True,
            dtype=torch.float16)
        self.config = config


def sdpa_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    bs: int,
    dim: int,
    num_heads: int,
    head_dim: int,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    query, key, value = qkv.split([dim, dim, dim], dim=-1)

    pos_offsets = (position_ids == 0).nonzero(as_tuple=True)[0].int()
    end_offset = torch.Tensor([sys.maxsize]).int()
    pos_offsets = torch.cat((pos_offsets, end_offset))
    start = 0
    attn_outs = []
    for offset in pos_offsets:
        if not offset:
            continue
        end = offset.item()
        pos_ids = position_ids[start:end]
        q = query[start:end, :]
        k = key[start:end, :]
        v = value[start:end, :]
        q, k = module.rotary_emb(positions=pos_ids, query=q, key=k)
        q = q.view(1, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(1, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(1, -1, num_heads, head_dim).transpose(1, 2)
        attn_output = (F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
        ).transpose(1, 2).contiguous())
        attn_output = attn_output.view(-1, dim)
        attn_outs.append(attn_output)
        start = end

    attn_output = torch.cat(attn_outs, dim=0)
    return attn_output


class ModernBertAttention(nn.Module):

    def __init__(self,
                 config: ModernBertConfig,
                 layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size ({config.hidden_size}) "
                             f"is not a multiple of the number of "
                             f"attention heads ({config.num_attention_heads})")

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size,
                              3 * self.all_head_size,
                              bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2,
                                    config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        if self.local_attention != (
                -1, -1) and config.local_rope_theta is not None:
            rope_theta = config.local_rope_theta
        self.rotary_emb = ModernBertRotaryEmbedding(config=config,
                                                    head_size=self.head_dim,
                                                    dim=self.head_dim,
                                                    base=rope_theta)
        self.Wo = nn.Linear(config.hidden_size,
                            config.hidden_size,
                            bias=config.attention_bias)
        self.out_drop = nn.Dropout(
            config.attention_dropout
        ) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)
        attn_outputs = sdpa_attention_forward(
            self,
            qkv=qkv,
            position_ids=position_ids,
            rotary_emb=self.rotary_emb,
            local_attention=self.local_attention,
            bs=1,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            **kwargs,
        )
        hidden_states = attn_outputs
        hidden_states = self.Wo(hidden_states)

        return hidden_states


class GELUActivation(nn.Module):

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(input)


class ModernBertMLP(nn.Module):

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size,
                            int(config.intermediate_size) * 2,
                            bias=config.mlp_bias)
        self.act = GELUActivation()
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size,
                            config.hidden_size,
                            bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class ModernBertLayer(nn.Module):

    def __init__(self,
                 config: ModernBertConfig,
                 prefix: str = "",
                 layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size,
                                          eps=config.norm_eps,
                                          bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(config.hidden_size,
                                     eps=config.norm_eps,
                                     bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        attn_outputs = self.attn(self.attn_norm(hidden_states),
                                 position_ids=position_ids)
        hidden_states = hidden_states + attn_outputs
        mlp_output = (self.compiled_mlp(hidden_states)
                      if self.config.reference_compile else self.mlp(
                          self.mlp_norm(hidden_states)))
        hidden_states = hidden_states + mlp_output
        return hidden_states


class ModernBertEncoderLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.layers = nn.ModuleList([
            ModernBertLayer(config=config, layer_id=layer_id)
            for layer_id in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids)
        return hidden_states


class ModernBertModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"layers.": "encoder_layer.layers."})

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.encoder_layer = ModernBertEncoderLayer(vllm_config)
        self.final_norm = nn.LayerNorm(config.hidden_size,
                                       eps=config.norm_eps,
                                       bias=config.norm_bias)
        self.gradient_checkpointing = False
        self.dtype = torch.float16

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        weights = self.hf_to_vllm_mapper.apply(weights)
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids=input_ids,
                                            inputs_embeds=inputs_embeds)

        outputs = self.encoder_layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
        )
        norm_outputs = self.final_norm(outputs)
        return norm_outputs


class ModernBertPooler(nn.Module):

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size,
                               config.classifier_bias)
        self.act = GELUActivation()
        self.norm = nn.LayerNorm(config.hidden_size,
                                 eps=config.norm_eps,
                                 bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states
        pooled_output = pooled_output.mean(dim=0, keepdim=False)
        pooled_output = self.norm(self.act(self.dense(pooled_output)))
        return pooled_output


class ModernBertForSequenceClassification(nn.Module, SupportsCrossEncoding):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = ModernBertModel(vllm_config,
                                     maybe_prefix(prefix, "modernbert"))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self._pooler = CrossEncodingPooler(config, self.classifier,
                                           ModernBertPooler(config))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        self_weights = []

        def weight_filter():
            for name, weight in weights:
                if name.startswith("model."):
                    yield name[len("model."):], weight
                else:
                    self_weights.append((name, weight))

        self.model.load_weights(weight_filter())

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in self_weights:
            if name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            if name.startswith("head"):
                param = params_dict["_pooler.pooler." + name[len("head") + 1:]]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
        )
