# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import (get_act_and_mul_fn,
                                                   get_act_fn)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import SupportsV0Only
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.model_executor.models.utils import WeightsMapper
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class BertWithRopeEmbedding(nn.Module):

    def __init__(self, config: PretrainedConfig):

        super().__init__()
        if config.position_embedding_type not in ["rope", "rotary"]:
            raise ValueError("Only 'rotary'('rope') position_embedding_type" +
                             " is supported")

        self.word_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = VocabParallelEmbedding(
                config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape,
                                             dtype=torch.long,
                                             device=inputs_embeds.device)

            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BertWithRopeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = True,
        rotary_kwargs: Optional[dict] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_attention_heads
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        assert self.head_dim * self.total_num_heads == self.hidden_size

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj")

        self.rotary_emb = get_rope(**rotary_kwargs)

        self.attn = Attention(num_heads=self.num_heads,
                              head_size=self.head_dim,
                              scale=self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=AttentionType.ENCODER_ONLY)

        self.out_proj = RowParallelLinear(input_size=hidden_size,
                                          output_size=hidden_size,
                                          bias=bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.dense")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class BertWithRopeGatedMLP(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 bias: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.act_fn = get_act_and_mul_fn(hidden_act)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class BertWithRopeMLP(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 bias: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.act_fn = get_act_fn(hidden_act)
        self.up_proj = ColumnParallelLinear(input_size=hidden_size,
                                            output_size=intermediate_size,
                                            bias=bias,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.up_proj")
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class NomicRouter(nn.Module):

    def __init__(self, hidden_size: int, moe_num_experts: int, moe_top_k: int):
        super().__init__()
        self.moe_top_k = moe_top_k
        self.layer = ReplicatedLinear(hidden_size, moe_num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        weights = self.layer(x.view(-1, x.shape[-1]))[0].softmax(
            dim=-1, dtype=torch.float32)
        top_weights, top_experts = torch.topk(weights, self.moe_top_k, dim=-1)
        weights = weights.to(x.dtype)
        top_weights = top_weights.to(x.dtype)
        return weights, top_weights, top_experts  # type: ignore


class NomicExpertMLP(nn.Module):

    def __init__(self, hidden_size: int, ffn_hidden_size: int,
                 moe_num_experts: int, ffn_act_fn: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts

        self.w1 = nn.Parameter(
            torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.w2 = nn.Parameter(
            torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.activation_fn = get_act_fn(ffn_act_fn)

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        expert_w1 = self.w1.view(self.moe_num_experts, self.ffn_hidden_size,
                                 self.hidden_size)[expert_idx]
        expert_w2 = self.w2.view(self.moe_num_experts, self.ffn_hidden_size,
                                 self.hidden_size)[expert_idx]

        x1 = x.matmul(expert_w1.t())
        act_out = self.activation_fn(x1)
        x2 = act_out.matmul(expert_w2)
        return x2


class NomicExperts(nn.Module):

    def __init__(self, config, hidden_size: int, ffn_hidden_size: int,
                 moe_num_experts: int):
        super().__init__()
        self.moe_num_experts = moe_num_experts

        self.mlp = NomicExpertMLP(hidden_size=config.n_embd,
                                  ffn_hidden_size=config.n_inner,
                                  moe_num_experts=moe_num_experts,
                                  ffn_act_fn=config.hidden_act)
        self.bias = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x: torch.Tensor, weights: torch.Tensor,
                top_weights: torch.Tensor,
                top_experts: torch.LongTensor) -> torch.Tensor:
        q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        expert_mask = nn.functional.one_hot(
            top_experts, num_classes=self.moe_num_experts).permute(2, 1, 0)
        for expert_idx in range(0, self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue

            token_list = token_idx.tolist()
            topk_list = topk_idx.tolist()

            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            expert_out = self.mlp(
                expert_tokens, expert_idx) * top_weights[token_list, topk_list,
                                                         None]

            out.index_add_(0, token_idx, expert_out)

        out = out.reshape(q_len, hidden_size)
        return out + self.bias


class NomicMoELayer(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.router = NomicRouter(
            config.n_embd,
            moe_num_experts=config.num_experts,
            moe_top_k=config.moe_top_k,
        )

        self.experts = NomicExperts(
            config,
            hidden_size=config.n_embd,
            ffn_hidden_size=config.n_inner,
            moe_num_experts=config.num_experts,
        )

    def forward(self, x: torch.Tensor):
        weights, top_weights, top_experts = self.router(x)
        out = self.experts(x, weights, top_weights, top_experts)
        return out


class BertWithRopeBlock(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 moe: bool = False,
                 bias: bool = True,
                 rotary_kwargs: Optional[dict] = None,
                 prefix: str = ""):
        super().__init__()
        self.attn = BertWithRopeAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            bias=bias,
            rotary_kwargs=rotary_kwargs,
            prefix=f"{prefix}.attention")

        if moe:
            self.mlp = NomicMoELayer(config=config, )
        else:
            if config.hidden_act in ["silu", "geglu"]:
                self.mlp = BertWithRopeGatedMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    bias=bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp")
            else:
                self.mlp = BertWithRopeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    bias=bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp")

        self.attn_ln = nn.LayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
        self.mlp_ln = nn.LayerNorm(config.hidden_size,
                                   eps=config.layer_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        attn_output = self.attn(positions, hidden_states)
        hidden_states = self.attn_ln(hidden_states + attn_output)
        mlp_out = self.mlp(hidden_states)
        hidden_states = self.mlp_ln(hidden_states + mlp_out)
        return hidden_states


@support_torch_compile
class BertWithRopeEncoder(nn.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 bias: bool = True,
                 rotary_kwargs: Optional[dict] = None,
                 prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        every_n = getattr(config, "moe_every_n_layers", 0)
        self.layers = nn.ModuleList([
            BertWithRopeBlock(config=config,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              bias=bias,
                              moe=every_n > 0 and (layer_idx % every_n == 1),
                              rotary_kwargs=rotary_kwargs,
                              prefix=f"{prefix}.layer.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return hidden_states


class BertWithRope(nn.Module, SupportsV0Only, SupportsQuant):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = self.config_verify(vllm_config)
        self.embeddings = BertWithRopeEmbedding(self.config)
        self.encoder = BertWithRopeEncoder(
            vllm_config=vllm_config,
            bias=getattr(self.config, "bias", True),
            rotary_kwargs=self.config.rotary_kwargs,
            prefix=f"{prefix}.encoder")

    def config_verify(self, vllm_config):
        raise NotImplementedError

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids=input_ids,
                                            token_type_ids=token_type_ids)
        return self.encoder(positions, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        weights = self.hf_to_vllm_mapper.apply(weights)

        if self.config.hidden_act in ["silu", "geglu"]:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]
        else:
            stacked_params_mapping = []

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "pooler" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class NomicBertModel(BertWithRope):
    # for https://huggingface.co/nomic-ai/nomic-bert-2048

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "emb_ln": "embeddings.LayerNorm",
            "attn.Wqkv": "attn.qkv_proj",
            "norm1": "attn_ln",
            "mlp.fc1.": "mlp.up_proj.",
            "mlp.fc11": "mlp.up_proj",
            "mlp.fc12": "mlp.gate_proj",
            "mlp.fc2": "mlp.down_proj",
            "norm2": "mlp_ln",
        })

    def config_verify(self, vllm_config):
        config = vllm_config.model_config.hf_config

        assert config.__class__.__name__ == "NomicBertConfig"
        assert config.activation_function in ["swiglu", "gelu"]
        config.position_embedding_type = getattr(config,
                                                 "position_embedding_type",
                                                 "rope")

        if config.activation_function == "swiglu":
            config.hidden_act = "silu"
        else:
            config.hidden_act = config.activation_function

        assert (config.mlp_fc1_bias == config.mlp_fc2_bias ==
                config.qkv_proj_bias)
        config.bias = config.qkv_proj_bias

        assert config.rotary_emb_scale_base is None
        assert not config.rotary_emb_interleaved

        config.layer_norm_eps = config.layer_norm_epsilon
        config.intermediate_size = config.n_inner
        config.hidden_size = config.n_embd
        config.num_hidden_layers = config.n_layer

        head_dim = config.hidden_size // config.num_attention_heads
        rotary_emb_dim = head_dim * config.rotary_emb_fraction
        max_trained_positions = getattr(config, "max_trained_positions", 2048)
        config.rotary_kwargs = {
            "head_size": head_dim,
            "rotary_dim": rotary_emb_dim,
            "max_position": max_trained_positions,
            "base": getattr(config, "rope_theta", config.rotary_emb_base),
            "rope_scaling": getattr(config, "rope_scaling", None)
        }

        # we ignore config.rotary_scaling_factor so that for datasets shorter
        # than max_trained_positions 2048, the results are consistent
        # with SentenceTransformer.
        # The context extension uses vllm style rope_theta and rope_scaling.
        # See #17785 #18755
        if (not vllm_config.model_config.hf_overrides
                and vllm_config.model_config.original_max_model_len is None):
            # Default
            # Reset max_model_len to max_trained_positions.
            # nomic-embed-text-v2-moe the length is set to 512
            # by sentence_bert_config.json.
            max_model_len_before = vllm_config.model_config.max_model_len
            max_model_len = min(vllm_config.model_config.max_model_len,
                                max_trained_positions)

            vllm_config.recalculate_max_model_len(max_model_len)
            logger.warning(
                "Nomic context extension is disabled. "
                "Changing max_model_len from %s to %s. "
                "To enable context extension, see: "
                "https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/context_extension.html",
                max_model_len_before, vllm_config.model_config.max_model_len)
        else:
            # We need to re-verify max_model_len to avoid lengths
            # greater than position_embedding.
            model_config = vllm_config.model_config
            hf_text_config = model_config.hf_text_config

            if isinstance(model_config.hf_overrides, dict):
                # hf_overrides_kw
                max_model_len = model_config.hf_overrides.get(
                    "max_model_len", vllm_config.model_config.max_model_len)
            else:
                # hf_overrides_fn
                # This might be overridden by sentence_bert_config.json.
                max_model_len = vllm_config.model_config.max_model_len

            # reset hf_text_config for recalculate_max_model_len.
            if hasattr(hf_text_config, "max_model_len"):
                delattr(hf_text_config, "max_model_len")
            hf_text_config.max_position_embeddings = max_trained_positions
            hf_text_config.rope_scaling = config.rotary_kwargs["rope_scaling"]

            # The priority of sentence_bert_config.json is higher
            # than max_position_embeddings
            encoder_config = deepcopy(model_config.encoder_config)
            encoder_config.pop("max_seq_length", None)
            model_config.encoder_config = encoder_config

            vllm_config.recalculate_max_model_len(max_model_len)
        return config


class GteNewModel(BertWithRope):
    # for https://huggingface.co/Alibaba-NLP/new-impl

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "new.": "",
            "layer": "layers",
            "attention.qkv_proj": "attn.qkv_proj",
            "attention.o_proj": "attn.out_proj",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # GteNewModel only gate_up_proj does not have bias.
        # Hack method learned from vllm/model_executor/models/glm.py
        for layer in self.encoder.layers:
            layer.mlp.gate_up_proj.bias = None
            layer.mlp.gate_up_proj.skip_bias_add = True

    def config_verify(self, vllm_config):
        config = vllm_config.model_config.hf_config

        assert config.__class__.__name__ == "NewConfig"
        assert config.hidden_act == "gelu"

        config.hidden_act = "geglu"

        head_dim = config.hidden_size // config.num_attention_heads
        config.rotary_kwargs = {
            "head_size": head_dim,
            "rotary_dim": getattr(config, "rotary_emb_dim", head_dim),
            "max_position": config.max_position_embeddings,
            "base": config.rope_theta,
            "rope_scaling": getattr(config, "rope_scaling", None)
        }
        return config

    def split_up_gate_proj(self, weights: Iterable[tuple[str, torch.Tensor]]):
        n = "mlp.up_gate_proj"
        for name, weight in weights:
            if n in name:
                up, gate = weight.chunk(2, dim=0)
                yield name.replace(n, "mlp.up_proj"), up
                yield name.replace(n, "mlp.gate_proj"), gate
            else:
                yield name, weight

    def ignore_unnecessary_layers(self,
                                  weights: Iterable[tuple[str, torch.Tensor]]):
        for name, weight in weights:
            if name.startswith("classifier"):
                continue
            yield name, weight

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        weights = self.ignore_unnecessary_layers(weights)
        weights = self.split_up_gate_proj(weights)
        return super().load_weights(weights)


class SnowflakeGteNewModel(GteNewModel):
    # for Snowflake/snowflake-arctic-embed-m-v2.0

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "layer": "layers",
            "attention.qkv_proj": "attn.qkv_proj",
            "attention.o_proj": "attn.out_proj",
        })

    def config_verify(self, vllm_config):
        config = vllm_config.model_config.hf_config

        assert config.__class__.__name__ == "GteConfig"
        assert config.hidden_act == "gelu"

        config.hidden_act = "geglu"

        head_dim = config.hidden_size // config.num_attention_heads
        config.rotary_kwargs = {
            "head_size": head_dim,
            "rotary_dim": getattr(config, "rotary_emb_dim", head_dim),
            "max_position": config.max_position_embeddings,
            "base": config.rope_theta,
            "rope_scaling": getattr(config, "rope_scaling", None)
        }
        return config


class JinaRobertaModel(BertWithRope):
    # for https://huggingface.co/jinaai/jina-embeddings-v3

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "emb_ln": "embeddings.LayerNorm",
            "mixer.Wqkv": "attn.qkv_proj",
            "mixer.out_proj": "attn.out_proj",
            "norm1": "attn_ln",
            "mlp.fc1.": "mlp.up_proj.",
            "mlp.fc2": "mlp.down_proj",
            "norm2": "mlp_ln",
        })

    def config_verify(self, vllm_config):
        config = vllm_config.model_config.hf_config

        assert config.__class__.__name__ == "XLMRobertaFlashConfig"

        head_dim = config.hidden_size // config.num_attention_heads
        config.rotary_kwargs = {
            "head_size": head_dim,
            "rotary_dim": getattr(config, "rotary_emb_dim", head_dim),
            "max_position": config.max_position_embeddings,
            "base": getattr(config, "rope_theta", config.rotary_emb_base),
            "rope_scaling": getattr(config, "rope_scaling", None)
        }
        return config

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(input_ids=input_ids,
                               positions=position_ids,
                               intermediate_tensors=intermediate_tensors,
                               inputs_embeds=inputs_embeds,
                               token_type_ids=token_type_ids)

    @torch.inference_mode()
    def jina_merge_lora_weights(self, weights: Iterable[tuple[str,
                                                              torch.Tensor]]):
        # use for jina-embeddings-v3
        # Merge Lora weights into a single weight tensor.
        # This is a temporary solution until we have a better way to handle

        scaling = self.config.lora_alpha / self.config.lora_rank
        device = self.vllm_config.device_config.device

        weights = {name: weight for name, weight in weights}

        o = ".original"
        a = ".0.lora_A"
        b = ".0.lora_B"

        # text-matching
        i = -1

        for name in list(weights.keys()):
            if o in name:
                dtype = weights[name].dtype
                shape = weights[name].shape
                weight_name = name[:-len(o)]

                if "embeddings" in weight_name:
                    B = weights[weight_name + a][i].to(device).float()
                    A = weights[weight_name + b][i].to(device).float()
                else:
                    B = weights[weight_name + b][i].to(device).float()
                    A = weights[weight_name + a][i].to(device).float()

                weight = (weights[weight_name + o].to(device) +
                          torch.matmul(B, A).view(shape) * scaling)
                weight = weight.cpu().to(dtype)

                weights[weight_name.replace(".parametrizations", "")] = weight

                del weights[weight_name + o], weights[weight_name +
                                                      a], weights[weight_name +
                                                                  b]

        return [(name, weight) for name, weight in weights.items()]

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        weights = self.jina_merge_lora_weights(weights)
        return super().load_weights(weights)
