# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import get_act_and_mul_fn, get_act_fn
from vllm.model_executor.layers.attention.encoder_only_attention import (
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.fused_moe import activation_without_mul, fused_topk
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .bert import BertPooler
from .interfaces import SupportsCrossEncoding, SupportsQuant
from .interfaces_base import default_pooling_type


class BertWithRopeEmbedding(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        if config.position_embedding_type not in ["rope", "rotary"]:
            raise ValueError(
                "Only 'rotary'('rope') position_embedding_type" + " is supported"
            )

        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        if config.type_vocab_size > 0:
            self.token_type_embeddings = VocabParallelEmbedding(
                config.type_vocab_size, config.hidden_size
            )
        else:
            self.token_type_embeddings = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=inputs_embeds.device
                )

            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BertWithRopeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        bias: bool = True,
        rotary_kwargs: dict | None = None,
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
            prefix=f"{prefix}.qkv_proj",
        )

        self.rotary_emb = get_rope(**rotary_kwargs)

        self.attn = EncoderOnlyAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.out_proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

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
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.act_fn = get_act_and_mul_fn(hidden_act)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class BertWithRopeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.act_fn = get_act_fn(hidden_act)
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class NomicMoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        params_dtype: torch.dtype | None = None,
        tp_size: int | None = None,
    ):
        super().__init__()

        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.total_intermediate_size = intermediate_size
        self.intermediate_size = divide(intermediate_size, self.tp_size)
        self.hidden_act = activation_without_mul(hidden_act)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.router = ReplicatedLinear(
            self.hidden_size, self.num_total_experts, bias=False
        )
        self.w1 = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                self.intermediate_size,
                self.hidden_size,
                device=current_platform.device_type,
                dtype=self.params_dtype,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                self.hidden_size,
                self.intermediate_size,
                device=current_platform.device_type,
                dtype=self.params_dtype,
            )
        )
        self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        set_weight_attrs(
            self.w1,
            {
                "weight_loader": self.weight_loader,
            },
        )
        set_weight_attrs(
            self.w2,
            {
                "weight_loader": self.weight_loader,
            },
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
    ):
        # NOTE: Nomic-MoE has fused experts weights with shape
        # (num_experts * intermediate_size, hidden_size)
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w1"):
            loaded_weight = loaded_weight.reshape(
                self.num_total_experts,
                self.total_intermediate_size,
                self.hidden_size,
            )[:, shard]
        if weight_name.endswith("w2"):
            loaded_weight = loaded_weight.reshape(
                self.num_total_experts,
                self.total_intermediate_size,
                self.hidden_size,
            )[:, shard].transpose(1, 2)
        param_data.copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.router(hidden_states)
        # FIXME(Isotr0py): This implementation is too tricky,
        # we should use FusedMoE instead in the future
        # after supporting ungated activation for it.
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, router_logits, self.top_k, renormalize=False
        )

        final_hidden_states = torch.ops.vllm.outplace_fused_experts(
            hidden_states=hidden_states,
            w1=self.w1,
            w2=self.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=self.hidden_act,
        )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_size) + self.bias


class BertWithRopeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        moe: bool = False,
        bias: bool = True,
        rotary_kwargs: dict | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attn = BertWithRopeAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            bias=bias,
            rotary_kwargs=rotary_kwargs,
            prefix=f"{prefix}.attention",
        )

        if moe:
            self.mlp = NomicMoE(
                num_experts=config.num_experts,
                top_k=config.moe_top_k,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        else:
            if config.hidden_act in ["silu", "geglu"]:
                self.mlp = BertWithRopeGatedMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    bias=bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )
            else:
                self.mlp = BertWithRopeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    bias=bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )

        self.attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        attn_output = self.attn(positions, hidden_states)
        hidden_states = self.attn_ln(hidden_states + attn_output)
        mlp_out = self.mlp(hidden_states)
        hidden_states = self.mlp_ln(hidden_states + mlp_out)
        return hidden_states


class BertWithRopeEncoder(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        bias: bool = True,
        rotary_kwargs: dict | None = None,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        every_n = getattr(config, "moe_every_n_layers", 0)
        self.layers = nn.ModuleList(
            [
                BertWithRopeBlock(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    bias=bias,
                    moe=every_n > 0 and (layer_idx % every_n == 1),
                    rotary_kwargs=rotary_kwargs,
                    prefix=f"{prefix}.layer.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return hidden_states


@support_torch_compile
@default_pooling_type(seq_pooling_type="CLS")
class BertWithRope(nn.Module, SupportsQuant):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        add_pooling_layer: bool = False,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.add_pooling_layer = add_pooling_layer
        self.config = vllm_config.model_config.hf_config
        self.embeddings = BertWithRopeEmbedding(self.config)
        self.encoder = BertWithRopeEncoder(
            vllm_config=vllm_config,
            bias=getattr(self.config, "bias", True),
            rotary_kwargs=self.config.rotary_kwargs,
            prefix=f"{prefix}.encoder",
        )
        self.pooler = BertPooler(self.config) if add_pooling_layer else None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(
                input_ids=input_ids, token_type_ids=token_type_ids
            )
        return self.encoder(positions, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
            if not self.add_pooling_layer and "pooler" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
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
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if name.endswith((".w1", ".w2")):
                    # Nomic-MoE has fused experts weights
                    weight_loader(param, loaded_weight, name)
                else:
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
            # MoE mapping
            "experts.mlp.": "",
            "experts.": "",
            "router.layer": "router",
        }
    )


class GteNewModel(BertWithRope):
    # for https://huggingface.co/Alibaba-NLP/new-impl

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "new.": "",
            "layer": "layers",
            "attention.qkv_proj": "attn.qkv_proj",
            "attention.o_proj": "attn.out_proj",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

        # GteNewModel only gate_up_proj does not have bias.
        # Hack method learned from vllm/model_executor/models/glm.py
        for layer in self.encoder.layers:
            layer.mlp.gate_up_proj.bias = None
            layer.mlp.gate_up_proj.skip_bias_add = True

    def split_up_gate_proj(self, weights: Iterable[tuple[str, torch.Tensor]]):
        n = "mlp.up_gate_proj"
        for name, weight in weights:
            if n in name:
                up, gate = weight.chunk(2, dim=0)
                yield name.replace(n, "mlp.up_proj"), up
                yield name.replace(n, "mlp.gate_proj"), gate
            else:
                yield name, weight

    def ignore_unnecessary_layers(self, weights: Iterable[tuple[str, torch.Tensor]]):
        for name, weight in weights:
            if name.startswith("classifier"):
                continue
            yield name, weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
        }
    )


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
        }
    )

    @torch.inference_mode()
    def jina_merge_lora_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
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
                weight_name = name[: -len(o)]

                if "embeddings" in weight_name:
                    B = weights[weight_name + a][i].to(device).float()
                    A = weights[weight_name + b][i].to(device).float()
                else:
                    B = weights[weight_name + b][i].to(device).float()
                    A = weights[weight_name + a][i].to(device).float()

                weight = (
                    weights[weight_name + o].to(device)
                    + torch.matmul(B, A).view(shape) * scaling
                )
                weight = weight.cpu().to(dtype)

                weights[weight_name.replace(".parametrizations", "")] = weight

                del (
                    weights[weight_name + o],
                    weights[weight_name + a],
                    weights[weight_name + b],
                )

        return [(name, weight) for name, weight in weights.items()]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = self.jina_merge_lora_weights(weights)
        return super().load_weights(weights)


@default_pooling_type(seq_pooling_type="CLS")
class GteNewForSequenceClassification(nn.Module, SupportsCrossEncoding):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.new = GteNewModel(
            vllm_config=vllm_config, prefix=prefix, add_pooling_layer=True
        )
        self.classifier = ReplicatedLinear(
            config.hidden_size,
            config.num_labels,
            bias=True,
            quant_config=quant_config,
            params_dtype=vllm_config.model_config.head_dtype,
            prefix=maybe_prefix(prefix, "classifier"),
            return_bias=False,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            pooling=self.new.pooler,
            classifier=self.classifier,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights)
        return loaded_params

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.new.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.new(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
