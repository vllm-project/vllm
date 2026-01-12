# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch
from torch import nn
from transformers import ModernBertConfig
from transformers.activations import ACT2FN

from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention.encoder_only_attention import (
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.activations import LambdaPoolerActivation
from vllm.model_executor.layers.pooler.seqwise import (
    EmbeddingPoolerHead,
    SequencePooler,
    get_seq_pooling_method,
)
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsCrossEncoding
from .interfaces_base import attn_type, default_pooling_type
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        eps = (
            getattr(config, "norm_eps", None)
            or getattr(config, "layer_norm_eps", None)
            or 1e-5
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=eps, bias=config.norm_bias)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return self.norm(inputs_embeds)
        else:
            inputs_embeds = self.tok_embeddings(input_ids)
            embeddings = self.norm(inputs_embeds)
            return embeddings


class ModernBertAttention(nn.Module):
    def __init__(
        self, config: ModernBertConfig, layer_id: int | None = None, prefix: str = ""
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        assert self.num_heads % tp_size == 0
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.scaling = self.head_dim**-0.5
        self.Wqkv = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            bias=config.attention_bias,
            prefix=f"{prefix}.Wqkv",
        )

        if layer_types := getattr(config, "layer_types", None):
            # Transformers v5
            layer_type = layer_types[layer_id]
            rope_parameters = config.rope_parameters[layer_type]
            sliding_window: int | None = None
            if layer_type == "sliding_attention":
                sliding_window = config.local_attention // 2
        else:
            # Transformers v4
            sliding_window = None
            if layer_id % config.global_attn_every_n_layers != 0:
                sliding_window = config.local_attention // 2
                rope_theta = (
                    config.local_rope_theta
                    if config.local_rope_theta is not None
                    else config.global_rope_theta
                )
            else:
                rope_theta = config.global_rope_theta
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            dtype=torch.float16,
        )
        self.attn = EncoderOnlyAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            prefix=f"{layer_id}.attn",
            per_layer_sliding_window=sliding_window,
        )
        self.Wo = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
            prefix=f"{prefix}.Wo",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        q, k, v = qkv.split([self.all_head_size] * 3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_outputs = self.attn(q, k, v)
        hidden_states = attn_outputs
        hidden_states, _ = self.Wo(hidden_states)
        return hidden_states


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(
            config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias
        )
        self.act = nn.GELU()
        self.Wo = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            prefix=f"{prefix}.Wo",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.act(input) * gate)[0]


class ModernBertLayer(nn.Module):
    def __init__(
        self, config: ModernBertConfig, prefix: str = "", layer_id: int | None = None
    ):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        self.attn = ModernBertAttention(
            config=config, layer_id=layer_id, prefix=f"{prefix}.attn"
        )
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attn_outputs = self.attn(
            hidden_states=self.attn_norm(hidden_states), position_ids=position_ids
        )
        hidden_states = hidden_states + attn_outputs
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_output
        return hidden_states


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.layers = nn.ModuleList(
            [
                ModernBertLayer(
                    config=config,
                    layer_id=layer_id,
                    prefix=f"{prefix}.layers.{layer_id}",
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids)
        return hidden_states


@support_torch_compile
@default_pooling_type(seq_pooling_type="CLS")
class ModernBertModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"layers.": "encoder_layer.layers."}
    )

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.encoder_layer = ModernBertEncoderLayer(
            vllm_config, prefix=f"{prefix}.encoder_layer"
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = self.hf_to_vllm_mapper.apply(weights)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(
                input_ids=input_ids, inputs_embeds=inputs_embeds
            )

        outputs = self.encoder_layer(
            hidden_states=hidden_states,
            position_ids=positions,
        )
        norm_outputs = self.final_norm(outputs)
        return norm_outputs


class ModernBertPooler(SequencePooler):
    def __init__(self, model_config: ModelConfig):
        pooler_config = model_config.pooler_config
        assert pooler_config is not None

        config: ModernBertConfig = model_config.hf_config
        hf_pooling_type = config.classifier_pooling.upper()
        # vllm_pooling_type = pooler_config.seq_pooling_type
        # Currently we don't have a way to see if the user set the pooling type
        # explicitly or not, so we always use the HF pooling type for now.

        super().__init__(
            pooling=get_seq_pooling_method(hf_pooling_type),
            # We set this dummy to avoid adding parameters to nn.Module too early
            head=nn.Identity(),
        )

        head_dtype = model_config.head_dtype
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            config.classifier_bias,
            dtype=head_dtype,
        )
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
        )

        # Use lambdas so that weights are not registered under `self.head`
        self.head = EmbeddingPoolerHead(
            projector=lambda x: self.dense(x),
            head_dtype=head_dtype,
            activation=LambdaPoolerActivation(lambda x: self.norm(self.act(x))),
        )


@default_pooling_type(seq_pooling_type="CLS")
class ModernBertForSequenceClassification(nn.Module, SupportsCrossEncoding):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.config = config
        self.model = ModernBertModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "modernbert")
        )
        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels,
            dtype=vllm_config.model_config.head_dtype,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooling = ModernBertPooler(vllm_config.model_config)

        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            pooling=self.pooling,
            classifier=self.classifier,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        self_weights = []

        def weight_filter():
            for name, weight in weights:
                if name.startswith("model."):
                    yield name[len("model.") :], weight
                else:
                    self_weights.append((name, weight))

        self.model.load_weights(weight_filter())

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in self_weights:
            if name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            if name.startswith("head"):
                param = params_dict["pooling." + name[len("head") + 1 :]]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def forward(
        self,
        input_ids: torch.LongTensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            positions=positions,
        )


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.classifier_bias
        )
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=getattr(config, "norm_eps", 1e-5),
            bias=getattr(config, "norm_bias", True),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


@attn_type("encoder_only")
@default_pooling_type(tok_pooling_type="ALL")
class ModernBertForTokenClassification(nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.head_dtype = vllm_config.model_config.head_dtype
        self.num_labels = config.num_labels
        self.model = ModernBertModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "modernbert")
        )
        self.head = ModernBertPredictionHead(config)
        self.classifier = nn.Linear(
            config.hidden_size, config.num_labels, dtype=self.head_dtype
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = pooler_for_token_classify(pooler_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self, skip_prefixes=["drop"])
        loaded_params = loader.load_weights(weights)
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        hidden_states = self.head(hidden_states)
        hidden_states = hidden_states.to(self.head_dtype)
        return self.classifier(hidden_states)
