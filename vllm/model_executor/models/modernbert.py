# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch
from torch import nn
from transformers import ModernBertConfig
from transformers.activations import ACT2FN
import os
import json
from pathlib import Path
import importlib
from vllm.config.utils import config
from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import (
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import (
    EmbeddingPoolerHead,
    SequencePooler,
    get_seq_pooling_method,
)
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from .interfaces import SupportsCrossEncoding, SupportsLoRA
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
        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        embeddings = self.norm(inputs_embeds)
        return embeddings


class ModernBertAttention(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: int | None = None,
        prefix: str = "",
        dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
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
            quant_config=quant_config,
            prefix=f"{prefix}.Wqkv",
        )

        layer_type = config.layer_types[layer_id]
        rope_parameters = config.rope_parameters[layer_type]
        sliding_window: int | None = None
        if layer_type == "sliding_attention":
            # Treats the local attention boundary as inclusive
            sliding_window = config.sliding_window + 1

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            dtype=dtype,
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
            quant_config=quant_config,
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
    def __init__(
        self,
        config: ModernBertConfig,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.Wi = MergedColumnParallelLinear(
            config.hidden_size,
            [int(config.intermediate_size)] * 2,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.Wi",
        )
        self.act = ACT2FN[config.hidden_activation]
        self.Wo = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.Wo",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_gate, _ = self.Wi(hidden_states)
        input, gate = input_gate.chunk(2, dim=-1)
        return self.Wo(self.act(input) * gate)[0]


class ModernBertLayer(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        prefix: str = "",
        layer_id: int | None = None,
        dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
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
            config=config,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
            dtype=dtype,
            quant_config=quant_config,
        )
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(
            config,
            prefix=f"{prefix}.mlp",
            quant_config=quant_config,
        )

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
        dtype = vllm_config.model_config.dtype
        quant_config = vllm_config.quant_config
        self.layers = nn.ModuleList(
            [
                ModernBertLayer(
                    config=config,
                    layer_id=layer_id,
                    prefix=f"{prefix}.layers.{layer_id}",
                    dtype=dtype,
                    quant_config=quant_config,
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
class ModernBertModel(nn.Module, SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.layers.": "encoder_layer.layers.",
            "layers.": "encoder_layer.layers.",
            "model.": "",
        }
    )

    packed_modules_mapping = {
        "Wqkv": ["Wqkv"],
        "Wi": ["Wi"],
    }

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
            if name not in params_dict:
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
    def __init__(self, model_config: ModelConfig, module_configs):
        pooler_config = model_config.pooler_config
        assert pooler_config is not None

        config: ModernBertConfig = model_config.hf_config
        hf_pooling_type = config.classifier_pooling.upper()
        # vllm_pooling_type = pooler_config.seq_pooling_type
        # Currently we don't have a way to see if the user set the pooling type
        # explicitly or not, so we always use the HF pooling type for now.
        
        pooling_config = module_configs.get("1_Pooling")
        if not pooling_config:
            hf_pooling_type = pooling_config.pooling_mode

        super().__init__(
            pooling=get_seq_pooling_method(hf_pooling_type),
            # We set this dummy to avoid adding parameters to nn.Module too early
            head=nn.Identity(),
        )

        head_dtype = model_config.head_dtype
        dense_in_features = config.hidden_size
        dense_out_features = config.hidden_size
        dense_bias = config.classifier_bias
        dense_config = module_configs.get("2_Dense")
        activation_function = nn.GELU()
        if dense_config is not None:
            dense_in_features = dense_config.in_features
            dense_out_features = dense_config.out_features
            dense_bias = dense_config.bias
            module_path, object_name = dense_config.activation_function.rsplit(".", 1)
            module = importlib.import_module(module_path)
            activation_function = getattr(module, object_name)()

        self.dense = nn.Linear(
            dense_in_features,
            dense_out_features,
            dense_bias,
            dtype=head_dtype,
        )
        self.act = activation_function

        normalized_shape = config.hidden_size
        layer_norm_eps = config.norm_eps
        layer_norm_bias = config.norm_bias
        layer_norm_config = module_configs.get("3_LayerNorm")
        if layer_norm_config is not None:
            normalized_shape = layer_norm_config.dimension
            layer_norm_bias = layer_norm_config.norm_bias
        self.norm = nn.LayerNorm(
            normalized_shape,
            eps=layer_norm_eps,
            bias=layer_norm_bias,
            dtype=head_dtype,
        )

        # Use lambdas so that weights are not registered under `self.head`
        self.head = EmbeddingPoolerHead(
            head_dtype=head_dtype,
            projector=lambda x: self.norm(self.act(self.dense(x))),
        )

@config
class PoolingConfig:
    path: str
    embedding_dimension: int
    pooling_mode: str
    include_prompt: bool
    prefix: str

@config
class LayerNormConfig:
    path: str
    dimension: int
    norm_bias: bool
    prefix: str

@config
class DenseConfig:
    path: str
    in_features: int
    out_features: int 
    bias: bool
    activation_function: str
    module_input_name: str
    module_output_name: str
    prefix: str

@default_pooling_type(seq_pooling_type="CLS")
class ModernBertForSequenceClassification(nn.Module, SupportsCrossEncoding):
    is_pooling_model = True
    config_class_for_modules = {
        "PoolingConfig": PoolingConfig,
        "LayerNormConfig": LayerNormConfig,
        "DenseConfig": DenseConfig
    }

    def load_other_modules_config(self, config):
        self.hf_to_vllm_mapper = None
        self.module_configs = {}
        modules_path = config.name_or_path+"/modules.json"
        modules_file_path = Path(modules_path)
        if modules_file_path.is_file():
            with open(modules_file_path, "r", encoding="utf-8") as f:
                modules_json_array = json.load(f)
        else:
            return
        
        for folder_json in modules_json_array:
            folder_name = folder_json.get("path")
            if not folder_name:
                continue
            module_config = config.name_or_path+f"/{folder_name}/config.json"
            module_config_path = Path(module_config)
            if not module_config_path.is_file():
                continue
            with open(module_config_path, "r", encoding="utf-8") as f:
                module_config_json = json.load(f)
                module_config_json["path"] = config.name_or_path+f"/{folder_name}"
            module_config_class_name = folder_name.split("_")[-1] + "Config"
            self.module_configs[folder_name] = self.config_class_for_modules[module_config_class_name](**module_config_json)
        
        weight_mapper_file_path = Path(config.name_or_path+"/weight_mapper.json")
        if weight_mapper_file_path.is_file():
            with open(weight_mapper_file_path, "r", encoding="utf-8") as f:
                weight_mapper_json = json.load(f)
            self.hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix=weight_mapper_json)


    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        self.load_other_modules_config(config)

        self.model = ModernBertModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "modernbert")
        )
        classifier_in_features = config.hidden_size
        classifier_out_features = config.num_labels
        classifier_bias = vllm_config.model_config.hf_config.classifier_bias
        classifier_config = self.module_configs.get("4_Dense")
        if classifier_config is not None:
            classifier_in_features = classifier_config.in_features
            classifier_out_features = classifier_config.out_features
            classifier_bias = classifier_config.bias

        self.classifier = nn.Linear(
            classifier_in_features,
            classifier_out_features,
            classifier_bias,
            dtype=vllm_config.model_config.head_dtype,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooling = ModernBertPooler(vllm_config.model_config, self.module_configs)

        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            pooling=self.pooling,
            classifier=self.classifier,
        )
        
        self.secondary_weights = []
        for _, module_config in self.module_configs.items(): 
            if not module_config.prefix:
                continue      
            self.secondary_weights.append(DefaultModelLoader.Source(
                                            module_config.path, 
                                            vllm_config.model_config.revision, 
                                            prefix=module_config.prefix)
                                        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        self_weights = []
        if self.hf_to_vllm_mapper is not None:
            weights = self.hf_to_vllm_mapper.apply(weights)
        def weight_filter():
            for name, weight in weights:
                if name.startswith("model."):
                    yield name[len("model.") :], weight
                else:
                    self_weights.append((name, weight))

        self.model.load_weights(weight_filter())

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in self_weights:
            if name.startswith("classifier") or name.startswith("pooling"):
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

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"drop": None})

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
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
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
