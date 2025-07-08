# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, Union

import torch
from torch import nn
from transformers import BatchFeature, BertConfig

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, PoolerConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.pooler import (ClassifierPooler, Pooler,
                                               PoolingType)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalFlatField, MultiModalInputs,
                                    MultiModalKwargs, MultiModalKwargsItem,
                                    PlaceholderRange)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors, PoolerOutput

from .interfaces import (MultiModalEmbeddings, SupportsCrossEncoding,
                         SupportsMultiModal, SupportsQuant)
from .utils import WeightsMapper, maybe_prefix


class BertEmbedding(nn.Module):

    def __init__(self, config: BertConfig):

        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
        self.position_embeddings = VocabParallelEmbedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = VocabParallelEmbedding(
            config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.position_ids = nn.Parameter(
            torch.empty((1, config.max_position_embeddings)), )

        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type != "absolute":
            raise ValueError("Only 'absolute' position_embedding_type" +
                             " is supported")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        apply_layer_norm: bool = True,
    ) -> torch.Tensor:

        # forward was called directly without going
        # throught the multi-modal flow
        if input_ids is not None and position_ids is not None \
            and inputs_embeds is None and token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.size(),
                                         dtype=torch.long,
                                         device=input_ids.device)

        tensors_to_add: list[torch.Tensor] = []

        if inputs_embeds is not None:
            tensors_to_add.append(inputs_embeds)

        if token_type_ids is not None:
            tensors_to_add.append(self.token_type_embeddings(token_type_ids))

        if position_ids is not None:
            tensors_to_add.append(self.position_embeddings(position_ids))

        if input_ids is not None:
            tensors_to_add.append(self.word_embeddings(input_ids))

        embeds = torch.stack(tensors_to_add, dim=0).sum(dim=0)

        if apply_layer_norm:
            return self.LayerNorm(embeds)
        else:
            return embeds


class BertPooler(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertEncoder(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.layer = nn.ModuleList([
            BertLayer(config=config,
                      cache_config=cache_config,
                      quant_config=quant_config,
                      prefix=f"{prefix}.layer.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self,
                 config: BertConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.attention = BertAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention")

        self.intermediate = BertIntermediate(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.intermediate")

        self.output = BertOutput(hidden_size=config.hidden_size,
                                 intermediate_size=config.intermediate_size,
                                 layer_norm_eps=config.layer_norm_eps,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.output")

    def forward(self, hidden_states: torch.Tensor):
        attn_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attn_output)
        output = self.output(intermediate_output, attn_output)
        return output


class BertAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_norm_eps: float,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.self = BertSelfAttention(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.output")

        self.output = BertSelfOutput(hidden_size=hidden_size,
                                     layer_norm_eps=layer_norm_eps,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.output")

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self_output = self.self(hidden_states)
        return self.output(self_output, hidden_states)


class BertSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
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
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj")

        self.attn = Attention(num_heads=self.num_heads,
                              head_size=self.head_dim,
                              scale=self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=AttentionType.ENCODER_ONLY)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        output = self.attn(q, k, v)
        return output


class BertSelfOutput(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 layer_norm_eps: float,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.dense = RowParallelLinear(input_size=hidden_size,
                                       output_size=hidden_size,
                                       bias=True,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.dense")
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.dense = ColumnParallelLinear(input_size=hidden_size,
                                          output_size=intermediate_size,
                                          bias=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.dense")
        self.intermediate_act_fn = get_act_fn(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 layer_norm_eps: float,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.dense = RowParallelLinear(input_size=intermediate_size,
                                       output_size=hidden_size,
                                       bias=True,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.dense")

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


@support_torch_compile
class BertModel(nn.Module, SupportsQuant):
    packed_modules_mapping = {"qkv_proj": ["query", "key", "value"]}

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 embedding_class: type = BertEmbedding,
                 add_pooling_layer: bool = False):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embeddings = embedding_class(config)
        self.encoder = BertEncoder(vllm_config=vllm_config,
                                   prefix=f"{prefix}.encoder")
        self.pooler = BertPooler(config) if add_pooling_layer else None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(input_ids=input_ids,
                                        position_ids=position_ids,
                                        token_type_ids=token_type_ids,
                                        inputs_embeds=inputs_embeds)
        return self.encoder(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.pooler is None and "pooler" in name:
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


class BertEmbeddingModel(nn.Module, SupportsQuant):
    """A model that uses Bert to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        pooler_config = vllm_config.model_config.pooler_config
        self.model = self._build_model(vllm_config=vllm_config,
                                       prefix=maybe_prefix(prefix, "model"))
        self._pooler = self._build_pooler(pooler_config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids,
                          position_ids=positions,
                          token_type_ids=token_type_ids,
                          inputs_embeds=inputs_embeds,
                          intermediate_tensors=intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        self.model.load_weights(weights)

    def _build_model(self,
                     vllm_config: VllmConfig,
                     prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config,
                         prefix=prefix,
                         embedding_class=BertEmbedding)

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        return Pooler.from_config_with_defaults(pooler_config,
                                                pooling_type=PoolingType.CLS,
                                                normalize=True,
                                                softmax=False)


TOKEN_TYPES = "token_type_ids"


class TokenTypeMultiModalProcessor(BaseMultiModalProcessor):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        raise NotImplementedError

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:

        assert isinstance(prompt, list)

        mm_data_item = mm_data[TOKEN_TYPES]
        if isinstance(mm_data_item, list):
            mm_data_item = torch.tensor(mm_data_item)

        prompt_len = len(prompt)

        mm_placeholders = {
            TOKEN_TYPES: [PlaceholderRange(
                offset=0,
                length=prompt_len,
            )]
        }

        field = MultiModalFlatField([slice(0, prompt_len)])
        mm_item = MultiModalKwargsItem.from_elems(
            field.build_elems(modality=TOKEN_TYPES,
                              key=TOKEN_TYPES,
                              data=mm_data_item))

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt,
            mm_kwargs=MultiModalKwargs.from_items([mm_item]),
            mm_hashes=None,
            mm_placeholders=mm_placeholders,
        )


class TokenTypeProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {TOKEN_TYPES: 1}


class TokenTypeInputBuilder(BaseDummyInputsBuilder[TokenTypeProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        raise NotImplementedError

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        raise NotImplementedError

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:

        dummy_prompt = [0] * seq_len
        dummy_mm_data = {
            TOKEN_TYPES: torch.zeros(seq_len, dtype=torch.int32),
        }
        return ProcessorInputs(prompt=dummy_prompt, mm_data=dummy_mm_data)


class BertMMTokenIdsMixin(SupportsMultiModal):

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        token_type_ids = kwargs.pop(TOKEN_TYPES, None)

        if token_type_ids is None:
            return []

        if not isinstance(token_type_ids, torch.Tensor):
            raise ValueError("Incorrect type token_type_ids. "
                             f"Got type: {type(token_type_ids)}")

        return self.get_language_model().embeddings(
            token_type_ids=token_type_ids, apply_layer_norm=False)

    def maybe_store_input_ids(self, input_ids: torch.Tensor):
        pass

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        token_type_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        # save for forward()
        self.maybe_store_input_ids(input_ids)

        token_type_ids: Optional[torch.Tensor] = None

        if token_type_embeddings is not None:
            assert isinstance(token_type_embeddings, list)
            token_type_embeddings = torch.cat(token_type_embeddings)
        else:
            token_type_ids = torch.zeros(input_ids.size(),
                                         dtype=torch.long,
                                         device=input_ids.device)

        return self.get_language_model().embeddings(
            input_ids=input_ids,
            inputs_embeds=token_type_embeddings,
            token_type_ids=token_type_ids,
            apply_layer_norm=False)


@MULTIMODAL_REGISTRY.register_processor(TokenTypeMultiModalProcessor,
                                        info=TokenTypeProcessingInfo,
                                        dummy_inputs=TokenTypeInputBuilder)
class BertForSequenceClassification(nn.Module, SupportsCrossEncoding,
                                    BertMMTokenIdsMixin, SupportsQuant):
    """A model that uses Bert to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.num_labels = config.num_labels
        self.bert = BertModel(vllm_config=vllm_config,
                              prefix=maybe_prefix(prefix, "bert"),
                              embedding_class=BertEmbedding,
                              add_pooling_layer=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self._pooler = ClassifierPooler(vllm_config.model_config,
                                        self.classifier, self.bert.pooler)

    def get_language_model(self) -> torch.nn.Module:
        return self.bert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        self_weights = []

        def weight_filter():
            for name, weight in weights:
                if name.startswith("bert."):
                    yield (name[len("bert."):], weight)
                else:
                    self_weights.append((name, weight))

        self.bert.load_weights(weight_filter())

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in self_weights:
            if name.startswith("classifier"):
                param = params_dict[name]
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
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.bert(input_ids=input_ids,
                         position_ids=positions,
                         inputs_embeds=inputs_embeds,
                         intermediate_tensors=intermediate_tensors,
                         token_type_ids=token_type_ids)
