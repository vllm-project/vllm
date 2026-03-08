# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from torch import nn
from transformers import BertConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.sequence import IntermediateTensors

from .bert import (
    TOKEN_TYPE_SHIFT,
    BertEmbedding,
    BertEmbeddingModel,
    BertModel,
    BertPoolingModel,
    _decode_token_type_ids,
    _encode_token_type_ids,
)
from .interfaces import SupportsCrossEncoding, SupportsQuant
from .interfaces_base import attn_type, default_pooling_type
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

_LEGACY_SUFFIX_MAPPER = WeightsMapper(
    orig_to_new_suffix={
        ".gamma": ".weight",
        ".beta": ".bias",
    }
)


class ErnieEmbedding(BertEmbedding):
    def __init__(self, config: BertConfig):
        super().__init__(config)

        task_type_vocab_size = max(1, getattr(config, "task_type_vocab_size", 1))
        self.task_type_embeddings = VocabParallelEmbedding(
            task_type_vocab_size, config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_type_ids = _decode_token_type_ids(input_ids)
        task_type_ids = torch.zeros_like(token_type_ids)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        task_type_embeddings = self.task_type_embeddings(task_type_ids)

        embeddings = (
            inputs_embeds
            + token_type_embeddings
            + task_type_embeddings
            + position_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        return embeddings


@default_pooling_type(seq_pooling_type="CLS")
class ErnieModel(BertModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            embedding_class=ErnieEmbedding,
        )


class ErniePoolingModel(BertPoolingModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            embedding_class=ErnieEmbedding,
        )


@default_pooling_type(seq_pooling_type="CLS")
class ErnieEmbeddingModel(BertEmbeddingModel):
    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> ErnieModel:
        return ErnieModel(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights_list = list(weights)
        has_model_prefix = any(name.startswith("model.") for name, _ in weights_list)
        has_ernie_prefix = any(name.startswith("ernie.") for name, _ in weights_list)

        mapper: WeightsMapper | None = None
        if not has_model_prefix:
            if has_ernie_prefix:
                mapper = WeightsMapper(orig_to_new_prefix={"ernie.": "model."})
            else:
                mapper = WeightsMapper(orig_to_new_prefix={"": "model."})
        if mapper is None:
            mapper = _LEGACY_SUFFIX_MAPPER
        else:
            mapper = mapper | _LEGACY_SUFFIX_MAPPER

        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head.", "cls."])
        return loader.load_weights(weights_list, mapper=mapper)


@default_pooling_type(seq_pooling_type="CLS")
class ErnieForSequenceClassification(nn.Module, SupportsCrossEncoding, SupportsQuant):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.num_labels = config.num_labels
        self.ernie = ErniePoolingModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "ernie"),
        )
        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels,
            dtype=vllm_config.model_config.head_dtype,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            pooling=self.ernie.pooler,
            classifier=self.classifier,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.ernie.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights_list = list(weights)
        has_ernie_prefix = any(name.startswith("ernie.") for name, _ in weights_list)
        has_bert_prefix = any(name.startswith("bert.") for name, _ in weights_list)

        mapper: WeightsMapper | None = None
        if has_bert_prefix and not has_ernie_prefix:
            mapper = WeightsMapper(orig_to_new_prefix={"bert.": "ernie."})
        if mapper is None:
            mapper = _LEGACY_SUFFIX_MAPPER
        else:
            mapper = mapper | _LEGACY_SUFFIX_MAPPER

        loader = AutoWeightsLoader(self, skip_prefixes=["cls.", "lm_head."])
        loaded_params = loader.load_weights(weights_list, mapper=mapper)
        if loaded_params is not None:
            loaded_params.update({"classifier.weight", "classifier.bias"})
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is not None:
            assert self.ernie.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
            assert input_ids is not None
            _encode_token_type_ids(input_ids, token_type_ids)

        return self.ernie(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )


@attn_type("encoder_only")
@default_pooling_type(tok_pooling_type="ALL")
class ErnieForTokenClassification(nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.head_dtype = vllm_config.model_config.head_dtype
        self.num_labels = config.num_labels
        self.ernie = ErnieModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "ernie"),
        )
        self.classifier = nn.Linear(
            config.hidden_size, config.num_labels, dtype=self.head_dtype
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = pooler_for_token_classify(pooler_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.ernie.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights_list = list(weights)
        has_ernie_prefix = any(name.startswith("ernie.") for name, _ in weights_list)
        has_bert_prefix = any(name.startswith("bert.") for name, _ in weights_list)

        mapper: WeightsMapper | None = None
        if has_bert_prefix and not has_ernie_prefix:
            mapper = WeightsMapper(orig_to_new_prefix={"bert.": "ernie."})
        if mapper is None:
            mapper = _LEGACY_SUFFIX_MAPPER
        else:
            mapper = mapper | _LEGACY_SUFFIX_MAPPER

        loader = AutoWeightsLoader(self, skip_prefixes=["cls.", "lm_head."])
        loaded_params = loader.load_weights(weights_list, mapper=mapper)
        if loaded_params is not None:
            loaded_params.update({"classifier.weight", "classifier.bias"})
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is not None:
            assert self.ernie.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
            assert input_ids is not None
            _encode_token_type_ids(input_ids, token_type_ids)

        hidden_states = self.ernie(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

        hidden_states = hidden_states.to(self.head_dtype)
        return self.classifier(hidden_states)
