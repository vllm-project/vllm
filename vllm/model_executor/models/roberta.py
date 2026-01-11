# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from torch import nn
from transformers import RobertaConfig

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import CLSPool
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.bert import (
    TOKEN_TYPE_SHIFT,
    BertEmbeddingModel,
    BertModel,
    _decode_token_type_ids,
    _encode_token_type_ids,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .bert_with_rope import BertWithRope, JinaRobertaModel
from .interfaces import SupportsCrossEncoding
from .interfaces_base import default_pooling_type


class RobertaEmbedding(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_type_ids = _decode_token_type_ids(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


# Adapted from transformers
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, model_config: "ModelConfig"):
        super().__init__()
        config = model_config.hf_config
        head_dtype = model_config.head_dtype
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, dtype=head_dtype)
        self.out_proj = nn.Linear(
            config.hidden_size, config.num_labels, dtype=head_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLSPool has already been applied in `pooling`
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


@default_pooling_type(seq_pooling_type="CLS")
class RobertaEmbeddingModel(BertEmbeddingModel):
    """A model that uses Roberta to provide embedding functionalities."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.padding_idx: int = vllm_config.model_config.hf_config.pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Fix Roberta positions here outside of the CUDA graph.
        # Because we need the to extract the sequences from
        # input_ids the control flow is data dependent.
        replace_roberta_positions(
            input_ids=input_ids, position_ids=positions, padding_idx=self.padding_idx
        )

        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def _build_model(
        self, vllm_config: VllmConfig, prefix: str = ""
    ) -> BertModel | BertWithRope:
        hf_config = vllm_config.model_config.hf_config
        kwargs = dict(vllm_config=vllm_config, prefix=prefix)
        if getattr(hf_config, "position_embedding_type", "absolute") == "absolute":
            return BertModel(**kwargs, embedding_class=RobertaEmbedding)
        else:
            return JinaRobertaModel(**kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights_list = list(weights)
        has_roberta_prefix = any(
            name.startswith("roberta.") for name, _ in weights_list
        )
        if has_roberta_prefix:
            # For models with the `roberta.` prefix e.g.
            # `FacebookAI/roberta-base`
            mapper = WeightsMapper(orig_to_new_prefix={"roberta.": "model."})
        else:
            # For models without the `roberta.` prefix e.g.
            # `sentence-transformers/stsb-roberta-base-v2`
            mapper = WeightsMapper(orig_to_new_prefix={"": "model."})

        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head."])
        return loader.load_weights(weights_list, mapper=mapper)


@default_pooling_type(seq_pooling_type="CLS")
class RobertaForSequenceClassification(nn.Module, SupportsCrossEncoding):
    """A model that uses Roberta to provide embedding functionalities.

    This class encapsulates the BertModel and provides an interface for
    embedding operations and customized pooling functions.

    Attributes:
        roberta: An instance of BertModel used for forward operations.
        _pooler: An instance of Pooler used for pooling operations.
    """

    is_pooling_model = True
    jina_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "emb_ln": "embeddings.LayerNorm",
            "layers": "layer",
            "mixer.Wqkv": "attention.self.qkv_proj",
            "mixer.out_proj": "attention.output.dense",
            "norm1": "attention.output.LayerNorm",
            "mlp.fc1": "intermediate.dense",
            "mlp.fc2": "output.dense",
            "norm2": "output.LayerNorm",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.padding_idx: int = vllm_config.model_config.hf_config.pad_token_id

        self.num_labels = config.num_labels
        self.roberta = BertModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "bert"),
            embedding_class=RobertaEmbedding,
        )
        self.classifier = RobertaClassificationHead(vllm_config.model_config)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            pooling=CLSPool(),
            classifier=self.classifier,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.jina_to_vllm_mapper)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.roberta.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        replace_roberta_positions(
            input_ids=input_ids, position_ids=positions, padding_idx=self.padding_idx
        )
        if token_type_ids is not None:
            assert self.roberta.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
            assert input_ids is not None
            _encode_token_type_ids(input_ids, token_type_ids)
        return self.roberta(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )


def replace_roberta_positions(
    input_ids: torch.Tensor, position_ids: torch.Tensor, padding_idx: int
) -> None:
    # Replace position ids because in RoBERTa models
    # they have to start at padding_idx + 1 and ignore
    # existing padding tokens
    # References:
    # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L133
    # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L1669
    # vllm does not use padding tokens, let's make things simpler
    position_ids += padding_idx + 1
