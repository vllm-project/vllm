# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch

from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.model_executor.models.utils import AutoWeightsLoader

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Online convert ForCausalLM into ForSequenceClassification model.
# - from_2_way_softmax:
#   - Qwen3ForCausalLM
#     - Qwen3-Reranker
#   - Qwen2ForCausalLM
#     - mxbai-rerank-v2


class SequenceClassificationConfig(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config
        method = getattr(config, "method", None)
        tokens = getattr(config, "classifier_from_token", None)

        if method is None:
            return

        assert tokens is not None
        assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"

        if method == "from_2_way_softmax":
            assert len(tokens) == 2
            config.num_labels = 1
        else:
            config.num_labels = len(tokens)


def load_weights_using_from_2_way_softmax(
        model, weights: Iterable[tuple[str, torch.Tensor]]):
    # refer to https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead)

    model_config = model.vllm_config.model_config
    tokens = getattr(model.config, "classifier_from_token", [])
    tokens = cast(list[int], tokens)
    assert len(tokens) == 2

    device = model.score.weight.device

    if model.config.tie_word_embeddings:
        model.lm_head = model.model.embed_tokens
    else:
        model.lm_head = ParallelLMHead(model.config.vocab_size,
                                       model.config.hidden_size,
                                       quant_config=model.quant_config)

    loader = AutoWeightsLoader(model)
    loaded_weights = loader.load_weights(weights)

    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(model_config.tokenizer,
                              revision=model_config.tokenizer_revision,
                              tokenizer_mode=model_config.tokenizer_mode,
                              trust_remote_code=model_config.trust_remote_code)

    false_id = tokenizer.convert_tokens_to_ids(tokens[0])
    true_id = tokenizer.convert_tokens_to_ids(tokens[1])
    weight = model.lm_head.weight.data[true_id].to(device).to(
        torch.float32) - model.lm_head.weight.data[false_id].to(device).to(
            torch.float32)
    model.score.weight.data.copy_(weight)

    del model.lm_head
    loaded_weights.add("score.weight")
    loaded_weights.discard("lm_head.weight")
    return loaded_weights


SEQ_CLS_LOAD_METHODS = {
    "from_2_way_softmax": load_weights_using_from_2_way_softmax,
}


def seq_cls_model_loader(model, weights: Iterable[tuple[str, torch.Tensor]]):
    config = model.vllm_config.model_config.hf_config
    method = getattr(config, "method", None)
    assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"
    return SEQ_CLS_LOAD_METHODS[method](model, weights)
