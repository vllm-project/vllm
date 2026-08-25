# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconcile word embedding tying with what the checkpoint actually contains."""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

logger = init_logger(__name__)


@dataclass
class _UntiedLMHead:
    name: str
    lm_head: ParallelLMHead
    embed_tokens: VocabParallelEmbedding

    @property
    def weight_name(self) -> str:
        return f"{self.name}.weight"

    def tie(self, model: nn.Module) -> None:
        parent_name, _, attr = self.name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, self.lm_head.tie_weights(self.embed_tokens))


def _get_untied_lm_head(model: nn.Module) -> _UntiedLMHead | None:
    """Locate an `lm_head` that could be tied to the input embeddings.

    Returns `None` unless the model has exactly one of each, so that models with
    several heads (such as MTP) or with already tied weights are left alone.
    Both are found by type rather than by name, because `get_input_embeddings`
    takes different arguments on multimodal models and the `lm_head` of a
    multimodal model is nested inside its language model.

    A quantized `lm_head` is also left alone. Its weights may be packed under
    another name, and online quantization creates them after loading, so
    neither their contents nor whether they were loaded can be established here.
    Note that this is a property of the layer, not of the model: most quantized
    models leave the `lm_head` and the input embeddings unquantized.
    """
    heads = list[tuple[str, ParallelLMHead]]()
    embeddings = list[VocabParallelEmbedding]()
    for name, module in model.named_modules():
        if not isinstance(module, VocabParallelEmbedding):
            continue
        if not isinstance(module.quant_method, UnquantizedEmbeddingMethod):
            continue
        if isinstance(module, ParallelLMHead):
            heads.append((name, module))
        else:
            embeddings.append(module)

    if len(heads) != 1 or len(embeddings) != 1:
        return None

    (name, lm_head), embed_tokens = heads[0], embeddings[0]
    if lm_head.weight.shape != embed_tokens.weight.shape:
        return None
    return _UntiedLMHead(name, lm_head, embed_tokens)


def maybe_retie_word_embeddings(model: nn.Module, model_config: ModelConfig) -> None:
    """Re-tie word embeddings that
    [ModelConfig.maybe_untie_word_embeddings][vllm.config.ModelConfig.maybe_untie_word_embeddings]
    untied, if the loaded `lm_head` turned out to be identical to the input embeddings
    after all.

    Checkpoints produced by quantization or fine-tuning tooling often keep a redundant
    copy of the tied `lm_head`. Sharing the storage again reclaims the memory it would
    otherwise cost."""
    if not model_config.word_embeddings_untied_by_checkpoint:
        return
    if (untied := _get_untied_lm_head(model)) is None:
        return

    # On device, torch.equal segfaults on ROCm when sleep mode is enabled
    if not torch.equal(untied.lm_head.weight.cpu(), untied.embed_tokens.weight.cpu()):
        logger.warning(
            "The config for %s says the word embeddings are tied, but the checkpoint "
            "contains a different %s, which has been used instead of tying. "
            "Set `tie_word_embeddings=False` in the config to silence this warning.",
            model_config.model,
            untied.weight_name,
        )
        return

    logger.debug("Re-tying %s, which is identical to the input embeddings", untied.name)
    untied.tie(model)
