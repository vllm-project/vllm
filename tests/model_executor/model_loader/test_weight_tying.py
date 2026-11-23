# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    finish_direct_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_tying import maybe_retie_word_embeddings

VOCAB_SIZE = 16
HIDDEN_SIZE = 4


class UntiedModel(nn.Module):
    """Nests the head like a multimodal model, which is the harder case."""

    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = VocabParallelEmbedding(
            VOCAB_SIZE, HIDDEN_SIZE
        )
        self.language_model.lm_head = ParallelLMHead(VOCAB_SIZE, HIDDEN_SIZE)
        self.embed_tokens.weight.data.fill_(1.0)
        self.lm_head.weight.data.fill_(1.0)

    @property
    def embed_tokens(self) -> VocabParallelEmbedding:
        return self.language_model.model.embed_tokens

    @property
    def lm_head(self) -> ParallelLMHead:
        return self.language_model.lm_head


def make_model_config(untied_by_checkpoint=False):
    return SimpleNamespace(
        model="dummy-model",
        word_embeddings_untied_by_checkpoint=untied_by_checkpoint,
        quantization=None,
    )


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize("identical", [True, False])
def test_retie_only_when_identical(identical: bool):
    """A redundant copy of a tied lm_head is shared again to reclaim memory."""
    model = UntiedModel()
    if not identical:
        model.lm_head.weight.data.fill_(2.0)

    maybe_retie_word_embeddings(model, make_model_config(untied_by_checkpoint=True))

    assert (model.lm_head.weight is model.embed_tokens.weight) is identical
    if not identical:
        assert torch.all(model.lm_head.weight == 2.0)


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_quantized_lm_head_is_left_alone():
    """A quantized head may store its weights packed under another name."""
    model = UntiedModel()
    model.lm_head.quant_method = SimpleNamespace()

    maybe_retie_word_embeddings(model, make_model_config(untied_by_checkpoint=True))

    assert model.lm_head.weight is not model.embed_tokens.weight


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_no_retie_without_checkpoint_override():
    """Word embeddings the config genuinely unties are left alone."""
    model = UntiedModel()

    maybe_retie_word_embeddings(model, make_model_config())

    assert model.lm_head.weight is not model.embed_tokens.weight


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_direct_reload_keeps_cold_start_tying(monkeypatch):
    """A direct reload does not re-tie: once lm_head is the input embeddings,
    every later update's lm_head would be dropped."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    model_config = make_model_config(untied_by_checkpoint=True)
    device = torch.device("cpu")
    model = UntiedModel()
    model.lm_head.weight.data.fill_(2.0)
    process_weights_after_loading(model, model_config, device)
    assert model.lm_head.weight is not model.embed_tokens.weight

    # The first update's lm_head equals the input embeddings, the second's does not.
    for lm_head_value in (1.0, 3.0):
        begin_direct_reload(model)
        model.embed_tokens.weight.data.fill_(1.0)
        model.lm_head.weight.data.fill_(lm_head_value)
        finish_direct_reload(model, model_config, device)

        assert model.lm_head.weight is not model.embed_tokens.weight
        assert torch.all(model.lm_head.weight == lm_head_value)
        assert torch.all(model.embed_tokens.weight == 1.0)
