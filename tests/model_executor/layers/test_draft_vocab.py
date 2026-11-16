# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.draft_vocab import DraftVocab
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

VOCAB = 200
HIDDEN = 32
DRAFT_IDS = [0, 3, 7, 50, 51, 128, 130, 199]


def _head(vocab: int = VOCAB, device: str = "cpu") -> ParallelLMHead:
    head = ParallelLMHead(vocab, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen))
    head = head.to(device)
    head.quant_method.process_weights_after_loading(head)
    return head


def _hidden(n: int = 64, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    return torch.randn(n, HIDDEN, generator=gen).to(device)


@pytest.mark.cpu_test
def test_pruned_head_maps_draft_ids_to_target_ids(default_vllm_config):
    """EAGLE3/DFlash pruned heads: logits land on their d2t target ids."""
    head = _head(len(DRAFT_IDS))
    draft_vocab = DraftVocab(LogitsProcessor(len(DRAFT_IDS)), VOCAB)
    d2t = draft_vocab.draft_id_to_target_id
    assert d2t is not None
    d2t.data.copy_(torch.tensor(DRAFT_IDS) - torch.arange(len(DRAFT_IDS)))

    hidden = _hidden()
    logits = draft_vocab.compute_logits(head, hidden)
    assert torch.equal(logits[:, DRAFT_IDS], draft_vocab.logits_processor(head, hidden))
    assert torch.isinf(logits).sum() == hidden.shape[0] * (VOCAB - len(DRAFT_IDS))
    assert torch.equal(draft_vocab.get_top_tokens(head, hidden), logits.argmax(-1))


@pytest.mark.cpu_test
def test_dspark_reads_d2t_of_subclasses_that_skip_init(default_vllm_config):
    """Gemma4, MiniMax-M3 and Muse Spark DSpark drafters skip the parent
    __init__ and set only `draft_id_to_target_id`; inherited methods use it."""
    model = Qwen3DSparkForCausalLM.__new__(Qwen3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = _head(len(DRAFT_IDS))
    model.logits_processor = LogitsProcessor(len(DRAFT_IDS))
    model.config = SimpleNamespace(vocab_size=VOCAB)
    model.draft_id_to_target_id = nn.Parameter(
        torch.tensor(DRAFT_IDS) - torch.arange(len(DRAFT_IDS)), requires_grad=False
    )

    ids = torch.arange(len(DRAFT_IDS))
    assert model.map_draft_to_target(ids).tolist() == DRAFT_IDS
    hidden = _hidden()
    draft_logits = model.logits_processor(model.lm_head, hidden)
    logits = model.compute_logits(hidden)
    assert logits is not None
    assert torch.equal(logits[:, DRAFT_IDS], draft_logits)
