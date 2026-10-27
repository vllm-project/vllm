# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.draft_vocab import DraftVocab, load_draft_token_ids
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

VOCAB = 200
HIDDEN = 32
DRAFT_IDS = [0, 3, 7, 50, 51, 128, 130, 199]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _head(vocab: int = VOCAB, device: str = "cpu") -> ParallelLMHead:
    head = ParallelLMHead(vocab, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen))
    return head.to(device)


def _hidden(n: int = 64, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    return torch.randn(n, HIDDEN, generator=gen).to(device)


@pytest.mark.cpu_test
def test_load_draft_token_ids(tmp_path):
    path = tmp_path / "map.pt"
    torch.save(torch.tensor([9, 2, 9, 4]), path)
    assert load_draft_token_ids(str(path), extra_ids=[7, 2]).tolist() == [2, 4, 7, 9]


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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_rows", [0, 16])
def test_token_map_drafts_over_listed_and_picked_rows(
    default_vllm_config, dist_init, device, num_rows
):
    """Each token drafts over the list plus `num_rows` picked rows, with the
    exact logits of the full head on all of them."""
    head = _head(device=device)
    hidden = _hidden(device=device)
    draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB)
    full = draft_vocab.compute_logits(head, hidden).float()
    draft_vocab.load_token_map(head, torch.tensor(DRAFT_IDS), num_rows)

    logits = draft_vocab.compute_logits(head, hidden).float()
    kept = logits.isfinite()
    assert torch.all(kept[:, DRAFT_IDS])
    assert torch.all(kept.sum(dim=-1) == len(DRAFT_IDS) + num_rows)
    torch.testing.assert_close(logits[kept], full[kept], atol=1e-4, rtol=1e-4)
    assert torch.equal(draft_vocab.get_top_tokens(head, hidden), logits.argmax(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantized rows need CUDA")
@pytest.mark.parametrize(
    ("quantization", "max_rel_err"), [("fp8", 0.05), ("nvfp4", 0.15)]
)
def test_quantized_rows_track_bf16_rows(
    default_vllm_config, dist_init, quantization, max_rel_err
):
    """Listed and dynamic rows in fp8/nvfp4 stay close to the bf16 head."""
    vocab, hidden_size = 4000, 512
    head = ParallelLMHead(vocab, hidden_size, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen) * 0.02)
    head = head.to("cuda", torch.bfloat16)
    hidden = torch.randn(512, hidden_size, device="cuda", dtype=torch.bfloat16)

    draft_vocab = DraftVocab(LogitsProcessor(vocab), vocab)
    draft_vocab.load_token_map(head, torch.arange(0, vocab, 4), 256, quantization)
    out = draft_vocab.compute_logits(head, hidden).float()
    full = LogitsProcessor(vocab)(head, hidden).float()
    kept = out.isfinite()
    assert (out[kept] - full[kept]).norm() / full[kept].norm() < max_rel_err
