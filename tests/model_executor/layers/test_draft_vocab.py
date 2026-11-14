# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.draft_vocab import (
    DraftVocab,
    load_draft_token_ids,
    load_draft_token_map,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.online.lm_head import (
    refresh_quantized_lm_heads,
    restore_quantized_lm_heads,
    share_source_lm_heads,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

VOCAB = 200
HIDDEN = 32
DRAFT_IDS = [0, 3, 7, 50, 51, 128, 130, 199]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


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


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="quantized rows need CUDA"
)


def _bf16_head(
    monkeypatch, vocab: int, hidden_size: int, cls=ParallelLMHead
) -> VocabParallelEmbedding:
    # Online methods read the model dtype from the current config.
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    return cls(vocab, hidden_size, disable_tp=True)


@requires_cuda
@pytest.mark.parametrize(
    ("quantization", "max_rel_err"), [("fp8", 0.05), ("nvfp4", 0.15)]
)
@pytest.mark.parametrize("extra_listed_id", [False, True])
def test_quantized_rows_track_bf16_rows(
    default_vllm_config,
    dist_init,
    monkeypatch,
    quantization,
    max_rel_err,
    extra_listed_id,
):
    """Quantized rows preserve logical token IDs, including unaligned maps."""
    vocab, hidden_size = 4000, 512
    head = _bf16_head(monkeypatch, vocab, hidden_size)
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(head.weight.shape, generator=gen) * 0.02
    token_ids = torch.arange(0, vocab, 4)
    if extra_listed_id:
        token_ids = torch.cat((token_ids, torch.tensor([1]))).sort().values
        weight = weight.abs()
    head.weight.data.copy_(weight)
    head = head.to("cuda", torch.bfloat16)
    hidden = torch.randn(
        512,
        hidden_size,
        generator=torch.Generator(device="cuda").manual_seed(1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    if extra_listed_id:
        # Dummy zero rows must not win when every real logit is negative.
        hidden[-1].fill_(-1)

    draft_vocab = DraftVocab(LogitsProcessor(vocab), vocab)
    draft_vocab.load_token_map(head, token_ids, 256, quantization)
    out = draft_vocab.compute_logits(head, hidden).float()
    full = LogitsProcessor(vocab)(head, hidden).float()
    kept = out.isfinite()
    assert out.shape == (hidden.shape[0], vocab)
    assert torch.all(kept[:, token_ids])
    assert torch.all(kept.sum(dim=-1) == len(token_ids) + 256)
    if extra_listed_id:
        assert torch.all(out[-1, kept[-1]] < 0)
    assert (out[kept] - full[kept]).norm() / full[kept].norm() < max_rel_err


@requires_cuda
@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_quantized_listed_rows_follow_target_head(
    default_vllm_config, dist_init, monkeypatch, tmp_path, quantization
):
    """Refreshing the drafter's quantized heads re-derives the listed rows from
    the updated target lm_head, in place. The head is a tied input embedding,
    as in Qwen3.5."""
    vocab, hidden_size = 4000, 512
    head = _bf16_head(monkeypatch, vocab, hidden_size, VocabParallelEmbedding)
    head = head.to("cuda", torch.bfloat16)
    head.weight.data.normal_(std=0.02)
    token_ids = list(range(1, vocab, 3))
    token_map = tmp_path / "map.json"
    token_map.write_text(str(token_ids))
    drafter = torch.nn.Module()
    drafter.lm_head = head
    drafter.draft_vocab = DraftVocab(LogitsProcessor(vocab), vocab)
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_token_map=str(token_map),
            draft_token_map_dynamic_rows=0,
            draft_lm_head_quantization=quantization,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(), try_get_generation_config=dict
        ),
    )
    load_draft_token_map(drafter, vllm_config)
    rows = drafter.draft_token_map_head
    assert rows is drafter.draft_vocab.rows
    ptrs = {k: t.data_ptr() for k, t in rows.state_dict().items()}

    # A copy of some rows is never swapped in for the full head.
    share_source_lm_heads(drafter)
    assert drafter.draft_token_map_head is rows
    restore_quantized_lm_heads(drafter, refresh=False)

    head.weight.data.normal_(std=0.02)
    hidden = torch.randn(64, hidden_size, device="cuda", dtype=torch.bfloat16)
    fresh = DraftVocab(LogitsProcessor(vocab), vocab)
    fresh.load_token_map(head, torch.tensor(token_ids), 0, quantization)
    expected = fresh.compute_logits(head, hidden)
    assert not torch.equal(drafter.draft_vocab.compute_logits(head, hidden), expected)

    refresh_quantized_lm_heads(drafter)
    assert torch.equal(drafter.draft_vocab.compute_logits(head, hidden), expected)
    assert {k: t.data_ptr() for k, t in rows.state_dict().items()} == ptrs
