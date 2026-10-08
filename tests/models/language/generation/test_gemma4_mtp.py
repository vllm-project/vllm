# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gemma4_mtp import (
    Gemma4MTP,
    Gemma4MTPMaskedEmbedder,
)


@pytest.mark.parametrize("H", [256, 1024])
@pytest.mark.parametrize("T", [1, 4, 16])
def test_gemma4_mtp_masked_embedder_select_and_score_and_top_tokens(H: int, T: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton kernel tests")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    vocab_size = 2048
    num_centroids = 4
    top_k = 2

    torch.manual_seed(42)
    embedder = Gemma4MTPMaskedEmbedder(
        hidden_size=H,
        vocab_size=vocab_size,
        num_centroids=num_centroids,
        centroid_intermediate_top_k=top_k,
    ).to(device=device, dtype=dtype)
    embedder.token_ordering.copy_(torch.randperm(vocab_size, device=device))
    embedder.centroids.weight.data.normal_(0.0, 0.02)

    lm_head_weight = torch.randn(vocab_size, H, device=device, dtype=dtype)
    h = torch.randn(T, H, device=device, dtype=dtype)

    logits, selected = embedder._select_and_score(h, lm_head_weight)
    S = embedder.num_selected

    ref_scores = torch.einsum(
        "td,tsd->ts",
        h,
        lm_head_weight[selected.reshape(-1)].view(T, S, H),
    )
    torch.testing.assert_close(logits, ref_scores, atol=1e-2, rtol=1e-2)

    top_tokens = embedder.get_top_tokens(h, lm_head_weight)
    expected_tokens = selected[
        torch.arange(T, device=device), ref_scores.argmax(dim=-1)
    ]
    torch.testing.assert_close(top_tokens, expected_tokens)

    # Full forward pass verification
    full_logits = embedder(h, lm_head_weight)
    assert full_logits.shape == (T, vocab_size)
    assert torch.equal(full_logits.argmax(dim=-1), top_tokens)

    # Empty tensor handling (T=0)
    h_empty = h[:0]
    empty_logits, empty_selected = embedder._select_and_score(h_empty, lm_head_weight)
    assert empty_logits.shape == (0, S)
    assert empty_selected.shape == (0, S)
    assert embedder.get_top_tokens(h_empty, lm_head_weight).shape == (0,)


@pytest.mark.usefixtures("dist_init")
def test_gemma4_mtp_get_top_tokens_masked_and_unmasked():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for test")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    H = 256
    T = 4
    vocab_size = 1024

    torch.manual_seed(42)
    h = torch.randn(T, H, device=device, dtype=dtype)
    lm_head_weight = torch.randn(vocab_size, H, device=device, dtype=dtype)

    mtp = Gemma4MTP.__new__(Gemma4MTP)
    torch.nn.Module.__init__(mtp)
    mtp._stable_full_lm_head_weight = lm_head_weight

    # 1. Masked embedding path (use_ordered_embeddings=True)
    embedder = Gemma4MTPMaskedEmbedder(
        hidden_size=H,
        vocab_size=vocab_size,
        num_centroids=4,
        centroid_intermediate_top_k=2,
    ).to(device=device, dtype=dtype)
    embedder.token_ordering.copy_(torch.arange(vocab_size, device=device))
    embedder.centroids.weight.data.normal_(0.0, 0.02)
    mtp.masked_embedding = embedder

    tokens_masked = mtp.get_top_tokens(h)
    expected_masked = embedder.get_top_tokens(h, lm_head_weight)
    assert torch.equal(tokens_masked, expected_masked)

    # 2. Unmasked embedding path (use_ordered_embeddings=False)
    mtp.masked_embedding = None
    mtp.logits_processor = LogitsProcessor(vocab_size)
    lm_head = ParallelLMHead(vocab_size, H)
    lm_head.to(device=device, dtype=dtype)
    lm_head.weight.data.copy_(lm_head_weight)  # type: ignore[operator]
    mtp.lm_head = lm_head

    tokens_unmasked = mtp.get_top_tokens(h)
    expected_unmasked = torch.matmul(h.float(), lm_head_weight.float().t()).argmax(
        dim=-1
    )
    assert torch.equal(tokens_unmasked, expected_unmasked)


def test_gemma4_proposer_use_local_argmax_reduction(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    def mock_init_greedy(self, *args, **kwargs):
        self._enable_probabilistic_draft_probs = False

    monkeypatch.setattr(SpecDecodeBaseProposer, "__init__", mock_init_greedy)
    proposer = Gemma4Proposer(MagicMock(), torch.device("cpu"))
    assert proposer.use_local_argmax_reduction is True

    def mock_init_prob(self, *args, **kwargs):
        self._enable_probabilistic_draft_probs = True

    monkeypatch.setattr(SpecDecodeBaseProposer, "__init__", mock_init_prob)
    proposer = Gemma4Proposer(MagicMock(), torch.device("cpu"))
    assert proposer.use_local_argmax_reduction is False
