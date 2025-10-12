# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.bert import (
    BertMLMHead,
    SPLADESparsePooler,
)

# ---------------------------------------------------------------------
# 1) Functional test: SPLADE formula correctness (no HF download needed)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("B,T,H,V", [(2, 3, 5, 7)])
def test_splade_pooler_matches_reference_formula(B, T, H, V):
    """Ensure SPLADESparsePooler forward() matches the mathematical formula:
    log1p(relu(logits)) -> max over sequence length (after masking)."""
    torch.manual_seed(0)

    # Prepare [B] sequences of shape [T, H]
    hs_list = [torch.randn(T, H) for _ in range(B)]

    # Simulate PoolingMetadata (only required fields)
    prompt_lens = [T, T - 1]
    token_ids = torch.tensor(
        [
            [101, 5, 102],  # Batch 0: [CLS], token, [SEP]
            [101, 6, 6],  # Batch 1: [CLS], token, token (last token ignored)
        ],
        dtype=torch.long,
    )
    meta = types.SimpleNamespace(prompt_lens=prompt_lens, prompt_token_ids=token_ids)

    # MLM head (prefer BertMLMHead, fallback to Linear if unavailable)
    try:
        mlm_head = BertMLMHead(hidden_size=H, vocab_size=V, layer_norm_eps=1e-12)
    except Exception:
        mlm_head = nn.Linear(H, V, bias=True)

    # Forward pass through SPLADE pooler
    pooler = SPLADESparsePooler(mlm_head=mlm_head, pooling="max", remove_cls_sep=True)
    pooled = pooler(hidden_states=hs_list, pooling_metadata=meta)  # list of [V]

    # Basic output checks
    assert isinstance(pooled, list) and len(pooled) == B
    for vec in pooled:
        assert vec.shape == (V,)
        assert torch.isfinite(vec).all()
        assert (vec >= 0).all(), "SPLADE outputs must be non-negative."

    # Reference implementation for comparison
    def ref_one(hs: torch.Tensor, L: int, tid_row: torch.Tensor) -> torch.Tensor:
        keep = torch.ones(L, dtype=torch.bool)
        if L > 0 and tid_row[0].item() == 101:  # remove CLS
            keep[0] = False
        if L > 0 and tid_row[L - 1].item() == 102:  # remove SEP
            keep[L - 1] = False

        valid = hs[:L][keep[:L]]
        if valid.numel() == 0:
            return torch.zeros(V, dtype=torch.float32)

        logits = mlm_head(valid)  # [L', V]
        scores = torch.log1p(torch.relu(logits))  # [L', V]
        return scores.max(dim=0).values.to(torch.float32)

    torch.testing.assert_close(
        pooled[0],
        ref_one(hs_list[0], prompt_lens[0], token_ids[0]),
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        pooled[1],
        ref_one(hs_list[1], prompt_lens[1], token_ids[1]),
        rtol=1e-4,
        atol=1e-4,
    )


# ---------------------------------------------------------------------
# 2) Integration smoke test: end-to-end embedding path wiring
# ---------------------------------------------------------------------


@pytest.mark.cpu_model
def test_bert_splade_sparse_embed_smoke(vllm_runner, monkeypatch):
    """Ensure BertSpladeSparseEmbeddingModel loads and produces sparse embeddings."""
    from transformers import AutoTokenizer

    MODEL_ID = "hf-internal-testing/tiny-random-bert"
    hf_overrides = {"architectures": ["BertSpladeSparseEmbeddingModel"]}

    # Enforce CPU-only execution (optional)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("VLLM_USE_TRITON_FLASH_ATTN", "False")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    vocab_size = tok.vocab_size

    # The embed path should route through SPLADESparsePooler
    with vllm_runner(
        MODEL_ID,
        runner="pooling",
        max_model_len=64,
        hf_overrides=hf_overrides,
    ) as vm:
        outs = vm.embed(["hello world", "splade sparse test"])

        # Basic sanity checks
        assert len(outs) == 2
        assert outs[0].shape[0] == vocab_size
        assert outs[1].shape[0] == vocab_size
        assert np.isfinite(outs[0]).all() and (outs[0] >= 0).all()
        assert np.isfinite(outs[1]).all() and (outs[1] >= 0).all()
