# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.bert import (
    BertMLMHead,
    SPLADESparsePooler,
)

# ---------------------------------------------------------------------
# Functional test: SPLADE formula correctness (no HF download needed)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("B,T,H,V", [(2, 3, 5, 7)])
@torch.inference_mode
def test_splade_pooler_matches_reference_formula(B, T, H, V):
    """Ensure SPLADESparsePooler forward() matches the mathematical formula:
    log1p(relu(logits)) -> max over sequence length (after masking)."""
    torch.manual_seed(0)

    # Prepare [B] sequences of shape [T, H]
    hs_list = [torch.randn(T, H) for _ in range(B)]
    hs_tenser = torch.cat(hs_list)

    # Simulate PoolingMetadata (only required fields)
    prompt_lens = [T, T - 1]
    prompt_lens_tenser = torch.tensor(prompt_lens, dtype=torch.int32)
    token_ids = torch.tensor(
        [
            [101, 5, 102],  # Batch 0: [CLS], token, [SEP]
            [101, 6, 6],  # Batch 1: [CLS], token, token (last token ignored)
        ],
        dtype=torch.long,
    )
    meta = types.SimpleNamespace(
        prompt_lens=prompt_lens_tenser, prompt_token_ids=token_ids
    )

    # MLM head (prefer BertMLMHead, fallback to Linear if unavailable)
    try:
        mlm_head = BertMLMHead(hidden_size=H, vocab_size=V, layer_norm_eps=1e-12)
    except Exception:
        mlm_head = nn.Linear(H, V, bias=True)

    # Forward pass through SPLADE pooler
    pooler = SPLADESparsePooler(mlm_head=mlm_head, pooling="max", remove_cls_sep=True)
    pooled = pooler(hidden_states=hs_tenser, pooling_metadata=meta)  # list of [V]

    # Basic output checks
    assert isinstance(pooled, torch.Tensor) and len(pooled) == B
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
