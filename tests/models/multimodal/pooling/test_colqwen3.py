# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ColQwen3 multimodal late interaction retrieval model."""

import pytest
import torch

MODEL_NAME = "TomoroAI/tomoro-colqwen3-embed-4b"


def test_colqwen3_registry():
    """Test that ColQwen3 is properly registered and has correct flags."""
    from vllm.model_executor.models.colqwen3_vl import ColQwen3ForRetrieval

    assert ColQwen3ForRetrieval.is_pooling_model is True
    assert ColQwen3ForRetrieval.supports_late_interaction is True
    assert ColQwen3ForRetrieval.default_tok_pooling_type == "ALL"


@pytest.fixture(scope="module")
def vllm_model():
    from vllm import LLM

    model = LLM(
        model=MODEL_NAME,
        runner="pooling",
        enforce_eager=True,
        max_model_len=512,
    )
    yield model
    del model


PROMPTS = [
    "What is machine learning?",
    "The capital of France is Paris.",
]


def test_colqwen3_text_embeddings(vllm_model):
    """Test that vLLM produces non-trivial per-token embeddings."""
    outputs = vllm_model.encode(PROMPTS, pooling_task="token_embed")

    for prompt, output in zip(PROMPTS, outputs):
        emb = output.outputs.data
        # Should be 2D: (num_tokens, embed_dim)
        assert emb.dim() == 2, f"Expected 2D tensor, got {emb.dim()}D"
        # embed_dim should be 320 for ColQwen3
        assert emb.shape[-1] == 320, f"Expected dim=320, got {emb.shape[-1]}"
        # Should have at least as many tokens as words
        assert emb.shape[0] >= len(prompt.split()), (
            f"Too few tokens: {emb.shape[0]} for '{prompt}'"
        )
        # Embeddings should be L2 normalized (norm ≈ 1.0 for non-padding)
        norms = emb.norm(dim=-1)
        nonzero_mask = norms > 0.1
        if nonzero_mask.any():
            active_norms = norms[nonzero_mask]
            assert torch.allclose(
                active_norms,
                torch.ones_like(active_norms),
                atol=0.05,
            ), f"Embeddings not L2 normalized: norms={active_norms[:5]}"


def test_colqwen3_embedding_similarity(vllm_model):
    """Test that similar queries produce higher MaxSim scores."""
    query = "What is deep learning?"
    docs = [
        "Deep learning is a subset of machine learning using neural networks.",
        "The weather in London is often rainy and cold.",
    ]

    q_out = vllm_model.encode([query], pooling_task="token_embed")
    d_outs = vllm_model.encode(docs, pooling_task="token_embed")

    q_emb = q_out[0].outputs.data

    scores = []
    for d_out in d_outs:
        d_emb = d_out.outputs.data
        # MaxSim: sum of max similarities per query token
        sim = torch.matmul(q_emb, d_emb.T)
        score = sim.amax(dim=-1).sum().item()
        scores.append(score)

    # The relevant document should score higher
    assert scores[0] > scores[1], (
        f"Relevant doc scored {scores[0]:.2f} <= irrelevant {scores[1]:.2f}"
    )
