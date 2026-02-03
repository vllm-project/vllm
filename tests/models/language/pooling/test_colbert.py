# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ColBERT late interaction scoring."""

import pytest
import torch

from vllm.entrypoints.pooling.score.utils import compute_maxsim_score

# ColBERT model - using answerai-colbert-small-v1 as it's a smaller model
# suitable for testing (based on BERT-base)
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
COLBERT_DIM = 96  # This model uses 96-dimensional output

TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]

DTYPE = "half"


@pytest.fixture(scope="module")
def colbert_model_name():
    return COLBERT_MODEL


def test_colbert_token_embed(vllm_runner, colbert_model_name):
    """Test that ColBERT model produces token embeddings."""
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        hf_overrides={"architectures": ["ColBERTModel"], "dim": COLBERT_DIM},
    ) as vllm_model:
        # Get token embeddings for a single text
        outputs = vllm_model.token_embed([TEXTS_1[0]])

        assert len(outputs) == 1
        # Token embeddings should be 2D: [num_tokens, colbert_dim]
        emb = torch.tensor(outputs[0])
        assert emb.dim() == 2
        assert emb.shape[1] == COLBERT_DIM
        # Should have at least a few tokens
        assert emb.shape[0] > 1


def test_colbert_late_interaction_1_to_1(vllm_runner, colbert_model_name):
    """Test ColBERT late interaction scoring with 1:1 query-document pair."""
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        hf_overrides={"architectures": ["ColBERTModel"], "dim": COLBERT_DIM},
    ) as vllm_model:
        # Get token embeddings
        q_outputs = vllm_model.token_embed([TEXTS_1[0]])
        d_outputs = vllm_model.token_embed([TEXTS_2[0]])

        q_emb = torch.tensor(q_outputs[0])
        d_emb = torch.tensor(d_outputs[0])

        # Compute MaxSim manually
        manual_score = compute_maxsim_score(q_emb, d_emb).item()

        # Use the score API (which should internally use _late_interaction_score)
        vllm_scores = vllm_model.score(TEXTS_1[0], TEXTS_2[0])

        assert len(vllm_scores) == 1
        assert vllm_scores[0] == pytest.approx(manual_score, rel=0.01)


def test_colbert_late_interaction_1_to_N(vllm_runner, colbert_model_name):
    """Test ColBERT late interaction scoring with 1:N query-documents."""
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        hf_overrides={"architectures": ["ColBERTModel"], "dim": COLBERT_DIM},
    ) as vllm_model:
        # Get token embeddings
        q_outputs = vllm_model.token_embed([TEXTS_1[0]])
        d_outputs = vllm_model.token_embed(TEXTS_2)

        q_emb = torch.tensor(q_outputs[0])

        # Compute MaxSim manually for each document
        manual_scores = []
        for d_out in d_outputs:
            d_emb = torch.tensor(d_out)
            manual_scores.append(compute_maxsim_score(q_emb, d_emb).item())

        # Use the score API
        vllm_scores = vllm_model.score(TEXTS_1[0], TEXTS_2)

        assert len(vllm_scores) == 2
        for i in range(2):
            assert vllm_scores[i] == pytest.approx(manual_scores[i], rel=0.01)


def test_colbert_late_interaction_N_to_N(vllm_runner, colbert_model_name):
    """Test ColBERT late interaction scoring with N:N query-documents."""
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        hf_overrides={"architectures": ["ColBERTModel"], "dim": COLBERT_DIM},
    ) as vllm_model:
        # Get token embeddings
        q_outputs = vllm_model.token_embed(TEXTS_1)
        d_outputs = vllm_model.token_embed(TEXTS_2)

        # Compute MaxSim manually for each pair
        manual_scores = []
        for q_out, d_out in zip(q_outputs, d_outputs):
            q_emb = torch.tensor(q_out)
            d_emb = torch.tensor(d_out)
            manual_scores.append(compute_maxsim_score(q_emb, d_emb).item())

        # Use the score API
        vllm_scores = vllm_model.score(TEXTS_1, TEXTS_2)

        assert len(vllm_scores) == 2
        for i in range(2):
            assert vllm_scores[i] == pytest.approx(manual_scores[i], rel=0.01)


def test_colbert_relevance_ordering(vllm_runner, colbert_model_name):
    """Test that ColBERT scores relevant documents higher than irrelevant ones."""
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks.",
    ]

    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        hf_overrides={"architectures": ["ColBERTModel"], "dim": COLBERT_DIM},
    ) as vllm_model:
        scores = vllm_model.score(query, documents)

        assert len(scores) == 3
        # ML-related documents should score higher than unrelated Python doc
        # Document 0 (ML definition) should be most relevant
        # Document 2 (Deep learning) should also be relevant
        # Document 1 (Python) should be least relevant
        assert scores[0] > scores[1], "ML doc should score higher than Python doc"
        assert scores[2] > scores[1], "DL doc should score higher than Python doc"


def test_colbert_embed_not_supported(vllm_runner, colbert_model_name):
    """Test that ColBERT model does not support 'embed' task."""
    with (
        vllm_runner(
            colbert_model_name,
            runner="pooling",
            dtype=DTYPE,
            max_model_len=512,
            enforce_eager=True,
            enable_prefix_caching=False,
            hf_overrides={"architectures": ["ColBERTModel"], "dim": COLBERT_DIM},
        ) as vllm_model,
        pytest.raises(ValueError, match="Task embed is not supported"),
    ):
        vllm_model.embed([TEXTS_1[0]])


def test_colbert_hf_comparison(vllm_runner, colbert_model_name):
    """Test that vLLM ColBERT produces same embeddings as HuggingFace."""
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, BertModel

    test_texts = [TEXTS_1[0], TEXTS_2[0]]

    # Get vLLM embeddings first (to avoid GPU memory contention)
    # Use fp32 to match HuggingFace default precision for fair comparison
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype="float32",
        max_model_len=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        hf_overrides={"architectures": ["ColBERTModel"], "dim": COLBERT_DIM},
    ) as vllm_model:
        vllm_outputs = vllm_model.token_embed(test_texts)

    # Get HuggingFace reference embeddings on CPU
    # Load the base BERT model and manually apply the ColBERT linear projection
    hf_tokenizer = AutoTokenizer.from_pretrained(colbert_model_name)
    hf_bert = BertModel.from_pretrained(colbert_model_name)
    hf_bert.eval()

    # Load the ColBERT linear weights from safetensors
    weights_path = hf_hub_download(colbert_model_name, filename="model.safetensors")
    weights = load_file(weights_path)
    linear_weight = weights["linear.weight"]  # [96, 384]

    hf_embeddings = []
    for text in test_texts:
        inputs = hf_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_bert(**inputs)
            # Get last hidden state: [1, seq_len, 384]
            hidden_states = outputs.last_hidden_state
            # Apply ColBERT linear projection: [1, seq_len, 96]
            token_emb = F.linear(hidden_states, linear_weight)
            # L2 normalize
            token_emb = F.normalize(token_emb, p=2, dim=-1)
            hf_embeddings.append(token_emb.squeeze(0).float())

    # Compare embeddings
    for i, (hf_emb, vllm_out) in enumerate(zip(hf_embeddings, vllm_outputs)):
        vllm_emb = torch.tensor(vllm_out).float()

        # Print first few components for debugging
        print(f"\n=== Text {i}: '{test_texts[i][:30]}...' ===")
        print(f"HF shape: {hf_emb.shape}, vLLM shape: {vllm_emb.shape}")
        print(f"HF first token, first 10 dims:   {hf_emb[0, :10].tolist()}")
        print(f"vLLM first token, first 10 dims: {vllm_emb[0, :10].tolist()}")
        print(f"HF last token, first 10 dims:    {hf_emb[-1, :10].tolist()}")
        print(f"vLLM last token, first 10 dims:  {vllm_emb[-1, :10].tolist()}")

        # Should have same shape
        assert hf_emb.shape == vllm_emb.shape, (
            f"Shape mismatch for text {i}: HF {hf_emb.shape} vs vLLM {vllm_emb.shape}"
        )

        # Should have same values (with tolerance for fp16)
        torch.testing.assert_close(
            vllm_emb,
            hf_emb,
            rtol=1e-2,
            atol=1e-2,
            msg=f"Embedding mismatch for text {i}",
        )
