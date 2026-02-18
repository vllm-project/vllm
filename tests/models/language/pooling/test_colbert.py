# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ColBERT late interaction scoring.

Tests are parametrized across multiple ColBERT backbones to ensure the
generic ColBERT support works with different encoder architectures.
"""

import pytest
import torch

from vllm.entrypoints.pooling.score.utils import compute_maxsim_score

# -----------------------------------------------------------------------
# Model definitions: (model_name, colbert_dim, extra vllm_runner kwargs)
# -----------------------------------------------------------------------
COLBERT_MODELS = {
    "bert": {
        "model": "answerdotai/answerai-colbert-small-v1",
        "colbert_dim": 96,
        "max_model_len": 512,
        "extra_kwargs": {},
    },
    "modernbert": {
        "model": "lightonai/GTE-ModernColBERT-v1",
        "colbert_dim": 128,
        "max_model_len": 299,
        "extra_kwargs": {
            "hf_overrides": {
                "architectures": ["ColBERTModernBertModel"],
            },
        },
    },
    "jina": {
        "model": "jinaai/jina-colbert-v2",
        "colbert_dim": 128,
        "max_model_len": 8192,
        "extra_kwargs": {
            "hf_overrides": {
                "architectures": ["ColBERTJinaRobertaModel"],
            },
        },
    },
}

TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]

DTYPE = "half"


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture(params=list(COLBERT_MODELS.keys()), scope="module")
def colbert_spec(request):
    """Return the model spec dict for the current parametrization."""
    return COLBERT_MODELS[request.param]


@pytest.fixture(scope="module")
def colbert_model_name(colbert_spec):
    return colbert_spec["model"]


@pytest.fixture(scope="module")
def colbert_dim(colbert_spec):
    return colbert_spec["colbert_dim"]


@pytest.fixture(scope="module")
def colbert_max_model_len(colbert_spec):
    return colbert_spec["max_model_len"]


@pytest.fixture(scope="module")
def colbert_extra_kwargs(colbert_spec):
    return colbert_spec["extra_kwargs"]


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


def test_colbert_token_embed(
    vllm_runner,
    colbert_model_name,
    colbert_dim,
    colbert_max_model_len,
    colbert_extra_kwargs,
):
    """Test that ColBERT model produces token embeddings."""
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=colbert_max_model_len,
        enforce_eager=True,
        **colbert_extra_kwargs,
    ) as vllm_model:
        outputs = vllm_model.token_embed([TEXTS_1[0]])

        assert len(outputs) == 1
        emb = torch.tensor(outputs[0])
        assert emb.dim() == 2
        assert emb.shape[1] == colbert_dim
        assert emb.shape[0] > 1


def test_colbert_late_interaction_1_to_1(
    vllm_runner,
    colbert_model_name,
    colbert_max_model_len,
    colbert_extra_kwargs,
):
    """Test ColBERT late interaction scoring with 1:1 query-document pair."""
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=colbert_max_model_len,
        enforce_eager=True,
        **colbert_extra_kwargs,
    ) as vllm_model:
        q_outputs = vllm_model.token_embed([TEXTS_1[0]])
        d_outputs = vllm_model.token_embed([TEXTS_2[0]])

        q_emb = torch.tensor(q_outputs[0])
        d_emb = torch.tensor(d_outputs[0])

        manual_score = compute_maxsim_score(q_emb, d_emb).item()

        vllm_scores = vllm_model.score(TEXTS_1[0], TEXTS_2[0])

        assert len(vllm_scores) == 1
        assert vllm_scores[0] == pytest.approx(manual_score, rel=0.01)


def test_colbert_late_interaction_1_to_N(
    vllm_runner,
    colbert_model_name,
    colbert_max_model_len,
    colbert_extra_kwargs,
):
    """Test ColBERT late interaction scoring with 1:N query-documents."""
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=colbert_max_model_len,
        enforce_eager=True,
        **colbert_extra_kwargs,
    ) as vllm_model:
        q_outputs = vllm_model.token_embed([TEXTS_1[0]])
        d_outputs = vllm_model.token_embed(TEXTS_2)

        q_emb = torch.tensor(q_outputs[0])

        manual_scores = []
        for d_out in d_outputs:
            d_emb = torch.tensor(d_out)
            manual_scores.append(compute_maxsim_score(q_emb, d_emb).item())

        vllm_scores = vllm_model.score(TEXTS_1[0], TEXTS_2)

        assert len(vllm_scores) == 2
        for i in range(2):
            assert vllm_scores[i] == pytest.approx(manual_scores[i], rel=0.01)


def test_colbert_late_interaction_N_to_N(
    vllm_runner,
    colbert_model_name,
    colbert_max_model_len,
    colbert_extra_kwargs,
):
    """Test ColBERT late interaction scoring with N:N query-documents."""
    with vllm_runner(
        colbert_model_name,
        runner="pooling",
        dtype=DTYPE,
        max_model_len=colbert_max_model_len,
        enforce_eager=True,
        **colbert_extra_kwargs,
    ) as vllm_model:
        q_outputs = vllm_model.token_embed(TEXTS_1)
        d_outputs = vllm_model.token_embed(TEXTS_2)

        manual_scores = []
        for q_out, d_out in zip(q_outputs, d_outputs):
            q_emb = torch.tensor(q_out)
            d_emb = torch.tensor(d_out)
            manual_scores.append(compute_maxsim_score(q_emb, d_emb).item())

        vllm_scores = vllm_model.score(TEXTS_1, TEXTS_2)

        assert len(vllm_scores) == 2
        for i in range(2):
            assert vllm_scores[i] == pytest.approx(manual_scores[i], rel=0.01)


def test_colbert_relevance_ordering(
    vllm_runner,
    colbert_model_name,
    colbert_max_model_len,
    colbert_extra_kwargs,
):
    """Test that ColBERT scores relevant documents higher than irrelevant."""
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
        max_model_len=colbert_max_model_len,
        enforce_eager=True,
        **colbert_extra_kwargs,
    ) as vllm_model:
        scores = vllm_model.score(query, documents)

        assert len(scores) == 3
        assert scores[0] > scores[1], "ML doc should score higher than Python doc"
        assert scores[2] > scores[1], "DL doc should score higher than Python doc"


def test_colbert_embed_not_supported(
    vllm_runner,
    colbert_model_name,
    colbert_max_model_len,
    colbert_extra_kwargs,
):
    """Test that ColBERT model does not support 'embed' task."""
    with (
        vllm_runner(
            colbert_model_name,
            runner="pooling",
            dtype=DTYPE,
            max_model_len=colbert_max_model_len,
            enforce_eager=True,
            **colbert_extra_kwargs,
        ) as vllm_model,
        pytest.raises(ValueError, match="Embedding API is not supported"),
    ):
        vllm_model.embed([TEXTS_1[0]])


# -----------------------------------------------------------------------
# Per-model HuggingFace comparison tests
# -----------------------------------------------------------------------


def _assert_embeddings_close(vllm_outputs, hf_embeddings):
    """Assert that vLLM and HuggingFace embeddings match."""
    for i, (hf_emb, vllm_out) in enumerate(zip(hf_embeddings, vllm_outputs)):
        vllm_emb = torch.tensor(vllm_out).float()

        assert hf_emb.shape == vllm_emb.shape, (
            f"Shape mismatch for text {i}: HF {hf_emb.shape} vs vLLM {vllm_emb.shape}"
        )

        torch.testing.assert_close(
            vllm_emb,
            hf_emb,
            rtol=1e-2,
            atol=1e-2,
            msg=f"Embedding mismatch for text {i}",
        )


def test_colbert_hf_comparison_bert(vllm_runner):
    """Test that vLLM ColBERT produces same embeddings as HuggingFace (BERT)."""
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, BertModel

    model_name = COLBERT_MODELS["bert"]["model"]
    test_texts = [TEXTS_1[0], TEXTS_2[0]]

    with vllm_runner(
        model_name,
        runner="pooling",
        dtype="float32",
        max_model_len=512,
        enforce_eager=True,
    ) as vllm_model:
        vllm_outputs = vllm_model.token_embed(test_texts)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_bert = BertModel.from_pretrained(model_name)
    hf_bert.eval()

    weights_path = hf_hub_download(model_name, filename="model.safetensors")
    weights = load_file(weights_path)
    linear_weight = weights["linear.weight"]  # [96, 384]

    hf_embeddings = []
    for text in test_texts:
        inputs = hf_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_bert(**inputs)
            hidden_states = outputs.last_hidden_state
            token_emb = F.linear(hidden_states, linear_weight)
            token_emb = F.normalize(token_emb, p=2, dim=-1)
            hf_embeddings.append(token_emb.squeeze(0).float())

    _assert_embeddings_close(vllm_outputs, hf_embeddings)


def test_colbert_hf_comparison_modernbert(vllm_runner):
    """Test that vLLM ColBERT produces same embeddings as HuggingFace
    (ModernBERT)."""
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    spec = COLBERT_MODELS["modernbert"]
    model_name = spec["model"]
    test_texts = [TEXTS_1[0], TEXTS_2[0]]

    with vllm_runner(
        model_name,
        runner="pooling",
        dtype="float32",
        max_model_len=spec["max_model_len"],
        enforce_eager=True,
        **spec["extra_kwargs"],
    ) as vllm_model:
        vllm_outputs = vllm_model.token_embed(test_texts)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModel.from_pretrained(model_name)
    hf_model.eval()

    # Load projection from sentence-transformers 1_Dense layer
    dense_path = hf_hub_download(model_name, filename="1_Dense/model.safetensors")
    dense_weights = load_file(dense_path)
    linear_weight = dense_weights["linear.weight"]  # [128, 768]

    hf_embeddings = []
    for text in test_texts:
        inputs = hf_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs)
            hidden_states = outputs.last_hidden_state
            token_emb = F.linear(hidden_states, linear_weight)
            token_emb = F.normalize(token_emb, p=2, dim=-1)
            hf_embeddings.append(token_emb.squeeze(0).float())

    _assert_embeddings_close(vllm_outputs, hf_embeddings)


def test_colbert_hf_comparison_jina(vllm_runner):
    """Test that vLLM ColBERT produces same embeddings as HuggingFace
    (Jina XLM-RoBERTa)."""
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    spec = COLBERT_MODELS["jina"]
    model_name = spec["model"]
    test_texts = [TEXTS_1[0], TEXTS_2[0]]

    with vllm_runner(
        model_name,
        runner="pooling",
        dtype="float32",
        max_model_len=spec["max_model_len"],
        enforce_eager=True,
        **spec["extra_kwargs"],
    ) as vllm_model:
        vllm_outputs = vllm_model.token_embed(test_texts)

    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    hf_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    hf_model.eval()

    # Load projection from main checkpoint
    weights_path = hf_hub_download(model_name, filename="model.safetensors")
    weights = load_file(weights_path)
    linear_weight = weights["linear.weight"]  # [128, 1024]

    hf_embeddings = []
    for text in test_texts:
        inputs = hf_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs)
            hidden_states = outputs.last_hidden_state
            token_emb = F.linear(hidden_states.float(), linear_weight.float())
            token_emb = F.normalize(token_emb, p=2, dim=-1)
            hf_embeddings.append(token_emb.squeeze(0).float())

    _assert_embeddings_close(vllm_outputs, hf_embeddings)
