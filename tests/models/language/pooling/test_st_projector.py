# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import pytest
from scipy.spatial.distance import cosine

from ...utils import EmbedModelInfo


def _get_vllm_embeddings(vllm_runner, model_info: EmbedModelInfo,
                         test_texts: list[str]):
    """Get embeddings from vLLM."""
    vllm_extra_kwargs: dict[str, Any] = {}
    if model_info.architecture == "GteNewModel":
        vllm_extra_kwargs["hf_overrides"] = {"architectures": ["GteNewModel"]}

    with vllm_runner(
            model_info.name,
            runner="pooling",
            max_model_len=None,
            trust_remote_code=True,
            **vllm_extra_kwargs,
    ) as vllm_model:
        embeddings = vllm_model.encode(test_texts)

        # Extract tensor/numpy data
        data = []
        for emb in embeddings:
            if hasattr(emb, "outputs"):
                data.append(emb.outputs.data.cpu().numpy())
            else:
                data.append(emb.cpu().numpy() if hasattr(emb, "cpu") else emb)
        return np.array(data)


def _get_hf_embeddings(hf_runner, model_info: EmbedModelInfo,
                       test_texts: list[str]):
    """Get embeddings from HuggingFace ST interface."""
    with hf_runner(
            model_info.name,
            is_sentence_transformer=True,
            dtype="float32",
    ) as hf_model:
        embeddings = hf_model.encode(test_texts)
        if hasattr(embeddings, "cpu"):
            return embeddings.cpu().numpy()
        return np.array(embeddings)


# ST models with projector (Dense) layers
ST_PROJECTOR_MODELS = [
    EmbedModelInfo(
        "TencentBAC/Conan-embedding-v1",
        architecture="BertModel",
        enable_test=True,
    ),
]


@pytest.mark.parametrize("model_info", ST_PROJECTOR_MODELS)
def test_st_projector_loading(vllm_runner, model_info: EmbedModelInfo) -> None:
    """Ensure projector models load and output expected dim."""
    if not model_info.enable_test:
        pytest.skip("Skipping test.")

    test_texts = ["This is a test sentence."]
    embeddings_data = _get_vllm_embeddings(vllm_runner, model_info, test_texts)

    actual_dim = embeddings_data.shape[-1]
    expected_dim = 1792
    assert actual_dim == expected_dim, (
        f"Expected {expected_dim}, got {actual_dim}")


@pytest.mark.parametrize("model_info", ST_PROJECTOR_MODELS)
def test_compare_with_hf_dimensions(hf_runner, vllm_runner,
                                    model_info: EmbedModelInfo) -> None:
    """Compare embedding dimensions between vLLM and HuggingFace."""
    if not model_info.enable_test:
        pytest.skip("Skipping test.")

    test_texts = ["This is a test sentence for dimension comparison."]

    vllm_data = _get_vllm_embeddings(vllm_runner, model_info, test_texts)
    hf_data = _get_hf_embeddings(hf_runner, model_info, test_texts)

    vllm_dim = vllm_data.shape[-1]
    hf_dim = hf_data.shape[-1]

    assert vllm_dim == hf_dim, ("Embedding dim mismatch: "
                                f"vLLM {vllm_dim} vs HF {hf_dim}")
    print(f"✓ Embedding dimensions match: {vllm_dim}")


@pytest.mark.parametrize("model_info", ST_PROJECTOR_MODELS)
def test_embedding_numerical_similarity(hf_runner, vllm_runner,
                                        model_info: EmbedModelInfo) -> None:
    """Numerical similarity between vLLM and HF embeddings."""
    if not model_info.enable_test:
        pytest.skip("Skipping test.")

    test_texts = [
        "This is a test sentence for numerical comparison.",
        "Another sentence to verify embedding quality.",
        "机器学习是人工智能的一个重要分支。",  # Chinese test
    ]

    vllm_data = _get_vllm_embeddings(vllm_runner, model_info, test_texts)
    hf_data = _get_hf_embeddings(hf_runner, model_info, test_texts)

    assert vllm_data.shape == hf_data.shape, (
        "Shape mismatch: "
        f"vLLM {vllm_data.shape} vs HF {hf_data.shape}")

    print(f"Embedding shape: {vllm_data.shape}")
    print(f"Embedding dimension: {vllm_data.shape[-1]}")

    similarities = []
    for i, text in enumerate(test_texts):
        vllm_emb = vllm_data[i]
        hf_emb = hf_data[i]

        similarity = 1 - cosine(vllm_emb, hf_emb)
        similarities.append(similarity)

        preview = text[:50] + ("..." if len(text) > 50 else "")
        print(f"Text {i + 1}: '{preview}'")
        print(f"  Cosine similarity: {similarity:.6f}")

        min_similarity = 0.95
        assert similarity > min_similarity, (
            f"Text {i + 1} similarity too low: "
            f"{similarity:.6f} < {min_similarity}\n"
            f"vLLM norm: {np.linalg.norm(vllm_emb):.6f}, "
            f"HF norm: {np.linalg.norm(hf_emb):.6f}")

    avg_similarity = np.mean(similarities)
    print(f"\nAverage cosine similarity: {avg_similarity:.6f}")

    assert avg_similarity > 0.98, (
        f"Average similarity too low: {avg_similarity:.6f} < 0.98")
    print("✓ All numerical similarity tests passed!")


@pytest.mark.parametrize("model_info", ST_PROJECTOR_MODELS)
def test_embedding_quality_checks(vllm_runner,
                                  model_info: EmbedModelInfo) -> None:
    """Basic quality checks: non-zero, non-constant, distinct."""
    if not model_info.enable_test:
        pytest.skip("Skipping test.")

    test_texts = [
        "First test sentence.",
        "Second different sentence.",
        "Completely different content here.",
    ]

    embeddings_data = _get_vllm_embeddings(vllm_runner, model_info, test_texts)

    print(f"Embeddings shape: {embeddings_data.shape}")

    # Non-zero and non-constant
    for i, emb in enumerate(embeddings_data):
        norm = np.linalg.norm(emb)
        print(f"Embedding {i + 1} L2 norm: {norm:.6f}")
        assert norm > 1e-6, (
            f"Embedding {i + 1} too close to zero: norm={norm}")

        std = np.std(emb)
        print(f"Embedding {i + 1} std: {std:.6f}")
        assert std > 1e-6, (
            f"Embedding {i + 1} too close to constant: std={std}")

    # Different texts should differ
    for i in range(len(embeddings_data)):
        for j in range(i + 1, len(embeddings_data)):
            sim = 1 - cosine(embeddings_data[i], embeddings_data[j])
            print(f"Similarity between text {i + 1} and {j + 1}: {sim:.6f}")
            assert sim < 0.99, ("Embeddings too similar: "
                                f"{i + 1} vs {j + 1} -> {sim:.6f}")

    print("✓ All embedding quality checks passed!")
