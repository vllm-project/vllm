# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ColQwen3.5 late interaction model for multi-modal retrieval.

ColQwen3.5 is a multi-vector retrieval model based on Qwen3.5 backbone with
ColBERT-style late interaction scoring (MaxSim). It produces per-token
embeddings for both text and image inputs.
"""

from types import SimpleNamespace

import pytest
import torch

from ....conftest import VllmRunner

MODELS = [
    "athrael-soju/colqwen3.5-4.5B-v3",
]

EMBED_DIMS = {
    "athrael-soju/colqwen3.5-4.5B-v3": 320,
}

TEXT_QUERIES = [
    "What is the capital of France?",
    "Describe the contents of the document.",
]

TEXT_DOCUMENTS = [
    "The capital of France is Paris.",
    "This document contains important financial data.",
]

DTYPE = "half"


def _run_token_embed_test(
    vllm_runner: type[VllmRunner],
    model: str,
    *,
    dtype: str,
) -> None:
    """Verify per-token embedding shape and L2 normalization."""
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=4096,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.token_embed([TEXT_QUERIES[0]])

        assert len(outputs) == 1
        emb = torch.tensor(outputs[0])
        # Token embeddings should be 2D: [num_tokens, embed_dim]
        assert emb.dim() == 2
        assert emb.shape[1] == EMBED_DIMS[model]
        assert emb.shape[0] > 1

        # Verify L2 normalization
        norms = torch.norm(emb, p=2, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            rtol=1e-2,
            atol=1e-2,
        )


def _run_late_interaction_test(
    vllm_runner: type[VllmRunner],
    model: str,
    *,
    dtype: str,
) -> None:
    """Verify MaxSim scoring matches manual computation."""
    from vllm.entrypoints.pooling.scoring.utils import compute_maxsim_score

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=4096,
        enforce_eager=True,
    ) as vllm_model:
        q_outputs = vllm_model.token_embed([TEXT_QUERIES[0]])
        d_outputs = vllm_model.token_embed([TEXT_DOCUMENTS[0]])

        q_emb = torch.tensor(q_outputs[0])
        d_emb = torch.tensor(d_outputs[0])

        manual_score = compute_maxsim_score(q_emb, d_emb).item()

        vllm_scores = vllm_model.score(TEXT_QUERIES[0], TEXT_DOCUMENTS[0])

        assert len(vllm_scores) == 1
        assert vllm_scores[0] == pytest.approx(manual_score, rel=0.01)


def _run_relevance_test(
    vllm_runner: type[VllmRunner],
    model: str,
    *,
    dtype: str,
) -> None:
    """Verify that relevant documents score higher than irrelevant ones."""
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather forecast shows rain tomorrow.",
        "Deep learning uses neural networks for complex tasks.",
    ]

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=4096,
        enforce_eager=True,
    ) as vllm_model:
        scores = vllm_model.score(query, documents)

        assert len(scores) == 3
        assert scores[0] > scores[1], "ML doc should score higher than weather doc"
        assert scores[2] > scores[1], "DL doc should score higher than weather doc"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
def test_colqwen3_5_token_embed(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    _run_token_embed_test(vllm_runner, model, dtype=dtype)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
def test_colqwen3_5_late_interaction_scoring(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    _run_late_interaction_test(vllm_runner, model, dtype=dtype)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
def test_colqwen3_5_relevance_ordering(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    _run_relevance_test(vllm_runner, model, dtype=dtype)


@pytest.mark.parametrize(
    ("contract", "expected_is_causal"),
    [("causal", True), ("bidirectional", False)],
)
def test_colqwen3_5_config_applies_declared_attention_contract(
    contract: str,
    expected_is_causal: bool,
) -> None:
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        ColQwen3_5Config,
    )

    assert MODELS_CONFIG_MAP["ColQwen3_5"] is ColQwen3_5Config

    hf_config = SimpleNamespace(retrieval_attention_contract=contract)
    text_config = SimpleNamespace()
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=text_config,
    )
    ColQwen3_5Config.verify_and_update_model_config(model_config)
    assert hf_config.is_causal is expected_is_causal
    assert text_config.is_causal is expected_is_causal


@pytest.mark.parametrize(
    "hf_config",
    [
        SimpleNamespace(),
        SimpleNamespace(retrieval_attention_contract="unsupported"),
        SimpleNamespace(
            retrieval_attention_contract="causal",
            text_config=SimpleNamespace(retrieval_attention_contract="bidirectional"),
        ),
    ],
)
def test_colqwen3_5_config_rejects_invalid_attention_contract(hf_config) -> None:
    from vllm.model_executor.models.config import ColQwen3_5Config

    text_config = getattr(hf_config, "text_config", SimpleNamespace())
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=text_config,
    )
    with pytest.raises(ValueError, match="retrieval_attention_contract"):
        ColQwen3_5Config.verify_and_update_model_config(model_config)


def test_colqwen3_5_bidirectional_contract_builds_encoder_only_attention(
    monkeypatch,
) -> None:
    from vllm.model_executor.models import qwen3_next
    from vllm.model_executor.models.config import ColQwen3_5Config
    from vllm.v1.attention.backend import AttentionType

    hf_config = SimpleNamespace(retrieval_attention_contract="bidirectional")
    text_config = SimpleNamespace(
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=4096,
        rope_parameters={},
        rms_norm_eps=1e-6,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=text_config,
    )
    ColQwen3_5Config.verify_and_update_model_config(model_config)

    captured = {}

    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            captured["attn_type"] = kwargs["attn_type"]

    monkeypatch.setattr(qwen3_next, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        qwen3_next, "QKVParallelLinear", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        qwen3_next, "RowParallelLinear", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        qwen3_next,
        "get_rope",
        lambda *args, **kwargs: SimpleNamespace(is_neox_style=False),
    )
    monkeypatch.setattr(
        qwen3_next, "Qwen3NextRMSNorm", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(qwen3_next, "Attention", FakeAttention)

    qwen3_next.Qwen3NextAttention(text_config)

    assert captured["attn_type"] is AttentionType.ENCODER_ONLY


def test_colqwen3_5_encoder_only_attention_has_no_kv_cache_spec() -> None:
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    attention = SimpleNamespace(attn_type=AttentionType.ENCODER_ONLY)
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))

    assert Attention.get_kv_cache_spec(attention, vllm_config) is None
