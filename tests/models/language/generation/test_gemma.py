# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import gemma
from vllm.model_executor.models.gemma3n import (
    Gemma3nTextModel,
    _kv_sharing_weights_mapper,
)
from vllm.model_executor.models.gemma4 import (
    Gemma4ForCausalLM,
    Gemma4Model,
    _gemma4_layer_weights_mapper,
)

MODELS = ["google/gemma-2b", "google/gemma-2-2b", "google/gemma-3-4b-it"]


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_checkpoint_lm_head_can_override_tied_config(monkeypatch) -> None:
    """A physical LM head must load after checkpoint-driven untying."""

    class StubGemmaModel(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            self.embed_tokens = VocabParallelEmbedding(4, 2)
            self.make_empty_intermediate_tensors = None

    monkeypatch.setattr(gemma, "GemmaModel", StubGemmaModel)
    config = SimpleNamespace(
        vocab_size=4,
        hidden_size=2,
        tie_word_embeddings=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
    )
    model = gemma.GemmaForCausalLM(vllm_config=cast(VllmConfig, vllm_config))
    embedding_weight = torch.full((4, 2), 1.0)
    lm_head_weight = torch.full((4, 2), 2.0)

    loaded = model.load_weights(
        [
            ("model.embed_tokens.weight", embedding_weight),
            ("lm_head.weight", lm_head_weight),
        ]
    )

    assert loaded == {"model.embed_tokens.weight", "lm_head.weight"}
    assert torch.equal(model.model.embed_tokens.weight[:4], embedding_weight)
    assert torch.equal(model.lm_head.weight[:4], lm_head_weight)


@pytest.mark.cpu_test
def test_gemma4_attention_mapper() -> None:
    """Layers with a qkv_proj pack q/k/v; `attention_k_eq_v` full-attention
    layers also load K as the V shard; KV-shared layers keep q_proj and drop
    the K/V tensors original checkpoints still ship for them."""
    config = SimpleNamespace(
        num_hidden_layers=3,
        num_kv_shared_layers=1,
        attention_k_eq_v=True,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )
    weights = [
        (f"model.layers.{i}.self_attn.{tensor}.weight", torch.full((2, 2), i + 1.0))
        for i in range(3)
        for tensor in ("q_proj", "k_proj", "k_norm")
    ] + [("model.layers.0.mlp.up_proj.weight", torch.empty(0))]

    mapper = Gemma4Model.hf_to_vllm_mapper | _gemma4_layer_weights_mapper(config)
    mapped = list(mapper.apply(weights))

    assert [(name, getattr(w, "shard_id", None)) for name, w in mapped] == [
        ("model.layers.0.self_attn.qkv_proj.weight", "q"),
        ("model.layers.0.self_attn.qkv_proj.weight", "k"),
        ("model.layers.0.self_attn.k_norm.weight", None),
        ("model.layers.1.self_attn.qkv_proj.weight", "q"),
        ("model.layers.1.self_attn.qkv_proj.weight", "k"),
        ("model.layers.1.self_attn.qkv_proj.weight", "v"),
        ("model.layers.1.self_attn.k_norm.weight", None),
        ("model.layers.2.self_attn.q_proj.weight", None),
        ("model.layers.0.mlp.gate_up_proj.weight", 1),
    ]
    k_weight, v_weight = weights[4][1], mapped[5][1]
    assert torch.equal(v_weight, k_weight) and v_weight is not k_weight


@pytest.mark.cpu_test
def test_gemma4_expert_names_map_onto_moe() -> None:
    """HF keeps experts and per_expert_scale on the layer/router; vLLM nests
    them under `moe`, so fused and per-expert tensors reach RoutedExperts."""
    prefix = "model.language_model.layers.0."
    weights = [
        (prefix + name, torch.empty(0))
        for name in (
            "experts.gate_up_proj",
            "experts.3.down_proj.weight_packed",
            "router.per_expert_scale",
        )
    ]

    mapped = [
        (name, getattr(w, "shard_id", None))
        for name, w in Gemma4ForCausalLM.hf_to_vllm_mapper.apply(weights)
    ]

    assert mapped == [
        ("model.layers.0.moe.experts.gate_up_proj", None),
        ("model.layers.0.moe.experts.3.down_proj.weight_packed", None),
        ("model.layers.0.moe.per_expert_scale", None),
    ]


@pytest.mark.cpu_test
def test_gemma3n_kv_shared_layer_mapper() -> None:
    """Only non-shared layers pack q/k/v into qkv_proj; KV-shared layers keep
    q_proj and drop the redundant K/V tensors original checkpoints ship."""
    config = SimpleNamespace(num_hidden_layers=4, num_kv_shared_layers=2)
    mapper = Gemma3nTextModel.hf_to_vllm_mapper | _kv_sharing_weights_mapper(config)
    weights = [
        (f"layers.{i}.self_attn.{tensor}.weight", torch.empty(0))
        for i in (1, 3)
        for tensor in ("q_proj", "k_proj", "v_proj", "k_norm", "o_proj")
    ]

    mapped = [
        (name, getattr(weight, "shard_id", None))
        for name, weight in mapper.apply(weights)
    ]

    assert mapped == [
        ("layers.1.self_attn.qkv_proj.weight", "q"),
        ("layers.1.self_attn.qkv_proj.weight", "k"),
        ("layers.1.self_attn.qkv_proj.weight", "v"),
        ("layers.1.self_attn.k_norm.weight", None),
        ("layers.1.self_attn.o_proj.weight", None),
        ("layers.3.self_attn.q_proj.weight", None),
        ("layers.3.self_attn.o_proj.weight", None),
    ]


@pytest.mark.parametrize("model", MODELS)
def test_dummy_loader(vllm_runner, monkeypatch, model: str) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(
            model,
            load_format="dummy",
        ) as llm:
            if model == "google/gemma-3-4b-it":
                normalizers = llm.llm.collective_rpc(
                    lambda self: self.model_runner.model.language_model.model.normalizer.cpu().item()  # noqa: E501
                )
                config = llm.llm.llm_engine.model_config.hf_config.text_config
            else:
                normalizers = llm.llm.collective_rpc(
                    lambda self: self.model_runner.model.model.normalizer.cpu().item()
                )
                config = llm.llm.llm_engine.model_config.hf_config
            assert np.allclose(normalizers, config.hidden_size**0.5, rtol=2e-3)
