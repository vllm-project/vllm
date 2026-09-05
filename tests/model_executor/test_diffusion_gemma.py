# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.models.diffusion_gemma as diffusion_gemma


def test_diffusion_gemma_embedding_weight_helper_uses_weight_attr():
    weight = torch.randn(4, 3)
    embed_tokens = SimpleNamespace(weight=weight)

    assert diffusion_gemma._get_diffusion_gemma_embedding_weight(embed_tokens) is weight


def test_diffusion_gemma_embedding_weight_helper_uses_quant_method_contract():
    weight = torch.randn(4, 3)
    calls = []

    class FakeQuantMethod:
        def get_embedding_weight(self, layer):
            calls.append(layer)
            return weight

    embed_tokens = SimpleNamespace(quant_method=FakeQuantMethod())

    assert diffusion_gemma._get_diffusion_gemma_embedding_weight(embed_tokens) is weight
    assert calls == [embed_tokens]


def test_diffusion_gemma_embedding_weight_helper_gathers_tp_shards(monkeypatch):
    local_weight = torch.tensor([[1.0], [2.0]])
    gathered_weight = torch.tensor([[1.0], [2.0], [3.0], [4.0], [0.0]])
    embed_tokens = SimpleNamespace(
        weight=local_weight,
        tp_size=2,
        org_vocab_size=4,
    )
    calls = []

    def fake_all_gather(weight, dim=-1):
        calls.append((weight, dim))
        return gathered_weight

    monkeypatch.setattr(
        diffusion_gemma,
        "tensor_model_parallel_all_gather",
        fake_all_gather,
    )

    full_weight = diffusion_gemma._get_diffusion_gemma_embedding_weight(embed_tokens)

    assert torch.equal(full_weight, gathered_weight[:4])
    assert calls == [(local_weight, 0)]


def test_diffusion_gemma_embedding_weight_helper_can_keep_local_tp_shard(monkeypatch):
    local_weight = torch.tensor([[1.0], [2.0]])
    embed_tokens = SimpleNamespace(
        weight=local_weight,
        tp_size=2,
        org_vocab_size=4,
    )
    calls = []

    def fake_all_gather(weight, dim=-1):
        calls.append((weight, dim))
        return torch.tensor([[1.0], [2.0], [3.0], [4.0]])

    monkeypatch.setattr(
        diffusion_gemma,
        "tensor_model_parallel_all_gather",
        fake_all_gather,
    )

    local = diffusion_gemma._get_diffusion_gemma_embedding_weight(
        embed_tokens,
        gather_tp_shards=False,
    )

    assert local is local_weight
    assert calls == []


def test_diffusion_gemma_embedding_weight_helper_requires_weight_provider():
    with pytest.raises(RuntimeError, match="get_embedding_weight"):
        diffusion_gemma._get_diffusion_gemma_embedding_weight(SimpleNamespace())
