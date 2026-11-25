# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-free tests for the proposed TP2 embedding replica."""

from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers import vocab_parallel_embedding as embedding_module
from vllm.models.minicpm5_moe.xpu import model as model_module


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_replica_loads_full_vocab_and_never_calls_tp_collective(monkeypatch, dtype):
    def unexpected(*args, **kwargs):
        raise AssertionError("replicated embedding must not use the TP group")

    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", unexpected)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", unexpected
    )
    monkeypatch.setattr(
        embedding_module, "tensor_model_parallel_all_reduce", unexpected
    )
    layer = embedding_module.VocabParallelEmbedding(
        128, 16, params_dtype=dtype, disable_tp=True
    )
    weight = (torch.arange(128 * 16).reshape(128, 16).float() / 127 - 0.5).to(dtype)
    layer.weight_loader(layer.weight, weight)
    ids = torch.tensor([0, 63, 64, 127])
    assert layer.tp_size == 1 and layer.weight.shape == weight.shape
    torch.testing.assert_close(layer(ids), F.embedding(ids, weight), rtol=0, atol=0)


def tiny_base(monkeypatch, dtype, vllm_config, world=2):
    monkeypatch.setattr(
        model_module, "get_tensor_model_parallel_world_size", lambda: world
    )
    return NS(
        config=vllm_config.model_config.hf_config,
        quant_config=None,
        embed_tokens=embedding_module.VocabParallelEmbedding(
            128,
            16,
            params_dtype=dtype,
            parallel_group=NS(world_size=2, rank_in_group=1),
        ),
    )


def config(enabled=True, tied=False, lora=None):
    return NS(
        model_config=NS(
            hf_config=NS(
                vocab_size=128,
                hidden_size=16,
                tie_word_embeddings=tied,
            )
        ),
        additional_config={"minicpm5_replicated_embedding": enabled},
        kernel_config=NS(monokernel=False),
        lora_config=lora,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enabled", [False, True])
def test_model_option_changes_only_input_embedding_layout(monkeypatch, dtype, enabled):
    vllm_config = config(enabled)
    model = tiny_base(monkeypatch, dtype, vllm_config)
    model_module.configure_xpu_model(NS(model=model), vllm_config)
    assert model.embed_tokens.tp_size == (1 if enabled else 2)
    assert model.embed_tokens.weight.shape == (128 if enabled else 64, 16)
    assert model.embed_tokens.weight.dtype == dtype
    assert model_module.get_tensor_model_parallel_world_size() == 2


@pytest.mark.parametrize(
    "world,tied,lora", [(1, False, None), (2, True, None), (2, False, NS(enabled=True))]
)
def test_incompatible_replica_requests_are_rejected(monkeypatch, world, tied, lora):
    vllm_config = config(tied=tied, lora=lora)
    model = tiny_base(monkeypatch, torch.bfloat16, vllm_config, world=world)
    with pytest.raises(ValueError, match="Replicated MiniCPM5 embedding requires"):
        model_module.configure_xpu_model(NS(model=model), vllm_config)
