# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models import diffusion_gemma


def test_soft_embeddings_from_probs_matches_full_vocab_embedding():
    embed_tokens = nn.Embedding(3, 2)
    embed_tokens.weight.data = torch.tensor(
        [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]
    )
    probs = torch.tensor([[[0.2, 0.3, 0.5]]])
    normalizer = torch.tensor(2.0)

    soft_embeds = diffusion_gemma._soft_embeddings_from_probs(
        probs, embed_tokens, normalizer
    )

    expected = torch.matmul(probs, embed_tokens.weight) * normalizer
    torch.testing.assert_close(soft_embeds, expected)


def test_soft_embeddings_from_probs_uses_tensor_parallel_vocab_shard(monkeypatch):
    embed_tokens = nn.Embedding(5, 2)
    embed_tokens.weight.data = torch.tensor(
        [
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 3.0],
            [40.0, 4.0],
            [50.0, 5.0],
        ]
    )
    embed_tokens.shard_indices = SimpleNamespace(
        org_vocab_start_index=1,
        org_vocab_end_index=3,
        added_vocab_start_index=6,
        added_vocab_end_index=8,
        num_org_elements_padded=3,
    )
    probs = torch.tensor([[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]])
    normalizer = torch.tensor(0.5)

    all_reduce_inputs = []

    def fake_all_reduce(tensor):
        all_reduce_inputs.append(tensor.clone())
        return tensor

    monkeypatch.setattr(
        diffusion_gemma, "tensor_model_parallel_all_reduce", fake_all_reduce
    )

    soft_embeds = diffusion_gemma._soft_embeddings_from_probs(
        probs, embed_tokens, normalizer
    )

    local_probs = torch.tensor([[[0.1, 0.2, 0.0, 0.6, 0.7]]])
    expected_local = torch.matmul(local_probs, embed_tokens.weight)
    torch.testing.assert_close(all_reduce_inputs[0], expected_local)
    torch.testing.assert_close(soft_embeds, expected_local * normalizer)


def test_soft_embeddings_from_probs_requires_shard_metadata_for_vocab_mismatch():
    embed_tokens = nn.Embedding(2, 2)
    probs = torch.zeros(1, 1, 3)

    with pytest.raises(RuntimeError, match="shapes are incompatible"):
        diffusion_gemma._soft_embeddings_from_probs(
            probs, embed_tokens, torch.tensor(1.0)
        )
