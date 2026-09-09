# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _markov_head(weight: torch.Tensor) -> DSparkMarkovHead:
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
    )
    head.markov_w2.weight.data.copy_(weight)
    return head


def test_gathered_markov_bias_overwrites_dense_logits():
    weight = torch.arange(21, dtype=torch.float32).view(7, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]])
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    values, index = logits.topk(3, dim=-1)
    values = torch.stack((values, torch.zeros_like(values)), dim=1)[:, 0]
    expected = values + torch.bmm(weight[index], markov_embed.unsqueeze(-1)).squeeze(-1)
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index
    )

    assert result is logits
    torch.testing.assert_close(result.gather(1, index), expected)
    selected = torch.zeros_like(result, dtype=torch.bool).scatter_(1, index, True)
    assert torch.isneginf(result.masked_select(~selected)).all()


def test_gathered_markov_bias_matches_dense_at_full_vocab():
    weight = torch.arange(15, dtype=torch.float32).view(5, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25]])
    logits = torch.tensor([[0.1, 0.4, -0.2, 0.3, 0.0]])
    original = logits.clone()
    values, index = logits.topk(logits.shape[-1], dim=-1)
    scale = 0.5
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index, scale
    )

    expected = original + markov_embed @ weight.T * scale
    torch.testing.assert_close(result, expected)


def test_gathered_markov_scores_match_dense_candidates():
    weight = torch.arange(21, dtype=torch.float32).view(7, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]])
    values = torch.tensor([[0.7, 0.4, 0.1], [0.6, 0.3, 0.2]])
    indices = torch.tensor([[6, 3, 0], [5, 2, 1]])
    scale = 0.5

    result = _markov_head(weight).score_gathered(markov_embed, values, indices, scale)

    dense_bias = markov_embed @ weight.T * scale
    expected = values + dense_bias.gather(1, indices)
    torch.testing.assert_close(result, expected)


class _GreedyTopKModel:
    def compute_draft_topk(
        self, hidden_states: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.shape == (4, 1)
        assert k == 2
        token_ids = torch.tensor([[5, 9], [6, 10], [15, 19], [16, 20]])
        return token_ids, torch.zeros_like(token_ids, dtype=torch.float32)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.float().unsqueeze(-1)

    def score_draft_candidates(
        self,
        markov_embed: torch.Tensor,
        values: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        expected_next = markov_embed.to(token_ids.dtype) + 1
        return values + 10 * (token_ids == expected_next)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids


def test_deepseek_greedy_topk_preserves_sequential_markov_dependency():
    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator._draft_topk = 2
    speculator.num_speculative_steps = 2
    speculator.sample_indices = torch.arange(4)
    speculator.input_buffers = SimpleNamespace(input_ids=torch.tensor([4, 0, 14, 0]))
    speculator._anchor_idx = torch.tensor([0, 2])
    speculator.draft_tokens = torch.full((2, 2), -1, dtype=torch.long)
    speculator.enable_adaptive_verification = False
    speculator.draft_logits = None
    speculator.model = _GreedyTopKModel()

    speculator._sample_sequential(2, torch.arange(4, dtype=torch.float32).view(4, 1))

    torch.testing.assert_close(
        speculator.draft_tokens, torch.tensor([[5, 6], [15, 16]])
    )


def test_vocab_parallel_topk_handles_k_larger_than_local_shard(
    monkeypatch, default_vllm_config
):
    logits_processor = LogitsProcessor(vocab_size=6)
    local_logits = torch.tensor([[4.0, 3.0, 2.0]])
    monkeypatch.setattr(
        logits_processor,
        "_apply_head",
        lambda lm_head, hidden_states, embedding_bias: local_logits,
    )
    lm_head = SimpleNamespace(
        tp_size=2,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=0,
            org_vocab_start_index=0,
        ),
    )
    gather_calls = 0

    def all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        nonlocal gather_calls
        gather_calls += 1
        assert dim == -1
        if tensor.dtype.is_floating_point:
            torch.testing.assert_close(tensor, torch.tensor([[4.0, 3.0, 2.0]]))
            remote = torch.tensor([[10.0, 1.0, 0.0]])
        else:
            torch.testing.assert_close(tensor, torch.tensor([[0, 1, 2]]))
            remote = torch.tensor([[3, 4, 5]])
        return torch.cat([tensor, remote], dim=dim)

    monkeypatch.setattr(
        "vllm.model_executor.layers.logits_processor.tensor_model_parallel_all_gather",
        all_gather,
    )

    token_ids, values = logits_processor.get_top_k_tokens(
        lm_head, torch.empty(1, 1), k=4
    )

    assert gather_calls == 2
    torch.testing.assert_close(token_ids, torch.tensor([[3, 0, 1, 2]]))
    torch.testing.assert_close(values, torch.tensor([[10.0, 4.0, 3.0, 2.0]]))
