# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor


def _fake_lm_head(tp_size: int = 2, num_embeddings_padded: int = 32):
    return SimpleNamespace(
        tp_size=tp_size,
        num_embeddings_padded=num_embeddings_padded,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=0,
            org_vocab_start_index=0,
        ),
    )


def test_get_top_k_tokens_uses_one_packed_all_gather(default_vllm_config, monkeypatch):
    all_logits = torch.arange(64, dtype=torch.float32).view(2, 32)
    rank_logits = all_logits.split(4, dim=-1)
    local_logits = rank_logits[0]
    hidden_states = torch.empty(2, 1)
    top_k = 2
    processor = LogitsProcessor(vocab_size=32)
    monkeypatch.setattr(processor, "_apply_head", lambda *args: local_logits)

    calls = 0

    def fake_all_gather(local_candidates: torch.Tensor, dim: int):
        nonlocal calls
        calls += 1
        assert dim == -1
        assert local_candidates.dtype == torch.float32
        assert local_candidates.shape == (2, 2 * top_k)
        candidates = [local_candidates]
        for rank, logits in enumerate(rank_logits[1:], start=1):
            remote_values, remote_ids = torch.topk(logits, top_k, dim=-1)
            remote_ids += rank * local_logits.shape[-1]
            candidates.append(
                torch.cat(
                    (remote_values, remote_ids.to(torch.int32).view(torch.float32)),
                    dim=-1,
                )
            )
        return torch.cat(candidates, dim=-1)

    monkeypatch.setattr(
        "vllm.model_executor.layers.logits_processor.tensor_model_parallel_all_gather",
        fake_all_gather,
    )

    ids, values = processor.get_top_k_tokens(
        _fake_lm_head(tp_size=8), hidden_states, top_k
    )

    expected_values, expected_ids = torch.topk(all_logits, top_k, dim=-1)
    assert calls == 1
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(values, expected_values)


def test_get_top_k_tokens_packs_large_ids_losslessly(default_vllm_config, monkeypatch):
    local_logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    remote_logits = torch.tensor([[8.0, 7.0, 6.0, 5.0]])
    hidden_states = torch.empty(1, 1)
    top_k = 2
    processor = LogitsProcessor(vocab_size=2**24 + 1)
    monkeypatch.setattr(processor, "_apply_head", lambda *args: local_logits)

    calls = 0

    def fake_all_gather(local_candidates: torch.Tensor, dim: int):
        nonlocal calls
        calls += 1
        assert dim == -1
        assert local_candidates.dtype == torch.float32
        remote_values, remote_ids = torch.topk(remote_logits, top_k, dim=-1)
        remote_ids = remote_ids.to(torch.int64) + 2**24 - 4
        remote_candidates = [
            torch.cat(
                (
                    remote_values - rank * 10,
                    remote_ids.to(torch.int32).view(torch.float32),
                ),
                dim=-1,
            )
            for rank in range(7)
        ]
        return torch.cat((local_candidates, *remote_candidates), dim=-1)

    monkeypatch.setattr(
        "vllm.model_executor.layers.logits_processor.tensor_model_parallel_all_gather",
        fake_all_gather,
    )

    ids, values = processor.get_top_k_tokens(
        _fake_lm_head(tp_size=8, num_embeddings_padded=2**24 + 1),
        hidden_states,
        top_k,
    )

    assert calls == 1
    assert torch.equal(ids, torch.tensor([[2**24 - 4, 2**24 - 3]]))
    torch.testing.assert_close(values, torch.tensor([[8.0, 7.0]]))


def test_get_top_k_tokens_keeps_separate_gathers_below_tp8(
    default_vllm_config, monkeypatch
):
    local_logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    hidden_states = torch.empty(1, 1)
    top_k = 2
    processor = LogitsProcessor(vocab_size=8)
    monkeypatch.setattr(processor, "_apply_head", lambda *args: local_logits)

    gathered_dtypes = []

    def fake_all_gather(local_candidates: torch.Tensor, dim: int):
        gathered_dtypes.append(local_candidates.dtype)
        return torch.cat((local_candidates, local_candidates), dim=dim)

    monkeypatch.setattr(
        "vllm.model_executor.layers.logits_processor.tensor_model_parallel_all_gather",
        fake_all_gather,
    )

    processor.get_top_k_tokens(_fake_lm_head(), hidden_states, top_k)

    assert gathered_dtypes == [torch.float32, torch.int64]
