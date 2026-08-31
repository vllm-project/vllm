# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for GLM-only TP-local target prompt logprobs."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed
import torch.multiprocessing

from vllm.model_executor.layers.logits_processor import LocalLogits
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from vllm.v1.worker.gpu.model_runner import _get_prompt_logprobs_local_logits_fn
from vllm.v1.worker.gpu.sample import prompt_logprob


class _SingleRankTPGroup:
    world_size = 1
    device_group = None

    @staticmethod
    def all_reduce(values: torch.Tensor) -> torch.Tensor:
        return values


class _GlooTPGroup:
    world_size = 2
    device_group = torch.distributed.group.WORLD

    @staticmethod
    def all_reduce(values: torch.Tensor) -> torch.Tensor:
        torch.distributed.all_reduce(values, group=torch.distributed.group.WORLD)
        return values


def _local_logits(
    logits: torch.Tensor, start: int = 0, org_width: int | None = None
) -> LocalLogits:
    org_width = logits.shape[-1] if org_width is None else org_width
    return LocalLogits(
        logits=logits,
        shard_indices=VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=start,
            padded_org_vocab_end_index=start + logits.shape[-1],
            padded_added_vocab_start_index=start + logits.shape[-1],
            padded_added_vocab_end_index=start + logits.shape[-1],
            org_vocab_start_index=start,
            org_vocab_end_index=start + org_width,
            added_vocab_start_index=start + org_width,
            added_vocab_end_index=start + org_width,
        ),
    )


def test_target_only_matches_one_rank_log_softmax(monkeypatch):
    monkeypatch.setattr(prompt_logprob, "get_tp_group", _SingleRankTPGroup)
    logits = torch.tensor([[1.0, -2.0, 3.0], [0.5, 2.0, -1.0]])
    target_ids = torch.tensor([2, 0])

    result = prompt_logprob.compute_distributed_token_logprobs(
        _local_logits(logits), target_ids
    )

    expected = torch.log_softmax(logits, -1)[torch.arange(2), target_ids]
    torch.testing.assert_close(result.logprobs[:, 0], expected)
    assert result.logprob_token_ids.tolist() == [[2], [0]]
    assert result.selected_token_ranks.tolist() == [1, 2]
    assert result.selected_token_ranks.dtype == torch.int64


def test_target_only_excludes_dominant_padded_and_added_tail(monkeypatch):
    monkeypatch.setattr(prompt_logprob, "get_tp_group", _SingleRankTPGroup)
    local = LocalLogits(
        logits=torch.tensor([[1.0, 2.0, -float("inf"), -float("inf"), 1e6]]),
        shard_indices=VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=4,
            padded_org_vocab_end_index=8,
            padded_added_vocab_start_index=6,
            padded_added_vocab_end_index=7,
            org_vocab_start_index=4,
            org_vocab_end_index=6,
            added_vocab_start_index=6,
            added_vocab_end_index=7,
        ),
    )

    result = prompt_logprob.compute_distributed_token_logprobs(local, torch.tensor([5]))

    torch.testing.assert_close(
        result.logprobs[:, 0], torch.log_softmax(torch.tensor([[1.0, 2.0]]), -1)[:, 1]
    )
    assert result.selected_token_ranks.tolist() == [1]


def _chunking_fallback_test(monkeypatch, num_prompt_logprobs: int) -> None:
    calls = SimpleNamespace(gathered=0, local=0)
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered += 1
        return logits[: hidden_states.shape[0]]

    def local(_: torch.Tensor) -> LocalLogits:
        calls.local += 1
        raise AssertionError(
            "gathered prompt-logprob requests must not project locally"
        )

    def fake_topk(
        prompt_logits: torch.Tensor,
        requested_num: int,
        _: torch.Tensor,
        **__: object,
    ) -> prompt_logprob.LogprobsTensors:
        assert requested_num == (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=torch.zeros(
                prompt_logits.shape[0], requested_num + 1, dtype=torch.int64
            ),
            logprobs=torch.zeros(prompt_logits.shape[0], requested_num + 1),
            selected_token_ranks=torch.ones(prompt_logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_topk)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.tensor([1, 0]),
        torch.empty(2, 1),
        gathered,
        local,
        num_prompt_logprobs,
    )
    assert calls.gathered == 1
    assert calls.local == 0


def test_chunking_gathers_for_prompt_topn(monkeypatch):
    _chunking_fallback_test(monkeypatch, num_prompt_logprobs=1)


def test_chunking_gathers_for_full_prompt_logprobs_minus_one(monkeypatch):
    """``prompt_logprobs=-1`` never invokes the local target-only route."""
    _chunking_fallback_test(monkeypatch, num_prompt_logprobs=-1)


@pytest.mark.parametrize("logprobs_mode", ["raw_logits", "raw_logprobs"])
def test_chunking_uses_local_target_only_at_minimum_row_count(
    monkeypatch, logprobs_mode
):
    calls = SimpleNamespace(gathered=0, local=0, distributed=0)

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered += 1
        return torch.zeros(hidden_states.shape[0], 2)

    def local(hidden_states: torch.Tensor) -> LocalLogits:
        calls.local += 1
        return _local_logits(torch.zeros(hidden_states.shape[0], 2))

    def distributed(
        _: LocalLogits, token_ids: torch.Tensor, mode: object
    ) -> prompt_logprob.LogprobsTensors:
        assert mode == logprobs_mode
        calls.distributed += 1
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=token_ids[:, None],
            logprobs=torch.zeros(token_ids.shape[0], 1),
            selected_token_ranks=torch.ones(token_ids.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(
        prompt_logprob, "compute_distributed_token_logprobs", distributed
    )
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.zeros(prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS, dtype=torch.int64),
        torch.empty(prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS, 1),
        gathered,
        local,
        num_prompt_logprobs=0,
        logprobs_mode=logprobs_mode,
    )

    assert calls.gathered == 0
    assert calls.local == calls.distributed == 1


@pytest.mark.parametrize("logprobs_mode", ["raw_logits", "raw_logprobs"])
def test_chunking_gathers_below_minimum_without_local_projection(
    monkeypatch, logprobs_mode
):
    calls = SimpleNamespace(gathered=0, local=0)
    logged: list[str] = []
    rows = prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS - 1

    class Logger:
        @staticmethod
        def info_once(message: str, *args: object) -> None:
            logged.append(message % args)

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered += 1
        return torch.zeros(hidden_states.shape[0], 2)

    def local(_: torch.Tensor) -> LocalLogits:
        calls.local += 1
        raise AssertionError("short chunks must not project local logits")

    def fake_topk(
        prompt_logits: torch.Tensor, _: int, token_ids: torch.Tensor, **kwargs: object
    ) -> prompt_logprob.LogprobsTensors:
        assert kwargs["logits_mode"] == (logprobs_mode == "raw_logits")
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=token_ids[:, None],
            logprobs=torch.zeros(prompt_logits.shape[0], 1),
            selected_token_ranks=torch.ones(prompt_logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "logger", Logger())
    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_topk)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.zeros(rows, dtype=torch.int64),
        torch.empty(rows, 1),
        gathered,
        local,
        num_prompt_logprobs=0,
        logprobs_mode=logprobs_mode,
    )

    assert calls.gathered == 1
    assert calls.local == 0
    assert logged == [
        (
            f"{prompt_logprob.SHORT_CHUNK_GATHERED_MARKER} rows={rows} "
            f"threshold={prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS}"
        )
    ]


def test_chunking_mixed_1025_rows_uses_local_prefix_and_gathered_tail(monkeypatch):
    calls = SimpleNamespace(gathered_rows=[], local_rows=[], distributed_rows=[])
    rows = 1025

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered_rows.append(hidden_states.shape[0])
        return torch.zeros(hidden_states.shape[0], 2)

    def local(hidden_states: torch.Tensor) -> LocalLogits:
        calls.local_rows.append(hidden_states.shape[0])
        return _local_logits(torch.zeros(hidden_states.shape[0], 2))

    def distributed(
        _: LocalLogits, token_ids: torch.Tensor, __: object
    ) -> prompt_logprob.LogprobsTensors:
        calls.distributed_rows.append(token_ids.shape[0])
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=token_ids[:, None],
            logprobs=torch.zeros(token_ids.shape[0], 1),
            selected_token_ranks=torch.ones(token_ids.shape[0], dtype=torch.int64),
        )

    def fake_topk(
        prompt_logits: torch.Tensor, _: int, token_ids: torch.Tensor, **__: object
    ) -> prompt_logprob.LogprobsTensors:
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=token_ids[:, None],
            logprobs=torch.zeros(prompt_logits.shape[0], 1),
            selected_token_ranks=torch.ones(prompt_logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(
        prompt_logprob, "compute_distributed_token_logprobs", distributed
    )
    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_topk)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.zeros(rows, dtype=torch.int64),
        torch.empty(rows, 1),
        gathered,
        local,
        num_prompt_logprobs=0,
    )

    assert calls.local_rows == calls.distributed_rows == [1024]
    assert calls.gathered_rows == [1]


def _tp2_asymmetric_worker(rank: int, init_method: str, queue) -> None:
    torch.distributed.init_process_group(
        backend="gloo", init_method=init_method, world_size=2, rank=rank
    )
    try:
        prompt_logprob.get_tp_group = lambda: _GlooTPGroup
        full = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        if rank == 0:
            org = full[:, :4]
            indices = VocabParallelEmbeddingShardIndices(
                padded_org_vocab_start_index=0,
                padded_org_vocab_end_index=4,
                padded_added_vocab_start_index=5,
                padded_added_vocab_end_index=9,
                org_vocab_start_index=0,
                org_vocab_end_index=4,
                added_vocab_start_index=5,
                added_vocab_end_index=7,
            )
            tail_width = 4
        else:
            org = full[:, 4:]
            indices = VocabParallelEmbeddingShardIndices(
                padded_org_vocab_start_index=4,
                padded_org_vocab_end_index=8,
                padded_added_vocab_start_index=9,
                padded_added_vocab_end_index=13,
                org_vocab_start_index=4,
                org_vocab_end_index=5,
                added_vocab_start_index=7,
                added_vocab_end_index=7,
            )
            tail_width = 7
        local = torch.cat((org, torch.full((2, tail_width), 1e6)), dim=-1)
        result = prompt_logprob.compute_distributed_token_logprobs(
            LocalLogits(local, indices), torch.tensor([4, 1])
        )
        queue.put(
            (rank, result.logprobs[:, 0].tolist(), result.selected_token_ranks.tolist())
        )
    finally:
        torch.distributed.destroy_process_group()


def test_tp2_asymmetric_org_shards_exclude_padded_and_added_tails(tmp_path):
    context = torch.multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    torch.multiprocessing.spawn(
        _tp2_asymmetric_worker,
        args=(f"file://{tmp_path / 'tp2-asymmetric-tail'}", queue),
        nprocs=2,
        join=True,
    )
    full = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    targets = torch.tensor([4, 1])
    expected_scores = torch.log_softmax(full, -1)[torch.arange(2), targets]
    expected_ranks = (full >= full[torch.arange(2), targets, None]).sum(-1)
    for _, scores, ranks in sorted(queue.get() for _ in range(2)):
        torch.testing.assert_close(torch.tensor(scores), expected_scores)
        assert ranks == expected_ranks.tolist()


def test_nvidia_dsa_local_logits_is_glm_only():
    from vllm.models.deepseek_v32.nvidia.model import DeepseekV32ForCausalLM

    class Processor:
        def __init__(self):
            self.calls = 0

        def get_local_logits(self, _: object, __: torch.Tensor) -> LocalLogits:
            self.calls += 1
            return _local_logits(torch.zeros(1, 2))

    processor = Processor()
    glm = SimpleNamespace(
        config=SimpleNamespace(model_type="glm_moe_dsa"),
        logits_processor=processor,
        lm_head=object(),
    )
    other = SimpleNamespace(
        config=SimpleNamespace(model_type="deepseek_v32"),
        logits_processor=processor,
        lm_head=object(),
    )
    assert DeepseekV32ForCausalLM.compute_local_logits(glm, torch.empty(1, 1))
    assert DeepseekV32ForCausalLM.compute_local_logits(other, torch.empty(1, 1)) is None
    assert processor.calls == 1


def test_dcp_falls_back_from_local_logits():
    class GLM:
        config = SimpleNamespace(model_type="glm_moe_dsa")

        @staticmethod
        def compute_local_logits(hidden_states: torch.Tensor) -> LocalLogits:
            return _local_logits(hidden_states)

    assert _get_prompt_logprobs_local_logits_fn(GLM(), dcp_size=1) is not None
    assert _get_prompt_logprobs_local_logits_fn(GLM(), dcp_size=4) is None
