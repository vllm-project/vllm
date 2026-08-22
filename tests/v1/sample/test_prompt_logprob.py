# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.warmup.kernel_warmup import (
    _warmup_compact_prompt_logprobs,
)
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample import prompt_logprob


@pytest.mark.parametrize("num_prompt_logprobs", [0, 5])
def test_compact_prompt_logprobs_callback_preserves_chunking(
    num_prompt_logprobs: int,
) -> None:
    num_rows = 2050
    hidden_states = torch.arange(num_rows * 2, dtype=torch.float32).view(num_rows, 2)
    target_token_ids = torch.arange(num_rows, dtype=torch.int32)
    chunk_sizes = []

    def compact_fn(
        chunk_hidden_states: torch.Tensor,
        chunk_token_ids: torch.Tensor,
        num_logprobs: int,
    ) -> LogprobsTensors:
        assert chunk_token_ids.dtype == torch.int64
        assert num_logprobs == num_prompt_logprobs
        chunk_sizes.append(chunk_hidden_states.shape[0])
        columns = torch.arange(num_logprobs + 1)
        return LogprobsTensors(
            logprob_token_ids=chunk_token_ids[:, None] + columns,
            logprobs=chunk_hidden_states[:, :1] + columns,
            selected_token_ranks=chunk_token_ids.to(torch.int32) + 1,
        )

    def unexpected_logits_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        raise AssertionError("compact path must not materialize logits")

    token_ids, logprobs, ranks = prompt_logprob.compute_prompt_logprobs_with_chunking(
        target_token_ids,
        hidden_states,
        unexpected_logits_fn,
        num_prompt_logprobs,
        compact_prompt_logprobs_fn=compact_fn,
    )

    columns = torch.arange(num_prompt_logprobs + 1)
    assert chunk_sizes == [1024, 1024, 2]
    assert torch.equal(token_ids, target_token_ids.to(torch.int64)[:, None] + columns)
    assert torch.equal(logprobs, hidden_states[:, :1] + columns)
    assert torch.equal(ranks, target_token_ids + 1)


def test_compact_prompt_logprobs_rejects_unsupported_mode() -> None:
    def unused_fn(*args) -> LogprobsTensors:
        raise AssertionError("invalid compact request must fail before execution")

    with pytest.raises(ValueError, match="compact prompt logprobs"):
        prompt_logprob.compute_prompt_logprobs_with_chunking(
            torch.tensor([1]),
            torch.zeros((1, 2)),
            unused_fn,
            5,
            "raw_logits",
            unused_fn,
        )


@pytest.mark.parametrize(
    ("num_prompt_logprobs", "with_compact_callback"),
    [(2, False), (-1, True), (33, True)],
)
def test_prompt_logprobs_uses_logits_fn_when_compact_is_unavailable(
    num_prompt_logprobs: int,
    with_compact_callback: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_sizes = []

    def logits_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        chunk_sizes.append(hidden_states.shape[0])
        return torch.zeros((hidden_states.shape[0], 40))

    def unexpected_compact_fn(*args) -> LogprobsTensors:
        raise AssertionError("unsupported K must use the native logits path")

    def fake_compute_topk_scores(
        logits: torch.Tensor,
        num_logprobs: int,
        target_token_ids: torch.Tensor,
        **kwargs,
    ) -> LogprobsTensors:
        expected_num_logprobs = 40 if num_prompt_logprobs == -1 else num_prompt_logprobs
        assert num_logprobs == expected_num_logprobs
        return LogprobsTensors(
            logprob_token_ids=target_token_ids[:, None],
            logprobs=torch.zeros((logits.shape[0], 1)),
            selected_token_ranks=torch.ones(logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_compute_topk_scores)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.arange(1025),
        torch.zeros((1025, 2)),
        logits_fn,
        num_prompt_logprobs,
        compact_prompt_logprobs_fn=(
            unexpected_compact_fn if with_compact_callback else None
        ),
    )

    assert chunk_sizes == [1024, 1]


@pytest.mark.parametrize("wrapped_model", [False, True])
def test_init_compact_prompt_logprobs_resolves_model_components(
    wrapped_model: bool,
) -> None:
    token_ids = torch.tensor([[1, 2]], dtype=torch.int32)
    logprobs = torch.tensor([[-0.1, -0.2]])
    ranks = torch.tensor([1], dtype=torch.int32)
    logits_processor = Mock()
    logits_processor.get_prompt_logprobs.return_value = token_ids, logprobs, ranks
    lm_head = Mock()
    language_model = SimpleNamespace(
        logits_processor=logits_processor,
        lm_head=lm_head,
    )
    model = language_model
    if wrapped_model:
        model = SimpleNamespace(get_language_model=Mock(return_value=language_model))

    compact_prompt_logprobs = prompt_logprob.init_compact_prompt_logprobs(
        model=model,
        hidden_dtype=torch.bfloat16,
        logprobs_mode="raw_logprobs",
    )

    logits_processor.validate_prompt_logprobs.assert_called_once_with(
        lm_head, torch.bfloat16
    )
    logits_processor.warmup_prompt_logprobs.assert_not_called()
    hidden_states = torch.empty((2, 4), dtype=torch.bfloat16)
    target_token_ids = torch.tensor([1, 2])
    output = compact_prompt_logprobs.compute(hidden_states, target_token_ids, 5)
    assert output.logprob_token_ids is token_ids
    assert output.logprobs is logprobs
    assert output.selected_token_ranks is ranks
    logits_processor.get_prompt_logprobs.assert_called_once_with(
        lm_head,
        hidden_states,
        target_token_ids,
        5,
    )
    compact_prompt_logprobs.warmup()
    logits_processor.warmup_prompt_logprobs.assert_called_once_with(lm_head)


def test_kernel_warmup_delegates_to_compact_prompt_logprobs() -> None:
    compact_prompt_logprobs = Mock()
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(
            compact_prompt_logprobs=compact_prompt_logprobs,
        ),
    )

    _warmup_compact_prompt_logprobs(worker)

    compact_prompt_logprobs.warmup.assert_called_once_with()


def test_kernel_warmup_skips_legacy_model_runner() -> None:
    model_runner = Mock()
    worker = SimpleNamespace(
        use_v2_model_runner=False,
        model_runner=model_runner,
    )

    _warmup_compact_prompt_logprobs(worker)

    assert not model_runner.mock_calls


def test_init_compact_prompt_logprobs_rejects_unsupported_model() -> None:
    with pytest.raises(RuntimeError, match="standard LM head and LogitsProcessor"):
        prompt_logprob.init_compact_prompt_logprobs(
            model=SimpleNamespace(),
            hidden_dtype=torch.bfloat16,
            logprobs_mode="raw_logprobs",
        )


def test_kernel_warmup_skips_without_compact_prompt_logprobs() -> None:
    model_runner = SimpleNamespace(compact_prompt_logprobs=None)
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=model_runner,
    )

    _warmup_compact_prompt_logprobs(worker)
