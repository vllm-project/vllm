# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for sampled-token dedup in top-k logprobs.

``compute_topk_scores`` launches Triton kernels for log-softmax and ranks.
This module patches those with torch equivalents so the production fast path
(topk + ``_drop_sampled_token_from_topk`` + concat) can run on CPU tensors.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch

from vllm.logprobs import append_logprobs_for_next_position, create_sample_logprobs

_MISSING = object()
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _try_import_logprob_module():
    try:
        return importlib.import_module("vllm.v1.worker.gpu.sample.logprob")
    except Exception:
        return None


def _stub_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_logprob_module_isolated():
    saved: dict[str, object] = {}

    def _install(name: str, module: types.ModuleType) -> None:
        saved[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = module

    _install(
        "vllm.sampling_params",
        _stub_module(
            "vllm.sampling_params",
            MAX_LOGPROB_TOKEN_IDS=128,
            SamplingParams=type("SamplingParams", (), {}),
        ),
    )

    class LogprobsTensors(NamedTuple):
        logprob_token_ids: torch.Tensor
        logprobs: torch.Tensor
        selected_token_ranks: torch.Tensor
        cu_num_generated_tokens: list[int] | None = None
        cu_num_generated_tokens_tensor: torch.Tensor | None = None

    _install(
        "vllm.v1.outputs",
        _stub_module("vllm.v1.outputs", LogprobsTensors=LogprobsTensors),
    )
    _install(
        "vllm.v1.worker.gpu.buffer_utils",
        _stub_module(
            "vllm.v1.worker.gpu.buffer_utils",
            StagedWriteTensor=object,
            UvaBackedTensor=object,
        ),
    )

    path = _REPO_ROOT / "vllm" / "v1" / "worker" / "gpu" / "sample" / "logprob.py"
    spec = importlib.util.spec_from_file_location("_vllm_logprob_cpu_dedup_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous  # type: ignore[assignment]
    return module


def _install_cpu_kernels(module) -> None:
    def compute_token_logprobs(
        logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        token_ids = token_ids.to(torch.int64)
        return torch.log_softmax(logits.float(), dim=-1).gather(-1, token_ids)

    def _ranks_launch(
        output: torch.Tensor,
        logits: torch.Tensor,
        logits_stride: int,
        token_ids: torch.Tensor,
        vocab_size: int,
        BLOCK_SIZE: int,
    ) -> None:
        sampled_logits = logits.gather(-1, token_ids.view(-1, 1))
        output.copy_((logits >= sampled_logits).sum(dim=-1).to(output.dtype))

    class _CpuKernel:
        def __getitem__(self, grid: Any) -> Callable[..., None]:
            return _ranks_launch

    module.compute_token_logprobs = compute_token_logprobs
    module._ranks_kernel = _CpuKernel()


_LOGPROB_MOD = _try_import_logprob_module() or _load_logprob_module_isolated()
_install_cpu_kernels(_LOGPROB_MOD)
_drop_sampled_token_from_topk = _LOGPROB_MOD._drop_sampled_token_from_topk
compute_topk_scores = _LOGPROB_MOD.compute_topk_scores


def _expected_fast_path_ranks(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    num_logprobs: int,
) -> torch.Tensor:
    sampled_rank = (logits >= logits.gather(-1, sampled_token_ids.view(-1, 1))).sum(
        dim=-1
    )
    if num_logprobs <= 0:
        return sampled_rank.unsqueeze(-1).to(torch.int64)
    k = min(num_logprobs + 1, logits.shape[1])
    topk_ids = torch.topk(logits, k, dim=-1).indices
    _, topk_ranks = _drop_sampled_token_from_topk(
        topk_ids, sampled_token_ids, num_logprobs
    )
    return torch.cat((sampled_rank.unsqueeze(-1), topk_ranks), dim=1).to(torch.int64)


def _reconstruct_row(result, row_idx: int, num_logprobs: int):
    request_logprobs = create_sample_logprobs(flat_logprobs=False)
    append_logprobs_for_next_position(
        request_logprobs,
        result.logprob_token_ids[row_idx].tolist(),
        result.logprobs[row_idx].tolist(),
        itertools.repeat(None),
        result.selected_token_ranks[row_idx].tolist(),
        num_logprobs,
    )
    return request_logprobs[0]


def _assert_cpu_topk_result(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    num_logprobs: int,
) -> None:
    result = compute_topk_scores(logits, num_logprobs, sampled_token_ids)
    token_ids = result.logprob_token_ids
    vocab_size = logits.shape[1]
    expected_width = 1 + min(num_logprobs, max(vocab_size - 1, 0))
    assert token_ids.shape[1] == expected_width
    assert result.selected_token_ranks.ndim == 2
    assert result.selected_token_ranks.shape == token_ids.shape
    expected_ranks = _expected_fast_path_ranks(logits, sampled_token_ids, num_logprobs)
    assert torch.equal(result.selected_token_ranks, expected_ranks)
    for row_idx in range(token_ids.shape[0]):
        row = token_ids[row_idx].tolist()
        assert row[0] == int(sampled_token_ids[row_idx].item())
        assert len(row) == len(set(row))
        reconstructed = _reconstruct_row(result, row_idx, num_logprobs)
        expected = dict(zip(row, result.selected_token_ranks[row_idx].tolist()))
        actual = {token_id: logprob.rank for token_id, logprob in reconstructed.items()}
        assert actual == expected


@pytest.mark.cpu_test
def test_drop_sampled_token_inside_topk_each_position() -> None:
    topk_token_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.int64)
    for sampled_id in (10, 20, 30, 40):
        token_ids, ranks = _drop_sampled_token_from_topk(
            topk_token_ids, torch.tensor([sampled_id], dtype=torch.int64), 3
        )
        kept = [token for token in [10, 20, 30, 40] if token != sampled_id]
        kept_ranks = [
            rank
            for rank, token in enumerate([10, 20, 30, 40], start=1)
            if token != sampled_id
        ]
        assert token_ids[0].tolist() == kept
        assert ranks[0].tolist() == kept_ranks
        assert sampled_id not in token_ids[0].tolist()
        assert token_ids.shape[1] == 3


@pytest.mark.cpu_test
def test_drop_sampled_token_outside_topk() -> None:
    topk_token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    token_ids, ranks = _drop_sampled_token_from_topk(
        topk_token_ids, torch.tensor([9], dtype=torch.int64), 3
    )
    assert torch.equal(token_ids, torch.tensor([[0, 1, 2]], dtype=torch.int64))
    assert torch.equal(ranks, torch.tensor([[1, 2, 3]], dtype=torch.int64))


@pytest.mark.cpu_test
def test_drop_sampled_token_when_k_exceeds_vocab() -> None:
    topk_token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    token_ids, ranks = _drop_sampled_token_from_topk(
        topk_token_ids, torch.tensor([0], dtype=torch.int64), 5
    )
    assert torch.equal(token_ids, torch.tensor([[1, 2, 3]], dtype=torch.int64))
    assert torch.equal(ranks, torch.tensor([[2, 3, 4]], dtype=torch.int64))
    assert token_ids.shape[1] == 3


@pytest.mark.cpu_test
def test_drop_sampled_token_mixed_batch() -> None:
    topk_token_ids = torch.tensor([[1, 2, 3], [0, 1, 2], [4, 5, 6]], dtype=torch.int64)
    sampled_token_ids = torch.tensor([1, 9, 6], dtype=torch.int64)
    token_ids, ranks = _drop_sampled_token_from_topk(
        topk_token_ids, sampled_token_ids, 2
    )
    assert torch.equal(
        token_ids, torch.tensor([[2, 3], [0, 1], [4, 5]], dtype=torch.int64)
    )
    assert torch.equal(ranks, torch.tensor([[2, 3], [1, 2], [1, 2]], dtype=torch.int64))
    for row, sampled in zip(token_ids.tolist(), sampled_token_ids.tolist()):
        assert sampled not in row
        assert len(row) == len(set(row))
        assert len(row) == 2


@pytest.mark.cpu_test
def test_drop_sampled_token_ties_keep_first_match_only() -> None:
    topk_token_ids = torch.tensor([[7, 7, 8]], dtype=torch.int64)
    token_ids, ranks = _drop_sampled_token_from_topk(
        topk_token_ids, torch.tensor([7], dtype=torch.int64), 2
    )
    assert token_ids[0].tolist() == [7, 8]
    assert ranks[0].tolist() == [2, 3]


@pytest.mark.cpu_test
def test_compute_topk_scores_sampled_inside_topk_each_position() -> None:
    logits = torch.tensor(
        [
            [9.0, 8.0, 7.0, 6.0, 5.0],
            [8.0, 9.0, 7.0, 6.0, 5.0],
            [8.0, 7.0, 6.0, 9.0, 5.0],
        ],
        dtype=torch.float32,
    )
    sampled_token_ids = torch.tensor([0, 1, 3], dtype=torch.int64)
    _assert_cpu_topk_result(logits, sampled_token_ids, 2)


@pytest.mark.cpu_test
def test_compute_topk_scores_sampled_outside_topk() -> None:
    logits = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]], dtype=torch.float32)
    sampled_token_ids = torch.tensor([4], dtype=torch.int64)
    _assert_cpu_topk_result(logits, sampled_token_ids, 3)
    result = compute_topk_scores(logits, 3, sampled_token_ids)
    assert torch.equal(
        result.logprob_token_ids, torch.tensor([[4, 0, 1, 2]], dtype=torch.int64)
    )
    assert torch.equal(
        result.selected_token_ranks, torch.tensor([[5, 1, 2, 3]], dtype=torch.int64)
    )


@pytest.mark.cpu_test
def test_compute_topk_scores_ties_in_logits() -> None:
    logits = torch.tensor(
        [
            [5.0, 5.0, 5.0, 1.0, 0.0],
            [9.0, 5.0, 5.0, 5.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sampled_token_ids = torch.tensor([1, 4], dtype=torch.int64)
    _assert_cpu_topk_result(logits, sampled_token_ids, 2)


@pytest.mark.cpu_test
def test_compute_topk_scores_when_k_plus_one_exceeds_vocab() -> None:
    logits = torch.tensor([[9.0, 8.0, 7.0, 6.0]], dtype=torch.float32)
    sampled_in = torch.tensor([0], dtype=torch.int64)
    sampled_last = torch.tensor([3], dtype=torch.int64)
    _assert_cpu_topk_result(logits, sampled_in, 5)
    _assert_cpu_topk_result(logits, sampled_in, 4)
    _assert_cpu_topk_result(logits, sampled_last, 5)
    result = compute_topk_scores(logits, 5, sampled_in)
    assert result.logprob_token_ids.shape == (1, 4)
    assert result.logprob_token_ids[0].tolist() == [0, 1, 2, 3]
    assert result.selected_token_ranks[0].tolist() == [1, 2, 3, 4]


@pytest.mark.cpu_test
def test_compute_topk_scores_mixed_batch() -> None:
    logits = torch.tensor(
        [
            [0.0, 9.0, 8.0, 7.0, 6.0],
            [9.0, 8.0, 7.0, 6.0, 5.0],
            [9.0, 8.0, 7.0, 6.0, 5.0],
        ],
        dtype=torch.float32,
    )
    sampled_token_ids = torch.tensor([1, 2, 4], dtype=torch.int64)
    _assert_cpu_topk_result(logits, sampled_token_ids, 2)
    result = compute_topk_scores(logits, 2, sampled_token_ids)
    assert torch.equal(
        result.logprob_token_ids,
        torch.tensor([[1, 2, 3], [2, 0, 1], [4, 0, 1]], dtype=torch.int64),
    )
    assert torch.equal(
        result.selected_token_ranks,
        torch.tensor([[1, 2, 3], [3, 1, 2], [5, 1, 2]], dtype=torch.int64),
    )


@pytest.mark.cpu_test
def test_compute_topk_scores_vocab_size_one() -> None:
    logits = torch.tensor([[3.0]], dtype=torch.float32)
    sampled_token_ids = torch.tensor([0], dtype=torch.int64)
    _assert_cpu_topk_result(logits, sampled_token_ids, 2)
    result = compute_topk_scores(logits, 2, sampled_token_ids)
    assert torch.equal(result.logprob_token_ids, torch.tensor([[0]], dtype=torch.int64))
    assert torch.equal(
        result.selected_token_ranks, torch.tensor([[1]], dtype=torch.int64)
    )
