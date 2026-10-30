# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check inverse values, untouched padding, and graph replay for varlen GDN."""

import importlib
from itertools import accumulate

import pytest
import torch

solve_module = importlib.import_module(
    "vllm.third_party.flash_linear_attention.ops.solve_tril"
)


def make_inputs(lengths, padding=16, heads=32):
    torch.manual_seed(42)
    capacity = sum(lengths) + padding
    a = torch.zeros((1, capacity, heads, 64), device="cuda", dtype=torch.float32)
    chunks = []
    start = 0
    for seq, length in enumerate(lengths):
        for offset in range(0, length, 64):
            n = min(64, length - offset)
            values = torch.randn((heads, n, n), device="cuda") * 0.02
            a[0, start + offset : start + offset + n, :, :n] = torch.tril(
                values, diagonal=-1
            ).permute(1, 0, 2)
            chunks.append((seq, offset // 64))
        start += length
    cu = torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)
    indices = torch.tensor(chunks[::-1], device="cuda", dtype=torch.int32)
    return a, cu, indices


@pytest.mark.parametrize(
    "lengths", [[1], [15], [16], [17], [33], [49], [63], [64], [65, 79], [3216]]
)
def test_parallel_inverse_preserves_values_and_zero_padding(monkeypatch, lengths):
    if solve_module.is_tma_supported:
        pytest.skip("The opt-in implementation excludes TMA")
    a, cu, indices = make_inputs(lengths)
    monkeypatch.setenv("FLA_TRIL_PARALLEL_DIAGONALS", "0")
    expected = solve_module.solve_tril(a, cu, indices, torch.bfloat16)
    monkeypatch.setenv("FLA_TRIL_PARALLEL_DIAGONALS", "1")
    actual = solve_module.solve_tril(a, cu, indices, torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert torch.count_nonzero(actual[:, sum(lengths) :]).item() == 0
    start = 0
    for length in lengths:
        for offset in range(0, length, 64):
            n = min(64, length - offset)
            result = actual[0, start + offset : start + offset + n].permute(1, 0, 2)
            assert torch.count_nonzero(torch.triu(result, diagonal=1)).item() == 0
            matrix = a[0, start + offset : start + offset + n, :, :n].permute(1, 0, 2)
            identity = torch.eye(n, device=a.device).expand(32, n, n)
            reference = torch.linalg.solve_triangular(
                matrix + identity, identity, upper=False
            )
            torch.testing.assert_close(
                result[:, :, :n].float(), reference, atol=2e-3, rtol=2e-3
            )
        start += length


def test_parallel_inverse_graph_replay_uses_current_lengths(monkeypatch):
    if solve_module.is_tma_supported:
        pytest.skip("The opt-in implementation excludes TMA")
    a, cu, indices = make_inputs([65, 79], padding=0)
    monkeypatch.setenv("FLA_TRIL_PARALLEL_DIAGONALS", "1")
    solve_module.solve_tril(a, cu, indices, torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = solve_module.solve_tril(a, cu, indices, torch.bfloat16)
    changed, changed_cu, changed_indices = make_inputs([65, 78], padding=1)
    a.copy_(changed)
    cu.copy_(changed_cu)
    indices.copy_(changed_indices)
    graph.replay()
    monkeypatch.setenv("FLA_TRIL_PARALLEL_DIAGONALS", "0")
    expected = solve_module.solve_tril(a, cu, indices, torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert torch.count_nonzero(actual[:, 143:]).item() == 0


@pytest.mark.parametrize("heads,dtype", [(24, torch.bfloat16), (32, torch.float32)])
def test_parallel_option_preserves_unsupported_configurations(
    monkeypatch, heads, dtype
):
    a, cu, indices = make_inputs([17, 65], heads=heads)
    monkeypatch.setenv("FLA_TRIL_PARALLEL_DIAGONALS", "0")
    expected = solve_module.solve_tril(a, cu, indices, dtype)
    monkeypatch.setenv("FLA_TRIL_PARALLEL_DIAGONALS", "1")
    actual = solve_module.solve_tril(a, cu, indices, dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
