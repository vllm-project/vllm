# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact equivalence of the Triton and PyTorch Mamba block-table gathers."""

import pytest
import torch

from vllm.v1.attention.backends.utils import (
    mamba_get_block_table_tensor,
    mamba_get_block_table_tensor_reference,
)
from vllm.v1.kv_cache_interface import MambaSpec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size", [0, 1, 17])
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("num_spec", [0, 1, 15])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_aligned_blocks_match_reference(
    batch_size, block_size, num_spec, dtype, strided
):
    spec = _make_spec(block_size, num_spec)
    table, lengths = _make_inputs(
        batch_size, block_size, num_spec, dtype, strided, "cuda"
    )
    expected = mamba_get_block_table_tensor_reference(table, lengths, spec, "align")
    actual = mamba_get_block_table_tensor(table, lengths, spec, "align")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["all", "none", "align"])
def test_cpu_behavior_matches_reference(mode):
    spec = _make_spec(16, 3)
    table, lengths = _make_inputs(17, 16, 3, torch.int32, True, "cpu")
    expected = mamba_get_block_table_tensor_reference(table, lengths, spec, mode)
    actual = mamba_get_block_table_tensor(table, lengths, spec, mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if mode != "align":
        assert actual is table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["all", "none"])
def test_cuda_passthrough_preserves_identity(mode):
    spec = _make_spec(16, 3)
    table, lengths = _make_inputs(17, 16, 3, torch.int64, True, "cuda")
    assert mamba_get_block_table_tensor_reference(table, lengths, spec, mode) is table
    assert mamba_get_block_table_tensor(table, lengths, spec, mode) is table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_spec", [0, 15])
def test_graph_replay_uses_updated_block_table_and_lengths(num_spec):
    spec = _make_spec(16, num_spec)
    table, lengths = _make_inputs(17, 16, num_spec, torch.int32, True, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mamba_get_block_table_tensor(table, lengths, spec, "align")
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = mamba_get_block_table_tensor(table, lengths, spec, "align")

    for shift in (0, 1, 16):
        table.add_(97)
        lengths.copy_((torch.arange(17, device="cuda") * 16 + shift) % 129)
        graph.replay()
        expected = mamba_get_block_table_tensor_reference(table, lengths, spec, "align")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _make_spec(block_size, num_spec):
    return MambaSpec(
        block_size=block_size,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        num_speculative_blocks=num_spec,
    )


def _make_inputs(batch_size, block_size, num_spec, dtype, strided, device):
    num_columns = 8 + num_spec
    step = 2 if strided else 1
    # Distinct values expose both row and column indexing errors.
    table = torch.arange(
        (batch_size + 1) * num_columns * step, dtype=dtype, device=device
    ).reshape(batch_size + 1, num_columns * step)[:batch_size, ::step]
    boundaries = [
        0,
        1,
        block_size - 1,
        block_size,
        block_size + 1,
        2 * block_size - 1,
        2 * block_size,
        2 * block_size + 1,
        8 * block_size,
    ]
    lengths = torch.zeros(batch_size * step, dtype=dtype, device=device)[::step]
    lengths.copy_(
        torch.tensor(
            [boundaries[i % len(boundaries)] for i in range(batch_size)],
            dtype=dtype,
            device=device,
        )
    )
    return table, lengths
