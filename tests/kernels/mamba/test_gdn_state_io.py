# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.gdn_state_io import (
    gather_gdn_initial_state,
    scatter_gdn_final_state,
)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 3])
def test_gather_gdn_initial_state_matches_pytorch(
    cache_dtype: torch.dtype,
    batch: int,
) -> None:
    torch.manual_seed(0)
    cache = torch.randn(7, 4, 8, 16, device="cuda", dtype=cache_dtype)
    indices = torch.tensor([5, 1, 6], device="cuda", dtype=torch.int32)[:batch]
    has_initial_state = torch.tensor(
        [True, False, True], device="cuda", dtype=torch.bool
    )[:batch]

    actual = gather_gdn_initial_state(cache, indices, has_initial_state)
    expected = cache[indices].float()
    expected[~has_initial_state] = 0

    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 3])
def test_scatter_gdn_final_state_matches_pytorch(
    cache_dtype: torch.dtype,
    batch: int,
) -> None:
    torch.manual_seed(0)
    cache = torch.randn(7, 4, 8, 16, device="cuda", dtype=cache_dtype)
    indices = torch.tensor([5, 1, 6], device="cuda", dtype=torch.int32)[:batch]
    final_state = torch.randn(batch, 4, 8, 16, device="cuda", dtype=torch.float32)

    expected = cache.clone()
    expected[indices.long()] = final_state.to(cache_dtype)
    actual = cache.clone()
    scatter_gdn_final_state(actual, indices, final_state)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_gdn_state_io_supports_distinct_source_and_destination_indices() -> None:
    torch.manual_seed(0)
    cache = torch.randn(6, 4, 8, 16, device="cuda", dtype=torch.bfloat16)
    source_indices = torch.tensor([1, 4], device="cuda", dtype=torch.int32)
    destination_indices = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
    has_initial_state = torch.ones(2, device="cuda", dtype=torch.bool)

    initial_state = gather_gdn_initial_state(cache, source_indices, has_initial_state)
    expected = cache.clone()
    expected[destination_indices.long()] = initial_state.to(cache.dtype)
    scatter_gdn_final_state(cache, destination_indices, initial_state)

    torch.testing.assert_close(cache, expected, atol=0, rtol=0)


def test_gdn_state_io_supports_noncontiguous_final_state() -> None:
    torch.manual_seed(0)
    cache = torch.randn(5, 4, 8, 16, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    backing = torch.randn(2, 4, 8, 32, device="cuda", dtype=torch.float32)
    final_state = backing[..., ::2]

    expected = cache.clone()
    expected[indices.long()] = final_state.to(cache.dtype)
    scatter_gdn_final_state(cache, indices, final_state)

    torch.testing.assert_close(cache, expected, atol=0, rtol=0)
