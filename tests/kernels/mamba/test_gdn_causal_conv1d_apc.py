# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn as official_causal_conv1d_fn,
)
from vllm.model_executor.layers.mamba.ops.gdn_causal_conv1d import (
    causal_conv1d_fn as replacement_causal_conv1d_fn,
)


@pytest.mark.parametrize("has_initial", [False, True])
@pytest.mark.parametrize(
    "tokens,computed,first,last,block_size",
    [(1024, 0, 1, 3, 128), (1021, 3, 1, 2, 128)],
)
def test_apc_output_and_all_cache_writes_match_official(
    has_initial: bool,
    tokens: int,
    computed: int,
    first: int,
    last: int,
    block_size: int,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dim = 256
    width = 4
    backing = torch.randn(tokens, dim + 64, device=device, dtype=torch.bfloat16)
    x = backing[:, :dim].transpose(0, 1)
    weight = torch.randn(dim, width, device=device, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=device, dtype=torch.bfloat16)
    states = torch.randn(8, width - 1, dim, device=device, dtype=torch.bfloat16)
    states = states.transpose(1, 2)
    cache_indices = torch.tensor([[1, 2, 3, 4]], device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    has_initial_state = torch.tensor([has_initial], device=device)
    first_tensor = torch.tensor([first], device=device, dtype=torch.int32)
    last_tensor = torch.tensor([last], device=device, dtype=torch.int32)
    initial_state_idx = torch.tensor([0], device=device, dtype=torch.int32)
    num_computed_tokens = torch.tensor([computed], device=device, dtype=torch.int32)

    def run(fn, state_pool):
        return fn(
            x,
            weight,
            bias,
            state_pool,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
            block_idx_first_scheduled_token=first_tensor,
            block_idx_last_scheduled_token=last_tensor,
            initial_state_idx=initial_state_idx,
            num_computed_tokens=num_computed_tokens,
            block_size_to_align=block_size,
        )

    official_states = states.clone()
    expected = run(official_causal_conv1d_fn, official_states)
    replacement_states = states.clone()
    actual = run(replacement_causal_conv1d_fn, replacement_states)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)
    torch.testing.assert_close(replacement_states, official_states, rtol=0, atol=0)


def test_apc_multi_sequence_cache_rows_match_official() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    dim = 256
    width = 4
    lengths = (512, 512)
    total_tokens = sum(lengths)
    x = torch.randn(total_tokens, dim + 64, device=device, dtype=torch.bfloat16)
    x = x[:, :dim].transpose(0, 1)
    weight = torch.randn(dim, width, device=device, dtype=torch.bfloat16)
    states = torch.randn(12, width - 1, dim, device=device, dtype=torch.bfloat16)
    states = states.transpose(1, 2)
    query_start_loc = torch.tensor([0, 512, 1024], device=device, dtype=torch.int32)
    cache_indices = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8]], device=device, dtype=torch.int32
    )
    has_initial_state = torch.tensor([True, False], device=device)
    first = torch.tensor([1, 1], device=device, dtype=torch.int32)
    last = torch.tensor([3, 2], device=device, dtype=torch.int32)
    initial = torch.tensor([0, 0], device=device, dtype=torch.int32)
    computed = torch.tensor([0, 3], device=device, dtype=torch.int32)

    def run(fn, state_pool):
        return fn(
            x,
            weight,
            None,
            state_pool,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=None,
            block_idx_first_scheduled_token=first,
            block_idx_last_scheduled_token=last,
            initial_state_idx=initial,
            num_computed_tokens=computed,
            block_size_to_align=128,
        )

    official_states = states.clone()
    expected = run(official_causal_conv1d_fn, official_states)
    replacement_states = states.clone()
    actual = run(replacement_causal_conv1d_fn, replacement_states)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)
    torch.testing.assert_close(replacement_states, official_states, rtol=0, atol=0)
