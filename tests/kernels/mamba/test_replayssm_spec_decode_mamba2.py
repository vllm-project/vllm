# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for fold-every-commit ReplaySSM speculative decode."""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_spec import (  # noqa: E501
    ReplaySSMSpecCommitContext,
    selective_state_update_replayssm_spec,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

DEV = "cuda"
NUM_LAYERS = 3
NUM_BLOCKS = 8
BATCH = 3
NHEADS = 4
HEAD_DIM = 32
DSTATE = 16
NGROUPS = 2
SPEC_QUERY_LEN = 4
CONV_DIM = 48
CONV_HISTORY_LEN = 3
CONV_STATE_LEN = CONV_HISTORY_LEN + SPEC_QUERY_LEN - 1

_PRECISIONS = [
    (torch.float32, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16),
]


def _tolerances(
    state_dtype: torch.dtype, activation_dtype: torch.dtype
) -> tuple[float, float]:
    if state_dtype == activation_dtype == torch.float32:
        return 2e-4, 2e-3
    return 6e-2, 2e-1


def _padded_states(dtype: torch.dtype) -> list[torch.Tensor]:
    natural_size = NHEADS * HEAD_DIM * DSTATE
    states = []
    for layer_idx in range(NUM_LAYERS):
        storage = torch.randn(
            NUM_BLOCKS,
            natural_size + 8 * (layer_idx + 1),
            device=DEV,
            dtype=dtype,
        )
        states.append(
            storage[:, :natural_size].view(NUM_BLOCKS, NHEADS, HEAD_DIM, DSTATE)
        )
    return states


def _conv_states(dtype: torch.dtype) -> list[torch.Tensor]:
    states = []
    for layer_idx in range(NUM_LAYERS):
        if layer_idx % 2 == 0:
            state = torch.randn(
                NUM_BLOCKS,
                CONV_DIM,
                CONV_STATE_LEN,
                device=DEV,
                dtype=dtype,
            )
        else:
            state = torch.randn(
                NUM_BLOCKS,
                CONV_STATE_LEN,
                CONV_DIM,
                device=DEV,
                dtype=dtype,
            ).transpose(-1, -2)
        states.append(state)
    return states


def _padded_caches(shape: tuple[int, ...], dtype: torch.dtype) -> list[torch.Tensor]:
    natural_size = 1
    for size in shape:
        natural_size *= size
    caches = []
    for layer_idx in range(NUM_LAYERS):
        storage = torch.empty(
            NUM_BLOCKS,
            natural_size + 8 * (layer_idx + 1),
            device=DEV,
            dtype=dtype,
        )
        caches.append(storage[:, :natural_size].view(NUM_BLOCKS, *shape))
    return caches


def _make_context(
    state_dtype: torch.dtype,
    activation_dtype: torch.dtype,
) -> tuple[
    ReplaySSMSpecCommitContext,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    states = _padded_states(state_dtype)
    conv_states = _conv_states(activation_dtype)
    x_caches = _padded_caches((NHEADS, SPEC_QUERY_LEN, HEAD_DIM), activation_dtype)
    dt_caches = _padded_caches((NHEADS, SPEC_QUERY_LEN), torch.float32)
    B_caches = _padded_caches((NGROUPS, SPEC_QUERY_LEN, DSTATE), activation_dtype)
    A = [
        -torch.rand(NHEADS, device=DEV, dtype=torch.float32) - 1.0
        for _ in range(NUM_LAYERS)
    ]
    dt_bias = [
        torch.rand(NHEADS, device=DEV, dtype=torch.float32) - 4.0
        for _ in range(NUM_LAYERS)
    ]
    context = ReplaySSMSpecCommitContext.create(
        conv_states,
        states,
        x_caches,
        dt_caches,
        B_caches,
        A,
        dt_bias,
        ngroups=NGROUPS,
        spec_query_len=SPEC_QUERY_LEN,
    )
    return context, A, dt_bias


def _make_inputs(
    query_lens: list[int], activation_dtype: torch.dtype
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    num_tokens = sum(query_lens)
    return [
        (
            torch.randn(
                num_tokens,
                NHEADS,
                HEAD_DIM,
                device=DEV,
                dtype=activation_dtype,
            ),
            torch.randn(num_tokens, NHEADS, device=DEV, dtype=activation_dtype),
            torch.randn(
                num_tokens,
                NGROUPS,
                DSTATE,
                device=DEV,
                dtype=activation_dtype,
            ),
            torch.randn(
                num_tokens,
                NGROUPS,
                DSTATE,
                device=DEV,
                dtype=activation_dtype,
            ),
            torch.randn(NHEADS, HEAD_DIM, device=DEV, dtype=activation_dtype),
        )
        for _ in range(NUM_LAYERS)
    ]


def _run_baseline(
    state: torch.Tensor,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    A: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    query_lens: list[int],
    run_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    x, dt, B, C, D = inputs
    expected_state = state.clone()
    expected_out = torch.full_like(x, torch.nan)
    A_expanded = A[:, None, None].expand(NHEADS, HEAD_DIM, DSTATE)
    dt_bias_expanded = dt_bias[:, None].expand(NHEADS, HEAD_DIM)
    token_start = 0
    for batch_idx, (query_len, run_len) in enumerate(zip(query_lens, run_lens)):
        state_idx = int(state_indices[batch_idx].item())
        if state_idx == NULL_BLOCK_ID:
            token_start += query_len
            continue
        state_index = state_indices[batch_idx : batch_idx + 1]
        for offset in range(run_len):
            token_idx = token_start + offset
            out = expected_out[token_idx : token_idx + 1]
            selective_state_update(
                expected_state,
                x[token_idx : token_idx + 1],
                dt[token_idx : token_idx + 1, :, None].expand(1, NHEADS, HEAD_DIM),
                A_expanded,
                B[token_idx : token_idx + 1],
                C[token_idx : token_idx + 1],
                D=D,
                dt_bias=dt_bias_expanded,
                dt_softplus=True,
                state_batch_indices=state_index,
                out=out,
            )
        token_start += query_len
    return expected_state, expected_out


def _query_start_loc(query_lens: list[int]) -> torch.Tensor:
    starts = [0]
    for query_len in query_lens:
        starts.append(starts[-1] + query_len)
    return torch.tensor(starts, device=DEV, dtype=torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16])
def test_commit_compacts_conv_state_to_canonical_history(activation_dtype):
    context, _, _ = _make_context(torch.float32, activation_dtype)
    initial_states = [state.clone() for state in context.conv_states]
    query_lens = [SPEC_QUERY_LEN, 2, SPEC_QUERY_LEN]
    accepted = [3, 1, SPEC_QUERY_LEN]
    state_indices = torch.tensor([6, 2, NULL_BLOCK_ID], device=DEV, dtype=torch.int32)

    context.commit(
        torch.tensor(accepted, device=DEV, dtype=torch.int32),
        state_indices,
        _query_start_loc(query_lens),
    )

    for actual, initial in zip(context.conv_states, initial_states):
        for batch_idx, block_idx in enumerate(state_indices.tolist()):
            if block_idx == NULL_BLOCK_ID:
                continue
            offset = accepted[batch_idx] - 1
            torch.testing.assert_close(
                actual[block_idx, :, :CONV_HISTORY_LEN],
                initial[
                    block_idx,
                    :,
                    offset : offset + CONV_HISTORY_LEN,
                ],
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_verify_rejects_window_larger_than_activation_capacity():
    context, A, dt_bias = _make_context(torch.float32, torch.float32)
    x, dt, B, C, D = _make_inputs([SPEC_QUERY_LEN + 1], torch.float32)[0]

    with pytest.raises(ValueError, match="activation capacity"):
        selective_state_update_replayssm_spec(
            context.state_checkpoints[0],
            context.x_caches[0],
            context.dt_caches[0],
            context.B_caches[0],
            x,
            dt,
            B,
            C,
            A[0][:, None, None].expand(NHEADS, HEAD_DIM, DSTATE),
            query_start_loc=_query_start_loc([SPEC_QUERY_LEN + 1]),
            state_batch_indices=torch.tensor([1], device=DEV, dtype=torch.int32),
            spec_query_len=SPEC_QUERY_LEN,
            D=D,
            dt_bias=dt_bias[0],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
def test_group_verify_and_all_layer_commit_match_baseline(precision):
    """One group launch commits every layer and only each accepted prefix."""
    state_dtype, activation_dtype = precision
    set_random_seed(0)
    context, A, dt_bias = _make_context(state_dtype, activation_dtype)
    initial_states = [state.clone() for state in context.state_checkpoints]
    query_lens = [SPEC_QUERY_LEN, 2, SPEC_QUERY_LEN]
    accepted = [2, 1, SPEC_QUERY_LEN]
    state_indices = torch.tensor([6, 2, NULL_BLOCK_ID], device=DEV, dtype=torch.int32)
    query_start_loc = _query_start_loc(query_lens)
    inputs = _make_inputs(query_lens, activation_dtype)
    rtol, atol = _tolerances(state_dtype, activation_dtype)
    expected_committed_states = []

    for layer_idx in range(NUM_LAYERS):
        x, dt, B, C, D = inputs[layer_idx]
        x_cache = context.x_caches[layer_idx]
        dt_cache = context.dt_caches[layer_idx]
        B_cache = context.B_caches[layer_idx]
        out = torch.full_like(x, torch.nan)
        selective_state_update_replayssm_spec(
            context.state_checkpoints[layer_idx],
            x_cache,
            dt_cache,
            B_cache,
            x,
            dt,
            B,
            C,
            A[layer_idx][:, None, None].expand(NHEADS, HEAD_DIM, DSTATE),
            query_start_loc=query_start_loc,
            state_batch_indices=state_indices,
            spec_query_len=SPEC_QUERY_LEN,
            D=D,
            dt_bias=dt_bias[layer_idx],
            out=out,
        )
        _, expected_out = _run_baseline(
            initial_states[layer_idx],
            inputs[layer_idx],
            A[layer_idx],
            dt_bias[layer_idx],
            state_indices,
            query_lens,
            query_lens,
        )
        for batch_idx in range(BATCH - 1):
            start = int(query_start_loc[batch_idx].item())
            end = int(query_start_loc[batch_idx + 1].item())
            torch.testing.assert_close(
                out[start:end], expected_out[start:end], rtol=rtol, atol=atol
            )
        assert torch.isnan(out[-SPEC_QUERY_LEN:]).all()
        torch.testing.assert_close(
            context.state_checkpoints[layer_idx],
            initial_states[layer_idx],
            rtol=0,
            atol=0,
        )
        expected_state, _ = _run_baseline(
            initial_states[layer_idx],
            inputs[layer_idx],
            A[layer_idx],
            dt_bias[layer_idx],
            state_indices,
            query_lens,
            accepted,
        )
        expected_committed_states.append(expected_state)

    context.commit(
        torch.tensor(accepted, device=DEV, dtype=torch.int32),
        state_indices,
        query_start_loc,
    )

    for actual, expected in zip(context.state_checkpoints, expected_committed_states):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    assert all(cache.shape[0] == NUM_BLOCKS for cache in context.x_caches)
    assert context.x_caches[0].stride(0) != context.x_caches[1].stride(0)
    assert context.state_checkpoints[0].stride(0) != (
        context.state_checkpoints[1].stride(0)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("precision", _PRECISIONS)
def test_forced_single_prompt_token_is_committed(precision):
    """A discarded one-token prompt tail still advances every checkpoint."""
    state_dtype, activation_dtype = precision
    set_random_seed(1)
    context, A, dt_bias = _make_context(state_dtype, activation_dtype)
    initial_states = [state.clone() for state in context.state_checkpoints]
    query_lens = [1, SPEC_QUERY_LEN, SPEC_QUERY_LEN]
    accepted = [0, 1, 0]
    force_commit = [True, False, False]
    commit_lens = [1, 1, 0]
    state_indices = torch.tensor([7, 3, NULL_BLOCK_ID], device=DEV, dtype=torch.int32)
    query_start_loc = _query_start_loc(query_lens)
    inputs = _make_inputs(query_lens, activation_dtype)
    expected_states = []
    rtol, atol = _tolerances(state_dtype, activation_dtype)

    for layer_idx in range(NUM_LAYERS):
        x, dt, B, C, D = inputs[layer_idx]
        selective_state_update_replayssm_spec(
            context.state_checkpoints[layer_idx],
            context.x_caches[layer_idx],
            context.dt_caches[layer_idx],
            context.B_caches[layer_idx],
            x,
            dt,
            B,
            C,
            A[layer_idx][:, None, None].expand(NHEADS, HEAD_DIM, DSTATE),
            query_start_loc=query_start_loc,
            state_batch_indices=state_indices,
            spec_query_len=SPEC_QUERY_LEN,
            D=D,
            dt_bias=dt_bias[layer_idx],
        )
        expected_state, _ = _run_baseline(
            initial_states[layer_idx],
            inputs[layer_idx],
            A[layer_idx],
            dt_bias[layer_idx],
            state_indices,
            query_lens,
            commit_lens,
        )
        expected_states.append(expected_state)

    context.commit(
        torch.tensor(accepted, device=DEV, dtype=torch.int32),
        state_indices,
        query_start_loc,
        force_commit=torch.tensor(force_commit, device=DEV, dtype=torch.bool),
    )

    for actual, expected in zip(context.state_checkpoints, expected_states):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_fold_every_commit_reuses_current_window_buffers():
    """Successive verify windows leave no pending cursor or acceptance state."""
    set_random_seed(2)
    context, A, dt_bias = _make_context(torch.float32, torch.float32)
    expected_states = [state.clone() for state in context.state_checkpoints]
    query_lens = [SPEC_QUERY_LEN] * BATCH
    state_indices = torch.tensor([1, 4, 7], device=DEV, dtype=torch.int32)
    query_start_loc = _query_start_loc(query_lens)

    for step in range(5):
        inputs = _make_inputs(query_lens, torch.float32)
        accepted = [
            1 + (step + batch_idx) % SPEC_QUERY_LEN for batch_idx in range(BATCH)
        ]
        for layer_idx in range(NUM_LAYERS):
            x, dt, B, C, D = inputs[layer_idx]
            selective_state_update_replayssm_spec(
                context.state_checkpoints[layer_idx],
                context.x_caches[layer_idx],
                context.dt_caches[layer_idx],
                context.B_caches[layer_idx],
                x,
                dt,
                B,
                C,
                A[layer_idx][:, None, None].expand(NHEADS, HEAD_DIM, DSTATE),
                query_start_loc=query_start_loc,
                state_batch_indices=state_indices,
                spec_query_len=SPEC_QUERY_LEN,
                D=D,
                dt_bias=dt_bias[layer_idx],
            )
            expected_states[layer_idx], _ = _run_baseline(
                expected_states[layer_idx],
                inputs[layer_idx],
                A[layer_idx],
                dt_bias[layer_idx],
                state_indices,
                query_lens,
                accepted,
            )
        context.commit(
            torch.tensor(accepted, device=DEV, dtype=torch.int32),
            state_indices,
            query_start_loc,
        )

    for actual, expected in zip(context.state_checkpoints, expected_states):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)
