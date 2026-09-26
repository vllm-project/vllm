# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the CPU torch stand-ins of V2 model-runner Triton kernels.

The kernels cannot run here to be compared against, so each reference below is
a literal transcription of one kernel's pointer arithmetic, kept deliberately
scalar. The fallbacks are vectorized instead, so the two disagree readily if
either drifts.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.cpu.mamba_utils import (
    _full_rows,
    preprocess_mamba_align,
    run_fused_postprocess_align,
    run_fused_precopy,
)
from vllm.v1.worker.cpu.mm.rope import prepare_rope_positions
from vllm.v1.worker.cpu.model_states.mamba_hybrid import (
    fill_num_accepted,
    scatter_num_accepted,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def ref_scatter_num_accepted(grid, idx_mapping, num_sampled, num_accepted):
    for row in range(grid[0]):
        req_state_idx = int(idx_mapping[row])
        if req_state_idx < 0:
            continue
        num_accepted[req_state_idx] = max(int(num_sampled[row]), 1)


def ref_fill_num_accepted(grid, idx_mapping, num_accepted, num_sampled):
    for row in range(grid[0]):
        req_state_idx = int(idx_mapping[row])
        if req_state_idx < 0:
            continue
        num_accepted[req_state_idx] = num_sampled


def ref_prepare_rope_positions(
    grid,
    positions,
    positions_stride,
    prefill_positions,
    stride0,
    stride1,
    prefill_delta,
    idx_mapping,
    query_start_loc,
    prefill_lens,
    num_computed_tokens,
    num_dims,
):
    flat_positions = positions.view(-1)
    flat_prefill = prefill_positions.reshape(-1)
    for row in range(grid[0]):
        req = int(idx_mapping[row])
        is_prefill = int(num_computed_tokens[req]) < int(prefill_lens[req])
        start = int(query_start_loc[row])
        query_len = int(query_start_loc[row + 1]) - start
        num_computed = int(num_computed_tokens[req])
        delta = int(prefill_delta[req])
        for token in range(query_len):
            orig = num_computed + token
            for dim in range(num_dims):
                if is_prefill:
                    value = int(flat_prefill[req * stride0 + dim * stride1 + orig])
                else:
                    value = orig + delta
                flat_positions[dim * positions_stride + start + token] = value


def _idx_mapping(num_reqs, max_num_reqs, num_filtered, generator):
    """Distinct request slots, as the runner hands out, some replaced by -1."""
    idx = torch.randperm(max_num_reqs, generator=generator)[:num_reqs]
    idx = idx.to(torch.int32)
    if num_filtered:
        filtered = torch.randperm(num_reqs, generator=generator)[:num_filtered]
        idx[filtered] = -1
    return idx


@pytest.mark.parametrize(
    "num_reqs,num_filtered", [(8, 0), (8, 3), (8, 8), (1, 0), (64, 6)]
)
def test_scatter_num_accepted_matches_kernel(num_reqs, num_filtered):
    generator = torch.Generator().manual_seed(num_reqs * 100 + num_filtered)
    max_num_reqs = 64
    idx_mapping = _idx_mapping(num_reqs, max_num_reqs, num_filtered, generator)
    # 0 is reachable: a chunked-prefill step samples no token, and mamba takes
    # 1 as the neutral non-spec value.
    num_sampled = torch.randint(
        0, 5, (num_reqs,), dtype=torch.int32, generator=generator
    )

    actual = torch.ones(max_num_reqs, dtype=torch.int32)
    expected = actual.clone()
    scatter_num_accepted((num_reqs,), idx_mapping, num_sampled, actual)
    ref_scatter_num_accepted((num_reqs,), idx_mapping, num_sampled, expected)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "num_reqs,num_filtered", [(8, 0), (8, 3), (8, 8), (1, 1), (64, 6)]
)
def test_fill_num_accepted_matches_kernel(num_reqs, num_filtered):
    generator = torch.Generator().manual_seed(num_reqs * 100 + num_filtered)
    max_num_reqs = 64
    idx_mapping = _idx_mapping(num_reqs, max_num_reqs, num_filtered, generator)

    actual = torch.ones(max_num_reqs, dtype=torch.int32)
    expected = actual.clone()
    fill_num_accepted((num_reqs,), idx_mapping, actual, 3)
    ref_fill_num_accepted((num_reqs,), idx_mapping, expected, 3)

    torch.testing.assert_close(actual, expected)


def test_num_accepted_fallbacks_leave_an_empty_batch_alone():
    idx_mapping = torch.empty(0, dtype=torch.int32)
    num_sampled = torch.empty(0, dtype=torch.int32)

    actual = torch.ones(8, dtype=torch.int32)
    scatter_num_accepted((0,), idx_mapping, num_sampled, actual)
    fill_num_accepted((0,), idx_mapping, actual, 2)

    torch.testing.assert_close(actual, torch.ones(8, dtype=torch.int32))


@pytest.mark.parametrize("num_dims", [3, 4])
@pytest.mark.parametrize("num_reqs", [1, 8, 16])
def test_prepare_rope_positions_matches_kernel(num_dims, num_reqs):
    generator = torch.Generator().manual_seed(num_dims * 100 + num_reqs)
    max_num_reqs, max_model_len, max_num_tokens = 16, 2176, 4096

    prefill_positions = torch.randint(
        0,
        500,
        (max_num_reqs * num_dims, max_model_len),
        dtype=torch.int32,
        generator=generator,
    )
    prefill_delta = torch.randint(
        -5, 20, (max_num_reqs,), dtype=torch.int32, generator=generator
    )
    idx_mapping = _idx_mapping(num_reqs, max_num_reqs, 0, generator)
    prefill_lens = torch.randint(
        1, max_model_len // 2, (max_num_reqs,), dtype=torch.int32, generator=generator
    )

    # Alternate prefilling and decoding rows, which is what a mixed batch is.
    num_computed = torch.zeros(max_num_reqs, dtype=torch.int32)
    query_lens = []
    for row, req in enumerate(idx_mapping.tolist()):
        if row % 2:
            num_computed[req] = int(prefill_lens[req]) + int(
                torch.randint(0, 50, (1,), generator=generator)
            )
            query_lens.append(1)
        else:
            num_computed[req] = int(
                torch.randint(
                    0, max(1, int(prefill_lens[req])), (1,), generator=generator
                )
            )
            room = max(1, int(prefill_lens[req]) - int(num_computed[req]))
            query_lens.append(
                min(room, int(torch.randint(1, 40, (1,), generator=generator)))
            )
    query_start_loc = torch.tensor(
        [0] + torch.tensor(query_lens).cumsum(0).tolist(), dtype=torch.int32
    )

    args = (
        prefill_positions,
        num_dims * max_model_len,
        max_model_len,
        prefill_delta,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed,
    )
    actual = torch.zeros((num_dims, max_num_tokens + 1), dtype=torch.int64)
    expected = actual.clone()

    prepare_rope_positions(
        (num_reqs,),
        actual,
        actual.stride(0),
        *args,
        BLOCK_SIZE=1024,
        NUM_DIMS=num_dims,
    )
    ref_prepare_rope_positions(
        (num_reqs,), expected, expected.stride(0), *args, num_dims=num_dims
    )

    torch.testing.assert_close(actual, expected)


BLOCK_SIZE = 4
MAX_BLOCKS = 8
MAX_REQS = 16


def ref_copy_state_block(ctx, state_idx, bt_row_idx, src_col, dst_col, token_bias):
    """Scalar transcription of _copy_mamba_state_block, element by element.

    Copies low to high like the kernel, which is what makes a same-block left
    shift safe there and is worth reproducing rather than working around.
    """
    state = ctx._cpu_states[state_idx]
    block_table = ctx._cpu_block_tables[ctx._cpu_group_indices[state_idx]]
    row = block_table[bt_row_idx]
    conv_width = ctx._cpu_conv_widths[state_idx]
    dst_block = int(row[dst_col])

    if conv_width == 0:
        src_block = int(row[src_col + token_bias])
        if src_block == dst_block:
            return
        flat_src = state[src_block].reshape(-1)
        flat_dst = state[dst_block].reshape(-1)
        for i in range(flat_src.numel()):
            flat_dst[i] = flat_src[i]
        return

    src_block = int(row[src_col])
    if ctx._cpu_conv_dim_first:
        dim_rows = state.shape[1]
        for token_idx in range(conv_width - token_bias):
            for dim in range(dim_rows):
                state[dst_block, dim, token_idx] = state[
                    src_block, dim, token_idx + token_bias
                ]
        return

    inner = state[0, 0].numel()
    for token_idx in range(conv_width - token_bias):
        flat_src = state[src_block, token_idx + token_bias].reshape(-1)
        flat_dst = state[dst_block, token_idx].reshape(-1)
        for i in range(inner):
            flat_dst[i] = flat_src[i]


def ref_run_fused_precopy(ctx, num_reqs, state_idx, src_col, token_bias, idx_mapping):
    for batch_idx in range(num_reqs):
        req = batch_idx if idx_mapping is None else int(idx_mapping[batch_idx])
        if req < 0:
            continue
        src, dst = int(src_col[req]), int(state_idx[req])
        if src < 0 or src == dst:
            continue
        bt_row = batch_idx if idx_mapping is not None else req
        for state in range(ctx.num_states):
            ref_copy_state_block(ctx, state, bt_row, src, dst, int(token_bias[req]))


def ref_run_fused_postprocess_align(
    ctx, num_reqs, num_accepted, state_idx, new_num_computed, idx_mapping
):
    snapshot = num_accepted.clone()
    for batch_idx in range(num_reqs):
        req = int(idx_mapping[batch_idx])
        if req < 0:
            continue
        running = int(new_num_computed[req]) - int(snapshot[req]) + 1
        aligned = (int(new_num_computed[req]) // ctx.block_size) * ctx.block_size
        if aligned < running:
            continue
        bias = aligned - running
        src, dst = int(state_idx[req]), aligned // ctx.block_size - 1
        if src == dst:
            num_accepted[req] = 1
            if bias == 0:
                continue
        for state in range(ctx.num_states):
            ref_copy_state_block(ctx, state, batch_idx, src, dst, bias)


def ref_preprocess_mamba_align(
    num_reqs,
    idx_mapping,
    state_idx,
    num_computed_tokens,
    query_start_loc,
    num_accepted,
    src_col,
    src_off,
    mamba_block_size,
):
    for batch_idx in range(num_reqs):
        req = int(idx_mapping[batch_idx])
        if req < 0:
            continue
        previous = int(state_idx[req])
        src_col[req] = previous
        src_off[req] = max(int(num_accepted[req]) - 1, 0)
        computed_after = (
            int(num_computed_tokens[req])
            + int(query_start_loc[batch_idx + 1])
            - int(query_start_loc[batch_idx])
        )
        new_state_idx = -(-computed_after // mamba_block_size) - 1
        state_idx[req] = new_state_idx
        if previous >= 0 and previous != new_state_idx:
            num_accepted[req] = 1


def _make_ctx(kinds, generator, num_groups=2):
    """A stand-in context holding the tensors the CPU copy path reads.

    ``kinds`` names each state: "temporal", "conv_sd" or "conv_ds".
    """
    dim_first = any(kind == "conv_ds" for kind in kinds)
    states, conv_widths, group_indices = [], [], []
    for state_idx, kind in enumerate(kinds):
        if kind == "temporal":
            shape, width = (MAX_REQS * MAX_BLOCKS, 2, 3), 0
        elif kind == "conv_sd":
            shape, width = (MAX_REQS * MAX_BLOCKS, 4, 3), 4
        else:
            shape, width = (MAX_REQS * MAX_BLOCKS, 3, 4), 4
        states.append(
            torch.rand(shape, dtype=torch.float32, generator=generator) + state_idx
        )
        conv_widths.append(width)
        group_indices.append(state_idx % num_groups)

    # Distinct physical blocks per (group, request) so one request's copy
    # cannot land in another's state and hide a mis-indexed row.
    block_tables = [
        torch.arange(MAX_REQS * MAX_BLOCKS, dtype=torch.int32).reshape(
            MAX_REQS, MAX_BLOCKS
        )
        for _ in range(num_groups)
    ]

    ctx = SimpleNamespace(
        is_initialized=True,
        num_states=len(kinds),
        block_size=BLOCK_SIZE,
        _cpu_states=states,
        _cpu_block_tables=block_tables,
        _cpu_conv_widths=conv_widths,
        _cpu_group_indices=group_indices,
        _cpu_conv_dim_first=dim_first,
    )
    return ctx


def _clone_ctx(ctx):
    twin = copy.copy(ctx)
    twin._cpu_states = [state.clone() for state in ctx._cpu_states]
    return twin


def _assert_states_close(actual_ctx, expected_ctx):
    for actual, expected in zip(actual_ctx._cpu_states, expected_ctx._cpu_states):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "kinds",
    [
        ["temporal"],
        ["conv_sd", "temporal"],
        ["conv_ds", "temporal"],
        ["temporal", "temporal", "conv_sd"],
    ],
)
@pytest.mark.parametrize("num_filtered", [0, 2])
def test_precopy_matches_kernel(kinds, num_filtered):
    generator = torch.Generator().manual_seed(len(kinds) * 10 + num_filtered)
    num_reqs = 6
    ctx = _make_ctx(kinds, generator)
    expected_ctx = _clone_ctx(ctx)

    idx_mapping = _idx_mapping(num_reqs, MAX_REQS, num_filtered, generator)
    state_idx = torch.full((MAX_REQS,), -1, dtype=torch.int32)
    src_col = torch.full((MAX_REQS,), -1, dtype=torch.int32)
    token_bias = torch.zeros(MAX_REQS, dtype=torch.int32)
    # Cover every branch the copy takes: a fresh request, one still inside its
    # block, and boundary crossings with and without an accepted-token bias.
    cases = [(-1, 0, 0), (2, 2, 0), (0, 1, 0), (1, 3, 1), (2, 5, 2), (0, 4, 3)]
    for batch_idx, req in enumerate(idx_mapping.tolist()):
        if req < 0:
            continue
        src, dst, bias = cases[batch_idx % len(cases)]
        src_col[req], state_idx[req], token_bias[req] = src, dst, bias

    run_fused_precopy(ctx, num_reqs, state_idx, src_col, token_bias, idx_mapping)
    ref_run_fused_precopy(
        expected_ctx, num_reqs, state_idx, src_col, token_bias, idx_mapping
    )

    _assert_states_close(ctx, expected_ctx)


def test_precopy_without_idx_mapping_uses_batch_order():
    generator = torch.Generator().manual_seed(7)
    ctx = _make_ctx(["conv_sd", "temporal"], generator)
    expected_ctx = _clone_ctx(ctx)
    num_reqs = 4

    state_idx = torch.tensor([1, 2, 2, 4] + [0] * (MAX_REQS - 4), dtype=torch.int32)
    src_col = torch.tensor([0, 2, -1, 1] + [0] * (MAX_REQS - 4), dtype=torch.int32)
    token_bias = torch.tensor([1, 0, 0, 2] + [0] * (MAX_REQS - 4), dtype=torch.int32)

    run_fused_precopy(ctx, num_reqs, state_idx, src_col, token_bias, None)
    ref_run_fused_precopy(expected_ctx, num_reqs, state_idx, src_col, token_bias, None)

    _assert_states_close(ctx, expected_ctx)


def test_precopy_leaves_state_alone_when_nothing_crosses():
    generator = torch.Generator().manual_seed(11)
    ctx = _make_ctx(["conv_sd", "temporal"], generator)
    expected_ctx = _clone_ctx(ctx)

    idx_mapping = _idx_mapping(4, MAX_REQS, 0, generator)
    # Fresh (-1) and same-block (src == dst) requests have nothing to migrate.
    state_idx = torch.full((MAX_REQS,), 3, dtype=torch.int32)
    src_col = torch.tensor([-1, 3, -1, 3] + [3] * (MAX_REQS - 4), dtype=torch.int32)
    token_bias = torch.ones(MAX_REQS, dtype=torch.int32)

    run_fused_precopy(ctx, 4, state_idx, src_col, token_bias, idx_mapping)

    _assert_states_close(ctx, expected_ctx)


@pytest.mark.parametrize("captured_rows", [1, 3])
def test_full_rows_recovers_what_a_sliced_block_table_hides(captured_rows):
    """The runner hands over batch-order slices of the persistent block tables.

    The kernels reach later rows regardless, addressing the buffer by pointer
    and stride, so a fallback that kept the slice would raise as soon as a
    second request needed a copy.
    """
    persistent = torch.arange(MAX_REQS * MAX_BLOCKS, dtype=torch.int32).reshape(
        MAX_REQS, MAX_BLOCKS
    )
    captured = persistent[:captured_rows]

    restored = _full_rows(captured)

    assert restored.size(0) == MAX_REQS, (
        f"recovered {restored.size(0)} rows from a {captured_rows}-row slice"
    )
    torch.testing.assert_close(restored, persistent)


def test_full_rows_leaves_a_whole_block_table_alone():
    persistent = torch.arange(MAX_REQS * MAX_BLOCKS, dtype=torch.int32).reshape(
        MAX_REQS, MAX_BLOCKS
    )
    torch.testing.assert_close(_full_rows(persistent), persistent)


@pytest.mark.parametrize("kinds", [["temporal"], ["conv_sd", "temporal"]])
def test_postprocess_align_matches_kernel(kinds):
    generator = torch.Generator().manual_seed(len(kinds) + 3)
    num_reqs = 6
    ctx = _make_ctx(kinds, generator)
    expected_ctx = _clone_ctx(ctx)

    idx_mapping = _idx_mapping(num_reqs, MAX_REQS, 1, generator)
    num_accepted = torch.ones(MAX_REQS, dtype=torch.int32)
    state_idx = torch.zeros(MAX_REQS, dtype=torch.int32)
    new_num_computed = torch.zeros(MAX_REQS, dtype=torch.int32)
    # (accepted, state_idx, new_num_computed) spanning: no copy needed, an
    # aligned landing in the running block, and a crossing with a bias.
    cases = [(1, 1, 7), (1, 0, 4), (2, 1, 8), (3, 0, 4), (1, 2, 11), (2, 2, 12)]
    for batch_idx, req in enumerate(idx_mapping.tolist()):
        if req < 0:
            continue
        accepted, src, computed = cases[batch_idx % len(cases)]
        num_accepted[req], state_idx[req], new_num_computed[req] = (
            accepted,
            src,
            computed,
        )
    expected_accepted = num_accepted.clone()

    run_fused_postprocess_align(
        ctx, num_reqs, num_accepted, state_idx, new_num_computed, idx_mapping
    )
    ref_run_fused_postprocess_align(
        expected_ctx,
        num_reqs,
        expected_accepted,
        state_idx,
        new_num_computed,
        idx_mapping,
    )

    torch.testing.assert_close(num_accepted, expected_accepted)
    _assert_states_close(ctx, expected_ctx)


@pytest.mark.parametrize("num_reqs,num_filtered", [(8, 0), (8, 3), (1, 0), (16, 5)])
def test_preprocess_mamba_align_matches_kernel(num_reqs, num_filtered):
    generator = torch.Generator().manual_seed(num_reqs * 7 + num_filtered)
    mamba_block_size = 16
    idx_mapping = _idx_mapping(num_reqs, MAX_REQS, num_filtered, generator)

    state_idx = torch.randint(
        -1, 4, (MAX_REQS,), dtype=torch.int32, generator=generator
    )
    num_computed = torch.randint(
        0, 64, (MAX_REQS,), dtype=torch.int32, generator=generator
    )
    num_accepted = torch.randint(
        0, 4, (MAX_REQS,), dtype=torch.int32, generator=generator
    )
    query_lens = torch.randint(
        1, 20, (num_reqs,), dtype=torch.int32, generator=generator
    )
    query_start_loc = torch.tensor(
        [0] + query_lens.cumsum(0).tolist(), dtype=torch.int32
    )
    src_col = torch.zeros(MAX_REQS, dtype=torch.int32)
    src_off = torch.zeros(MAX_REQS, dtype=torch.int32)

    expected = [t.clone() for t in (state_idx, num_accepted, src_col, src_off)]

    preprocess_mamba_align(
        (1,),
        idx_mapping,
        state_idx,
        num_computed,
        query_start_loc,
        num_accepted,
        src_col,
        src_off,
        num_reqs,
        BLOCK_SIZE=1024,
        MAMBA_BLOCK_SIZE=mamba_block_size,
    )
    ref_preprocess_mamba_align(
        num_reqs,
        idx_mapping,
        expected[0],
        num_computed,
        query_start_loc,
        expected[1],
        expected[2],
        expected[3],
        mamba_block_size,
    )

    for actual, want in zip((state_idx, num_accepted, src_col, src_off), expected):
        torch.testing.assert_close(actual, want)
