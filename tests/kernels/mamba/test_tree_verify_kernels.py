# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree verify (``parents``) in the conv update, GDN recurrence and Mamba2
state update kernels: a tree matches the pure-PyTorch reference, each of its
root-to-leaf paths gives the bits of the same kernel run on that path alone,
and chain parents match no parents (today's chain kernels) up to fp32 rounding.

Chain parents need not give today's bits: ptxas may contract a multiply and an
add into an FMA differently once the tree branch is compiled in (GDN on B300:
the in-thread partial sum of ``tl.sum(b_h * b_k[None, :], 1)``). Only the same
compiled kernel guarantees the same bits, which the path tests check.

Runs on CUDA, or on CPU with ``TRITON_INTERPRET=1`` set before the import.
"""

import os

import numpy as np
import pytest
import torch

from tests.kernels.mamba.tree_verify_reference import (
    chain_parents,
    conv_tree_ref,
    gdn_tree_ref,
    ssu_tree_ref,
)
from tests.utils_tree import path_to, random_parents
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    pytest.skip("needs CUDA or TRITON_INTERPRET=1", allow_module_level=True)

LENGTHS = [5, 1, 7]  # rows per request; the longest sets max_query_len
MAX_QUERY_LEN = max(LENGTHS)
NUM_ACCEPTED = [2, 1, 4]  # the history starts at column num_accepted - 1


def _parents(kind: str, seed: int) -> torch.Tensor:
    if kind == "chain":
        return torch.cat([chain_parents(n) for n in LENGTHS])
    rng = np.random.default_rng(seed)
    return torch.tensor(
        [p for n in LENGTHS for p in random_parents(rng, n)], dtype=torch.int32
    )


def _starts(lengths=LENGTHS) -> list[int]:
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    return starts


def _meta(lengths=LENGTHS, num_accepted=NUM_ACCEPTED):
    return (
        torch.tensor(_starts(lengths), dtype=torch.int32, device=DEVICE),
        torch.tensor(num_accepted, dtype=torch.int32, device=DEVICE),
    )


def _path_batch(parents: torch.Tensor) -> list[tuple[int, list[int]]]:
    """Each root-to-leaf path of each request's tree as (request, batch rows)."""
    starts = _starts()
    paths = []
    for r, (begin, end) in enumerate(zip(starts, starts[1:])):
        local = parents[begin:end]
        for leaf in sorted(set(range(end - begin)) - set(local.tolist())):
            paths.append((r, [begin + row for row in path_to(local, leaf)]))
    return paths


def _path_meta(paths):
    """Lengths, num_accepted and chain parents of a batch with one request per path."""
    lengths = [len(rows) for _, rows in paths]
    num_accepted = [NUM_ACCEPTED[r] for r, _ in paths]
    parents = torch.cat([chain_parents(n) for n in lengths])
    return lengths, num_accepted, parents


def _assert_close_to_rounding(got, want):
    """Equal up to fp32 rounding: a few ulp per step, as a different FMA
    contraction gives (~K * 2**-24 relative for a K-term dot product, K <= 128);
    a bf16 tensor within one bf16 ulp of the rounded result."""
    if got.dtype == torch.float32:
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=1e-5)


def _run_conv(
    x,
    conv_state,
    weight,
    bias,
    lines,
    parents,
    lengths=LENGTHS,
    num_accepted=NUM_ACCEPTED,
):
    query_start_loc, num_accepted = _meta(lengths, num_accepted)
    state = conv_state.clone()
    out = causal_conv1d_update(
        x.clone(),
        state,
        weight,
        bias,
        "silu",
        conv_state_indices=lines,
        num_accepted_tokens=num_accepted,
        query_start_loc=query_start_loc,
        max_query_len=MAX_QUERY_LEN,
        parents=None if parents is None else parents.to(DEVICE),
    )
    return out, state


def _skip_interpreted_bf16_conv(dtype):
    # The Triton interpreter gives NaN for today's bf16 conv update, too.
    if DEVICE == "cpu" and dtype == torch.bfloat16:
        pytest.skip("bf16 conv update is wrong in the Triton interpreter")


def _conv_inputs(seed, width, dtype):
    _skip_interpreted_bf16_conv(dtype)
    g = torch.Generator().manual_seed(seed)
    dim, num_lines = 64, 6
    state_len = width - 1 + MAX_QUERY_LEN - 1
    x = torch.randn(sum(LENGTHS), dim, generator=g).to(DEVICE, dtype)
    conv_state = torch.randn(num_lines, dim, state_len, generator=g).to(DEVICE, dtype)
    weight = torch.randn(dim, width, generator=g).to(DEVICE, dtype)
    bias = torch.randn(dim, generator=g).to(DEVICE, dtype)
    lines = torch.tensor([4, 1, 3], dtype=torch.int32, device=DEVICE)
    return x, conv_state, weight, bias, lines


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seed", range(3))
def test_conv_update_tree_matches_reference(seed, width, dtype):
    x, conv_state, weight, bias, lines = _conv_inputs(seed, width, dtype)
    parents = _parents("tree", seed)
    out, state = _run_conv(x, conv_state, weight, bias, lines, parents)
    _, chain_state = _run_conv(x, conv_state, weight, bias, lines, None)
    # The window keeps the rows in row order, as for a chain.
    assert torch.equal(state, chain_state)
    starts = _starts()
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    for r, (begin, end) in enumerate(zip(starts, starts[1:])):
        offset = NUM_ACCEPTED[r] - 1
        history = conv_state[lines[r], :, offset : offset + width - 1]
        want = conv_tree_ref(
            x[begin:end].cpu(),
            history.cpu(),
            weight.cpu(),
            bias.cpu(),
            parents[begin:end],
        )
        torch.testing.assert_close(
            out[begin:end].float().cpu(), want, rtol=tol, atol=tol
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seed", range(3))
def test_conv_update_tree_paths_are_chain_bits(seed, width, dtype):
    # Taps are copies of the ancestors' inputs, so a tree row's output has the
    # bits of its path run as a chain (the windows differ: rows in row order).
    x, conv_state, weight, bias, lines = _conv_inputs(seed, width, dtype)
    parents = _parents("tree", seed)
    out, _ = _run_conv(x, conv_state, weight, bias, lines, parents)
    paths = _path_batch(parents)
    lengths, num_accepted, path_parents = _path_meta(paths)
    rows = torch.tensor([row for _, p in paths for row in p], device=DEVICE)
    # Line 0 is the null block, so path p takes line p + 1.
    path_lines = torch.arange(1, 1 + len(paths), dtype=torch.int32, device=DEVICE)
    path_state = torch.cat(
        [conv_state[:1], conv_state[lines[[r for r, _ in paths]].long()]]
    )
    path_out, _ = _run_conv(
        x[rows],
        path_state,
        weight,
        bias,
        path_lines,
        path_parents,
        lengths,
        num_accepted,
    )
    assert torch.equal(out[rows], path_out)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 4])
def test_conv_update_chain_parents_match_todays_kernel(width, dtype):
    x, conv_state, weight, bias, lines = _conv_inputs(0, width, dtype)
    out, state = _run_conv(x, conv_state, weight, bias, lines, _parents("chain", 0))
    want, want_state = _run_conv(x, conv_state, weight, bias, lines, None)
    _assert_close_to_rounding(out, want)
    # The window holds copies of the inputs either way.
    assert torch.equal(state, want_state)


def _state_indices(num_reqs: int = len(LENGTHS)) -> torch.Tensor:
    # A distinct state block per (request, row); 0 is the null block.
    blocks = torch.arange(1, 1 + num_reqs * MAX_QUERY_LEN, dtype=torch.int32)
    return blocks.view(num_reqs, MAX_QUERY_LEN).to(DEVICE)


def _path_states(state, paths):
    """A state pool for the path batch: path ``p`` of request ``r`` starts from
    request ``r``'s initial block. Returns (pool, per-path block table)."""
    src, dst = _state_indices(), _state_indices(len(paths))
    pool = torch.zeros(1 + len(paths) * MAX_QUERY_LEN, *state.shape[1:]).to(state)
    for p, (r, _) in enumerate(paths):
        pool[dst[p, NUM_ACCEPTED[r] - 1]] = state[src[r, NUM_ACCEPTED[r] - 1]]
    return pool, dst


def _assert_paths_bitwise(paths, out, state, path_out, path_state, path_blocks):
    """Each path row of the tree run has the path run's output and state bits."""
    starts, src = _starts(), _state_indices()
    path_row = 0
    for p, (r, rows) in enumerate(paths):
        n = len(rows)
        assert torch.equal(out[rows], path_out[path_row : path_row + n])
        tree_blocks = src[r, [row - starts[r] for row in rows]].long()
        assert torch.equal(state[tree_blocks], path_state[path_blocks[p, :n].long()])
        path_row += n


def _gdn_inputs(seed, dtype):
    g = torch.Generator().manual_seed(seed)
    tokens, heads, v_heads, k_dim, v_dim = sum(LENGTHS), 2, 4, 32, 64
    num_blocks = 1 + len(LENGTHS) * MAX_QUERY_LEN
    return dict(
        q=torch.randn(1, tokens, heads, k_dim, generator=g).to(DEVICE, dtype),
        k=torch.randn(1, tokens, heads, k_dim, generator=g).to(DEVICE, dtype),
        v=torch.randn(1, tokens, v_heads, v_dim, generator=g).to(DEVICE, dtype),
        a=torch.randn(tokens, v_heads, generator=g).to(DEVICE, dtype),
        b=torch.randn(tokens, v_heads, generator=g).to(DEVICE, dtype),
        A_log=torch.randn(v_heads, generator=g).to(DEVICE),
        dt_bias=torch.randn(v_heads, generator=g).to(DEVICE),
        initial_state=torch.randn(num_blocks, v_heads, v_dim, k_dim, generator=g).to(
            DEVICE
        ),
    )


def _run_gdn(inp, parents, lengths=LENGTHS, num_accepted=NUM_ACCEPTED):
    query_start_loc, num_accepted = _meta(lengths, num_accepted)
    args = dict(inp, initial_state=inp["initial_state"].clone())
    out, state = fused_sigmoid_gating_delta_rule_update(
        **args,
        inplace_final_state=True,
        cu_seqlens=query_start_loc,
        ssm_state_indices=_state_indices(len(lengths)),
        num_accepted_tokens=num_accepted,
        use_qk_l2norm_in_kernel=True,
        parents=None if parents is None else parents.to(DEVICE),
    )
    return out, state


@pytest.mark.parametrize("seed", range(3))
def test_gdn_tree_matches_reference(seed):
    inp = _gdn_inputs(seed, torch.float32)
    parents = _parents("tree", seed)
    out, state = _run_gdn(inp, parents)
    indices = _state_indices().cpu()
    starts = _starts()
    for r, (begin, end) in enumerate(zip(starts, starts[1:])):
        h0 = inp["initial_state"][indices[r, NUM_ACCEPTED[r] - 1]].cpu()
        want, want_states = gdn_tree_ref(
            inp["q"][0, begin:end].cpu(),
            inp["k"][0, begin:end].cpu(),
            inp["v"][0, begin:end].cpu(),
            inp["a"][begin:end].cpu(),
            inp["b"][begin:end].cpu(),
            inp["A_log"].cpu(),
            inp["dt_bias"].cpu(),
            h0,
            parents[begin:end],
            scale=inp["q"].shape[-1] ** -0.5,
        )
        torch.testing.assert_close(out[0, begin:end].cpu(), want, rtol=1e-4, atol=1e-4)
        got_states = state[indices[r, : end - begin].long()].cpu()
        torch.testing.assert_close(got_states, want_states, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tree_tile", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", range(3))
def test_gdn_tree_paths_are_chain_bits(seed, dtype, tree_tile, monkeypatch):
    # The fp32 state a branching row reloads is the register state its parent
    # stored, so a tree row has the bits of its path run as a chain. A tree
    # launch takes 4- or 8-wide tiles by grid size; check both.
    monkeypatch.setattr(
        fused_sigmoid_gating, "_tree_tile", lambda V, heads, sms: (tree_tile, 1)
    )
    inp = _gdn_inputs(seed, dtype)
    parents = _parents("tree", seed)
    out, state = _run_gdn(inp, parents)
    paths = _path_batch(parents)
    lengths, num_accepted, path_parents = _path_meta(paths)
    rows = torch.tensor([row for _, p in paths for row in p], device=DEVICE)
    pool, blocks = _path_states(inp["initial_state"], paths)
    path_inp = dict(
        inp,
        **{n: inp[n][:, rows] for n in "qkv"},
        a=inp["a"][rows],
        b=inp["b"][rows],
        initial_state=pool,
    )
    path_out, path_state = _run_gdn(path_inp, path_parents, lengths, num_accepted)
    _assert_paths_bitwise(paths, out[0], state, path_out[0], path_state, blocks)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gdn_chain_parents_match_todays_kernel(dtype):
    inp = _gdn_inputs(0, dtype)
    out, state = _run_gdn(inp, _parents("chain", 0))
    want, want_state = _run_gdn(inp, None)
    _assert_close_to_rounding(out, want)
    _assert_close_to_rounding(state, want_state)


def _ssu_inputs(seed, dtype):
    g = torch.Generator().manual_seed(seed)
    tokens, nheads, dim, dstate, ngroups = sum(LENGTHS), 4, 64, 16, 2
    num_blocks = 1 + len(LENGTHS) * MAX_QUERY_LEN
    return dict(
        x=torch.randn(tokens, nheads, dim, generator=g).to(DEVICE, dtype),
        dt=torch.randn(tokens, nheads, dim, generator=g).to(DEVICE, dtype),
        A=-torch.rand(nheads, dim, dstate, generator=g).to(DEVICE),
        B=torch.randn(tokens, ngroups, dstate, generator=g).to(DEVICE, dtype),
        C=torch.randn(tokens, ngroups, dstate, generator=g).to(DEVICE, dtype),
        D=torch.randn(nheads, dim, generator=g).to(DEVICE),
        dt_bias=torch.randn(nheads, dim, generator=g).to(DEVICE),
        state=torch.randn(num_blocks, nheads, dim, dstate, generator=g).to(DEVICE),
    )


def _run_ssu(inp, parents, lengths=LENGTHS, num_accepted=NUM_ACCEPTED):
    query_start_loc, num_accepted = _meta(lengths, num_accepted)
    state = inp["state"].clone()
    out = torch.empty_like(inp["x"])
    indices = _state_indices(len(lengths))
    selective_state_update(
        state,
        inp["x"],
        inp["dt"],
        inp["A"],
        inp["B"],
        inp["C"],
        D=inp["D"],
        dt_bias=inp["dt_bias"],
        dt_softplus=True,
        state_batch_indices=indices,
        dst_state_batch_indices=indices,
        out=out,
        num_accepted_tokens=num_accepted,
        cu_seqlens=query_start_loc,
        parents=None if parents is None else parents.to(DEVICE),
    )
    return out, state


@pytest.mark.parametrize("seed", range(3))
def test_ssu_tree_matches_reference(seed):
    inp = _ssu_inputs(seed, torch.float32)
    parents = _parents("tree", seed)
    out, state = _run_ssu(inp, parents)
    indices = _state_indices().cpu()
    starts = _starts()
    for r, (begin, end) in enumerate(zip(starts, starts[1:])):
        state0 = inp["state"][indices[r, NUM_ACCEPTED[r] - 1]].cpu()
        rows = slice(begin, end)
        want, want_states = ssu_tree_ref(
            inp["x"][rows].cpu(),
            inp["dt"][rows].cpu(),
            inp["A"].cpu(),
            inp["B"][rows].cpu(),
            inp["C"][rows].cpu(),
            inp["D"].cpu(),
            inp["dt_bias"].cpu(),
            state0,
            parents[rows],
        )
        torch.testing.assert_close(out[rows].cpu(), want, rtol=1e-4, atol=1e-4)
        got_states = state[indices[r, : end - begin].long()].cpu()
        torch.testing.assert_close(got_states, want_states, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", range(3))
def test_ssu_tree_paths_are_chain_bits(seed, dtype):
    inp = _ssu_inputs(seed, dtype)
    parents = _parents("tree", seed)
    out, state = _run_ssu(inp, parents)
    paths = _path_batch(parents)
    lengths, num_accepted, path_parents = _path_meta(paths)
    rows = torch.tensor([row for _, p in paths for row in p], device=DEVICE)
    pool, blocks = _path_states(inp["state"], paths)
    path_inp = dict(inp, **{n: inp[n][rows] for n in ("x", "dt", "B", "C")}, state=pool)
    path_out, path_state = _run_ssu(path_inp, path_parents, lengths, num_accepted)
    _assert_paths_bitwise(paths, out, state, path_out, path_state, blocks)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ssu_chain_parents_match_todays_kernel(dtype):
    inp = _ssu_inputs(0, dtype)
    out, state = _run_ssu(inp, _parents("chain", 0))
    want, want_state = _run_ssu(inp, None)
    _assert_close_to_rounding(out, want)
    _assert_close_to_rounding(state, want_state)
