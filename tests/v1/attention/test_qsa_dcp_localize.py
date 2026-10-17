# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA DCP selection localization and empty-owner behavior."""

from importlib import import_module

import pytest
import torch

PAD = -1


def _owner(pos, world, interleave):
    return (pos // interleave) % world


def _local_id(pos, world, interleave):
    return (pos // (world * interleave)) * interleave + (pos % interleave)


def _reference_localize(row, world, rank, interleave):
    """Plain enumeration. Returns (localized prefix, kept count)."""
    kept = []
    for g in row:
        if g < 0:
            continue
        if _owner(g, world, interleave) != rank:
            continue
        kept.append(_local_id(g, world, interleave))
    return kept, len(kept)


def _pack(rows, width, device="cpu"):
    """Build the packed buffer: selection columns plus a trailing count."""
    out = torch.full((len(rows), width + 1), PAD, dtype=torch.int32, device=device)
    for i, row in enumerate(rows):
        for j, g in enumerate(row[:width]):
            out[i, j] = g
        out[i, width] = len(row[:width])
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")
@pytest.mark.parametrize("world,interleave", [(2, 1), (2, 4), (4, 16)])
@pytest.mark.parametrize("width", [32, 2051])
def test_kernel_matches_the_reference(world, interleave, width):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    rows = (
        [list(range(0, 64, 2)), list(range(1, 40)), [], [7]]
        if width == 32
        else [list(range(width)), [0], []]
    )
    packed = _pack(rows, width).cuda()
    source = packed.clone()
    local = torch.empty_like(packed)
    for rank in range(world):
        expected = [
            _reference_localize(row[:width], world, rank, interleave) for row in rows
        ]
        for _ in range(5 if width == 2051 else 1):
            local.fill_(123)
            qsa_localize_dcp_indices(packed, local, world, rank, interleave)
            # The layer's selection buffer must survive: MTP steps read it again.
            assert torch.equal(packed, source)
            for row, (ids, count) in enumerate(expected):
                assert int(local[row, width]) == count
                assert local[row, :count].tolist() == ids
                assert bool((local[row, count:width] == PAD).all())


# --- contracts that keep the gate from being applied twice -------------------


@pytest.mark.cpu_test
def test_localize_leaves_the_source_alone_without_dcp():
    """World size 1 still copies, so one call site covers both paths."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    packed = _pack([[5, 6], [7]], 8)
    source = packed.clone()
    local = torch.empty_like(packed)
    qsa_localize_dcp_indices(packed, local, 1, 0, 1)
    assert torch.equal(packed, source)
    assert torch.equal(local, source)


@pytest.mark.cpu_test
def test_the_kernel_wrapper_rejects_a_gate_with_return_lse():
    """The output gate must be applied after DCP merging, not by the kernel."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    query = torch.zeros(2, 4, 64, dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 2, 64, dtype=torch.bfloat16)
    indices = _pack([[0], [0]], 8)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    token_to_req = torch.zeros(2, dtype=torch.int32)
    args = (query, cache, cache, indices, block_table, token_to_req, False)

    with pytest.raises(ValueError, match="do not pass one"):
        qsa_ops.qsa_sparse_paged_attention(
            *args, output_gate=torch.zeros_like(query), return_lse=True
        )
    with pytest.raises(ValueError, match="requires an output gate"):
        qsa_ops.qsa_sparse_paged_attention(*args)


@pytest.mark.cpu_test
def test_empty_qsa_batch_returns_the_dcp_output_and_lse_pair():
    """A rank with no query tokens still supplies both DCP merge inputs."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    query = torch.empty((0, 4, 64), dtype=torch.bfloat16)
    cache = torch.empty((1, 16, 2, 64), dtype=torch.bfloat16)
    indices = torch.empty((0, 9), dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    token_to_req = torch.empty((0,), dtype=torch.int32)

    output, lse = qsa_sparse_paged_attention(
        query,
        cache,
        cache,
        indices,
        block_table,
        token_to_req,
        False,
        return_lse=True,
    )

    assert output.shape == query.shape
    assert output.dtype == torch.float32
    assert lse.shape == query.shape[:2]
    assert lse.dtype == torch.float32
    assert lse.device == query.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA kernels need CUDA")
@pytest.mark.parametrize(
    "world,interleave,seq_len,cache_dtype,k_scale,v_scale",
    [
        (2, 1, 64, torch.bfloat16, None, None),
        (4, 16, 200, torch.bfloat16, None, None),
        (2, 16, 64, torch.float8_e4m3fn, 0.75, 1.5),
        (4, 4, 64, torch.float8_e4m3fn, 0.75, 1.5),
        (2, 16, 64, torch.uint8, 0.75, 1.5),
    ],
)
def test_sharded_ranks_reproduce_the_single_rank_result(
    world, interleave, seq_len, cache_dtype, k_scale, v_scale
):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices
    from vllm.v1.attention.ops.dcp import _lse_weighted_combine

    import_module("vllm.models.qwen4_exp.nvidia.model")
    qsa_backend = import_module(
        "vllm.models.qwen4_exp.nvidia.qsa"
    ).Qwen4ExpQSAFlashAttentionBackend
    assert qsa_backend.get_impl_cls().lse_base_on_e is False

    torch.manual_seed(1234 + seq_len)
    device = "cuda"
    page = 16
    num_kv_heads = 2
    head_dim = 64
    num_q_heads = 8

    full_blocks = (seq_len + page - 1) // page
    shape = (full_blocks, page, num_kv_heads, head_dim)
    k_source = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v_source = torch.randn(shape, dtype=torch.bfloat16, device=device)
    if cache_dtype in (torch.float8_e4m3fn, torch.uint8):
        k_full = (k_source / k_scale).to(torch.float8_e4m3fn)
        v_full = (v_source / v_scale).to(torch.float8_e4m3fn)
        if cache_dtype == torch.uint8:
            k_full = k_full.view(torch.uint8)
            v_full = v_full.view(torch.uint8)
    else:
        k_full, v_full = k_source, v_source

    local_len = seq_len // world + page
    local_blocks = (local_len + page - 1) // page
    local_shape = (local_blocks, page, num_kv_heads, head_dim)
    k_rank = [
        torch.zeros(local_shape, dtype=cache_dtype, device=device) for _ in range(world)
    ]
    v_rank = [
        torch.zeros(local_shape, dtype=cache_dtype, device=device) for _ in range(world)
    ]
    for pos in range(seq_len):
        rank = _owner(pos, world, interleave)
        local_id = _local_id(pos, world, interleave)
        k_rank[rank][local_id // page, local_id % page] = k_full[
            pos // page, pos % page
        ]
        v_rank[rank][local_id // page, local_id % page] = v_full[
            pos // page, pos % page
        ]

    num_rows = 6
    selections = [
        list(range(row, seq_len, row + 2))[: seq_len // 2] for row in range(num_rows)
    ]
    selections[0] = [0]
    width = max(len(s) for s in selections)
    global_sel = _pack(selections, width, device)

    q = torch.randn(
        (num_rows, num_q_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    token_to_req = torch.zeros(num_rows, dtype=torch.int32, device=device)
    full_table = torch.arange(full_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    local_table = torch.arange(
        local_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)

    reference, reference_lse = qsa_sparse_paged_attention(
        q,
        k_full,
        v_full,
        global_sel,
        full_table,
        token_to_req,
        False,
        k_scale=k_scale,
        v_scale=v_scale,
        return_lse=True,
    )
    assert reference.dtype == torch.float32
    if cache_dtype == torch.uint8:
        native_out, native_lse = qsa_sparse_paged_attention(
            q,
            k_full.view(torch.float8_e4m3fn),
            v_full.view(torch.float8_e4m3fn),
            global_sel,
            full_table,
            token_to_req,
            False,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
        )
        torch.testing.assert_close(reference, native_out, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(reference_lse, native_lse, rtol=1e-3, atol=1e-3)

    outs, lses = [], []
    for rank in range(world):
        local_sel = qsa_localize_dcp_indices(
            global_sel, torch.empty_like(global_sel), world, rank, interleave
        )
        empty = local_sel[:, -1] <= 0
        if rank:
            assert bool(empty[0])
        out, lse = qsa_sparse_paged_attention(
            q,
            k_rank[rank],
            v_rank[rank],
            local_sel,
            local_table,
            token_to_req,
            False,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
        )
        if rank:
            assert torch.isneginf(lse[0]).all()
        outs.append(out.float())
        lses.append(lse.float())

    merged, merged_lse = _lse_weighted_combine(
        torch.stack(outs),
        torch.stack(lses),
        return_lse=True,
        is_lse_base_on_e=False,
    )

    torch.testing.assert_close(
        merged.to(reference.dtype), reference, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(merged_lse, reference_lse.float(), rtol=1e-3, atol=1e-3)
