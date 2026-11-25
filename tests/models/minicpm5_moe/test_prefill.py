# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-free causal math and dispatch contracts for query-only prefill."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.minicpm5_moe.xpu.draft_attention import MiniCPM5DraftAttention
from vllm.models.minicpm5_moe.xpu.prefill import (
    MiniCPM5PrefillAttention,
    configure_prefill_attention,
    split_attention_queries,
)
from vllm.utils.math_utils import cdiv


def tensor_metadata(shape, dtype=torch.float16, device="xpu"):
    return NS(
        shape=shape,
        dtype=dtype,
        device=NS(type=device),
        ndim=len(shape),
        dim=lambda: len(shape),
        numel=lambda: math.prod(shape),
        stride=lambda dim: math.prod(shape[dim + 1 :]),
    )


def attention_arguments():
    return dict(
        q=tensor_metadata((4096, 16, 128)),
        k=tensor_metadata((1024, 64, 1, 128)),
        v=tensor_metadata((1024, 64, 1, 128)),
        max_seqlen_q=4096,
        max_seqlen_k=65536,
        cu_seqlens_q=tensor_metadata((2,), torch.int32),
        seqused_k=tensor_metadata((1,), torch.int32),
        block_table=tensor_metadata((1, 1024), torch.int32),
        causal=True,
    )


def draft_arguments():
    return dict(
        q=tensor_metadata((7, 16, 128)),
        k=tensor_metadata((1024, 64, 1, 128)),
        v=tensor_metadata((1024, 64, 1, 128)),
        max_seqlen_q=7,
        max_seqlen_k=65536,
        cu_seqlens_q=tensor_metadata((2,), torch.int32),
        seqused_k=tensor_metadata((1,), torch.int32),
        block_table=tensor_metadata((1, 1024), torch.int32),
        causal=False,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("causal", True),
        ("max_seqlen_q", 8),
        ("q", tensor_metadata((7, 16, 128), torch.float32)),
        ("q", tensor_metadata((7, 16, 128), device="cpu")),
        ("seqused_k", tensor_metadata((2,), torch.int32)),
        ("block_table", tensor_metadata((1, 1025), torch.int32)),
        ("window_size", (128, 0)),
        ("softcap", 1.0),
        ("return_softmax_lse", True),
        ("q_descale", object()),
        ("mask_mod", object()),
    ],
)
def test_unsupported_draft_attention_falls_back_before_any_submit(field, value):
    calls, sentinel = [], object()

    def kernel(**arguments):
        calls.append(arguments)
        return sentinel

    fast = MiniCPM5DraftAttention(kernel, 65536, 2, 7, 16)
    assert fast.eligible(draft_arguments())
    arguments = dict(draft_arguments(), **{field: value})
    assert fast(**arguments) is sentinel
    assert calls == [arguments]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("batch", [1, 2])
def test_draft_split_kv_graph_preserves_noncausal_owner_and_rollback(
    dtype, rank, batch
):
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.flash_attn import FA4DenseAttentionKernel

    if not current_platform.is_xpu() or torch.xpu.device_count() <= rank:
        pytest.skip("XPU Graph requires the selected Intel GPU")
    torch.xpu.set_device(rank)
    device = f"xpu:{rank}"
    torch.manual_seed(7)
    queries, heads, columns = 7, 16, 513
    rows = batch * queries
    cache = torch.randn(batch * columns, 64, 1, 256, dtype=dtype, device=device)
    k, v = cache[..., :128], cache[..., 128:]
    pages = torch.randperm(batch * columns, dtype=torch.int32, device=device).reshape(
        batch, columns
    )
    q = torch.randn(rows, heads, 128, dtype=dtype, device=device)
    output = torch.empty_like(q)
    starts = torch.arange(batch + 1, dtype=torch.int32, device=device) * queries
    lengths = torch.full((batch,), 32775, dtype=torch.int32, device=device)
    fast = MiniCPM5DraftAttention(
        FA4DenseAttentionKernel(), 65536, batch, queries, heads
    )
    arguments = dict(
        q=q,
        k=k,
        v=v,
        out=output,
        cu_seqlens_q=starts,
        seqused_k=lengths,
        max_seqlen_q=queries,
        max_seqlen_k=32775,
        softmax_scale=128**-0.5,
        causal=False,
        block_table=pages,
        window_size=(-1, -1),
        fa_version=2,
    )
    fast(**arguments)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=torch.xpu.current_stream()):
        fast(**arguments)
    epoch = int(fast.buffers["epochs"].cpu())
    cases = [
        ([i * queries for i in range(batch + 1)], [32775] * batch),
        ([i * queries for i in range(batch + 1)], [16449] + [73] * (batch - 1)),
        ([0, queries] if batch == 1 else [0, 0, queries], [126] * batch),
    ]
    for bounds, sizes in cases:
        starts.copy_(torch.tensor(bounds, dtype=torch.int32))
        lengths.copy_(torch.tensor(sizes, dtype=torch.int32))
        q.mul_(0.875)
        graph.replay()
        expected = torch.zeros_like(q)
        for req in range(batch):
            start, end = bounds[req : req + 2]
            if start == end:
                continue
            count = sizes[req]
            selected = pages[req, : cdiv(count, 64)].long()
            keys = k[selected].reshape(-1, 128)[:count].float()
            values = v[selected].reshape(-1, 128)[:count].float()
            scores = torch.einsum("qhd,kd->hqk", q[start:end].float(), keys) * 128**-0.5
            expected[start:end] = torch.einsum(
                "hqk,kd->qhd", scores.softmax(-1), values
            ).to(dtype)
        torch.testing.assert_close(
            output,
            expected,
            atol=0.002 if dtype == torch.float16 else 0.02,
            rtol=0.01 if dtype == torch.float16 else 0.05,
        )
    assert int(fast.buffers["epochs"].cpu()) - epoch == len(cases)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_seqlen_k", 32768),
        ("max_seqlen_q", 2048),
        ("q", tensor_metadata((4096, 8, 128))),
        ("q", tensor_metadata((4096, 16, 128), device="cpu")),
        ("q", tensor_metadata((4096, 16, 128), torch.bfloat16)),
        ("k", tensor_metadata((1024, 64, 2, 128))),
        ("cu_seqlens_q", tensor_metadata((3,), torch.int32)),
        ("seqused_k", tensor_metadata((2,), torch.int32)),
        ("seqused_k", None),
        ("block_table", tensor_metadata((2, 1024), torch.int32)),
        ("block_table", None),
        ("causal", False),
        ("return_softmax_lse", True),
        ("return_attn_probs", True),
        ("dropout_p", 0.1),
        ("window_size", (128, 0)),
        ("softcap", 1.0),
    ]
    + [
        (field, object())
        for field in (
            "cu_seqlens_k",
            "alibi_slopes",
            "q_v",
            "q_descale",
            "scheduler_metadata",
            "s_aux",
            "mask_mod",
            "aux_tensors",
            "dynamic_causal",
        )
    ],
)
def test_unsupported_attention_retains_original_call(field, value):
    arguments = dict(attention_arguments(), **{field: value})
    calls = []
    expected = object()

    def kernel(**kwargs):
        calls.append(kwargs)
        return expected

    attention = MiniCPM5PrefillAttention(kernel, 2048)
    assert attention(**arguments) is expected
    assert len(calls) == 1
    assert calls[0] == arguments


def test_half_precision_backend_descales_do_not_disable_chunking():
    arguments = dict(attention_arguments(), k_descale=object(), v_descale=object())
    assert MiniCPM5PrefillAttention(None, 2048).eligible(arguments)


@pytest.mark.parametrize("rows", [257, 512, 513, 749])
@pytest.mark.parametrize("prefix", [0, 37])
@pytest.mark.parametrize("provided_out", [False, True])
def test_query_chunks_preserve_absolute_causality_and_output_padding(
    rows, prefix, provided_out
):
    torch.manual_seed(7)
    context, page_size = rows + prefix, 16
    pages = (context + page_size - 1) // page_size
    q_storage = torch.randn(rows, 3, 4, dtype=torch.float64)
    q = q_storage[:, :2]
    kv = torch.randn(pages, page_size, 1, 8, dtype=torch.float64)
    k, v = kv[..., :4], kv[..., 4:]
    table = torch.randperm(pages, dtype=torch.int32).reshape(1, -1)
    lengths = torch.tensor([context], dtype=torch.int32)
    cu = torch.tensor([0, rows], dtype=torch.int32)
    storage = torch.full((rows, 3, 4), 7.0, dtype=torch.float64)
    out = storage[:, :2] if provided_out else None
    arguments = dict(
        q=q,
        k=k,
        v=v,
        out=out,
        seqused_k=lengths,
        cu_seqlens_q=cu,
        max_seqlen_q=rows,
        max_seqlen_k=context,
        block_table=table,
        causal=True,
    )
    calls = []

    def dense(query, active):
        keys = k[table[0].long()].reshape(-1, 4)[:active]
        values = v[table[0].long()].reshape(-1, 4)[:active]
        scores = torch.einsum("qhd,kd->hqk", query, keys) * 0.5
        query_positions = torch.arange(query.shape[0]) + active - query.shape[0]
        mask = torch.arange(active)[None, :] > query_positions[:, None]
        scores.masked_fill_(mask, -torch.inf)
        return torch.einsum("hqk,kd->qhd", scores.softmax(-1), values)

    def kernel(**piece):
        count = piece["q"].shape[0]
        calls.append(count)
        assert count > 1, "a one-query tail must remain on the prefill path"
        assert piece["cu_seqlens_q"].tolist() == [0, count]
        piece["out"].copy_(dense(piece["q"], int(piece["seqused_k"][0])))
        return piece["out"]

    for changed in (False, True):
        if changed:
            q.mul_(0.75)
            k.add_(0.02)
            v.mul_(0.85)
            table.copy_(table.flip(1).contiguous())
            if prefix:
                lengths.sub_(7)
        expected = dense(q, int(lengths[0]))
        result = split_attention_queries(kernel, arguments, 256)
        torch.testing.assert_close(result, expected, atol=2e-14, rtol=2e-14)
        assert sum(calls) == rows
        calls.clear()
        assert cu.tolist() == [0, rows]
        if provided_out:
            assert result is out
            assert torch.all(storage[:, 2:] == 7)


@pytest.mark.parametrize(
    "dtype,tp,chunk_size",
    [
        (torch.float16, 2, 0),
        (torch.bfloat16, 2, 2048),
        (torch.float16, 1, 2048),
        (torch.float16, 2, 2048),
    ],
)
def test_prefill_override_is_owned_by_qualified_model_layers(dtype, tp, chunk_size):
    from vllm.v1.attention.backends import flash_attn

    original = flash_attn._FA4_DENSE_ATTENTION_KERNEL
    impls = [
        flash_attn.FlashAttentionImpl.__new__(flash_attn.FlashAttentionImpl)
        for _ in range(2)
    ]
    for impl in impls:
        impl.dense_attention_kernel = original
        impl.vllm_flash_attn_version = 2
    model = NS(layers=[NS(self_attn=NS(attn=NS(impl=impl))) for impl in impls])
    config = NS(
        additional_config={"minicpm5_prefill_query_chunk_size": chunk_size},
        model_config=NS(dtype=dtype),
        parallel_config=NS(tensor_parallel_size=tp),
    )
    configure_prefill_attention(model, config)
    enabled = dtype == torch.float16 and tp == 2 and chunk_size != 0
    for impl in impls:
        if enabled:
            assert isinstance(impl.dense_attention_kernel, MiniCPM5PrefillAttention)
            assert impl.dense_attention_kernel.kernel is original
        else:
            assert impl.dense_attention_kernel is original
    assert flash_attn._FA4_DENSE_ATTENTION_KERNEL is original
    if enabled:
        assert impls[0].dense_attention_kernel is not impls[1].dense_attention_kernel


@pytest.mark.parametrize("value", [True, 2048.0, -1, 256, None])
def test_unqualified_query_chunk_sizes_are_rejected(value):
    config = NS(additional_config={"minicpm5_prefill_query_chunk_size": value})
    with pytest.raises(ValueError, match="must be 0 or 2048"):
        configure_prefill_attention(None, config)
