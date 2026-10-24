# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.supports_fp8():
    pytest.skip(
        "MiniMax-M3 FP8 sparse attention scale tests require an FP8 GPU.",
        allow_module_level=True,
    )

if current_platform.is_rocm():
    from vllm.models.minimax_m3.amd.ops.sparse_attn import (
        SPARSE_BLOCK_SIZE,
        minimax_m3_sparse_attn,
        minimax_m3_sparse_attn_decode,
    )
else:
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        SPARSE_BLOCK_SIZE,
        minimax_m3_sparse_attn,
        minimax_m3_sparse_attn_decode,
    )


DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 128
NUM_KV_HEADS = 1
NUM_HEADS = 2
K_SCALE = 0.25
V_SCALE = 0.5


def _scale_tensors(mode: str, num_blocks: int):
    if mode == "scalar":
        k_scale = torch.tensor(K_SCALE, dtype=torch.float32, device=DEVICE)
        v_scale = torch.tensor(V_SCALE, dtype=torch.float32, device=DEVICE)
    else:
        shape = (NUM_KV_HEADS, num_blocks * SPARSE_BLOCK_SIZE)
        k_scale = torch.full(shape, K_SCALE, dtype=torch.float32, device=DEVICE)
        v_scale = torch.full(shape, V_SCALE, dtype=torch.float32, device=DEVICE)
    return k_scale, v_scale


def _make_kv_cache(num_blocks: int, seed: int):
    torch.manual_seed(seed)
    fp8_dtype = current_platform.fp8_dtype()
    kv_ref = torch.randn(
        num_blocks,
        NUM_KV_HEADS,
        SPARSE_BLOCK_SIZE,
        2 * HEAD_DIM,
        dtype=DTYPE,
        device=DEVICE,
    )
    kv_fp8 = torch.empty_like(kv_ref, dtype=fp8_dtype)
    kv_fp8[..., :HEAD_DIM] = (kv_ref[..., :HEAD_DIM].float() / K_SCALE).to(fp8_dtype)
    kv_fp8[..., HEAD_DIM:] = (kv_ref[..., HEAD_DIM:].float() / V_SCALE).to(fp8_dtype)

    kv_dequant = torch.empty_like(kv_ref)
    kv_dequant[..., :HEAD_DIM] = (kv_fp8[..., :HEAD_DIM].float() * K_SCALE).to(DTYPE)
    kv_dequant[..., HEAD_DIM:] = (kv_fp8[..., HEAD_DIM:].float() * V_SCALE).to(DTYPE)
    return kv_fp8, kv_dequant


@pytest.mark.parametrize("scale_mode", ["scalar", "per_token_head"])
@torch.inference_mode()
def test_minimax_m3_sparse_prefill_fp8_kv_scales(scale_mode: str):
    total_q = 17
    num_blocks = 1
    torch.manual_seed(0)
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.1
    kv_fp8, kv_dequant = _make_kv_cache(num_blocks, seed=1)
    k_scale, v_scale = _scale_tensors(scale_mode, num_blocks)

    topk = torch.zeros(NUM_KV_HEADS, total_q, 1, dtype=torch.int32, device=DEVICE)
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
    cu_seqlens = torch.tensor([0, total_q], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([total_q], dtype=torch.int32, device=DEVICE)
    prefix_lens = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    got = torch.empty_like(q)
    ref = torch.empty_like(q)
    unscaled = torch.empty_like(q)

    minimax_m3_sparse_attn(
        q,
        kv_fp8,
        topk,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        total_q,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        got,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    minimax_m3_sparse_attn(
        q,
        kv_dequant,
        topk,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        total_q,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        ref,
    )
    minimax_m3_sparse_attn(
        q,
        kv_fp8,
        topk,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        total_q,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        unscaled,
    )

    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    assert not torch.allclose(unscaled, ref, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("scale_mode", ["scalar", "per_token_head"])
@torch.inference_mode()
def test_minimax_m3_sparse_decode_fp8_kv_scales(scale_mode: str):
    total_q = 2
    num_blocks = 1
    torch.manual_seed(2)
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.1
    kv_fp8, kv_dequant = _make_kv_cache(num_blocks, seed=3)
    k_scale, v_scale = _scale_tensors(scale_mode, num_blocks)

    topk = torch.zeros(NUM_KV_HEADS, total_q, 1, dtype=torch.int32, device=DEVICE)
    block_table = torch.zeros(total_q, 1, dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([64, 128], dtype=torch.int32, device=DEVICE)
    got = torch.empty_like(q)
    ref = torch.empty_like(q)
    unscaled = torch.empty_like(q)

    minimax_m3_sparse_attn_decode(
        q,
        kv_fp8,
        topk,
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        got,
        decode_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    minimax_m3_sparse_attn_decode(
        q,
        kv_dequant,
        topk,
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        ref,
        decode_query_len=1,
    )
    minimax_m3_sparse_attn_decode(
        q,
        kv_fp8,
        topk,
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        unscaled,
        decode_query_len=1,
    )

    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    assert not torch.allclose(unscaled, ref, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("page_capacity", [1, 4, 17])
@pytest.mark.parametrize("mode", ["bf16", "scalar", "per_token_head"])
@pytest.mark.parametrize("decode_query_len", [1, 3])
@pytest.mark.parametrize("num_reqs,num_kv_heads", [(1, 1), (2, 2), (4, 1)])
@torch.inference_mode()
def test_page_split_decode_matches_dense(
    mode, decode_query_len, num_reqs, num_kv_heads, page_capacity
):
    """Check causal tails, permuted physical pages and both FP8 scale layouts."""
    if not current_platform.is_cuda() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
    ):
        pytest.skip("Blackwell single-page path")
    if page_capacity > 1 and (
        torch.cuda.get_device_capability() != (12, 0)
        or (mode == "bf16" and num_reqs * decode_query_len * num_kv_heads > 16)
    ):
        pytest.skip("Outside page-split dispatch")
    torch.manual_seed(73)
    total_q = num_reqs * decode_query_len
    heads = num_kv_heads * 16
    q = torch.randn(total_q, heads, 128, device=DEVICE, dtype=DTYPE)
    kv = torch.randn(
        (num_reqs + 1) * page_capacity,
        num_kv_heads,
        128,
        256,
        device=DEVICE,
        dtype=DTYPE,
    )
    if mode != "bf16":
        kv = kv.to(torch.float8_e4m3fn)
    ks = vs = None
    if mode == "scalar":
        ks = torch.tensor([0.7], device=DEVICE)
        vs = torch.tensor([1.3], device=DEVICE)
    elif mode == "per_token_head":
        ks = 0.5 + torch.rand(
            num_kv_heads, (num_reqs + 1) * page_capacity * 128, device=DEVICE
        )
        vs = 0.5 + torch.rand_like(ks)
    table = torch.randperm(
        num_reqs * page_capacity, device=DEVICE, dtype=torch.int32
    ).reshape(num_reqs, page_capacity)
    lens = torch.tensor(
        ([1, 63, 127, 128][:num_reqs]), device=DEVICE, dtype=torch.int32
    )
    if page_capacity > 1:
        lens += (page_capacity - 1) * 128
    topk = torch.full(
        (num_kv_heads, total_q, 16), 2147483647, device=DEVICE, dtype=torch.int32
    )
    for token in range(total_q):
        req, local = divmod(token, decode_query_len)
        visible = max(int(lens[req]) - decode_query_len + local + 1, 0)
        pages = (visible + 127) // 128
        for head in range(num_kv_heads):
            topk[head, token, : min(16, pages)] = torch.randperm(
                pages, device=DEVICE, dtype=torch.int32
            )[:16]
    output = torch.full_like(q, float("nan"))
    minimax_m3_sparse_attn_decode(
        q,
        kv,
        topk,
        table,
        lens,
        num_kv_heads,
        128**-0.5,
        output,
        decode_query_len,
        ks,
        vs,
    )
    reference = torch.zeros_like(q)
    for token in range(total_q):
        req, local = divmod(token, decode_query_len)
        visible = max(int(lens[req]) - decode_query_len + local + 1, 0)
        if visible == 0:
            continue
        for head in range(num_kv_heads):
            keys, values = [], []
            for logical in topk[
                head, token, : min(16, (visible + 127) // 128)
            ].tolist():
                page = int(table[req, logical])
                count = min(128, visible - logical * 128)
                k = kv[page, head, :count, :128].to(DTYPE)
                v = kv[page, head, :count, 128:].to(DTYPE)
                if ks is not None:
                    assert vs is not None
                    k_scale = (
                        ks
                        if ks.numel() == 1
                        else ks[head, page * 128 : page * 128 + count, None]
                    )
                    v_scale = (
                        vs
                        if vs.numel() == 1
                        else vs[head, page * 128 : page * 128 + count, None]
                    )
                    k = (k.float() * k_scale).to(DTYPE)
                    v = (v.float() * v_scale).to(DTYPE)
                keys.append(k)
                values.append(v)
            k, v = torch.cat(keys), torch.cat(values)
            qq = q[token, head * 16 : (head + 1) * 16].float()
            reference[token, head * 16 : (head + 1) * 16] = (
                torch.softmax(qq @ k.float().T * 128**-0.5, dim=-1) @ v.float()
            ).to(DTYPE)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("page_capacity", [1, 4, 17])
@pytest.mark.parametrize("fp8", [False, True])
@torch.inference_mode()
def test_page_split_decode_graph_length_changes(fp8, page_capacity):
    """A reused output must become zero for empty rows, including after replay."""
    if not current_platform.is_cuda() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
    ):
        pytest.skip("Blackwell single-page path")
    if page_capacity > 1 and torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 page-split path")
    q = torch.randn(1, 16, 128, device=DEVICE, dtype=DTYPE)
    kv = torch.randn(page_capacity, 1, 128, 256, device=DEVICE, dtype=DTYPE)
    if fp8:
        kv = kv.to(torch.float8_e4m3fn)
    topk = torch.arange(16, device=DEVICE, dtype=torch.int32).reshape(1, 1, 16)
    table = torch.arange(page_capacity, device=DEVICE, dtype=torch.int32).reshape(
        1, page_capacity
    )
    lens = torch.tensor([128 * page_capacity], device=DEVICE, dtype=torch.int32)
    output = torch.empty_like(q)

    def run():
        minimax_m3_sparse_attn_decode(q, kv, topk, table, lens, 1, 128**-0.5, output, 1)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for length in [0, 1, min(129, page_capacity * 128), page_capacity * 128, 0]:
            lens.fill_(length)
            output.fill_(float("nan"))
            graph.replay()
            assert torch.isfinite(output).all()
            if length == 0:
                assert torch.count_nonzero(output) == 0
            elif length == 1:
                expected = kv[0, 0, 0, 128:].to(DTYPE).expand_as(output)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)
