# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.attention.ops.triton_fp8_ds_mla import (
    _dequantize_e4m3,
    decode_attention_fwd_fp8_ds_mla,
)

pytestmark = pytest.mark.skip_global_cleanup

DEVICE_TYPE = current_platform.device_type


@pytest.mark.parametrize("B", [3, 5])
@pytest.mark.parametrize("L", [1027, 1025])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D_QK", [128, 192, 576])
@pytest.mark.parametrize("D_V", [128, 512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
def test_decode_attention(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L  # This represents the number of tokens already in the sequence
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device=DEVICE_TYPE
    )
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device=DEVICE_TYPE).view(
        1, 1, -1
    )
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)

    # k_buffer and v_buffer represent all previous tokens
    # Page size is 1.
    k_buffer = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE)
    v_buffer = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE)

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)

    lse = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)

    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)

    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1),
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )

    # Call the original implementation.
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
    )

    # Page size can be larger than 1.
    k_buffer = k_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
    v_buffer = v_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)

    o1 = torch.zeros_like(o)
    lse1 = torch.zeros_like(lse)

    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o1,
        lse1,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
    )

    assert torch.allclose(o, o1)


def _quantize_to_fp8(tensor: torch.Tensor):
    """Quantize a BF16 tensor to FP8 e4m3fn with per-tensor scale.

    Returns (fp8_tensor, scale) where:
        fp8_tensor ≈ tensor / scale  (stored as float8_e4m3fn)
        tensor ≈ fp8_tensor.to(float32) * scale  (dequantized)
    """
    amax = tensor.abs().amax()
    # float8_e4m3fn max representable value is 448.0
    scale = (amax / 448.0).clamp(min=1e-12).to(torch.float32)
    fp8_tensor = (
        (tensor.to(torch.float32) / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return fp8_tensor, scale


@pytest.mark.parametrize("B", [3])
@pytest.mark.parametrize("L", [1025])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D_QK", [128, 576])
@pytest.mark.parametrize("D_V", [128, 512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
def test_decode_attention_fp8(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    """Test FP8 KV cache path: quantize K/V to FP8, run kernel with scales,
    and compare against BF16 reference output."""
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device=DEVICE_TYPE
    )
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device=DEVICE_TYPE).view(
        1, 1, -1
    )
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)

    # Create BF16 K/V as reference
    k_bf16 = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE)
    v_bf16 = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE)

    # --- BF16 reference ---
    o_ref = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
    lse_ref = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
    )

    if PAGE_SIZE == 1:
        decode_attention_fwd(
            q,
            k_bf16,
            v_bf16,
            o_ref,
            lse_ref,
            req_to_token,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
        )
    else:
        k_paged = k_bf16.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
        v_paged = v_bf16.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)
        decode_attention_fwd(
            q,
            k_paged,
            v_paged,
            o_ref,
            lse_ref,
            req_to_page,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            page_size=PAGE_SIZE,
        )

    # --- FP8 path ---
    k_fp8, k_scale = _quantize_to_fp8(k_bf16)
    v_fp8, v_scale = _quantize_to_fp8(v_bf16)

    o_fp8 = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
    lse_fp8 = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
    attn_logits_fp8 = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
    )

    if PAGE_SIZE == 1:
        decode_attention_fwd(
            q,
            k_fp8,
            v_fp8,
            o_fp8,
            lse_fp8,
            req_to_token,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits_fp8,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    else:
        k_fp8_paged = k_fp8.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
        v_fp8_paged = v_fp8.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)
        decode_attention_fwd(
            q,
            k_fp8_paged,
            v_fp8_paged,
            o_fp8,
            lse_fp8,
            req_to_page,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits_fp8,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            page_size=PAGE_SIZE,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # FP8 tolerances match test_mla_backends.py test_backend_correctness.
    torch.testing.assert_close(o_ref, o_fp8, atol=5e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "H_Q,H_KV,D_QK,D_V,is_mla",
    [
        (16, 1, 576, 512, True),  # MLA path (grouped kernel, v = trans(k))
        (32, 8, 128, 128, False),  # GQA path (grouped kernel)
        (32, 32, 128, 128, False),  # MHA path (normal kernel)
    ],
)
@pytest.mark.parametrize("PAGE_SIZE", [16])
def test_decode_attention_cross_layer_view(H_Q, H_KV, D_QK, D_V, is_mla, PAGE_SIZE):
    """The kernel must honor the cache's page-dim stride, not assume pages are
    packed back-to-back. A per-layer view into a cross-layer (block-major)
    cache has stride(0) inflated by num_layers; outputs must match a
    contiguous cache holding the same data exactly."""
    B = 3
    seq_len = 1027
    CACHE_SIZE = 16384
    NUM_LAYERS = 3
    LAYER_IDX = 1
    dtype = torch.bfloat16
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8
    num_pages = CACHE_SIZE // PAGE_SIZE

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, num_pages, (B, num_pages_per_batch), device=DEVICE_TYPE
    )

    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)
    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)

    # Reference: contiguous paged cache.
    k_ref = torch.randn(
        num_pages, PAGE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE
    )
    if is_mla:
        v_ref = k_ref[..., :D_V]
    else:
        v_ref = torch.randn(
            num_pages, PAGE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE
        )

    # Cross-layer cache: all layers' pages for a block are adjacent. The
    # per-layer view has the same shape as the contiguous cache but
    # stride(0) is NUM_LAYERS x larger. Neighbor layers hold random data so
    # any packed-pages addressing reads garbage rather than zeros.
    k_xl = torch.randn(
        num_pages, NUM_LAYERS, PAGE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE
    )
    k_view = k_xl[:, LAYER_IDX]
    k_view.copy_(k_ref)
    assert k_view.stride(0) == NUM_LAYERS * PAGE_SIZE * H_KV * D_QK
    if is_mla:
        v_view = k_view[..., :D_V]
    else:
        v_xl = torch.randn(
            num_pages, NUM_LAYERS, PAGE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE
        )
        v_view = v_xl[:, LAYER_IDX]
        v_view.copy_(v_ref)

    def run(k_buffer, v_buffer):
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
        lse = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
        )
        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_page,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            PAGE_SIZE,
            is_mla=is_mla,
        )
        return o, lse

    o_ref, lse_ref = run(k_ref, v_ref)
    o_xl, lse_xl = run(k_view, v_view)

    # Same data and same compute order; only addressing differs.
    assert torch.equal(o_ref, o_xl)
    assert torch.equal(lse_ref, lse_xl)


@pytest.mark.parametrize(
    "B,H_Q,H_KV,D_QK,D_V,is_mla",
    [
        (256, 32, 1, 576, 512, True),  # grouped kernel
        (1024, 32, 32, 128, 128, False),  # normal kernel
    ],
)
def test_decode_attention_int64_logits_offset(B, H_Q, H_KV, D_QK, D_V, is_mla):
    """attn_logits offsets must not overflow int32 when B * H_Q * num_kv_splits
    * (D_V + 1) exceeds 2**31 (long max_model_len, large decode batch)."""
    PAGE_SIZE = 16
    seq_len = 33
    num_kv_splits = 512
    dtype = torch.bfloat16
    sm_scale = 1.0 / (D_QK**0.5)
    assert B * H_Q * num_kv_splits * (D_V + 1) > 2**31

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.arange(
        B * num_pages_per_batch, device=DEVICE_TYPE, dtype=torch.int32
    ).view(B, num_pages_per_batch)
    k_buffer = torch.randn(
        B * num_pages_per_batch, PAGE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE
    )
    v_buffer = (
        k_buffer[..., :D_V]
        if is_mla
        else torch.randn(
            B * num_pages_per_batch,
            PAGE_SIZE,
            H_KV,
            D_V,
            dtype=dtype,
            device=DEVICE_TYPE,
        )
    )
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)
    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)

    def run(rows):
        n = rows.stop - rows.start
        o = torch.zeros(n, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
        lse = torch.zeros(n, H_Q, dtype=dtype, device=DEVICE_TYPE)
        attn_logits = torch.empty(
            (n, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
        )
        decode_attention_fwd(
            q[rows],
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_page[rows],
            b_seq_len[rows],
            attn_logits,
            num_kv_splits,
            sm_scale,
            PAGE_SIZE,
            is_mla=is_mla,
        )
        return o, lse

    o, lse = run(slice(0, B))
    o_last, lse_last = run(slice(B - 1, B))

    # Same compute order; the last request only differs in its logits offset.
    assert torch.equal(o[-1:], o_last)
    assert torch.equal(lse[-1:], lse_last)


@triton.jit
def _convert_packed_mla_e4m3(Bytes, Scales, Output):
    d = tl.arange(0, 256)
    tl.store(Output + d, _dequantize_e4m3(tl.load(Bytes + d), tl.load(Scales + d)))


@pytest.mark.parametrize("scale", [0.0, 0.003, 1.0, 2.0**-126])
def test_packed_fp8_software_e4m3_conversion(scale):
    raw = torch.arange(256, dtype=torch.uint8, device="cuda")
    scales = torch.full((256,), scale, dtype=torch.float32, device="cuda")
    actual = torch.empty((256,), dtype=torch.bfloat16, device="cuda")
    _convert_packed_mla_e4m3[(1,)](raw, scales, actual)
    expected = (raw.view(torch.float8_e4m3fn).float() * scales).bfloat16()
    finite = torch.isfinite(expected)
    torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(
        actual.view(torch.int16)[finite],
        expected.view(torch.int16)[finite],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "token_stride,block_padding,storage_offset,workspace_width,workspace_padding,error",
    [
        (657, 0, 0, 513, 0, "cache strides and storage offset"),
        (656, 1, 0, 513, 0, "cache strides and storage offset"),
        (656, 0, 1, 513, 0, "cache strides and storage offset"),
        (656, 0, 0, 512, 0, "workspace"),
        (656, 0, 0, 513, 1, "workspace"),
    ],
)
def test_packed_fp8_decode_rejects_invalid_layout(
    token_stride,
    block_padding,
    storage_offset,
    workspace_width,
    workspace_padding,
    error,
):
    """Reject unsafe typed cache views and incompatible split-KV workspace layouts."""
    storage = torch.empty(
        2 * 16 * token_stride + block_padding + storage_offset, dtype=torch.uint8
    )
    cache = storage.as_strided(
        (2, 16, 656),
        (16 * token_stride + block_padding, token_stride, 1),
        storage_offset,
    )
    q = torch.empty(2, 2, 576, dtype=torch.bfloat16)
    output = torch.empty(2, 2, 512, dtype=q.dtype)
    lse = torch.empty(2, 2, dtype=torch.float32)
    table = torch.zeros(2, 1, dtype=torch.int32)
    lengths = torch.ones(2, dtype=torch.int32)
    mid = torch.empty(2, 2, 2, workspace_width + workspace_padding)[
        ..., :workspace_width
    ]
    with pytest.raises(AssertionError, match=error):
        decode_attention_fwd_fp8_ds_mla(
            q, cache, output, lse, table, lengths, mid, 2, 576**-0.5
        )


def _make_packed_fp8_mla_case(batch, length, heads, page_size, padded):
    torch.manual_seed(42)
    pages_per_request = triton.cdiv(length, page_size)
    pages = batch * pages_per_request
    table = torch.randperm(pages, device="cuda").reshape(batch, -1).int()
    lengths = torch.tensor(
        [max(1, length - 17 * i) for i in range(batch)],
        device="cuda",
        dtype=torch.int32,
    )
    latent = torch.randn(pages, page_size, 512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(pages, page_size, 64, device="cuda", dtype=torch.bfloat16)
    grouped = latent.float().reshape(pages, page_size, 4, 128)
    scales = (grouped.abs().amax(-1) / 448).clamp_min(torch.finfo(torch.float32).tiny)
    fp8 = (grouped / scales[..., None]).to(torch.float8_e4m3fn)
    storage = torch.empty(
        pages, page_size, 672 if padded else 656, device="cuda", dtype=torch.uint8
    )
    cache = storage[..., :656]
    cache[..., :512] = fp8.reshape(pages, page_size, 512).view(torch.uint8)
    cache[..., 512:528] = scales.contiguous().view(torch.uint8)
    cache[..., 528:] = rope.view(torch.uint8)
    restored = (
        (fp8.float() * scales[..., None]).reshape(pages, page_size, 512).bfloat16()
    )
    unpacked = torch.cat((restored, rope), dim=-1)
    q = torch.randn(batch, heads, 576, device="cuda", dtype=torch.bfloat16)
    return q, cache, unpacked, table, lengths


def _assert_packed_mla_reference(q, unpacked, table, lengths, output, lse):
    for b, length in enumerate(lengths.cpu().tolist()):
        kv = unpacked[table[b].long()].reshape(-1, 576)[:length].float()
        logits = (q[b].float() @ kv.T) * 576**-0.5
        expected = logits.softmax(-1) @ kv[:, :512]
        torch.testing.assert_close(output[b].float(), expected, atol=0.005, rtol=0.03)
        torch.testing.assert_close(
            lse[b].float(), logits.logsumexp(-1), atol=0.01, rtol=0.001
        )


@pytest.mark.parametrize(
    "batch,length,heads,page_size,splits",
    [
        (1, 1, 1, 16, 4),
        (3, 31, 17, 16, 4),
        (3, 1027, 32, 16, 16),
        (5, 1025, 16, 32, 8),
        (2, 4097, 16, 16, 1),
        (2, 8193, 128, 16, 16),
    ],
)
@pytest.mark.parametrize("padded", [False, True])
def test_packed_fp8_decode(batch, length, heads, page_size, splits, padded):
    q, cache, unpacked, table, lengths = _make_packed_fp8_mla_case(
        batch, length, heads, page_size, padded
    )
    output = torch.empty(batch, heads, 512, device="cuda", dtype=q.dtype)
    lse = torch.empty(batch, heads, device="cuda", dtype=torch.float32)
    mid = torch.full(
        (batch, heads, splits, 513), float("nan"), device="cuda", dtype=torch.float32
    )
    decode_attention_fwd_fp8_ds_mla(
        q, cache, output, lse, table, lengths, mid, splits, 576**-0.5
    )
    _assert_packed_mla_reference(q, unpacked, table, lengths, output, lse)


def test_packed_fp8_cuda_graph_replay():
    q, cache, unpacked, table, lengths = _make_packed_fp8_mla_case(
        3, 1027, 17, 16, False
    )
    output = torch.empty(3, 17, 512, device="cuda", dtype=q.dtype)
    lse = torch.empty(3, 17, device="cuda", dtype=torch.float32)
    mid = torch.empty(3, 17, 16, 513, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            decode_attention_fwd_fp8_ds_mla(
                q, cache, output, lse, table, lengths, mid, 16, 576**-0.5
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_attention_fwd_fp8_ds_mla(
            q, cache, output, lse, table, lengths, mid, 16, 576**-0.5
        )
    lengths.copy_(torch.tensor([1, 31, 1025], device="cuda", dtype=torch.int32))
    q.mul_(0.5)
    graph.replay()
    _assert_packed_mla_reference(q, unpacked, table, lengths, output, lse)


def test_packed_fp8_batch_invariance():
    q, cache, _, table, lengths = _make_packed_fp8_mla_case(3, 1027, 17, 16, False)
    output = torch.empty(3, 17, 512, device="cuda", dtype=q.dtype)
    lse = torch.empty(3, 17, device="cuda", dtype=torch.float32)
    mid = torch.empty(3, 17, 1, 513, device="cuda", dtype=torch.float32)
    decode_attention_fwd_fp8_ds_mla(
        q, cache, output, lse, table, lengths, mid, 1, 576**-0.5
    )
    single_output = torch.empty_like(output[:1])
    single_lse = torch.empty_like(lse[:1])
    decode_attention_fwd_fp8_ds_mla(
        q[:1],
        cache,
        single_output,
        single_lse,
        table[:1],
        lengths[:1],
        mid[:1],
        1,
        576**-0.5,
    )
    torch.testing.assert_close(single_output, output[:1], atol=0, rtol=0)
    torch.testing.assert_close(single_lse, lse[:1], atol=0, rtol=0)
