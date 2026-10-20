# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

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


def _verify_reference(q, cache, block_table, seq_lens, real_seq_lens, qlen, scale):
    """fp32 causal MLA verify: row t of a request attends its first
    min(L - qlen + 1 + t, L_real) tokens; a row with none comes back zero."""
    kv_lora_rank = 512
    page_size = cache.shape[1]
    out = torch.zeros(q.shape[0], q.shape[1], kv_lora_rank, device=q.device)
    for b, seq_len in enumerate(seq_lens.tolist()):
        real = seq_len if real_seq_lens is None else real_seq_lens[b].item()
        num_pages = cdiv(max(seq_len, 1), page_size)
        kv = cache[block_table[b, :num_pages].long()].reshape(-1, cache.shape[-1])
        kv = kv[:seq_len].float()
        for t in range(qlen):
            lim = max(min(seq_len - qlen + 1 + t, real), 0)
            if lim == 0:
                continue
            qt = q[b * qlen + t].float()
            p = torch.softmax(qt @ kv[:lim].T * scale, dim=-1)
            out[b * qlen + t] = p @ kv[:lim, :kv_lora_rank]
    return out


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only the ROCm AITER MLA backend uses it"
)
@pytest.mark.parametrize("num_heads", [8, 16, 128])
@pytest.mark.parametrize("qlen", [5, 8])
# 2 requests stay on the small-tile config; 24 reach the 64-row tiles.
@pytest.mark.parametrize("batch", [2, 24])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_decode_attention_fwd_verify(num_heads, qlen, batch, page_size, kv_dtype):
    from vllm.v1.attention.ops.triton_decode_attention import (
        decode_attention_fwd_verify,
    )

    set_random_seed(0)
    dim, kv_lora_rank = 576, 512
    scale = 1.0 / (192**0.5)
    # Ragged lengths, one shorter than qlen, and a graph padding request.
    lens = torch.randint(qlen, 700, (batch,))
    lens[0] = qlen - 2
    lens[-1] = 0
    seq_lens = lens.to(torch.int32).to(DEVICE_TYPE)
    pages_per_req = cdiv(int(lens.max()), page_size) + 1
    num_pages = batch * pages_per_req + 3
    block_table = (
        torch.randperm(num_pages, device=DEVICE_TYPE)[: batch * pages_per_req]
        .view(batch, pages_per_req)
        .to(torch.int32)
    )
    cache = torch.randn(num_pages, page_size, dim, device=DEVICE_TYPE) * 0.5
    k_scale = torch.tensor(1.0, device=DEVICE_TYPE)
    if kv_dtype == "fp8":
        cache = cache.to(current_platform.fp8_dtype())
        k_scale = torch.tensor(3.0, device=DEVICE_TYPE)
        ref_cache = cache.float() * 3.0
    else:
        cache = cache.to(torch.bfloat16)
        ref_cache = cache
    q = torch.randn(batch * qlen, num_heads, dim, device=DEVICE_TYPE).to(torch.bfloat16)
    paged = cache.unsqueeze(2)
    # Error bound of the two bf16 roundings: P before the PV dot (at most
    # 2^-8 * max|V| absolute) and the output (2^-8 relative).
    atol = 2**-8 * ref_cache[..., :kv_lora_rank].float().abs().max().item()
    num_kv_splits = 4
    for real_seq_lens in (None, (seq_lens - 2).clamp(min=0)):
        o = torch.full(
            (batch * qlen, num_heads, kv_lora_rank),
            float("nan"),
            dtype=torch.bfloat16,
            device=DEVICE_TYPE,
        )
        lse = torch.empty(batch * qlen, num_heads, device=DEVICE_TYPE)
        logits = torch.full(
            (batch * qlen, num_heads, num_kv_splits, kv_lora_rank + 1),
            float("nan"),
            device=DEVICE_TYPE,
        )
        decode_attention_fwd_verify(
            q,
            paged,
            paged[..., :kv_lora_rank],
            o,
            lse,
            block_table,
            seq_lens,
            real_seq_lens,
            qlen,
            logits,
            num_kv_splits,
            scale,
            page_size,
            k_scale=k_scale,
        )
        ref = _verify_reference(
            q, ref_cache, block_table, seq_lens, real_seq_lens, qlen, scale
        )
        torch.testing.assert_close(o.float(), ref, atol=atol, rtol=2**-8)
