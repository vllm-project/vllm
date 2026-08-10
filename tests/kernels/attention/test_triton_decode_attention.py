# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.triton_decode_attention import (
    decode_attention_fwd,
    stage1_block_h,
    stage1_head_tiles,
)

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


def _reference_mla_decode(q, kv, req_to_page, seq_len, sm_scale, d_v, page_size):
    """Plain-PyTorch MLA decode. v is the leading d_v dims of the same latent."""
    B, H, _ = q.shape
    d_qk = kv.shape[-1]
    kv_flat = kv.reshape(-1, d_qk).float()
    tok = req_to_page[:, :, 0:1] * page_size + torch.arange(
        page_size, device=req_to_page.device
    ).view(1, 1, -1)
    tok = tok.reshape(B, -1)[:, :seq_len]
    out = torch.empty(B, H, d_v, dtype=torch.float32, device=q.device)
    for b in range(B):
        k = kv_flat[tok[b]]
        p = ((q[b].float() @ k.T) * sm_scale).softmax(-1)
        out[b] = p @ k[:, :d_v]
    return out


# 112 = 16 query heads per rank x a 7-token DSpark block folded into the head
# dim, i.e. the shape production actually launches; it is also not a multiple of
# the 64-wide tile, so the final head tile is partial. 64 is the tile boundary.
@pytest.mark.parametrize("H_Q", [64, 112])
@pytest.mark.parametrize("quant", ["bf16", "fp8_kv", "fp8_kv_and_q"])
def test_decode_attention_mla_wide_head_tile(H_Q, quant):
    """MLA decode with enough query heads to select the wide head tile.

    The other tests in this file compare one kernel launch against another, so a
    tile-shape or masking fault that both launches share cancels out. This one
    checks against PyTorch, and covers the >=64 kv_group_num branch that no
    other case reaches.
    """
    B, seq_len, D_QK, D_V = 2, 517, 576, 512
    PAGE_SIZE, CACHE_SIZE, num_kv_splits = 16, 4096, 8
    dtype = torch.bfloat16
    sm_scale = 1.0 / (D_QK**0.5)

    req_to_page = torch.randint(
        0,
        CACHE_SIZE // PAGE_SIZE,
        (B, cdiv(seq_len, PAGE_SIZE), 1),
        device=DEVICE_TYPE,
    )
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE) / 4
    kv = torch.randn(CACHE_SIZE, 1, D_QK, dtype=dtype, device=DEVICE_TYPE) / 4

    q_in, kv_in = q, kv
    q_scale = k_scale = None
    if quant != "bf16":
        kv_fp8, k_scale = _quantize_to_fp8(kv)
        kv_in = kv_fp8
        # The reference has to see what the kernel sees.
        kv = (kv_fp8.to(torch.float32) * k_scale).to(dtype)
    if quant == "fp8_kv_and_q":
        q_fp8, q_scale = _quantize_to_fp8(q)
        q_in = q_fp8
        q = (q_fp8.to(torch.float32) * q_scale).to(dtype)

    ref = _reference_mla_decode(q, kv, req_to_page, seq_len, sm_scale, D_V, PAGE_SIZE)

    kv_paged = kv_in.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, 1, D_QK)
    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
    lse = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
    )
    decode_attention_fwd(
        q_in,
        kv_paged,
        kv_paged[..., :D_V],
        o,
        lse,
        req_to_page,
        torch.full((B,), seq_len, device=DEVICE_TYPE),
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
        k_scale=k_scale,
        v_scale=k_scale,
        q_scale=q_scale,
        is_mla=True,
    )

    # Same tolerance as the fp8 case above; a tile or masking fault produces
    # garbage rather than a small drift, so this is wide enough to catch it.
    torch.testing.assert_close(ref.to(dtype), o, atol=5e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "q_head_num,kv_group_num,is_mla,expected_block_h",
    [
        (112, 112, True, 64),  # folded DSpark block
        (64, 64, True, 64),
        (32, 32, True, 16),  # below the wide-tile threshold
        (16, 16, True, 16),
        (32, 4, False, 16),  # GQA never takes the wide tile
    ],
)
def test_stage1_head_tiles_matches_launch_grid(
    q_head_num, kv_group_num, is_mla, expected_block_h
):
    """The split heuristic sizes itself from this, so it must equal the grid.

    _decode_grouped_att_m_fwd builds grid[1] as cdiv(q_head_num, min(BLOCK_H,
    kv_group_num)). A caller that sizes num_kv_splits from a different number
    over- or under-splits without failing, so pin the two together.
    """
    if not current_platform.is_rocm() and expected_block_h == 64:
        pytest.skip("the wide tile is gated to HIP")
    assert stage1_block_h(is_mla, kv_group_num) == expected_block_h
    assert stage1_head_tiles(q_head_num, kv_group_num, is_mla) == cdiv(
        q_head_num, min(expected_block_h, kv_group_num)
    )
