# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.attention.ops.triton_decode_attention import decode_attention_fwd


def cdiv(a, b):
    return (a + b - 1) // b


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
    req_to_page = torch.randint(0,
                                CACHE_SIZE // PAGE_SIZE,
                                (B, num_pages_per_batch, 1),
                                device="cuda")
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device="cuda").view(
        1, 1, -1)
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device="cuda")

    # k_buffer and v_buffer represent all previous tokens
    # Page size is 1.
    k_buffer = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device="cuda")
    v_buffer = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device="cuda")

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")

    b_seq_len = torch.full((B, ), seq_len, device="cuda")

    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1),
        dtype=torch.float32,
        device="cuda",
    )

    # Call the original implementation.
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
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

    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o1,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
    )

    assert torch.allclose(o, o1)
