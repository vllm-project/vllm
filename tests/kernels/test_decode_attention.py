import torch

from vllm.attention.ops.decode_attention import decode_attention_fwd


def cdiv(a, b):
    return (a + b - 1) // b


def test_decode_attention(B, H_Q, H_KV, D):
    dtype = torch.bfloat16
    seq_len = 128  # This represents the number of tokens already in the sequence
    total_tokens = B * seq_len
    sm_scale = 1.0 / (D**0.5)
    num_kv_splits = 8

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    # k_buffer and v_buffer represent all previous tokens
    # Page size is 1.
    k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")

    req_to_token = torch.arange(total_tokens, device="cuda").reshape(B, seq_len)
    b_req_idx = torch.arange(B, device="cuda")
    b_seq_len = torch.full((B,), seq_len, device="cuda")

    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D + 1),
        dtype=torch.float32,
        device="cuda",
    )

    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        req_to_token,
        b_req_idx,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
    )

    # Page size is larger than 1.
    page_size = 16
    num_pages = cdiv(total_tokens, page_size)
    k_buffer = k_buffer.view(num_pages, page_size, H_KV, D)
    v_buffer = v_buffer.view(num_pages, page_size, H_KV, D)

    o1 = torch.zeros_like(o)

    num_pages_per_batch = cdiv(seq_len, page_size)
    req_to_page = torch.arange(num_pages, device="cuda").reshape(B, num_pages_per_batch)
    b_req_idx = torch.arange(B, device="cuda")
    b_seq_len = torch.full((B,), seq_len, device="cuda")

    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D + 1),
        dtype=torch.float32,
        device="cuda",
    )

    # Trick: Flatten the KV cache so that inside the kernel, we use
    # page_size = 1.
    decode_attention_fwd(
        q,
        k_buffer.flatten(0, 1),
        v_buffer.flatten(0, 1),
        o1,
        req_to_page,
        b_req_idx,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        page_size,
    )
    assert torch.allclose(o, o1)


if __name__ == "__main__":
    test_decode_attention(B=3, H_Q=32, H_KV=8, D=128)
