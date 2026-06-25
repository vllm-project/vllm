# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.v1.attention.backends.oscar_attn import (
    OscarAttentionImpl,
    OscarMetadata,
    _build_hadamard,
)
from vllm.v1.attention.ops.triton_oscar_decode_attention import (
    triton_oscar_decode_attention,
)
from vllm.v1.attention.ops.triton_oscar_dequant import dequant_oscar_kv_cache
from vllm.v1.attention.ops.triton_oscar_store import (
    quantized_set_kv_int2_pretransformed_clip_triton,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_oscar_store_dequant_roundtrip(default_vllm_config):
    # Test that packing and then unpacking recovers the original values
    # with expected accuracy
    num_tokens = 64
    num_heads = 4
    head_dim = 128
    device = "cuda"
    dtype = torch.bfloat16

    # Generate random data
    torch.manual_seed(42)
    k = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=dtype)

    # Use OscarConfig for correct slot size (matches runtime layout)

    oscar_cfg = OscarConfig(head_dim=head_dim)
    slot_size = oscar_cfg.slot_size_aligned
    num_blocks = 2
    block_size = 64
    kv_cache = torch.zeros(
        num_blocks, block_size, num_heads, slot_size, dtype=torch.uint8, device=device
    )

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    nb, bs, Hk, S = kv_cache.shape
    cache_flat = kv_cache.view(nb * bs, Hk, S)

    # Sub-region offsets must match oscar_attn.py _store_kv and dequant kernel
    k_q_bytes = head_dim // 4
    sz_bytes = 4
    k_aligned_size = (k_q_bytes + sz_bytes + 15) // 16 * 16

    k_start = 0
    ksz_start = k_q_bytes
    v_start = k_aligned_size
    vsz_start = v_start + k_q_bytes

    k_buf = cache_flat[:, :, k_start : k_start + k_q_bytes]
    ksz_f16 = (
        cache_flat[:, :, ksz_start : ksz_start + sz_bytes]
        .view(torch.float16)
        .view(nb * bs, Hk, 2)
    )
    v_buf = cache_flat[:, :, v_start : v_start + k_q_bytes]
    vsz_f16 = (
        cache_flat[:, :, vsz_start : vsz_start + sz_bytes]
        .view(torch.float16)
        .view(nb * bs, Hk, 2)
    )

    Pi = _build_hadamard(head_dim, device)
    Pi_half = Pi.to(torch.bfloat16)

    k_rot = (k.float() @ Pi).to(dtype)

    # Pack
    quantized_set_kv_int2_pretransformed_clip_triton(
        cache_k=k_rot,
        cache_v=v,
        loc=slot_mapping,
        k_cache_buffer=k_buf,
        v_cache_buffer=v_buf,
        k_scales_zeros_buffer=ksz_f16,
        v_scales_zeros_buffer=vsz_f16,
        clip_ratio_k=0.96,
        clip_ratio_v=0.92,
    )

    # Unpack
    k_hat, v_hat = dequant_oscar_kv_cache(
        kv_cache=kv_cache, slot_mapping=slot_mapping, Pi_half=Pi_half, head_dim=head_dim
    )

    k_mse = torch.nn.functional.mse_loss(k, k_hat)
    v_mse = torch.nn.functional.mse_loss(v, v_hat)

    # Expected MSE for INT2 quantization of N(0,1) is roughly < 0.2
    assert k_mse < 0.2, f"K MSE too high: {k_mse.item()}"
    assert v_mse < 0.2, f"V MSE too high: {v_mse.item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_oscar_fused_decode(default_vllm_config):
    # Test the fused decode kernel matches a standard BF16 attention reference.
    # In a decode batch: 1 query token per request, multiple KV tokens in cache.
    num_reqs = 2
    num_heads = 4
    head_dim = 128
    device = "cuda"
    dtype = torch.bfloat16
    # Sequence lengths for each request (number of KV tokens already in cache)
    seq_len_0, seq_len_1 = 32, 48
    max_seq_len = max(seq_len_0, seq_len_1)
    total_kv_tokens = seq_len_0 + seq_len_1

    torch.manual_seed(42)
    # q has 1 token per request in decode
    q = torch.randn(num_reqs, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_kv_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_kv_tokens, num_heads, head_dim, device=device, dtype=dtype)

    block_size = 64

    slot_size = OscarConfig(head_dim=head_dim).slot_size_aligned
    num_blocks = 4
    kv_cache = torch.zeros(
        num_blocks, block_size, num_heads, slot_size, dtype=torch.uint8, device=device
    )
    # Slot mapping: req 0 uses slots 0..31, req 1 uses slots 64..111
    slot_mapping_r0 = torch.arange(seq_len_0, dtype=torch.int64, device=device)
    slot_mapping_r1 = torch.arange(64, 64 + seq_len_1, dtype=torch.int64, device=device)
    slot_mapping = torch.cat([slot_mapping_r0, slot_mapping_r1])
    # block_table: req 0 = block 0 (slots 0..63), req 1 = block 1 (slots 64..127)
    block_table = torch.tensor(
        [
            [0, -1, -1, -1],
            [1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )

    seq_lens = torch.tensor([seq_len_0, seq_len_1], dtype=torch.int32, device=device)
    # For decode: 1 query token per request
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    attn_metadata = OscarMetadata(
        seq_lens=seq_lens,
        slot_mapping=slot_mapping,
        block_table=block_table,
        query_start_loc=query_start_loc,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        max_seq_len=max_seq_len,
        is_prefill=False,
        num_decodes=num_reqs,
        num_decode_tokens=num_reqs,
    )

    class DummyLayer:
        _oscar_Pi: Any
        _oscar_PiT: Any
        _oscar_Pi_half: Any
        _oscar_centroids: Any
        _oscar_midpoints: Any
        _oscar_cached: bool

    H = _build_hadamard(head_dim, device)

    layer = DummyLayer()
    layer._oscar_Pi = H
    layer._oscar_PiT = H
    layer._oscar_Pi_half = H.to(torch.bfloat16)
    layer._oscar_centroids = torch.tensor(
        [-1.5, -0.5, 0.5, 1.5], dtype=torch.float32, device=device
    )
    layer._oscar_midpoints = torch.tensor(
        [-1.0, 0.0, 1.0], dtype=torch.float32, device=device
    )
    layer._oscar_cached = True

    impl = OscarAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / math.sqrt(head_dim),
        num_kv_heads=num_heads,
        kv_cache_dtype="oscar_int2",
    )

    # Store K, V into OSCAR INT2 cache
    impl.do_kv_cache_update(layer, k, v, kv_cache, slot_mapping)

    # Run fused decode
    output = impl.forward(
        layer=layer,
        query=q,
        key=k[:num_reqs],  # Dummy keys (ignored in decode path)
        value=v[:num_reqs],
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )
    # output: (num_reqs, num_heads * head_dim)

    # Build reference: dequantize cache, then run SDPA per request
    Pi_half = layer._oscar_Pi_half
    ref_outputs = []
    for i, req_slots in enumerate([slot_mapping_r0, slot_mapping_r1]):
        k_hat, v_hat = dequant_oscar_kv_cache(
            kv_cache=kv_cache,
            slot_mapping=req_slots,
            Pi_half=Pi_half,
            head_dim=head_dim,
        )
        q_i = q[i : i + 1].transpose(0, 1).unsqueeze(0)  # [1, H, 1, D]
        k_i = k_hat.to(dtype).transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
        v_i = v_hat.to(dtype).transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
        out_i = torch.nn.functional.scaled_dot_product_attention(
            q_i, k_i, v_i, scale=impl.scale
        )  # [1, H, 1, D]
        ref_outputs.append(out_i.squeeze(0).squeeze(1))  # [H, D]

    ref_output = torch.stack(ref_outputs, dim=0)  # (N, H, D)
    ref_flat = ref_output.to(output.dtype).reshape(num_reqs, -1)

    assert torch.allclose(output[:num_reqs], ref_flat, atol=0.1, rtol=0.1), (
        f"Fused decode does not match reference. "
        f"Max diff: {(output[:num_reqs] - ref_flat).abs().max().item():.4f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_oscar_fused_decode_gqa(default_vllm_config):
    """Test fused decode with GQA (multiple query heads per KV head)."""
    num_reqs = 2
    num_q_heads = 8
    num_kv_heads = 2  # GQA ratio = 4
    head_dim = 128
    device = "cuda"
    dtype = torch.bfloat16
    seq_len_0, seq_len_1 = 64, 96
    total_kv_tokens = seq_len_0 + seq_len_1

    torch.manual_seed(123)
    q = torch.randn(num_reqs, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_kv_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_kv_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)

    block_size = 64

    slot_size = OscarConfig(head_dim=head_dim).slot_size_aligned
    num_blocks = 4
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        slot_size,
        dtype=torch.uint8,
        device=device,
    )
    slot_mapping_r0 = torch.arange(seq_len_0, dtype=torch.int64, device=device)
    slot_mapping_r1 = torch.arange(
        block_size, block_size + seq_len_1, dtype=torch.int64, device=device
    )
    slot_mapping = torch.cat([slot_mapping_r0, slot_mapping_r1])
    block_table = torch.tensor(
        [[0, -1, -1, -1], [1, 2, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.tensor([seq_len_0, seq_len_1], dtype=torch.int32, device=device)

    class DummyLayer:
        _oscar_Pi: Any
        _oscar_PiT: Any
        _oscar_Pi_half: Any
        _oscar_centroids: Any
        _oscar_midpoints: Any
        _oscar_cached: bool

    H = _build_hadamard(head_dim, device)

    layer = DummyLayer()
    layer._oscar_Pi = H
    layer._oscar_PiT = H
    layer._oscar_Pi_half = H.to(torch.bfloat16)
    layer._oscar_centroids = torch.tensor(
        [-1.5, -0.5, 0.5, 1.5], dtype=torch.float32, device=device
    )
    layer._oscar_midpoints = torch.tensor(
        [-1.0, 0.0, 1.0], dtype=torch.float32, device=device
    )
    layer._oscar_cached = True

    impl = OscarAttentionImpl(
        num_heads=num_q_heads,
        head_size=head_dim,
        scale=1.0 / math.sqrt(head_dim),
        num_kv_heads=num_kv_heads,
        kv_cache_dtype="oscar_int2",
    )
    impl.do_kv_cache_update(layer, k, v, kv_cache, slot_mapping)

    # Run fused decode via direct kernel call

    output = triton_oscar_decode_attention(
        query=q,
        kv_cache=kv_cache,
        block_table=block_table[:num_reqs],
        seq_lens=seq_lens,
        Pi=layer._oscar_Pi,
        scale=impl.scale,
    )

    # Reference: dequant + SDPA with GQA expansion
    Pi_half = layer._oscar_Pi_half
    ref_outputs = []
    for i, req_slots in enumerate([slot_mapping_r0, slot_mapping_r1]):
        k_hat, v_hat = dequant_oscar_kv_cache(
            kv_cache=kv_cache,
            slot_mapping=req_slots,
            Pi_half=Pi_half,
            head_dim=head_dim,
        )
        q_i = q[i : i + 1].transpose(0, 1).unsqueeze(0)  # [1, Hq, 1, D]
        k_i = k_hat.to(dtype).transpose(0, 1).unsqueeze(0)  # [1, Hk, S, D]
        v_i = v_hat.to(dtype).transpose(0, 1).unsqueeze(0)  # [1, Hk, S, D]
        out_i = torch.nn.functional.scaled_dot_product_attention(
            q_i, k_i, v_i, scale=impl.scale, enable_gqa=True
        )
        ref_outputs.append(out_i.squeeze(0).squeeze(1))  # [Hq, D]

    ref_output = torch.stack(ref_outputs, dim=0)  # (N, Hq, D)

    max_diff = (output - ref_output.to(output.dtype)).abs().max().item()
    assert max_diff < 0.15, (
        f"GQA fused decode does not match reference. Max diff: {max_diff:.4f}"
    )
