# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the draft context-KV dummy-Q removal.

``DSparkDeepseekV4Model.precompute_and_store_context_kv`` ->
``_insert_context_kv`` (vllm/models/deepseek_v41/amd/dspark.py) used to
allocate a zero dummy query ``[N, n_local_heads, 512]`` and run the fused
Q+KV op with ``q_head_padded = attn.padded_heads``, discarding the
quantized-Q output. The op supports ``q_head_padded = 0`` -- a KV-only
launch that writes no Q at all -- so the optimized path passes an empty
``[N, 0, 512]`` q and 0 (kill switch VLLM_DISABLE_DRAFT_CTXKV_DUMMYQ).

These tests prove, at op level, the exact A/B: for identical
kv/slots/positions/cos_sin inputs, the dummy-Q call (stock) and the
KV-only call must produce BIT-IDENTICAL cache bytes (the Q pass writes
only its separate q_out tensor, never the cache), the wrapper's empty-q
contract must hold, and the KV-only call must be CUDA-graph capturable
and replay-stable (the DSpark draft step captures it inside the FULL
draft CUDA graph).
"""

import pytest
import torch

from tests.kernels.test_fused_deepseek_v4_qnorm_rope_kv_insert import (
    HEAD_BYTES,
    HEAD_DIM,
    _op_available,
    make_cos_sin_cache,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _op_available(),
    reason="CUDA not available or fused DeepseekV4 op not built in",
)


def _make_inputs(num_tokens, block_size, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(4096, 64, torch.float32, device)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    return kv, positions, cos_sin_cache, num_blocks, slot_mapping


def _stock_call(
    kv,
    positions,
    cos_sin_cache,
    num_blocks,
    slot_mapping,
    block_size,
    n_heads,
    padded_heads,
    kv_mxfp8,
):
    """The stock dspark call: zero dummy q, padded_heads, apply_q_norm."""
    cache = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device="cuda"
    )
    dummy_q = torch.zeros(
        kv.shape[0], n_heads, HEAD_DIM, dtype=kv.dtype, device=kv.device
    )
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        dummy_q,
        kv,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        padded_heads,
        1e-6,
        block_size,
        True,
        kv_mxfp8,
    )
    return cache, dummy_q


def _lever_call(
    kv,
    positions,
    cos_sin_cache,
    num_blocks,
    slot_mapping,
    block_size,
    kv_mxfp8,
):
    """The optimized call: empty q, q_head_padded=0 (KV-only launch)."""
    cache = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device="cuda"
    )
    empty_q = kv.new_empty((kv.shape[0], 0, HEAD_DIM), dtype=kv.dtype)
    q_out = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        empty_q,
        kv,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        0,
        1e-6,
        block_size,
        True,
        kv_mxfp8,
    )
    return cache, q_out


@pytest.mark.parametrize("num_tokens", [1, 6, 17])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("kv_mxfp8", [False, True])
@pytest.mark.parametrize("n_heads,padded_heads", [(16, 16), (16, 64), (1, 8)])
def test_kv_only_matches_dummy_q_cache_bytes(
    num_tokens, block_size, kv_mxfp8, n_heads, padded_heads
):
    """The exact A/B: identical inputs through the stock dummy-Q call
    and the KV-only call must leave BIT-IDENTICAL cache bytes. The Q pass
    writes only its own q_out tensor, so removing it cannot touch the KV
    path."""
    kv, positions, cos_sin, num_blocks, slots = _make_inputs(
        num_tokens, block_size, seed=num_tokens + block_size
    )
    cache_stock, dummy_q = _stock_call(
        kv,
        positions,
        cos_sin,
        num_blocks,
        slots,
        block_size,
        n_heads,
        padded_heads,
        kv_mxfp8,
    )
    cache_lever, q_out = _lever_call(
        kv,
        positions,
        cos_sin,
        num_blocks,
        slots,
        block_size,
        kv_mxfp8,
    )
    torch.accelerator.synchronize()
    assert torch.equal(cache_stock, cache_lever), (
        "KV-only insert changed the cache bytes vs the dummy-Q insert"
    )
    # The stock dummy q is never written through (q_in is const; the op
    # returns a separate q_out) and the KV-only call's q_out is empty.
    assert torch.count_nonzero(dummy_q) == 0
    assert q_out.numel() == 0


def test_kv_only_graph_replay():
    """The DSpark draft step captures _insert_context_kv inside the FULL
    draft CUDA graph: the KV-only call must capture and replay correctly,
    with replays on changed inputs matching an eager run bit-level."""
    num_tokens, block_size = 6, 32
    kv, positions, cos_sin, num_blocks, slots = _make_inputs(
        num_tokens, block_size, seed=7
    )
    # Static buffers the graph will read; replay with different contents.
    kv_s = kv.clone()
    slots_s = slots.clone()
    cache_g = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device="cuda"
    )
    empty_q = kv_s.new_empty((num_tokens, 0, HEAD_DIM), dtype=kv_s.dtype)

    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        empty_q,
        kv_s,
        cache_g,
        slots_s,
        positions,
        cos_sin,
        0,
        1e-6,
        block_size,
        True,
        False,
    )  # warmup outside capture
    torch.accelerator.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            empty_q,
            kv_s,
            cache_g,
            slots_s,
            positions,
            cos_sin,
            0,
            1e-6,
            block_size,
            True,
            False,
        )

    # Replay 1: fresh cache, original inputs.
    cache_g.zero_()
    g.replay()
    torch.accelerator.synchronize()
    cache_ref, _ = _lever_call(
        kv_s, positions, cos_sin, num_blocks, slots_s, block_size, False
    )
    assert torch.equal(cache_g, cache_ref)

    # Replay 2: changed kv + permuted slots must land exactly where the
    # eager run puts them (no stale-pointer or fixed-address effects).
    torch.manual_seed(8)
    kv_s.copy_(torch.randn_like(kv_s))
    slots_s.copy_(slots_s.flip(0))
    cache_g.zero_()
    g.replay()
    torch.accelerator.synchronize()
    cache_ref2, _ = _lever_call(
        kv_s, positions, cos_sin, num_blocks, slots_s, block_size, False
    )
    assert torch.equal(cache_g, cache_ref2)
