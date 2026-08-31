# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the horizontally-fused MiniMax-M3 attention pre-processing
kernel:

  fused_minimax_m3_qknorm_rope_kv_insert
    - q / k / index_q / index_k: Gemma RMSNorm + partial NeoX RoPE (in place)
    - sparse (insert) mode: scatter k/v into the paged bf16 KV cache and the
      index key into the index cache by its own slot mapping.

Reference: PyTorch Gemma RMSNorm with the same dtype materialization boundary
as the unfused path, followed by vLLM CUDA rotary_embedding-style NeoX RoPE.
"""

import pytest
import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform

HEAD_DIM = 128
ROTARY_DIM = 64


def _op_available() -> bool:
    return hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _op_available(),
    reason="CUDA not available or fused MiniMax-M3 op not built in",
)


def make_cos_sin_cache(max_pos, rotary_dim, base, dtype, device):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_pos, rotary_dim/2]
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rotary_dim]
    return cache.to(dtype)


def gemma_rmsnorm(x, weight, eps):
    """x: [..., 128]; weight: [128]. Returns original dtype."""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def apply_rope_neox_partial(x, positions, cos_sin_cache, rotary_dim):
    """NeoX-style RoPE on the leading rotary_dim dims; rest pass through.

    x: [num_tokens, num_heads, head_dim]
    cos_sin_cache: [max_pos, rotary_dim] (cos||sin), read as float (matches the
    kernel, which loads the bf16 cache and converts to fp32).
    """
    half = rotary_dim // 2
    cs = cos_sin_cache[positions].float()  # [num_tokens, rotary_dim]
    cos = cs[..., :half].unsqueeze(1)  # [nt, 1, half]
    sin = cs[..., half:].unsqueeze(1)

    rot = x[..., :rotary_dim].float()
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    out = x.clone()
    out[..., :half] = o1
    out[..., half:rotary_dim] = o2
    return out.to(x.dtype)


def norm_rope_ref(x, weight, positions, cos_sin_cache, eps):
    """[nt, nheads, 128] -> Gemma norm + neox partial rope."""
    normed = gemma_rmsnorm(x, weight, eps)
    roped = apply_rope_neox_partial(normed, positions, cos_sin_cache, ROTARY_DIM)
    return roped


def assert_fp8_cache_close(kv_cache, expected_kv_cache):
    """Compare two e4m3 caches allowing 1 ulp.

    On CUDA the fused kernel quantizes K from its fp32 intermediate, while the
    reshape_and_cache_flash reference quantizes the bf16-materialized value, so
    rounding-boundary values may differ by one e4m3 code.
    """
    byte_diff = (kv_cache.int() - expected_kv_cache.int()).abs()
    got = kv_cache.view(torch.float8_e4m3fn).float()
    exp = expected_kv_cache.view(torch.float8_e4m3fn).float()
    ok = (byte_diff <= 1) | ((got == 0) & (exp == 0))
    assert bool(ok.all()), (
        f"fp8 cache differs by more than 1 ulp in {int((~ok).sum())} elements"
    )


def assert_storage_equal(actual, expected):
    """Compare tensor storage bit-for-bit, including signed zero."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


# Model dispatch eligibility


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AITER dispatch is ROCm-only"
)
@pytest.mark.parametrize(
    "layout,expected",
    [
        pytest.param("bf16", True, id="bf16-eligible"),
        pytest.param("fp8", False, id="fp8-fallback"),
        pytest.param("strided_cache", False, id="strided-cache-fallback"),
        pytest.param("strided_mapping", False, id="strided-mapping-fallback"),
        pytest.param("misaligned_cache", False, id="misaligned-cache-fallback"),
    ],
)
def test_aiter_index_cache_insert_fusion_eligibility(layout, expected):
    from vllm.models.minimax_m3.amd.model import (
        _can_fuse_aiter_index_cache_insert,
    )

    num_tokens = 4
    cache_shape = (2, 128, HEAD_DIM)
    qkv = torch.empty(num_tokens, 1, dtype=torch.bfloat16)
    index_cache = torch.empty(cache_shape, dtype=torch.bfloat16)
    index_slot_mapping = torch.arange(num_tokens, dtype=torch.int64)

    if layout == "fp8":
        index_cache = torch.empty(cache_shape, dtype=torch.float8_e4m3fn)
    elif layout == "strided_cache":
        index_cache = torch.empty(2, 256, HEAD_DIM, dtype=torch.bfloat16)[:, ::2]
    elif layout == "strided_mapping":
        index_slot_mapping = torch.arange(num_tokens * 2, dtype=torch.int64)[::2]
    elif layout == "misaligned_cache":
        storage = torch.empty(index_cache.numel() + 1, dtype=torch.bfloat16)
        index_cache = storage[1:].view(cache_shape)
        assert index_cache.data_ptr() % 8 != 0

    assert (
        _can_fuse_aiter_index_cache_insert(qkv, index_cache, index_slot_mapping)
        is expected
    )


# Test 1: dense mode (norm+rope only, no index, no insert)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (16, 4), (64, 4)])
def test_dense_norm_rope(num_tokens, num_heads, num_kv_heads):
    torch.manual_seed(0)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    qkv = torch.randn(num_tokens, qsz + 2 * kvsz, dtype=dtype, device=device)
    qkv_orig = qkv.clone()

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_w,
        k_w,
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        eps,
        kv_cache_dtype="auto",
    )
    q_out, k_out, v_out = qkv.split([qsz, kvsz, kvsz], dim=-1)

    q_in, k_in, v_in = qkv_orig.split([qsz, kvsz, kvsz], dim=-1)
    q_ref = norm_rope_ref(
        q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps
    ).view(num_tokens, qsz)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        k_w,
        positions,
        cos_sin,
        eps,
    ).view(num_tokens, kvsz)

    # The fused kernel keeps an fp32 intermediate across norm->rope, while the
    # reference materializes bf16 after the norm (the unfused boundary), so
    # rounding-boundary elements can differ by ~1 bf16 ulp.
    torch.testing.assert_close(q_out, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=2e-2, atol=2e-2)
    # V is untouched.
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)


# ── Test 2: sparse mode (full: index branch + cache inserts) ─────────────────


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_sparse_full(num_tokens, block_size, kv_cache_dtype):
    torch.manual_seed(1)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096
    num_heads, num_kv_heads, num_idx_heads = 16, 4, 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz, iksz = num_idx_heads * HEAD_DIM, HEAD_DIM
    # Single fused tensor packing [q | k | v | index_q | index_k].
    qkv = torch.randn(
        num_tokens, qsz + 2 * kvsz + iqsz + iksz, dtype=dtype, device=device
    )
    qkv_orig = qkv.clone()
    splits = [qsz, kvsz, kvsz, iqsz, iksz]

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache_storage_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        block_size,
        2 * HEAD_DIM,
        dtype=kv_cache_storage_dtype,
        device=device,
    )
    index_cache = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=dtype, device=device
    )
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=device
    )[:num_tokens]
    index_slot_mapping = torch.roll(slot_mapping, shifts=1)

    # Contiguous gather targets: the kernel writes the normed/roped q and
    # index_q here (de-interleaved from the packed qkv); k/v/index_k stay in
    # place inside qkv and are scatter-inserted into the caches.
    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)
    q_fp8 = torch.empty(
        num_tokens,
        qsz,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    index_q = torch.empty(num_tokens, iqsz, dtype=dtype, device=device)

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_w,
        k_w,
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        eps,
        iq_w,
        ik_w,
        num_idx_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q,
        kv_cache_dtype,
        q_fp8_out=q_fp8,
        q_fp8_scale=0.5,
    )

    # ── norm+rope parity. q/index_q land in their gather buffers; k/index_k are
    # rewritten in place inside qkv. ──
    _, k_out, v_out, _, index_k = qkv.split(splits, dim=-1)
    q_in, k_in, v_in, iq_orig, ik_orig = qkv_orig.split(splits, dim=-1)
    q_ref = norm_rope_ref(
        q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps
    ).view(num_tokens, qsz)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        k_w,
        positions,
        cos_sin,
        eps,
    ).view(num_tokens, kvsz)
    iq_ref = norm_rope_ref(
        iq_orig.view(num_tokens, num_idx_heads, HEAD_DIM),
        iq_w,
        positions,
        cos_sin,
        eps,
    ).view(num_tokens, num_idx_heads * HEAD_DIM)
    ik_ref = norm_rope_ref(
        ik_orig.view(num_tokens, 1, HEAD_DIM), ik_w, positions, cos_sin, eps
    ).view(num_tokens, HEAD_DIM)

    # The fused kernel keeps an fp32 intermediate across norm->rope, while the
    # reference materializes bf16 after the norm (the unfused boundary), so
    # rounding-boundary elements can differ by ~1 bf16 ulp.
    torch.testing.assert_close(q_out, q_ref, rtol=2e-2, atol=2e-2)
    expected_q_fp8 = torch.empty_like(q_fp8)
    ops.scaled_fp8_quant(
        q_out,
        scale=torch.tensor(0.5, dtype=torch.float32, device=device),
        output=expected_q_fp8,
    )
    torch.testing.assert_close(q_fp8, expected_q_fp8, rtol=0, atol=0)
    torch.testing.assert_close(k_out, k_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(index_q, iq_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_k, ik_ref, rtol=1e-2, atol=1e-2)

    # ── Cache inserts. ──
    # Main cache layout is [num_blocks, num_kv_heads, block_size, 2*head_dim];
    # index cache is [num_blocks, block_size, head_dim].
    k_ref_h = k_ref.view(num_tokens, num_kv_heads, HEAD_DIM)
    v_ref_h = v_in.view(num_tokens, num_kv_heads, HEAD_DIM)  # v is raw (no norm/rope)
    if kv_cache_dtype == "fp8":
        expected_kv_cache = torch.zeros_like(kv_cache)
        expected_k_cache, expected_v_cache = expected_kv_cache.transpose(1, 2).split(
            HEAD_DIM, dim=-1
        )
        scale = torch.ones((), device=device)
        ops.reshape_and_cache_flash(
            k_out.view(num_tokens, num_kv_heads, HEAD_DIM),
            v_out.view(num_tokens, num_kv_heads, HEAD_DIM),
            expected_k_cache,
            expected_v_cache,
            slot_mapping,
            kv_cache_dtype,
            scale,
            scale,
        )
        assert_fp8_cache_close(kv_cache, expected_kv_cache)
    else:
        for t in range(num_tokens):
            s = slot_mapping[t].item()
            b, pos = s // block_size, s % block_size
            torch.testing.assert_close(
                kv_cache[b, :, pos, :HEAD_DIM], k_ref_h[t], rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(
                kv_cache[b, :, pos, HEAD_DIM:], v_ref_h[t], rtol=0, atol=0
            )

    expected_index_cache = torch.zeros_like(index_cache).view(-1, HEAD_DIM)
    expected_index_cache[index_slot_mapping] = index_k
    torch.testing.assert_close(
        index_cache.view(-1, HEAD_DIM), expected_index_cache, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "num_tokens,slot_case,use_graph",
    [
        pytest.param(0, "active", False, id="zero"),
        pytest.param(1, "active", False, id="one"),
        pytest.param(4, "boundary", True, id="four-graph-boundary"),
        pytest.param(12, "interspersed", False, id="twelve-padded"),
        pytest.param(16, "all_padding", False, id="sixteen-all-padding"),
        pytest.param(24, "boundary", True, id="twenty-four-graph-boundary"),
        pytest.param(64, "active", False, id="sixty-four"),
    ],
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_sparse_index_only_insert_matches_triton(
    num_tokens, slot_case, use_graph, kv_cache_dtype
):
    from vllm.models.minimax_m3.amd.ops.sparse_pa import (
        minimax_m3_insert_index_cache,
    )

    torch.manual_seed(3)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 131_072
    num_heads, num_kv_heads, num_idx_heads = 16, 1, 1
    block_size, num_blocks = 128, 3

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device) * 997
    if num_tokens:
        positions[-1] = max_pos - 1

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz = num_idx_heads * HEAD_DIM
    row_size = qsz + 2 * kvsz + iqsz + HEAD_DIM
    qkv_source = torch.randn(num_tokens, row_size, dtype=dtype, device=device)
    if qkv_source.numel():
        qkv_source.view(-1)[0] = 0.0
        qkv_source.view(-1)[1] = -0.0
        index_k_source = qkv_source[:, -HEAD_DIM:]
        index_k_source[0].zero_()
        index_k_source[0, 1] = -0.0

    live_prefix = [0, 1, 126, 127, 128, 255, 256]
    live_slots = live_prefix + [
        slot for slot in range(num_blocks * block_size) if slot not in live_prefix
    ]
    if slot_case == "all_padding":
        slots = [-1] * num_tokens
    elif slot_case == "interspersed":
        slots = [
            -1 if token % 3 == 1 else live_slots[token] for token in range(num_tokens)
        ]
    elif slot_case == "boundary":
        boundary_slots = [127, 128, 255, -1, 256]
        remaining_slots = [
            slot for slot in live_slots if slot not in {127, 128, 255, 256}
        ]
        slots = (boundary_slots + remaining_slots)[:num_tokens]
    else:
        slots = live_slots[:num_tokens]
    index_slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)

    poison = torch.full(
        (num_blocks, block_size, HEAD_DIM),
        -31.5,
        dtype=dtype,
        device=device,
    )

    def make_provider():
        return (
            qkv_source.clone(),
            torch.empty(num_tokens, qsz, dtype=dtype, device=device),
            torch.empty(num_tokens, iqsz, dtype=dtype, device=device),
            poison.clone(),
        )

    def run_fused(qkv, q_out, index_q_out, index_cache=None):
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            q_w,
            k_w,
            cos_sin,
            positions,
            num_heads,
            num_kv_heads,
            ROTARY_DIM,
            eps,
            iq_w,
            ik_w,
            num_idx_heads,
            index_slot_mapping=(
                index_slot_mapping if index_cache is not None else None
            ),
            index_cache=index_cache,
            q_out=q_out,
            index_q_out=index_q_out,
            kv_cache_dtype=kv_cache_dtype,
        )

    def run_baseline(qkv, q_out, index_q_out, index_cache):
        run_fused(qkv, q_out, index_q_out)
        minimax_m3_insert_index_cache(
            qkv[:, -HEAD_DIM:], index_cache, index_slot_mapping
        )

    baseline = make_provider()
    candidate = make_provider()
    if use_graph:
        warm_baseline = make_provider()
        warm_candidate = make_provider()
        run_baseline(*warm_baseline)
        run_fused(
            warm_candidate[0],
            warm_candidate[1],
            warm_candidate[2],
            warm_candidate[3],
        )
        torch.accelerator.synchronize()

        baseline_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(baseline_graph):
            run_baseline(*baseline)
        candidate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate_graph):
            run_fused(candidate[0], candidate[1], candidate[2], candidate[3])

        baseline[0].copy_(qkv_source)
        candidate[0].copy_(qkv_source)
        baseline[3].copy_(poison)
        candidate[3].copy_(poison)
        index_slot_mapping.copy_(index_slot_mapping.roll(1))
        baseline_graph.replay()
        candidate_graph.replay()
        torch.accelerator.synchronize()
    else:
        run_baseline(*baseline)
        run_fused(candidate[0], candidate[1], candidate[2], candidate[3])

    for baseline_tensor, candidate_tensor in zip(baseline, candidate):
        assert_storage_equal(candidate_tensor, baseline_tensor)

    live = index_slot_mapping[index_slot_mapping >= 0]
    untouched = torch.ones(num_blocks * block_size, dtype=torch.bool, device=device)
    untouched[live] = False
    assert_storage_equal(
        candidate[3].view(-1, HEAD_DIM)[untouched],
        poison.view(-1, HEAD_DIM)[untouched],
    )


@pytest.mark.parametrize("num_tokens", [1, 64, 513])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_sparse_skip_index_branch(num_tokens, block_size, kv_cache_dtype):
    torch.manual_seed(2)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096
    num_heads, num_kv_heads, num_idx_heads = 16, 4, 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz, iksz = num_idx_heads * HEAD_DIM, HEAD_DIM
    qkv = torch.randn(
        num_tokens, qsz + 2 * kvsz + iqsz + iksz, dtype=dtype, device=device
    )
    qkv_orig = qkv.clone()
    splits = [qsz, kvsz, kvsz, iqsz, iksz]

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache_storage_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        block_size,
        2 * HEAD_DIM,
        dtype=kv_cache_storage_dtype,
        device=device,
    )
    index_cache = torch.randn(
        num_blocks, block_size, HEAD_DIM, dtype=dtype, device=device
    )
    index_cache_orig = index_cache.clone()
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=device
    )[:num_tokens]
    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_w,
        k_w,
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        eps,
        num_index_heads=num_idx_heads,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        index_cache=index_cache,
        block_size=block_size,
        q_out=q_out,
        kv_cache_dtype=kv_cache_dtype,
        skip_index_branch=True,
    )

    _, k_out, v_out, index_q_out, index_k_out = qkv.split(splits, dim=-1)
    q_in, k_in, v_in, index_q_in, index_k_in = qkv_orig.split(splits, dim=-1)
    q_ref = norm_rope_ref(
        q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps
    ).view(num_tokens, qsz)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        k_w,
        positions,
        cos_sin,
        eps,
    ).view(num_tokens, kvsz)

    # The fused kernel keeps an fp32 intermediate across norm->rope, while the
    # reference materializes bf16 after the norm (the unfused boundary), so
    # rounding-boundary elements can differ by ~1 bf16 ulp.
    torch.testing.assert_close(q_out, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)
    torch.testing.assert_close(index_q_out, index_q_in, rtol=0, atol=0)
    torch.testing.assert_close(index_k_out, index_k_in, rtol=0, atol=0)
    torch.testing.assert_close(index_cache, index_cache_orig, rtol=0, atol=0)

    if kv_cache_dtype == "fp8":
        expected_kv_cache = torch.zeros_like(kv_cache)
        expected_k_cache, expected_v_cache = expected_kv_cache.transpose(1, 2).split(
            HEAD_DIM, dim=-1
        )
        scale = torch.ones((), device=device)
        ops.reshape_and_cache_flash(
            k_out.view(num_tokens, num_kv_heads, HEAD_DIM),
            v_out.view(num_tokens, num_kv_heads, HEAD_DIM),
            expected_k_cache,
            expected_v_cache,
            slot_mapping,
            kv_cache_dtype,
            scale,
            scale,
        )
        assert_fp8_cache_close(kv_cache, expected_kv_cache)
    else:
        k_ref_h = k_ref.view(num_tokens, num_kv_heads, HEAD_DIM)
        v_ref_h = v_in.view(num_tokens, num_kv_heads, HEAD_DIM)
        for t in range(num_tokens):
            s = slot_mapping[t].item()
            b, pos = s // block_size, s % block_size
            torch.testing.assert_close(
                kv_cache[b, :, pos, :HEAD_DIM], k_ref_h[t], rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(
                kv_cache[b, :, pos, HEAD_DIM:], v_ref_h[t], rtol=0, atol=0
            )


# ── Test 4: fp8 (e4m3) index outputs ─────────────────────────────────────────
# The fp8 score path stores index_q and the index-K cache as e4m3 while q/k/v +
# q_out stay bf16. Asserts: (1) q/k/v/q_out are bit-identical to the bf16 run
# (the index dtype must not perturb the main branch), and (2) the e4m3 index
# outputs dequantize close to the bf16 reference.


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="e4m3 conversion requires CUDA SM89+.",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("block_size", [16, 64])
def test_sparse_full_fp8_index(num_tokens, block_size):
    torch.manual_seed(1)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096
    num_heads, num_kv_heads, num_idx_heads = 16, 4, 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz, iksz = num_idx_heads * HEAD_DIM, HEAD_DIM
    qkv0 = torch.randn(
        num_tokens, qsz + 2 * kvsz + iqsz + iksz, dtype=dtype, device=device
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=device
    )[:num_tokens]
    index_slot_mapping = torch.roll(slot_mapping, shifts=1)

    def run(index_dtype):
        qkv = qkv0.clone()
        kv_cache = torch.zeros(
            num_blocks,
            num_kv_heads,
            block_size,
            2 * HEAD_DIM,
            dtype=dtype,
            device=device,
        )
        index_cache = torch.zeros(
            num_blocks, block_size, HEAD_DIM, dtype=index_dtype, device=device
        )
        q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)
        index_q = torch.empty(num_tokens, iqsz, dtype=index_dtype, device=device)
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            q_w,
            k_w,
            cos_sin,
            positions,
            num_heads,
            num_kv_heads,
            ROTARY_DIM,
            eps,
            iq_w,
            ik_w,
            num_idx_heads,
            slot_mapping,
            index_slot_mapping,
            kv_cache,
            index_cache,
            block_size,
            q_out,
            index_q,
        )
        return qkv, kv_cache, index_cache, q_out, index_q

    qkv_bf, kvc_bf, idxc_bf, qo_bf, iq_bf = run(torch.bfloat16)
    qkv_fp, kvc_fp, idxc_fp, qo_fp, iq_fp = run(torch.float8_e4m3fn)

    assert iq_fp.dtype == torch.float8_e4m3fn
    assert idxc_fp.dtype == torch.float8_e4m3fn

    # (1) The main branch (q/k/v in qkv, q_out, kv cache) must be bit-identical:
    # the index output dtype must not perturb anything else.
    torch.testing.assert_close(qo_fp, qo_bf, rtol=0, atol=0)
    torch.testing.assert_close(qkv_fp, qkv_bf, rtol=0, atol=0)
    torch.testing.assert_close(kvc_fp, kvc_bf, rtol=0, atol=0)

    # (2) Dequantized e4m3 index outputs match the bf16 reference within fp8 ulp.
    torch.testing.assert_close(iq_fp.float(), iq_bf.float(), rtol=0.13, atol=0.05)
    torch.testing.assert_close(idxc_fp.float(), idxc_bf.float(), rtol=0.13, atol=0.05)
