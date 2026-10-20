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


# ── Test 1: dense mode (norm+rope only, no index, no insert) ─────────────────


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

    # Without q_out, q is emitted only in fp8: the same bytes, while qkv keeps
    # its raw q and every other output matches the run above.
    outputs = [qkv_orig.clone(), torch.empty_like(q_fp8), torch.empty_like(index_q)]
    caches = [torch.zeros_like(kv_cache), torch.zeros_like(index_cache)]
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        outputs[0],
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
        *caches,
        block_size,
        None,
        outputs[2],
        kv_cache_dtype,
        q_fp8_out=outputs[1],
        q_fp8_scale=0.5,
    )
    for actual, expected in zip(
        outputs + caches, [qkv, q_fp8, index_q, kv_cache, index_cache]
    ):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


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
# (the index dtype must not perturb the main branch), and (2) index_q is the
# e4m3 cast of the bf16 run's index_q, and the index-K cache is within fp8 ulp.


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

    # (2) index_q is e4m3(bf16(q)); the index-K cache is within fp8 ulp.
    assert torch.equal(
        iq_fp.view(torch.uint8), iq_bf.to(torch.float8_e4m3fn).view(torch.uint8)
    )
    torch.testing.assert_close(idxc_fp.float(), idxc_bf.float(), rtol=0.13, atol=0.05)


# ── Test 5: sparse mode with an NVFP4 main KV cache ──────────────────────────
# The NVFP4 inserts quantize x * sf_scale, where sf_scale = 1 / the global
# scale: per 16-value block,
# scale = min(448, max(E4M3(max(amax, 1e-12) / 6), 1/512)) and
# code = sign | RNE E2M1 of min(|x / scale|, 6), with zero stored as +0.

E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _div(a, b):
    # Division by a Python scalar would become a reciprocal multiply.
    return a / torch.as_tensor(b, dtype=a.dtype, device=a.device)


def nvfp4_ref(x, sf_scale):
    """Returns the E2M1 codes [..., 128] and E4M3 scale bytes [..., 8] of x."""
    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = (sf_scale * _div(amax, 6.0)).clamp(max=448.0)
    scale = scale.to(torch.float8_e4m3fn).float().clamp(min=2.0**-9)
    q = _div(blocks * sf_scale, scale)
    # Alternating > / >= rounds exact midpoints to the even code.
    mag = sum(
        (q.abs() >= m) if i % 2 else (q.abs() > m) for i, m in enumerate(E2M1_MIDPOINTS)
    ).to(torch.uint8)
    codes = torch.where(mag > 0, mag | (torch.signbit(q).to(torch.uint8) << 3), 0)
    scale_bytes = scale.squeeze(-1).to(torch.float8_e4m3fn).view(torch.uint8)
    return codes.flatten(-2), scale_bytes


def nvfp4_pages(sides, slot_mapping, num_blocks, block_size):
    """Expected NVFP4 pages [nb, len(sides) * nkv, bs, 72] for the K (then V)
    side given as (x [N, nkv, 128], sf_scale). Slot h * len(sides) + side is
    head h's packed E2M1 data [bs, 64] (low nibble = even element), then its
    E4M3 scales [bs, 8]: linear for K, 4x4 token-quad swizzled for V."""
    nkv, ns, device = sides[0][0].shape[1], HEAD_DIM // 16, slot_mapping.device
    row_bytes, data_bytes = HEAD_DIM // 2 + ns, block_size * HEAD_DIM // 2
    pages = torch.zeros(
        num_blocks,
        nkv,
        len(sides),
        block_size * row_bytes,
        dtype=torch.uint8,
        device=device,
    )
    valid = slot_mapping >= 0
    blk, tok = slot_mapping[valid] // block_size, slot_mapping[valid] % block_size
    g, t = torch.arange(ns, device=device), tok[:, None]
    scale_offsets = (t * ns + g, ((t // 4) * 4 + g // 2) * ns + (g % 2) * 4 + t % 4)
    for side, (x, sf_scale) in enumerate(sides):
        codes, scales = nvfp4_ref(x[valid], sf_scale)
        data = pages[:, :, side, :data_bytes].view(num_blocks, nkv, block_size, -1)
        data[blk, :, tok] = codes[..., 0::2] | (codes[..., 1::2] << 4)
        scale_region = pages[:, :, side, data_bytes:]
        for h in range(nkv):
            scale_region[blk[:, None], h, scale_offsets[side]] = scales[:, h]
    return pages.view(num_blocks, -1, block_size, row_bytes)


def nvfp4_edge_values(num_rows, global_scale):
    """[num_rows, 128] bf16 whose 16-value blocks, divided by global_scale, hit
    the quantizer's corner cases: exact E2M1 midpoints under non-power-of-2 scales
    (blocks 0-1), amax under the 1/512 scale floor (2), all -0 (3), and
    saturation with |x| > 4096 up to bf16 max (4) or with inf (5)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(num_rows, 8, 16, generator=g)
    sign = torch.where(torch.rand(x.shape, generator=g) < 0.5, -1.0, 1.0)
    steps = torch.tensor(E2M1_MIDPOINTS + (0.5, 1.0, 3.0))
    mant = 1 + torch.randint(1, 8, (num_rows, 2, 1), generator=g) / 8
    scale = mant * 2.0 ** torch.randint(-6, 3, (num_rows, 2, 1), generator=g)
    x[:, :2] = steps[torch.randint(len(steps), (num_rows, 2, 16), generator=g)] * scale
    x[:, :2, :1] = 6 * scale  # amax = 6 * scale, so the block scale is `scale`
    x[:, 2] = torch.rand(num_rows, 16, generator=g) * 2.0**-8
    x[:, 4] *= 5000
    x = x * sign * global_scale
    x[:, 3] = -0.0
    x[:, 4, 0] = sign[:, 4, 0] * torch.finfo(torch.bfloat16).max
    x[:, 5, 0] = sign[:, 5, 0] * float("inf")
    return x.flatten(1).to(torch.bfloat16)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="NVFP4 KV caches need SM100",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("kv_scales", [(0.5, 2.0), (1.0, 1.0)])
def test_sparse_full_nvfp4(num_tokens, num_kv_heads, block_size, kv_scales):
    """The fused NVFP4 insert writes the reference quantizer's bytes (packed
    E2M1, linear K / swizzled V E4M3 block scales) for the kernel's own
    post-RoPE k and for a raw v that hits the quantizer's corner cases, and
    skips padded slots. Unit global scales take the kernel's separate fast
    path."""
    kv_cache_dtype = "nvfp4"
    torch.manual_seed(5)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096
    num_heads, num_idx_heads = 4 * num_kv_heads, 1

    weights = [
        torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1 for _ in range(4)
    ]
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )
    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz = num_idx_heads * HEAD_DIM
    qkv = torch.randn(
        num_tokens, qsz + 2 * kvsz + iqsz + HEAD_DIM, dtype=dtype, device=device
    )
    k_scale, v_scale = (
        torch.tensor(s, dtype=torch.float32, device=device) for s in kv_scales
    )
    qkv[:, qsz + kvsz : qsz + 2 * kvsz] = nvfp4_edge_values(
        num_tokens * num_kv_heads, v_scale.item()
    ).view(num_tokens, kvsz)
    qkv_orig = qkv.clone()

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache = torch.zeros(
        num_blocks,
        2 * num_kv_heads,
        block_size,
        HEAD_DIM // 2 + HEAD_DIM // 16,
        dtype=torch.uint8,
        device=device,
    )
    index_cache = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=dtype, device=device
    )
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=device
    )[:num_tokens]
    slot_mapping[::5] = -1  # padded / unscheduled tokens are not written
    index_slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=device
    )[:num_tokens]

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        *weights[:2],
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        eps,
        *weights[2:],
        num_idx_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        torch.empty(num_tokens, qsz, dtype=dtype, device=device),
        torch.empty(num_tokens, iqsz, dtype=dtype, device=device),
        kv_cache_dtype,
        kv_k_scale=k_scale,
        kv_v_scale=v_scale,
    )

    _, k_out, v_out, _, index_k = qkv.split([qsz, kvsz, kvsz, iqsz, HEAD_DIM], dim=-1)
    torch.testing.assert_close(
        v_out, qkv_orig.split([qsz, kvsz, kvsz, iqsz, HEAD_DIM], dim=-1)[2]
    )
    expected_kv_cache = nvfp4_pages(
        [
            (k_out.reshape(num_tokens, num_kv_heads, HEAD_DIM), 1 / k_scale),
            (v_out.reshape(num_tokens, num_kv_heads, HEAD_DIM), 1 / v_scale),
        ],
        slot_mapping,
        num_blocks,
        block_size,
    )
    assert torch.equal(kv_cache, expected_kv_cache), (
        f"{int((kv_cache != expected_kv_cache).sum())} NVFP4 cache bytes differ"
    )

    expected_index_cache = torch.zeros_like(index_cache).view(-1, HEAD_DIM)
    expected_index_cache[index_slot_mapping] = index_k
    torch.testing.assert_close(
        index_cache.view(-1, HEAD_DIM), expected_index_cache, rtol=0, atol=0
    )
