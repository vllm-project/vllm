# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone unit test for the horizontally-fused DeepseekV4-MLA kernel:

  fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    - Q side:  per-head RMSNorm (no weight) + GPT-J RoPE on last 64 dims
    - KV side: GPT-J RoPE on last 64 + UE8M0 FP8 quant + paged cache insert

We compare against:
  - PyTorch reference for RMSNorm + GPT-J RoPE on Q
  - Existing Triton `quantize_and_insert_k_cache` + round-trip via
    `dequantize_and_gather_k_cache` for KV

The kernel is imported via
`torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` and writes Q
into a caller-owned output tensor.
"""

import pytest
import torch

from tests.kernels.utils import bf16_ulp_distance, fp8_ulp_distance
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.models.deepseek_v4.common.ops import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm.platforms import current_platform

# ── Constants matching the kernel ────────────────────────────────────────────
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448
QUANT_BLOCK = 64
# Match the C++ SWA-K encoder: FNUZ on gfx942, OCP elsewhere.
USE_FNUZ = current_platform.is_fp8_fnuz()
_, FP8_MAX = get_fp8_min_max()
# The kernel emits FNUZ-encoded fp8 bytes on gfx942 (rocm_cvt_float_to_fp8_e4m3)
# but stores them into float8_e4m3fn-typed tensors, matching vLLM's ROCm cache
# convention. References must encode under the same scheme and the kernel's
# e4m3fn-typed outputs must be reinterpreted under it before decoding.
FP8_STORE_DTYPE = torch.float8_e4m3fnuz if USE_FNUZ else torch.float8_e4m3fn
HEAD_BYTES = NOPE_DIM + ROPE_DIM * 2 + 8  # 448 + 128 + 8 = 584


# ── PyTorch reference implementations ────────────────────────────────────────


def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
    """Build a cos||sin cache matching DeepseekV4ScalingRotaryEmbedding layout.
    cos_sin_cache[pos, :rope_dim/2] = cos(theta), [rope_dim/2:] = sin(theta).
    """
    base = 10000.0
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rope_dim/2]
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rope_dim]
    return cache.to(dtype)


def apply_rope_gptj_last_k(
    x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """GPT-J-style (interleaved-pair) RoPE on the LAST rope_dim elements.

    x: [..., head_dim] float32
    positions: [num_tokens] int64 (positions[i] corresponds to x[i, ...])
    cos_sin_cache: [max_pos, rope_dim] float (cos|sin layout)

    Returns rotated x (same shape/dtype).
    """
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim

    cs = cos_sin_cache[positions.long()].to(torch.float32)
    cos = cs[..., :half]
    sin = cs[..., half:]

    rope = x[..., nope_dim:].float()
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)
    even = rope[..., 0]
    odd = rope[..., 1]

    for _ in range(rope.ndim - 3):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # Use addcmul (an FMA) for the 2x2 rotation to mirror the kernel's
    # `e*c - o*s` fused form. This keeps the reference close to the kernel, but
    # the fp32 reference and the fp32 GPU kernel can still round to bf16 on
    # opposite sides of a round-to-nearest tie for a tiny number of elements at
    # high positions, so callers compare the RoPE region within 1 bf16 ULP.
    new_even = torch.addcmul(-odd * sin, even, cos)
    new_odd = torch.addcmul(odd * cos, even, sin)
    rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)

    out = x.clone().float()
    out[..., nope_dim:] = rope_rotated
    return out.to(x.dtype)


def rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with no learnable weight, matching
    `RMSNorm(head_dim, has_weight=False)`.

    Returns fp32 so callers can chain RoPE without an intermediate bf16 round
    (the kernel keeps the whole RMSNorm→RoPE pipeline in fp32 and rounds once
    at the final store).
    """
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps)


# ── Dispatch to the CUDA op (skip test cleanly if it isn't built in) ─────────


def _op_available() -> bool:
    return hasattr(torch.ops._C, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")


def _full_cache_fp8_op_available() -> bool:
    return hasattr(
        torch.ops._C, "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert"
    )


def _full_cache_bf16_op_available() -> bool:
    return hasattr(
        torch.ops._C, "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert"
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _op_available(),
    reason="CUDA not available or fused DeepseekV4 op not built in",
)


def _call_fused(
    q_in, q_head_padded, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs
):
    q_out = torch.empty(
        q_in.shape[0],
        q_head_padded,
        q_in.shape[2],
        dtype=q_in.dtype,
        device=q_in.device,
    )
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q_in,
        kv,
        q_out,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        bs,
    )
    return q_out


def _call_fused_out(
    q_in,
    q_out,
    kv,
    k_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    eps,
    bs,
):
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q_in,
        kv,
        q_out,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        bs,
    )
    return q_out


def _as_stored_fp8(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a float8_e4m3fn-typed kernel output under the real (FNUZ on
    gfx942) encoding the kernel actually wrote, without touching the bytes."""
    return t.contiguous().view(torch.uint8).view(FP8_STORE_DTYPE)


def _dequant_cache(k_cache_2d, num_tokens, num_blocks, block_size):
    """Round-trip a [num_blocks, block_size*HEAD_BYTES] K-cache back to bf16."""
    device = k_cache_2d.device
    out = torch.zeros(1, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    k_cache_3d = k_cache_2d.view(num_blocks, block_size, HEAD_BYTES)
    dequantize_and_gather_k_cache(
        out,
        k_cache_3d,
        seq_lens,
        None,
        block_table,
        block_size,
        offset=0,
        use_fnuz=USE_FNUZ,
    )
    return out[0, :num_tokens]


def _assert_kv_cache_parity(
    k_cache_fused, k_cache_ref, num_tokens, num_blocks, block_size
):
    """Assert the fused and reference K-caches agree after decoding.

    The NoPE region is deterministic UE8M0 FP8, so its round-trip must be
    bit-identical. The RoPE region is stored as bf16 after an fp32 rotation:
    the GPU kernel and the PyTorch reference can fall on opposite sides of a
    round-to-nearest tie and differ by at most one bf16 ULP. (Spot checks show
    the kernel value is the correctly-rounded one; the fp32 torch reference is
    the one that lands on the wrong side near a midpoint.) Allow <=1 ULP there.
    """
    rec_fused = _dequant_cache(k_cache_fused, num_tokens, num_blocks, block_size)
    rec_ref = _dequant_cache(k_cache_ref, num_tokens, num_blocks, block_size)
    torch.testing.assert_close(
        rec_fused[:, :NOPE_DIM], rec_ref[:, :NOPE_DIM], rtol=0, atol=0
    )
    max_ulp = int(
        bf16_ulp_distance(rec_fused[:, NOPE_DIM:], rec_ref[:, NOPE_DIM:]).max().item()
    )
    assert max_ulp <= 1, f"RoPE bf16 region differs by {max_ulp} ULP (>1)"


# ── Test 1: Q path numerical parity ──────────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 64, 2048])
@pytest.mark.parametrize(
    "n_heads,padded_heads",
    [
        # Each supported padded_heads instantiation: padded (n_heads <
        # padded_heads) and unpadded (n_heads == padded_heads).
        (1, 8),
        (8, 8),
        (8, 16),
        (16, 16),
        (16, 32),
        (32, 32),
        (8, 64),
        (64, 64),
        (64, 128),
    ],
)
def test_q_path_matches_reference(num_tokens: int, n_heads: int, padded_heads: int):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

    # Reference: RMSNorm (no weight) per head, then GPT-J RoPE on last 64.
    # Keep the chain in fp32 (rmsnorm_no_weight returns fp32) and round to
    # bf16 once at the end, matching the kernel.
    q_ref = rmsnorm_no_weight(q, eps)
    q_ref = apply_rope_gptj_last_k(q_ref, positions, cos_sin_cache).to(dtype)

    # Fused call with dummy KV tensors (KV branch will write slot_mapping=-1 → noop).
    num_blocks = 2
    bs = 16
    kv = torch.zeros(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    k_cache = torch.zeros(
        num_blocks, bs, HEAD_BYTES, dtype=torch.uint8, device=device
    ).view(num_blocks, -1)
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
    q_out = _call_fused(
        q, padded_heads, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs
    )

    torch.testing.assert_close(q_out[:, :n_heads], q_ref, rtol=1e-2, atol=1e-2)
    if n_heads < padded_heads:
        pad_region = q_out[:, n_heads:padded_heads]
        assert pad_region.abs().max().item() == 0.0, (
            "padded head slots must be exact zero"
        )


def test_quant_insert_writes_caller_owned_q_out():
    torch.manual_seed(4)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    num_tokens = 17
    n_heads = 8
    padded_heads = 16
    block_size = 16

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(4096, ROPE_DIM, torch.float32, device)
    k_cache = torch.zeros(
        2, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
    q_out = torch.full(
        (num_tokens, padded_heads, HEAD_DIM),
        -123.0,
        dtype=dtype,
        device=device,
    )

    returned = _call_fused_out(
        q, q_out, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size
    )

    assert returned.data_ptr() == q_out.data_ptr()
    q_ref = apply_rope_gptj_last_k(
        rmsnorm_no_weight(q, eps), positions, cos_sin_cache
    ).to(dtype)
    torch.testing.assert_close(q_out[:, :n_heads], q_ref, rtol=1e-2, atol=1e-2)
    assert q_out[:, n_heads:padded_heads].abs().max().item() == 0.0


# ── Test 2: KV path round-trip byte/value parity ─────────────────────────────


def _ue8m0_per_block_scales(kv_roped_nope_f32: torch.Tensor, qblock: int):
    """Return per-token per-block max scale (used to bound FP8 error)."""
    n_tok, nope = kv_roped_nope_f32.shape
    n_blocks = nope // qblock
    blocks = kv_roped_nope_f32.view(n_tok, n_blocks, qblock)
    absmax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    raw = absmax / FP8_MAX
    exponent = torch.ceil(torch.log2(raw))
    return torch.pow(2.0, exponent)  # [n_tok, n_blocks]


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 64, 2048])
@pytest.mark.parametrize("block_size", [16, 64])
def test_kv_path_matches_reference(num_tokens: int, block_size: int):
    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096

    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # ── Reference path: RoPE on kv, then existing Triton quant+insert ──────
    kv_ref = apply_rope_gptj_last_k(kv, positions, cos_sin_cache)
    k_cache_ref = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
    )
    quantize_and_insert_k_cache(
        kv_ref, k_cache_ref, slot_mapping, block_size=block_size, use_fnuz=USE_FNUZ
    )

    # ── Fused path (dummy q, padded to FlashMLA's min head count 64) ───────
    k_cache_fused = torch.zeros_like(k_cache_ref)
    q_dummy = torch.zeros(num_tokens, 1, HEAD_DIM, dtype=dtype, device=device)
    _ = _call_fused(
        q_dummy,
        64,
        kv,
        k_cache_fused,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    )

    # ── Round-trip compare via dequant+gather ──────────────────────────────
    def _dequant(k_cache_2d):
        num_reqs = 1
        max_blocks = num_blocks
        out = torch.zeros(
            num_reqs, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        block_table = torch.arange(
            max_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0)
        # gather_lens arg is None (use seq_lens)
        k_cache_3d = k_cache_2d.view(num_blocks, block_size, HEAD_BYTES)
        dequantize_and_gather_k_cache(
            out,
            k_cache_3d,
            seq_lens,
            None,
            block_table,
            block_size,
            offset=0,
            use_fnuz=USE_FNUZ,
        )
        return out[0, :num_tokens]

    recovered_ref = _dequant(k_cache_ref)
    recovered_fused = _dequant(k_cache_fused)

    # NoPE: per-block UE8M0 FP8 error bound (half-ULP at max = 16 * scale).
    scales = _ue8m0_per_block_scales(kv_ref[:, :NOPE_DIM].float(), QUANT_BLOCK)
    for t in range(num_tokens):
        max_allowed = 16.0 * scales[t].max().item()
        diff_ref = (
            (recovered_ref[t, :NOPE_DIM] - kv_ref[t, :NOPE_DIM]).abs().max().item()
        )
        diff_fused = (
            (recovered_fused[t, :NOPE_DIM] - kv_ref[t, :NOPE_DIM]).abs().max().item()
        )
        assert diff_ref <= max_allowed, (
            f"ref NoPE token {t} diff {diff_ref} > {max_allowed}"
        )
        assert diff_fused <= max_allowed, (
            f"fused NoPE token {t} diff {diff_fused} > {max_allowed}"
        )

    # Strong parity: NoPE FP8 round-trip bit-identical, RoPE bf16 within 1 ULP.
    _assert_kv_cache_parity(
        k_cache_fused, k_cache_ref, num_tokens, num_blocks, block_size
    )


# ── Test 2b: DP padding (slot_mapping shorter than q/kv) ─────────────────────


@pytest.mark.parametrize("num_tokens", [4, 17, 2048])
@pytest.mark.parametrize("pad", [1, 5])
@pytest.mark.parametrize("block_size", [16, 64])
def test_kv_path_with_dp_padding(num_tokens: int, pad: int, block_size: int):
    """slot_mapping.size(0) < q.size(0): the kernel must skip padded
    tokens in the KV branch while still running Q-norm+RoPE on all rows."""
    torch.manual_seed(3)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096
    total = num_tokens + pad

    kv = torch.randn(total, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(total, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Reference: only the first num_tokens kv rows get inserted.
    kv_ref = apply_rope_gptj_last_k(
        kv[:num_tokens], positions[:num_tokens], cos_sin_cache
    )
    k_cache_ref = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
    )
    quantize_and_insert_k_cache(
        kv_ref, k_cache_ref, slot_mapping, block_size=block_size, use_fnuz=USE_FNUZ
    )

    # Fused: pass full-sized q/kv/positions, shorter slot_mapping.
    q_dummy = torch.zeros(total, 1, HEAD_DIM, dtype=dtype, device=device)
    k_cache_fused = torch.zeros_like(k_cache_ref)
    _ = _call_fused(
        q_dummy,
        64,
        kv,
        k_cache_fused,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    )

    _assert_kv_cache_parity(
        k_cache_fused, k_cache_ref, num_tokens, num_blocks, block_size
    )


# ── Test 3: combined single-call Q + KV parity ───────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 2048])
@pytest.mark.parametrize(
    "n_heads,padded_heads",
    [
        # Each supported padded_heads instantiation: padded (n_heads <
        # padded_heads) and unpadded (n_heads == padded_heads).
        (1, 8),
        (8, 8),
        (8, 16),
        (16, 16),
        (16, 32),
        (32, 32),
        (8, 64),
        (64, 64),
        (64, 128),
    ],
)
@pytest.mark.parametrize("block_size", [16, 64])
def test_combined_q_and_kv(
    num_tokens: int, n_heads: int, padded_heads: int, block_size: int
):
    torch.manual_seed(2)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Reference.
    q_ref = rmsnorm_no_weight(q, eps)
    q_ref = apply_rope_gptj_last_k(q_ref, positions, cos_sin_cache).to(dtype)
    kv_ref = apply_rope_gptj_last_k(kv, positions, cos_sin_cache)
    k_cache_ref = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
    )
    quantize_and_insert_k_cache(
        kv_ref, k_cache_ref, slot_mapping, block_size=block_size, use_fnuz=USE_FNUZ
    )

    # Fused single call.
    k_cache_fused = torch.zeros_like(k_cache_ref)
    q_out = _call_fused(
        q,
        padded_heads,
        kv,
        k_cache_fused,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    )

    torch.testing.assert_close(q_out[:, :n_heads], q_ref, rtol=1e-2, atol=1e-2)
    if n_heads < padded_heads:
        pad_region = q_out[:, n_heads:padded_heads]
        assert pad_region.abs().max().item() == 0.0, (
            "padded head slots must be exact zero"
        )
    _assert_kv_cache_parity(
        k_cache_fused, k_cache_ref, num_tokens, num_blocks, block_size
    )


# ── Full-cache (FlashInfer) path parity ──────────────────────────────────────


def _call_full_cache_fp8_fused(
    q,
    kv,
    q_fp8,
    k_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    fp8_scale,
    q_fp8_scale_inv,
    eps,
    bs,
):
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
        q,
        kv,
        q_fp8,
        k_cache,
        slot_mapping,
        positions.long(),
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        eps,
        bs,
    )


def _call_full_cache_bf16_fused(
    q,
    kv,
    k_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    eps,
    bs,
):
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
        q,
        kv,
        k_cache,
        slot_mapping,
        positions.long(),
        cos_sin_cache,
        eps,
        bs,
    )


def _fp8_full_cache_reference(
    q,
    kv,
    k_cache,
    q_fp8,
    slot_mapping,
    positions,
    cos_sin_cache,
    eps,
    block_size,
    fp8_scale,
    q_fp8_scale_inv,
):
    q_ref = rmsnorm_no_weight(q, eps)
    q_ref = apply_rope_gptj_last_k(q_ref, positions, cos_sin_cache)
    q_fp8.copy_(
        torch.clamp(q_ref.float() * q_fp8_scale_inv, -FP8_MAX, FP8_MAX).to(
            FP8_STORE_DTYPE
        )
    )

    kv_ref = apply_rope_gptj_last_k(kv, positions, cos_sin_cache)
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    block_idx = slots // block_size
    pos_in_block = slots % block_size
    k_cache[block_idx, pos_in_block] = torch.clamp(
        kv_ref[valid].float() / fp8_scale, -FP8_MAX, FP8_MAX
    ).to(FP8_STORE_DTYPE)


def _bf16_full_cache_reference(
    q,
    kv,
    k_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    eps,
    block_size,
):
    q_ref = rmsnorm_no_weight(q, eps)
    # Kernel keeps RMSNorm+RoPE in fp32 and rounds to bf16 once at the store.
    q_ref = apply_rope_gptj_last_k(q_ref, positions, cos_sin_cache).to(q.dtype)

    kv_ref = apply_rope_gptj_last_k(kv, positions, cos_sin_cache)
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    block_idx = slots // block_size
    pos_in_block = slots % block_size
    k_cache[block_idx, pos_in_block] = kv_ref[valid]
    return q_ref


@pytest.mark.skipif(
    not _full_cache_fp8_op_available(),
    reason="full-cache per-tensor FP8 DeepseekV4 op not built in",
)
@pytest.mark.parametrize("num_tokens", [4, 17])
@pytest.mark.parametrize("n_heads", [8, 17])
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
def test_full_cache_per_tensor_fp8_matches_reference(
    num_tokens: int,
    n_heads: int,
    positions_dtype: torch.dtype,
):
    torch.manual_seed(4)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    block_size = 16
    max_pos = 4096

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=positions_dtype, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    fp8_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    q_fp8_scale_inv = torch.tensor([1.0], dtype=torch.float32, device=device)

    # References are encoded under the scheme the kernel actually writes
    # (FNUZ on gfx942); the kernel's own outputs must stay float8_e4m3fn-typed
    # because the op asserts that dtype.
    q_fp8_ref = torch.empty_like(q, dtype=FP8_STORE_DTYPE)
    q_fp8_fused = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_cache_ref = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=FP8_STORE_DTYPE, device=device
    )
    k_cache_fused = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )

    _fp8_full_cache_reference(
        q,
        kv,
        k_cache_ref,
        q_fp8_ref,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
        fp8_scale,
        q_fp8_scale_inv,
    )
    _call_full_cache_fp8_fused(
        q.clone(),
        kv,
        q_fp8_fused,
        k_cache_fused,
        slot_mapping,
        positions,
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        eps,
        block_size,
    )

    # Q is RMSNorm(no-weight)+RoPE in fp32 before fp8 quant; the RMSNorm
    # reduction and RoPE rotation can land the kernel and the torch reference on
    # opposite sides of an fp8 round-to-nearest tie, so allow <=1 fp8 ULP.
    q_fused = _as_stored_fp8(q_fp8_fused)
    q_max_ulp = int(fp8_ulp_distance(q_fused, q_fp8_ref).max().item())
    assert q_max_ulp <= 1, f"Q fp8 differs by {q_max_ulp} ULP (>1)"

    # K-cache NoPE region [0, NOPE_DIM) is a deterministic per-tensor fp8 quant
    # of the (un-rotated) KV input, so it must be bit-identical. The RoPE region
    # [NOPE_DIM, HEAD_DIM) is rotated in fp32 and may differ by <=1 fp8 ULP.
    k_fused = _as_stored_fp8(k_cache_fused)
    torch.testing.assert_close(
        k_fused[..., :NOPE_DIM].float(),
        k_cache_ref[..., :NOPE_DIM].float(),
        rtol=0,
        atol=0,
    )
    k_max_ulp = int(
        fp8_ulp_distance(k_fused[..., NOPE_DIM:], k_cache_ref[..., NOPE_DIM:])
        .max()
        .item()
    )
    assert k_max_ulp <= 1, f"K-cache RoPE fp8 differs by {k_max_ulp} ULP (>1)"


@pytest.mark.skipif(
    not _full_cache_bf16_op_available(),
    reason="full-cache BF16 DeepseekV4 op not built in",
)
@pytest.mark.parametrize("num_tokens", [4, 17])
@pytest.mark.parametrize("n_heads", [8, 17])
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
def test_full_cache_bf16_matches_reference(
    num_tokens: int,
    n_heads: int,
    positions_dtype: torch.dtype,
):
    torch.manual_seed(5)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    block_size = 16
    max_pos = 4096

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=positions_dtype, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    q_fused = q.clone()
    k_cache_ref = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    k_cache_fused = torch.zeros_like(k_cache_ref)
    q_ref = _bf16_full_cache_reference(
        q,
        kv,
        k_cache_ref,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    )
    _call_full_cache_bf16_fused(
        q_fused,
        kv,
        k_cache_fused,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    )

    torch.testing.assert_close(q_fused, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_cache_fused, k_cache_ref, rtol=0, atol=0)
