# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Hadamard rotation in the DeepSeek-V4 sparse indexer
quant kernels.

The indexer Q and K Triton kernels rotate the full 128-dim vector by a
Sylvester Hadamard matrix (scaled by head_dim**-0.5, computed as a
fixed-order fp32 butterfly) after RoPE and before quantization, matching the
reference implementation's rotate_activation. These tests assert bit-exact
equality against unfused rope → hadamard → quant references (using an fp8
quant oracle independent of per_token_group_quant_fp8), check that the
orthogonal rotation preserves indexer QK dot products, and measure the
rotation's quantization-quality value: reduced QK dot-product error on
outlier-heavy post-RoPE-like activations, neutrality on iid Gaussian input.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    fused_indexer_q_rope_quant,
)
from vllm.platforms import current_platform

from .test_compressor_kv_cache import _reference_kv_compress_norm_rope
from .test_fused_indexer_q_rope_quant import (
    _hadamard_rotate,
    _rope_gptj_tail,
    quantize_to_mxfp4,
)

HEAD_DIM = 128
ROPE_DIM = 64
N_HEAD = 8
MAX_POS = 4096
# The K-side indexer kernels pin tl.float8e4nv/448 on every platform, while
# the Q kernel follows current_platform.fp8_dtype() (fnuz/224 on gfx942) and
# is resolved at runtime in the test.
FP8_MAX = 448.0

requires_sm100 = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="MXFP4 indexer cache requires an SM100-family GPU",
)


def _sylvester_hadamard(n: int, device: torch.device) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h.to(device)


def _ue8m0_fp8_quant(x: torch.Tensor, fp8_dtype: torch.dtype, fp8_max: float):
    """Per-row ue8m0 fp8 quant over the last dim, mirroring the Triton
    kernels' math (fp32 absmax, power-of-two scale)."""
    rows = x.float().reshape(-1, x.shape[-1])
    out = torch.empty_like(rows, dtype=fp8_dtype)
    scales = torch.empty(rows.shape[0], dtype=torch.float32, device=x.device)
    for i, row in enumerate(rows):
        amax = max(row.abs().max().item(), 1e-4)
        scale = 2.0 ** math.ceil(math.log2(amax / fp8_max))
        out[i] = (row / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        scales[i] = scale
    return out.view(x.shape), scales.view(x.shape[:-1])


def _reference_q(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    use_fp4: bool,
    fp8_dtype: torch.dtype,
    fp8_max: float,
):
    """Unfused oracle: GPT-J RoPE + bf16 roundtrip → Hadamard rotation →
    quant, mirroring the fused Q kernels' math."""
    x = _rope_gptj_tail(q, positions, cos_sin_cache)
    x = _hadamard_rotate(x)
    if use_fp4:
        q_packed, ue8m0 = quantize_to_mxfp4(x)
        q_scale = ue8m0.view(torch.int32).squeeze(-1)
        weights_out = weights.float() * softmax_scale * head_scale
        return (q_packed, q_scale), weights_out
    q_fp8, q_scale = _ue8m0_fp8_quant(x, fp8_dtype, fp8_max)
    weights_out = weights.float() * q_scale * softmax_scale * head_scale
    return q_fp8, weights_out


def _assert_q_bitwise(expected, actual, use_fp4: bool):
    if use_fp4:
        (packed_exp, scale_exp), (packed_act, scale_act) = expected, actual
        assert torch.equal(scale_exp, scale_act), (
            f"ue8m0 scales differ: {(scale_exp != scale_act).sum().item()} bytes"
        )
        assert torch.equal(packed_exp, packed_act), (
            f"packed e2m1 bytes differ: {(packed_exp != packed_act).sum().item()}"
        )
    else:
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)), (
            "fp8 bytes differ: "
            f"{(expected.view(torch.uint8) != actual.view(torch.uint8)).sum().item()}"
        )


def _rope_gptj_tail_dims(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """_rope_gptj_tail generalized to arbitrary head/rope dims (unrotated)."""
    nope_dim = x.shape[-1] - rope_dim
    cos = cos_sin_cache[positions, : rope_dim // 2].double().unsqueeze(1)
    sin = cos_sin_cache[positions, rope_dim // 2 :].double().unsqueeze(1)
    out = x.double()
    ev, od = out[..., nope_dim::2], out[..., nope_dim + 1 :: 2]
    r_ev = (ev * cos - (od * sin).float().double()).float()
    r_od = (od * cos + (ev * sin).float().double()).float()
    out = out.clone()
    out[..., nope_dim::2] = r_ev
    out[..., nope_dim + 1 :: 2] = r_od
    return out.to(torch.bfloat16).float()


@pytest.mark.parametrize("num_tokens", [1, 37])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_fp4", [False, pytest.param(True, marks=requires_sm100)])
@torch.inference_mode()
def test_indexer_q_hadamard(num_tokens, cache_dtype, use_fp4):
    """Bit-exact check of the fused indexer Q RoPE+Hadamard+quant Triton
    kernel against an unfused reference with an independent fp8 oracle."""
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(num_tokens, N_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.randint(
        0, MAX_POS, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM, dtype=cache_dtype, device=device)
    weights = torch.randn(num_tokens, N_HEAD, dtype=torch.bfloat16, device=device)
    softmax_scale = HEAD_DIM**-0.5
    head_scale = N_HEAD**-0.5
    # Match the launcher, which resolves the fp8 flavor at runtime.
    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = 224.0 if fp8_dtype == torch.float8_e4m3fnuz else 448.0

    out, w_out = fused_indexer_q_rope_quant(
        positions,
        q.clone(),
        cos_sin_cache,
        weights,
        softmax_scale,
        head_scale,
        use_fp4,
    )
    ref, w_ref = _reference_q(
        positions,
        q,
        cos_sin_cache,
        weights,
        softmax_scale,
        head_scale,
        use_fp4,
        fp8_dtype,
        fp8_max,
    )
    _assert_q_bitwise(ref, out, use_fp4)
    assert torch.equal(w_ref, w_out)


@pytest.mark.parametrize("use_fp4", [False, pytest.param(True, marks=requires_sm100)])
@torch.inference_mode()
def test_indexer_k_hadamard(use_fp4):
    """Bit-exact check of the fused indexer K compress+RoPE+Hadamard+quant+
    insert Triton kernels via the shared launcher."""
    head_dim, rope_dim = 128, 64
    block_size = 16  # state cache block size
    rms_eps = 1e-6
    num_tokens = 7
    kv_block_size = 16
    compress_ratio = 4
    overlap = 1  # matching DeepseekCompressor logic at compress_ratio == 4

    if use_fp4:
        token_stride = head_dim // 2  # packed nibbles: 64 bytes
        scale_dim = head_dim // 32  # ue8m0 bytes: 4
        quant_block = 32
    else:
        token_stride = head_dim  # FP8 bytes: 128
        scale_dim = 4  # 1 float32: 4 bytes
        quant_block = head_dim

    device = "cuda"
    torch.manual_seed(42)
    coff = 1 + overlap
    num_pages = (compress_ratio * num_tokens - 1) // block_size + 2
    state_cache = torch.randn(
        num_pages,
        block_size,
        2 * coff * head_dim,  # kv_state + score_state, each coff*head_dim wide
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = torch.arange(
        compress_ratio - 1,
        compress_ratio * num_tokens,
        compress_ratio,
        dtype=torch.int64,
        device=device,
    )
    rms_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(compress_ratio * num_tokens, rope_dim, device=device)

    kv_n_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    # 3-D (blocks, tokens/block, bytes/token): the launcher reads
    # kv_cache.shape[1] as the paged cache block size in tokens.
    kv_cache = torch.zeros(
        kv_n_blocks,
        kv_block_size,
        token_stride + scale_dim,
        dtype=torch.uint8,
        device=device,
    )

    compress_norm_rope_store_triton(
        state_cache=state_cache,
        num_actual=num_tokens,
        token_to_req_indices=token_to_req,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=block_size,
        state_width=coff * head_dim,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=slot_mapping),
        pdl_kwargs={},
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        use_fp4_cache=use_fp4,
        rms_norm_weight=rms_weight,
        rms_norm_eps=rms_eps,
        quant_block=quant_block,
        token_stride=token_stride,
        scale_dim=scale_dim,
    )

    k_ref, s_ref = _reference_kv_compress_norm_rope(
        state_cache,
        block_table,
        positions,
        rms_weight,
        cos_sin_cache,
        compress_ratio,
        overlap,
        use_fp4,
        rms_eps=rms_eps,
        fp8_max=FP8_MAX,
        rotate=True,
    )

    kv_flat = kv_cache.view(kv_n_blocks, -1)
    if not use_fp4:
        k_ref = k_ref.view(torch.uint8)
    for i in range(num_tokens):
        blk, pos = i // kv_block_size, i % kv_block_size
        val_off = pos * token_stride
        val_actual = kv_flat[blk, val_off : val_off + token_stride]
        assert torch.equal(k_ref[i], val_actual), f"token {i}: values differ"
        scale_off = kv_block_size * token_stride + pos * scale_dim
        scale_actual = kv_flat[blk, scale_off : scale_off + scale_dim]
        if use_fp4:
            assert torch.equal(scale_actual, s_ref[i]), (
                f"token {i}: ue8m0 {scale_actual.tolist()} != {s_ref[i].tolist()}"
            )
        else:
            assert torch.equal(scale_actual.view(torch.float32), s_ref[i : i + 1]), (
                f"token {i}: scale differs"
            )


@pytest.mark.parametrize(
    # Parity is only guaranteed where main's Triton kernels work: fp8 needs a
    # power-of-2 nope dim (tl.arange), mxfp4 needs nope/rope dims divisible
    # by 32 and head_dim % 128 == 0 (launcher int32 scale view).
    "head_dim, rope_dim, use_fp4",
    [
        (96, 32, False),  # head_dim != 128
        pytest.param(128, 32, True, marks=requires_sm100),  # rope_dim != 64
    ],
)
@torch.inference_mode()
def test_indexer_q_hadamard_dimension_guard(
    head_dim, rope_dim, use_fp4, monkeypatch: pytest.MonkeyPatch
):
    """Off-production dims (head_dim != 128 or rope_dim != 64) bypass the
    Hadamard rotation and take main's original path: the kernel compiles
    (the 128/64 static_assert is scoped to HADAMARD) and the output is
    bit-exact against an unrotated rope → quant reference."""
    # Force the Triton path; the cutedsl dispatch for off-dims configs is
    # unchanged from main and not exercised here. Object-form setattr: the
    # shim pre-registers the submodule in sys.modules without a parent
    # attribute, so dotted-string resolution fails.
    import importlib

    fiq_mod = importlib.import_module(
        "vllm.models.deepseek_v4.common.ops.fused_indexer_q"
    )
    monkeypatch.setattr(fiq_mod, "has_cutedsl", lambda: False)
    device = "cuda"
    torch.manual_seed(0)
    num_tokens, n_head = 37, 8
    q = torch.randn(num_tokens, n_head, head_dim, dtype=torch.bfloat16, device=device)
    positions = torch.randint(
        0, MAX_POS, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = torch.randn(MAX_POS, rope_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(num_tokens, n_head, dtype=torch.bfloat16, device=device)
    softmax_scale = head_dim**-0.5
    head_scale = n_head**-0.5
    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = 224.0 if fp8_dtype == torch.float8_e4m3fnuz else 448.0

    out, w_out = fused_indexer_q_rope_quant(
        positions,
        q.clone(),
        cos_sin_cache,
        weights,
        softmax_scale,
        head_scale,
        use_fp4,
    )
    x = _rope_gptj_tail_dims(q, positions, cos_sin_cache, rope_dim)
    if use_fp4:
        packed_ref, ue8m0_ref = quantize_to_mxfp4(x)
        packed, scale = out
        assert torch.equal(ue8m0_ref.view(torch.int32).squeeze(-1), scale)
        assert torch.equal(packed_ref, packed)
        assert torch.equal(w_out, weights.float() * softmax_scale * head_scale)
    else:
        q_fp8, q_scale = _ue8m0_fp8_quant(x, fp8_dtype, fp8_max)
        assert torch.equal(q_fp8.view(torch.uint8), out.view(torch.uint8))
        assert torch.equal(
            w_out, weights.float() * q_scale * softmax_scale * head_scale
        )


@torch.inference_mode()
def test_hadamard_rotation_preserves_qk_dot():
    """The Sylvester matrix is symmetric with H @ H == n*I, so the scaled
    rotation is orthogonal and preserves indexer QK dot products."""
    device = torch.device("cuda")
    hadamard = _sylvester_hadamard(HEAD_DIM, device)
    assert torch.equal(hadamard, hadamard.t())
    identity = torch.eye(HEAD_DIM, device=device) * HEAD_DIM
    assert torch.equal(hadamard @ hadamard, identity)

    # The butterfly oracle computes the same rotation (up to fp32 rounding).
    torch.manual_seed(0)
    q = torch.randn(256, HEAD_DIM, device=device, dtype=torch.float64)
    k = torch.randn(256, HEAD_DIM, device=device, dtype=torch.float64)
    h64 = hadamard.double()
    q_rot = (q @ h64) * (HEAD_DIM**-0.5)
    k_rot = (k @ h64) * (HEAD_DIM**-0.5)
    torch.testing.assert_close(
        _hadamard_rotate(q.float()).double(), q_rot, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close((q_rot * k_rot).sum(-1), (q * k).sum(-1))


# ---------------------------------------------------------------------------
# Quantization-quality property test (value of the rotation, not bit-exactness)
# ---------------------------------------------------------------------------


def _dequant_mxfp4(packed: torch.Tensor, ue8m0: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_to_mxfp4: packed [..., D//2] uint8 (low nibble =
    even index) plus ue8m0 [..., D//32] scale bytes, back to fp32."""
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device
    )
    nibbles = torch.stack([packed & 0x0F, packed >> 4], dim=-1)
    nibbles = nibbles.reshape(*packed.shape[:-1], -1)
    vals = levels[(nibbles & 0x7).long()]
    vals = torch.where(nibbles > 7, -vals, vals)  # 0x8 is the sign bit
    scales = torch.exp2(ue8m0.float() - 127.0)
    out = vals.reshape(*vals.shape[:-1], -1, 32) * scales.unsqueeze(-1)
    return out.reshape(*packed.shape[:-1], -1)


def _power_law_spectrum() -> torch.Tensor:
    """Channel-pair magnitude spectrum shared by Q and K: a power law over
    frequency with a per-pair jitter, mimicking the energy concentration in
    low-frequency channel pairs of post-RoPE activations."""
    g = torch.Generator().manual_seed(20260906)
    freq = torch.arange(1, HEAD_DIM // 2 + 1, dtype=torch.float32)
    mag = 8.0 * freq**-1.2
    return mag * torch.exp(0.3 * torch.randn(HEAD_DIM // 2, generator=g))


def _rope_like_activation(
    batch: int,
    spectrum: torch.Tensor | None,
    seed_shift: int,
    device: str,
) -> torch.Tensor:
    """Synthetic post-RoPE activations: channel pairs sharing the power-law
    ``spectrum`` with independent random phase/amplitude per vector, plus a
    small noise floor. ``spectrum=None`` gives the iid Gaussian control."""
    g = torch.Generator().manual_seed(20260906 + seed_shift)
    if spectrum is None:
        return torch.randn(batch, HEAD_DIM, generator=g).to(device)
    n_pairs = HEAD_DIM // 2
    phase = torch.rand(batch, n_pairs, generator=g) * (2 * math.pi)
    amp = torch.randn(batch, n_pairs, generator=g).abs() + 0.3
    even = amp * spectrum * torch.cos(phase)
    odd = amp * spectrum * torch.sin(phase)
    x = torch.stack([even, odd], dim=-1).reshape(batch, HEAD_DIM)
    x += 0.05 * torch.randn(batch, HEAD_DIM, generator=g)
    return x.to(device)


def _q_quantized_variants(q, use_fp4, fp8_dtype, fp8_max):
    """Run the fused indexer Q kernel on bf16 activations with identity RoPE
    (cos=1, sin=0) and unit weights/scales, and return the dequantized kernel
    output alongside the dequantized unrotated oracle (main's math)."""
    num_tokens, n_head = q.shape[0], q.shape[1]
    device = q.device
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = torch.zeros(num_tokens, ROPE_DIM, device=device)
    cos_sin_cache[:, : ROPE_DIM // 2] = 1.0
    weights = torch.ones(num_tokens, n_head, dtype=torch.bfloat16, device=device)
    out, w_out = fused_indexer_q_rope_quant(
        positions, q.clone(), cos_sin_cache, weights, 1.0, 1.0, use_fp4
    )
    if use_fp4:
        packed, q_scale = out
        ue8m0 = q_scale.unsqueeze(-1).contiguous().view(torch.uint8)
        kernel_deq = _dequant_mxfp4(packed, ue8m0)
        oracle_deq = _dequant_mxfp4(*quantize_to_mxfp4(q.float()))
    else:
        # weights/softmax_scale/head_scale are all 1, so w_out == q_scale.
        kernel_deq = out.float() * w_out.unsqueeze(-1)
        q_fp8, q_scale_ref = _ue8m0_fp8_quant(q.float(), fp8_dtype, fp8_max)
        oracle_deq = q_fp8.float() * q_scale_ref.unsqueeze(-1)
    return kernel_deq, oracle_deq


def _k_quantized_variants(k_act, use_fp4):
    """Run the fused indexer K kernel on raw bf16 activations with
    compress_ratio=1, overlap=0, identity RoPE and unit RMSNorm weight, and
    return (dequantized kernel output, dequantized unrotated oracle, exact
    fp64 quant input)."""
    head_dim, rope_dim = HEAD_DIM, ROPE_DIM
    device = k_act.device
    num_tokens = k_act.shape[0]
    block_size = 16
    kv_block_size = 16
    if use_fp4:
        token_stride, scale_dim, quant_block = head_dim // 2, head_dim // 32, 32
    else:
        token_stride, scale_dim, quant_block = head_dim, 4, head_dim

    num_pages = (num_tokens - 1) // block_size + 2
    state_cache = torch.zeros(
        num_pages, block_size, 2 * head_dim, dtype=torch.bfloat16, device=device
    )
    state_cache.view(-1, 2 * head_dim)[:num_tokens, :head_dim] = k_act
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    rms_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.zeros(num_tokens, rope_dim, device=device)
    cos_sin_cache[:, : rope_dim // 2] = 1.0
    kv_cache = torch.zeros(
        (num_tokens + kv_block_size - 1) // kv_block_size + 1,
        kv_block_size,
        token_stride + scale_dim,
        dtype=torch.uint8,
        device=device,
    )
    compress_norm_rope_store_triton(
        state_cache=state_cache,
        num_actual=num_tokens,
        token_to_req_indices=token_to_req,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=block_size,
        state_width=head_dim,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=slot_mapping),
        pdl_kwargs={},
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        compress_ratio=1,
        overlap=False,
        use_fp4_cache=use_fp4,
        rms_norm_weight=rms_weight,
        rms_norm_eps=1e-6,
        quant_block=quant_block,
        token_stride=token_stride,
        scale_dim=scale_dim,
    )

    ref_args = (
        state_cache,
        block_table,
        positions,
        rms_weight,
        cos_sin_cache,
        1,  # compress_ratio
        0,  # overlap
        use_fp4,
    )
    # Exact reference: the kernel's quantization input (bf16 of the normed
    # activations), unrotated — the rotation preserves dot products.
    k_exact = _reference_kv_compress_norm_rope(
        *ref_args, rms_eps=1e-6, return_full_cache=True, rotate=False
    ).double()
    o_vals, o_scales = _reference_kv_compress_norm_rope(
        *ref_args, rms_eps=1e-6, fp8_max=FP8_MAX, rotate=False
    )

    kv_flat = kv_cache.view(kv_cache.shape[0], -1)
    val_region = kv_flat[:, : kv_block_size * token_stride]
    val_region = val_region.reshape(-1, token_stride)[:num_tokens]
    scale_region = kv_flat[:, kv_block_size * token_stride :]
    scale_region = scale_region.reshape(-1, scale_dim)[:num_tokens]
    if use_fp4:
        kernel_deq = _dequant_mxfp4(val_region, scale_region)
        oracle_deq = _dequant_mxfp4(o_vals, o_scales)
    else:
        scales = scale_region.view(torch.float32).squeeze(-1)
        kernel_deq = val_region.view(torch.float8_e4m3fn).float() * scales[:, None]
        oracle_deq = o_vals.float() * o_scales.unsqueeze(-1)
    return kernel_deq, oracle_deq, k_exact


@pytest.mark.parametrize("use_fp4", [False, pytest.param(True, marks=requires_sm100)])
@torch.inference_mode()
def test_indexer_hadamard_quant_quality(use_fp4):
    """The Hadamard rotation reduces indexer QK dot-product quantization
    error on outlier-heavy post-RoPE-like activations, and is neutral on
    outlier-free (iid Gaussian) ones.

    Post-RoPE activations concentrate energy in low-frequency channel pairs
    (power-law spectrum), so per-group quant scales are set by outlier
    channels and the rest of the group loses resolution. The orthogonal
    rotation spreads energy evenly — preserving exact dot products — and
    shrinks the dot-product error the indexer actually consumes. On real
    DSv4 indexer activations (official reference impl, TP4, 8192 tokens)
    the rotation cuts the fp8 dot-error mean by 1.25-1.46x and mxfp4 by
    1.18-1.66x across layers, landing on the iid-Gaussian baseline
    (rotated-real/iid ratio 0.97-1.02). Per-vector fp8 relative L2 error
    is neutral by design — the per-row scale absorbs outliers into the
    exponent (real K: 0.0262 unrotated vs 0.0267 rotated) — so this test
    asserts on dot-product error only.

    With identity RoPE and unit weights/scales, the fused kernels quantize
    the synthetic activations directly; the oracle applies the same
    quantizers without the rotation, which is exactly what main's kernels
    compute. On main the error ratio is therefore ~1.0 and the outlier
    assertion fails; here the rotation drives it well below the threshold.
    """
    device = "cuda"
    num_tokens, n_head, num_kv = 128, 64, 128
    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = 224.0 if fp8_dtype == torch.float8_e4m3fnuz else FP8_MAX
    spectrum = _power_law_spectrum()

    for label, spec in (("outlier", spectrum), ("iid", None)):
        q_act = _rope_like_activation(num_tokens * n_head, spec, 1, device)
        q_act = q_act.reshape(num_tokens, n_head, HEAD_DIM).to(torch.bfloat16)
        k_act = _rope_like_activation(num_kv, spec, 2, device).to(torch.bfloat16)
        q_kernel, q_oracle = _q_quantized_variants(q_act, use_fp4, fp8_dtype, fp8_max)
        k_kernel, k_oracle, k_exact = _k_quantized_variants(k_act, use_fp4)

        exact = torch.einsum("thd,sd->ths", q_act.double(), k_exact)
        err_kernel = (
            torch.einsum("thd,sd->ths", q_kernel.double(), k_kernel.double()) - exact
        ).abs()
        err_oracle = (
            torch.einsum("thd,sd->ths", q_oracle.double(), k_oracle.double()) - exact
        ).abs()
        mean_ratio = (err_kernel.mean() / err_oracle.mean()).item()
        max_ratio = (err_kernel.max() / err_oracle.max()).item()
        if label == "outlier":
            # Measured on this branch (SM100): 0.21 fp8 / 0.25 mxfp4; main's
            # unrotated kernels are identical to the oracle, giving ~1.0.
            # 0.5 keeps >=2x distance from both.
            assert mean_ratio < 0.5, (
                f"rotation should cut dot-product quantization error on "
                f"outlier activations (use_fp4={use_fp4}): mean-error ratio "
                f"{mean_ratio:.3f} (max {max_ratio:.3f}) >= 0.5"
            )
        else:
            assert 0.8 <= mean_ratio <= 1.25, (
                f"rotation should be neutral on iid activations "
                f"(use_fp4={use_fp4}): mean-error ratio {mean_ratio:.3f} "
                f"outside [0.8, 1.25]"
            )
