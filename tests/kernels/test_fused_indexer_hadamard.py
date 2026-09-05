# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Hadamard rotation in the DeepSeek-V4 sparse indexer
quant kernels.

The indexer Q and K Triton kernels rotate the full 128-dim vector by a
Sylvester Hadamard matrix (scaled by head_dim**-0.5, computed as a
fixed-order fp32 butterfly) after RoPE and before quantization, matching the
reference implementation's rotate_activation. These tests assert bit-exact
equality against unfused rope → hadamard → quant references (using an fp8
quant oracle independent of per_token_group_quant_fp8) and check that the
orthogonal rotation preserves indexer QK dot products.
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
