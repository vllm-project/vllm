# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests for the DeepSeek V4 aiter ASM sparse decode path.

The ASM path repacks the selected pages of both V4 caches into the two-buffer
layout `mla_decode_fwd_v4_nm` consumes, then issues one kernel call. It must
agree with the Triton path it replaces, including on the two inputs that path
tolerates: slots it masks out, and ragged index arrays whose allocated capacity
exceeds their live length.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only used by ROCm"
)

DIM_NOPE, DIM_ROPE, DIM_QK = 448, 64, 512
NTILES, TOK, FP8_MAX = 7, 576, 448.0


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import _ON_GFX950

        return bool(_ON_GFX950)
    except ImportError:
        return False


requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="aiter ships the v4 nm ASM decode for gfx950 only"
)


def _pack_tokens(latent: torch.Tensor):
    """Encode latents exactly as the V4 cache writers do.

    448 OCP E4M3 FP8 NoPE values, 64 BF16 RoPE values at byte offset 448, and
    seven UE8M0 scale bytes of ceil(log2(absmax / 448)) + 127.
    """
    n = latent.shape[0]
    nope = latent[:, :DIM_NOPE].float().view(n, NTILES, 64)
    absmax = nope.abs().amax(-1).clamp_min(1e-4)
    exp = torch.ceil(torch.log2(absmax / FP8_MAX))
    q = (nope / torch.pow(2.0, exp).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)

    tokens = torch.empty((n, TOK), dtype=torch.uint8, device=latent.device)
    tokens[:, :DIM_NOPE] = (
        q.to(torch.float8_e4m3fn).view(n, DIM_NOPE).view(torch.uint8)
    )
    tokens[:, DIM_NOPE:] = latent[:, DIM_NOPE:].contiguous().view(torch.uint8)
    scales = torch.zeros((n, 8), dtype=torch.uint8, device=latent.device)
    scales[:, :NTILES] = (exp.to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
    return tokens, scales


def _make_cache(num_blocks: int, block_size: int, device: torch.device):
    """A paged V4 cache: per block, all token data then all scale bytes."""
    latent = (
        torch.randn(
            (num_blocks * block_size, DIM_QK), device=device, dtype=torch.bfloat16
        )
        * 0.5
    )
    tokens, scales = _pack_tokens(latent)
    block_bytes = block_size * TOK + block_size * 8
    cache = torch.zeros((num_blocks, block_bytes), dtype=torch.uint8, device=device)
    cache[:, : block_size * TOK] = tokens.view(num_blocks, block_size * TOK)
    cache[:, block_size * TOK :] = scales.view(num_blocks, block_size * 8)
    return cache.view(num_blocks, block_size, block_bytes // block_size)


def _run_both(*, num_blocks, block_size, num_tokens, heads, len_extra, len_main,
              slack=0, corrupt=False, device="cuda"):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

    torch.manual_seed(0)
    dev = torch.device(device)
    extra_cache = _make_cache(num_blocks, block_size, dev)
    main_cache = _make_cache(num_blocks, block_size, dev)
    rows = num_blocks * block_size

    def ragged(per_query, n_slack):
        lens = torch.full((num_tokens,), per_query, dtype=torch.int32, device=dev)
        indptr = torch.nn.functional.pad(lens.cumsum(0), (1, 0)).int()
        idx = torch.randint(
            0, rows, (num_tokens * per_query + n_slack,), dtype=torch.int32, device=dev
        )
        return idx, indptr

    extra_idx, extra_indptr = ragged(len_extra, slack)
    main_idx, main_indptr = ragged(len_main, slack)
    if corrupt:
        # Slots the Triton path masks out: negative, and past the row count.
        extra_idx[3::17] = -1
        extra_idx[5::29] = rows + 7
        main_idx[2::11] = -1

    q = torch.randn((num_tokens, heads, DIM_QK), device=dev, dtype=torch.bfloat16) * 0.5
    sink = torch.randn((heads,), device=dev, dtype=torch.float32)
    common = dict(
        q=q,
        main_cache=main_cache,
        main_indices=main_idx,
        scale=1.0 / (DIM_QK**0.5),
        attn_sink=sink,
        nope_head_dim=DIM_NOPE,
        rope_head_dim=DIM_ROPE,
        extra_cache=extra_cache,
        extra_indices=extra_idx,
        main_ragged_indices=main_idx,
        main_ragged_indptr=main_indptr,
        extra_ragged_indices=extra_idx,
        extra_ragged_indptr=extra_indptr,
    )

    outs = []
    for enabled in (True, False):
        out = torch.empty(
            (num_tokens, heads, DIM_QK), dtype=torch.bfloat16, device=dev
        )
        saved = ops._V4_ASM_DECODE
        ops._V4_ASM_DECODE = enabled
        try:
            ops._rocm_sparse_attn_decode_triton(out=out, **common)
        finally:
            ops._V4_ASM_DECODE = saved
        torch.cuda.synchronize()
        outs.append(out.float())
    return outs


@requires_gfx950
@pytest.mark.parametrize("heads", [16, 128])
def test_asm_decode_matches_triton(heads):
    """The ASM path agrees with Triton on a plain decode batch."""
    asm, triton_out = _run_both(
        num_blocks=512,
        block_size=64,
        num_tokens=4,
        heads=heads,
        len_extra=96,
        len_main=32,
    )
    # Q is quantized to FP8 on the ASM path and kept BF16 on the Triton path,
    # so agreement is to BF16 accuracy rather than bitwise.
    torch.testing.assert_close(asm, triton_out, atol=6e-3, rtol=6e-2)


@requires_gfx950
def test_asm_decode_masks_invalid_slots():
    """Negative and out-of-range slots must be excluded, as Triton excludes them."""
    asm, triton_out = _run_both(
        num_blocks=512,
        block_size=64,
        num_tokens=4,
        heads=128,
        len_extra=96,
        len_main=32,
        corrupt=True,
    )
    torch.testing.assert_close(asm, triton_out, atol=6e-3, rtol=6e-2)


@requires_gfx950
def test_asm_decode_tolerates_oversized_ragged_workspace():
    """Ragged arrays are capacity-sized workspaces, so numel() exceeds indptr[T].

    Entries past the live region must not be gathered, and must not push the
    per-entry owner index past the end of the per-query length tensors.
    """
    asm, triton_out = _run_both(
        num_blocks=512,
        block_size=64,
        num_tokens=4,
        heads=128,
        len_extra=96,
        len_main=32,
        slack=512,
        corrupt=True,
    )
    torch.testing.assert_close(asm, triton_out, atol=6e-3, rtol=6e-2)


@requires_gfx950
def test_asm_decode_falls_back_on_small_caches():
    """A cache too small to be a served rank must stay on Triton.

    The memory-profiling dummy run allocates a handful of blocks, which would
    leave every sequence with zero KV length after the validity filter.
    """
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

    asm, triton_out = _run_both(
        num_blocks=8,
        block_size=64,
        num_tokens=4,
        heads=128,
        len_extra=96,
        len_main=32,
    )
    assert 8 * 64 < ops._V4_MIN_CACHE_ROWS
    # Both arms took the Triton path, so they agree exactly.
    torch.testing.assert_close(asm, triton_out, atol=0.0, rtol=0.0)
