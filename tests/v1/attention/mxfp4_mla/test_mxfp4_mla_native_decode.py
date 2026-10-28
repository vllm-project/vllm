# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx950 native mxfp4_mla decode: the V widening bit for bit against an
exact dequant, and the attention output against the bf16-unpack decode."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.mxfp4_mla.decode import (
    TILE_K,
    _fold_v_scales,
    _widen_v_128,
)

if current_platform.is_rocm() and torch.cuda.is_available():
    from vllm.platforms.rocm import on_gfx950

    _GFX950 = on_gfx950()
else:
    _GFX950 = False
pytestmark = pytest.mark.skipif(not _GFX950, reason="needs gfx950")

LATENT, ROW, GROUP = 512, 272, 32
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _dequantize(rows: torch.Tensor) -> torch.Tensor:
    """Packed ``[n, ROW]`` uint8 rows -> fp32 ``[n, LATENT]``."""
    rows = rows.cpu()
    data = rows[:, : LATENT // 2]
    codes = torch.stack([data & 0x0F, data >> 4], dim=-1).reshape(len(rows), LATENT)
    sign = torch.where(codes & 0x08 != 0, -1.0, 1.0)
    values = _E2M1_VALUES[(codes & 0x07).long()] * sign
    scales = torch.exp2(rows[:, LATENT // 2 :].float() - 127.0)
    return values * scales.repeat_interleave(GROUP, dim=1)


@triton.jit
def _widen_kernel(cache_ptr, valid_ptr, v_ptr, vs_ptr, BLOCK_K: tl.constexpr):
    tokens = tl.arange(0, BLOCK_K)
    valid = tl.load(valid_ptr + tokens) != 0
    row_ptr = cache_ptr + tokens.to(tl.int64) * 272
    k_scale = tl.load(
        row_ptr[:, None] + 256 + tl.arange(0, 16)[None, :],
        mask=valid[:, None],
        other=127,
    )
    v_mult, block_max = _fold_v_scales(k_scale, valid, BLOCK_K)
    for chunk in tl.static_range(4):
        v_fp8, v_scale = _widen_v_128(row_ptr, valid, v_mult, block_max, chunk, BLOCK_K)
        dims = chunk * 128 + tl.arange(0, 128)
        tl.store(v_ptr + tokens[:, None] * 512 + dims[None, :], v_fp8.to(tl.float32))
        tl.store(
            vs_ptr
            + dims[:, None] * (BLOCK_K // 32)
            + tl.arange(0, BLOCK_K // 32)[None, :],
            v_scale,
        )


def test_widen_v_matches_dequant():
    """Every e2m1 code, scales spread up to 2**8 below their 32-token block's
    maximum (the exact range), and masked tokens: widened e4m3 V times its
    block scale equals the dequantized row exactly."""
    set_random_seed(0)
    block_k = TILE_K.value
    cache = torch.randint(0, 256, (block_k, ROW), dtype=torch.uint8)
    top = torch.randint(110, 140, (block_k // 32, 1, LATENT // GROUP))
    drop = torch.randint(0, 9, (block_k // 32, 32, LATENT // GROUP))
    cache[:, LATENT // 2 :] = (top - drop).reshape(block_k, LATENT // GROUP)
    valid = torch.ones(block_k, dtype=torch.int32)
    valid[[3, 40, 127]] = 0

    v = torch.empty(block_k, LATENT, dtype=torch.float32, device="cuda")
    v_scale = torch.empty(LATENT, block_k // 32, dtype=torch.uint8, device="cuda")
    _widen_kernel[(1,)](cache.cuda(), valid.cuda(), v, v_scale, BLOCK_K=block_k)

    # v_scale[d, b] applies to tokens 32b..32b+31 of dim d.
    block_scale = torch.exp2(v_scale.cpu().float() - 127.0).T.repeat_interleave(32, 0)
    want = _dequantize(cache) * valid[:, None]
    assert torch.equal(v.cpu() * block_scale, want)


@pytest.mark.parametrize("num_splits", [1, 3, 8])
@pytest.mark.parametrize("heads", [16, 8])
def test_native_decode_matches_bf16_unpack(num_splits: int, heads: int):
    """Ragged rows (empty, shorter than a tile, tile edges, several tiles) with
    padded and out-of-range slots: the native decode stays within MXFP8 error
    of the decode that unpacks the same rows to bf16."""
    from vllm.v1.attention.ops.mxfp4_mla.store import store_mxfp4_mla
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_sparse_attn_decode_bf16,
    )

    set_random_seed(0)
    num_kv = 1024
    lens = torch.tensor([0, 1, 127, 128, 129, 600], dtype=torch.int32)
    q = torch.randn(len(lens), heads, LATENT, dtype=torch.bfloat16).cuda()
    latent = torch.randn(num_kv, LATENT, dtype=torch.bfloat16)
    latent[:, ::97] *= 8.0
    indptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(lens, 0)
    indices = torch.randint(0, num_kv, (int(indptr[-1]),), dtype=torch.int32)
    indices[5], indices[300] = -1, num_kv + 7

    packed = torch.zeros(num_kv, ROW, dtype=torch.uint8, device="cuda")
    store_mxfp4_mla(latent.cuda(), torch.arange(num_kv).cuda(), packed)

    def attend(native: bool) -> torch.Tensor:
        out = torch.zeros(len(lens), heads, LATENT, dtype=torch.bfloat16).cuda()
        rocm_sparse_attn_decode_bf16(
            q=q,
            kv=packed.unsqueeze(1),
            scale=LATENT**-0.5,
            head_dim=LATENT,
            nope_head_dim=LATENT,
            rope_head_dim=0,
            attn_sink=None,
            output=out,
            ragged_indices=indices.cuda(),
            ragged_indptr=indptr.cuda(),
            num_splits=num_splits,
            kv_cache_dtype="mxfp4_mla",
            native_mxfp4=native,
        )
        return out.float().cpu()

    want, got = attend(False), attend(True)
    assert torch.isfinite(got).all()
    nonempty = lens > 0
    err = (got - want)[nonempty].abs()
    ref = want[nonempty].abs()
    assert err.max() <= 0.05 * ref.max(), (err.max(), ref.max())
    assert err.mean() <= 0.02 * ref.mean(), (err.mean(), ref.mean())
