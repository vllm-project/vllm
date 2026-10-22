# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mxfp4_mla read path: the row unpack on its own, with the gfx950
hardware converter and with the software path used elsewhere, and the
attention kernels that read a packed cache through it."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.mxfp4_mla_read import _load_mxfp4_rows

if current_platform.is_rocm() and torch.cuda.is_available():
    from vllm.platforms.rocm import get_cdna_version

    _CDNA_VERSION = get_cdna_version()
else:
    _CDNA_VERSION = 0
pytestmark = pytest.mark.skipif(
    _CDNA_VERSION not in (3, 4), reason="mxfp4_mla needs CDNA 3 or 4 (gfx942 or gfx950)"
)

LATENT, ROW, GROUP, BLOCK_K = 512, 272, 32, 16
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
def _gather_kernel(
    cache_ptr,
    slots_ptr,
    out_ptr,
    num_slots,
    row_pitch,
    BLOCK_K: tl.constexpr,
    LATENT: tl.constexpr,
    HW_UNPACK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    slots = tl.load(slots_ptr + offsets).to(tl.int64)
    valid = (slots >= 0) & (slots < num_slots)
    values = _load_mxfp4_rows(
        cache_ptr,
        tl.where(valid, slots, 0),
        valid,
        row_pitch,
        BLOCK_K,
        LATENT,
        HW_UNPACK,
    )
    tl.store(
        out_ptr + offsets[:, None] * LATENT + tl.arange(0, LATENT)[None, :], values
    )


@pytest.mark.parametrize(
    "hw_unpack",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(_CDNA_VERSION != 4, reason="needs gfx950"),
        ),
    ],
    ids=["software", "gfx950"],
)
def test_load_rows_matches_reference(hw_unpack: bool):
    """Every code and a wide spread of scales, gathered from scattered slots;
    padded and out-of-range slots read as zero."""
    set_random_seed(0)
    num_rows = 64
    cache = torch.randint(0, 256, (num_rows, ROW), dtype=torch.uint8)
    cache[:, LATENT // 2 :] = torch.randint(100, 151, (num_rows, LATENT // GROUP))
    slots = torch.randperm(num_rows)[: 2 * BLOCK_K]
    slots[3], slots[20] = -1, num_rows + 5

    out = torch.empty(len(slots), LATENT, dtype=torch.bfloat16, device="cuda")
    _gather_kernel[(len(slots) // BLOCK_K,)](
        cache.cuda(),
        slots.cuda(),
        out,
        num_rows,
        ROW,
        BLOCK_K=BLOCK_K,
        LATENT=LATENT,
        HW_UNPACK=hw_unpack,
    )

    valid = (slots >= 0) & (slots < num_rows)
    want = torch.zeros(len(slots), LATENT)
    want[valid] = _dequantize(cache[slots[valid]])
    assert torch.equal(out.float().cpu(), want)


@pytest.mark.parametrize("num_splits", [1, 4], ids=["ragged", "split_k"])
def test_attention_matches_dequantized_cache(num_splits: int):
    """Attention over the packed cache equals attention over the same values
    stored as bf16, through the ragged kernel and the split-K decode kernel."""
    from vllm.v1.attention.ops.mxfp4_mla_store import store_mxfp4_mla
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_sparse_attn_decode_bf16,
        rocm_sparse_attn_prefill,
    )

    set_random_seed(0)
    num_queries, heads, num_kv, per_query = 4, 16, 128, 64
    q = torch.randn(num_queries, heads, LATENT, dtype=torch.bfloat16).cuda()
    latent = torch.randn(num_kv, LATENT, dtype=torch.bfloat16)
    latent[:, ::97] *= 8.0
    indices = torch.randint(0, num_kv, (num_queries * per_query,), dtype=torch.int32)
    indptr = torch.arange(num_queries + 1, dtype=torch.int32) * per_query

    packed = torch.zeros(num_kv, ROW, dtype=torch.uint8, device="cuda")
    store_mxfp4_mla(latent.cuda(), torch.arange(num_kv).cuda(), packed)
    dequantized = _dequantize(packed).to(torch.bfloat16).cuda()

    def attend(kv: torch.Tensor) -> torch.Tensor:
        out = torch.empty(num_queries, heads, LATENT, dtype=torch.bfloat16).cuda()
        common = dict(
            q=q,
            kv=kv.unsqueeze(1),
            scale=LATENT**-0.5,
            head_dim=LATENT,
            nope_head_dim=LATENT,
            rope_head_dim=0,
            attn_sink=None,
            output=out,
            ragged_indices=indices.cuda(),
            ragged_indptr=indptr.cuda(),
            kv_cache_dtype="mxfp4_mla" if kv.dtype == torch.uint8 else "auto",
        )
        if num_splits == 1:
            rocm_sparse_attn_prefill(indices=None, topk_length=None, **common)
        else:
            rocm_sparse_attn_decode_bf16(num_splits=num_splits, **common)
        return out.cpu()

    assert torch.equal(attend(packed), attend(dequantized))
