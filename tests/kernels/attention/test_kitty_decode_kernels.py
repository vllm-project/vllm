# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Kitty read kernels in `vllm/v1/attention/ops/kitty/decode.py`.

Every cache is written by the production write path (`write_cache`). Its
unwritten residual rows hold NaN and its unwritten pages random bytes, so a
kernel that reads anything it should not fails the comparison.

- `kitty_dequant_context` must return exactly the K/V that were written, for
  inputs that sealing stores without loss.
- `kitty_decode_attention` must match dense attention over the written K/V,
  with sealed blocks replaced by their pages as decoded by `ref_dequant_page`,
  plus the step's new token.
"""

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.kitty.decode import (
    NUM_SPLITS,
    kitty_decode_attention,
    kitty_dequant_context,
)
from vllm.v1.attention.ops.kitty.store_seal import (
    kitty_seal,
    kitty_write_residual,
    plan_kitty_step,
)
from vllm.v1.attention.ops.kitty.tiles import (
    KITTY_PAGE_SIZE,
    KITTY_RESIDUAL_ROWS,
    kitty_state_content_bytes,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="Kitty kernels need CUDA with compute capability >= 8.0",
)

PAGE = KITTY_PAGE_SIZE
D = 128  # the only head size Kitty supports
NUM_KV_HEADS = 2
SCALE = D**-0.5
DTYPES = [torch.float16, torch.bfloat16]


# --------------------------------------------------------------------------- #
# Inputs and cache setup
# --------------------------------------------------------------------------- #


def random_kv(length: int, dtype: torch.dtype, seed: int):
    """Random K/V `[length, H_KV, D]`. K channels have uneven magnitudes, so
    boosting picks some channels over others."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (length, NUM_KV_HEADS, D)
    channel_scale = torch.rand((1, NUM_KV_HEADS, D), generator=gen, device="cuda")
    k = torch.randn(shape, generator=gen, device="cuda") * (0.5 + 4 * channel_scale)
    v = torch.randn(shape, generator=gen, device="cuda")
    return k.to(dtype), v.to(dtype)


def lossless_kv(length: int, dtype: torch.dtype, seed: int):
    """Random K/V `[length, H_KV, D]` that sealing stores exactly.

    Within a page, each K channel takes values in {0, 5, 10, 15} / 8 and each
    V token values in {0, 1, 2, 3} / 2, with both extremes present. These are
    exact `code * scale + min` values for 2-bit K, 4-bit (boosted) K and 2-bit
    V.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (length, NUM_KV_HEADS, D)
    k = torch.randint(0, 4, shape, generator=gen, device="cuda") * 5 / 8
    v = torch.randint(0, 4, shape, generator=gen, device="cuda") / 2
    k[0::PAGE], k[1::PAGE] = 0, 15 / 8
    v[..., 0], v[..., 1] = 0, 3 / 2
    return k.to(dtype), v.to(dtype)


@dataclass
class KittyCache:
    """One layer's Kitty cache and the tables that map requests into it."""

    packed: torch.Tensor  # [num_pages, H_KV, 128, C] uint8
    residual: torch.Tensor  # [num_residual_blocks, H_KV, 640, D]
    block_table: torch.Tensor  # [num_reqs, max_pages] int32
    residual_block_ids: torch.Tensor  # [num_reqs] int32
    d_boost: int


def write_cache(keys: list, values: list, lens: list[int], d_boost: int):
    """A new cache holding the first lens[r] tokens of keys[r] / values[r],
    written in one step of the production write path.

    Unwritten residual rows hold NaN and unwritten page bytes are random.
    Pages and residual blocks are assigned in shuffled order; page 0 and
    residual block 0 are never assigned.
    """
    num_reqs = len(lens)
    max_pages = max(cdiv(max(lens), PAGE), 1)
    num_pages = num_reqs * max_pages + 1
    gen = torch.Generator(device="cuda").manual_seed(0)
    page_shape = (num_pages, NUM_KV_HEADS, PAGE, kitty_state_content_bytes(D, d_boost))
    cache = KittyCache(
        packed=torch.randint(
            0, 256, page_shape, generator=gen, dtype=torch.uint8, device="cuda"
        ),
        residual=torch.full(
            (num_reqs + 1, NUM_KV_HEADS, KITTY_RESIDUAL_ROWS, D),
            float("nan"),
            dtype=keys[0].dtype,
            device="cuda",
        ),
        block_table=(torch.randperm(num_pages - 1, generator=gen, device="cuda") + 1)
        .view(num_reqs, max_pages)
        .int(),
        residual_block_ids=(
            torch.randperm(num_reqs, generator=gen, device="cuda") + 1
        ).int(),
        d_boost=d_boost,
    )

    def gpu(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).cuda()

    # Every request is prefilled from empty to its length.
    query_start_loc = np.cumsum([0, *lens])
    plan = plan_kitty_step(query_start_loc, np.array(lens))
    key = torch.cat([k[:n] for k, n in zip(keys, lens)])
    value = torch.cat([v[:n] for v, n in zip(values, lens)])
    kitty_seal(
        key,
        value,
        cache.packed,
        cache.residual,
        cache.residual_block_ids,
        cache.block_table,
        gpu(plan.k_jobs),
        gpu(plan.v_jobs),
        gpu(plan.old_lens),
        gpu(query_start_loc),
        d_boost,
    )
    kitty_write_residual(
        key,
        value,
        cache.residual,
        cache.residual_block_ids,
        gpu(plan.k_writes),
        gpu(plan.v_writes),
    )
    return cache


# --------------------------------------------------------------------------- #
# kitty_dequant_context
# --------------------------------------------------------------------------- #


def dequant_context(cache: KittyCache, req: int, old_len: int, rows: int):
    """`kitty_dequant_context` into NaN-filled `[rows, H_KV, D]` outputs."""
    shape = (rows, NUM_KV_HEADS, D)
    k_out = torch.full(shape, float("nan"), dtype=cache.residual.dtype, device="cuda")
    v_out = torch.full_like(k_out, float("nan"))
    kitty_dequant_context(
        cache.packed,
        cache.residual,
        cache.residual_block_ids,
        cache.block_table,
        req,
        old_len,
        k_out,
        v_out,
        cache.d_boost,
    )
    return k_out, v_out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d_boost", [0, 32])
def test_dequant_context_returns_the_written_kv(dtype, d_boost):
    """Lengths around every page boundary cover the sink, sealed pages and the
    K and V tails. Rows past old_len are left untouched."""
    lens = [1, 127, 128, 129, 255, 256, 257, 383, 384, 385, 600]
    kv = [lossless_kv(n, dtype, seed=r) for r, n in enumerate(lens)]
    cache = write_cache([k for k, _ in kv], [v for _, v in kv], lens, d_boost)

    for r, n in enumerate(lens):
        k_out, v_out = dequant_context(cache, r, n, rows=n + 8)
        k, v = kv[r]
        torch.testing.assert_close(k_out[:n], k, rtol=0, atol=0, msg=f"len={n}")
        torch.testing.assert_close(v_out[:n], v, rtol=0, atol=0, msg=f"len={n}")
        assert k_out[n:].isnan().all() and v_out[n:].isnan().all()


def test_dequant_context_of_an_empty_cache_writes_nothing():
    k, v = lossless_kv(PAGE, torch.float16, seed=0)
    cache = write_cache([k], [v], [PAGE], d_boost=32)
    k_out, v_out = dequant_context(cache, req=0, old_len=0, rows=8)
    assert k_out.isnan().all() and v_out.isnan().all()


# --------------------------------------------------------------------------- #
# kitty_decode_attention
# --------------------------------------------------------------------------- #


def make_decode_batch(old_lens: list[int], group: int, dtype, d_boost: int):
    """A cache holding the first old_len tokens of each request, the decode
    step's queries `[B, H_KV * group, D]` and new K/V `[B, H_KV, D]`, and each
    request's K/V `[old_len + 1, H_KV, D]`."""
    # Each request's K/V include its new token, at position old_len.
    kv = [random_kv(n + 1, dtype, seed=r) for r, n in enumerate(old_lens)]
    keys, values = [k for k, _ in kv], [v for _, v in kv]
    cache = write_cache(keys, values, old_lens, d_boost)

    gen = torch.Generator(device="cuda").manual_seed(0)
    q_shape = (len(old_lens), NUM_KV_HEADS * group, D)
    q = torch.randn(q_shape, generator=gen, device="cuda").to(dtype)
    k_new = torch.stack([k[n] for k, n in zip(keys, old_lens)])
    v_new = torch.stack([v[n] for v, n in zip(values, old_lens)])
    return cache, q, k_new, v_new, kv


def run_decode(cache: KittyCache, q, k_new, v_new, old_lens: list[int]):
    """`kitty_decode_attention` with NaN-filled output and scratch, so any
    value the kernel fails to write, or reads before writing, shows up."""
    batch, num_heads, _ = q.shape
    out = torch.full_like(q, float("nan"))
    mid_o = torch.full(
        (batch, num_heads, NUM_SPLITS, D + 1), float("nan"), device="cuda"
    )
    lse = torch.full((batch, num_heads), float("nan"), device="cuda")
    num_chunks = [max(1, cdiv(n, PAGE)) for n in old_lens]
    kitty_decode_attention(
        q,
        k_new,
        v_new,
        out,
        cache.packed,
        cache.residual,
        cache.residual_block_ids,
        cache.block_table,
        torch.tensor(old_lens, device="cuda"),
        torch.tensor(num_chunks, device="cuda"),
        SCALE,
        cache.d_boost,
        mid_o,
        lse,
    )
    return out


def unpack_2bit(packed: torch.Tensor) -> torch.Tensor:
    """2-bit codes of uint8 `[..., n]`, first code in the lowest bits, as
    int64 `[..., 4 * n]`."""
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device)
    return ((packed[..., None].long() >> shifts) & 3).flatten(-2)


def ref_dequant_page(page: torch.Tensor, d_boost: int):
    """K and V `[128, D]` in fp32 of one packed page, per the layout in `tiles`:

    ```text
    K low plane   0             uint8 [D, 32], 2-bit codes along tokens
    V data        32*D          uint8 [128, D/4], 2-bit codes along D
    K high plane  64*D          uint8 [DB, 32], row r = rank-r channel
    K scale/min   64*D + 32*DB  fp16 [D, 2]
    V scale/min   + 4*D         fp16 [128, 2]
    K rank table  + 512         uint8 [D], DB means not boosted
    ```
    """
    page = page.flatten()
    k_scale_min_at = 64 * D + 32 * d_boost
    v_scale_min_at = k_scale_min_at + 4 * D
    rank_at = v_scale_min_at + 512
    k_scale_min = page[k_scale_min_at:v_scale_min_at].view(torch.float16).view(D, 2)
    v_scale_min = page[v_scale_min_at:rank_at].view(torch.float16).view(PAGE, 2)
    rank = page[rank_at : rank_at + D].long()

    k_codes = unpack_2bit(page[: 32 * D].view(D, 32))
    k_high = unpack_2bit(page[64 * D : k_scale_min_at].view(d_boost, 32))
    boosted = rank < d_boost
    k_codes[boosted] |= k_high[rank[boosted]] << 2
    v_codes = unpack_2bit(page[32 * D : 64 * D].view(PAGE, D // 4))

    k = k_codes * k_scale_min[:, :1].float() + k_scale_min[:, 1:].float()
    v = v_codes * v_scale_min[:, :1].float() + v_scale_min[:, 1:].float()
    return k.T, v


def ref_context(cache: KittyCache, req: int, k: torch.Tensor, v: torch.Tensor, n):
    """K/V `[n, H_KV, D]` that request req's cache holds at length n: the
    written k and v, except that sealed blocks read as their decoded pages.

    K blocks 1 to n // 128 - 1 are sealed, and V seals one block later.
    """
    k_ctx, v_ctx = k[:n].clone(), v[:n].clone()
    num_full = n // PAGE
    for b in range(1, num_full):
        rows = slice(b * PAGE, (b + 1) * PAGE)
        page_id = cache.block_table[req, b]
        for h in range(NUM_KV_HEADS):
            k_page, v_page = ref_dequant_page(cache.packed[page_id, h], cache.d_boost)
            k_ctx[rows, h] = k_page.to(k.dtype)
            if b < num_full - 1:
                v_ctx[rows, h] = v_page.to(v.dtype)
    return k_ctx, v_ctx


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """softmax(q k^T * scale) v in fp32 for one query token: q is `[H_Q, D]`,
    k and v are `[L, H_KV, D]`."""
    group = q.shape[0] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("hd,lhd->hl", q.float(), k) * SCALE
    return torch.einsum("hl,lhd->hd", scores.softmax(-1), v)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group", [1, 3, 8, 16])
@pytest.mark.parametrize("d_boost", [0, 32])
def test_decode_matches_dense_attention(dtype, group, d_boost):
    """Lengths around every page boundary cover the sink, sealed pages and the
    K and V tails; 0 attends to the new token only, and 4113 spans more chunks
    than splits. A group of 3 leaves padded query rows in every head tile."""
    old_lens = [0, 1, 127, 128, 129, 255, 256, 257, 383, 384, 4113]
    cache, q, k_new, v_new, kv = make_decode_batch(old_lens, group, dtype, d_boost)

    out = run_decode(cache, q, k_new, v_new, old_lens)

    for r, n in enumerate(old_lens):
        k, v = kv[r]
        k_ctx, v_ctx = ref_context(cache, r, k, v, n)
        k_all = torch.cat([k_ctx, k_new[r : r + 1]])
        v_all = torch.cat([v_ctx, v_new[r : r + 1]])
        want = dense_attention(q[r], k_all, v_all)
        torch.testing.assert_close(
            out[r].float(), want, atol=1.5e-2, rtol=1e-2, msg=f"old_len={n}"
        )


def test_decode_reads_strided_inputs():
    """The model passes q, k and v as views into one fused QKV tensor, and a
    block table with more columns than a request needs. Both must give the
    same output as contiguous inputs."""
    old_lens = [129, 300, 600]
    cache, q, k_new, v_new, _ = make_decode_batch(
        old_lens, group=4, dtype=torch.float16, d_boost=32
    )
    want = run_decode(cache, q, k_new, v_new, old_lens)

    qkv = torch.cat([q.flatten(1), k_new.flatten(1), v_new.flatten(1)], dim=1)
    q_view, k_view, v_view = (
        part.unflatten(1, (-1, D))
        for part in qkv.split([q[0].numel(), k_new[0].numel(), v_new[0].numel()], 1)
    )
    assert q_view.stride(0) == k_view.stride(0) == qkv.shape[1]
    # Extra columns point at page 0, which holds random bytes.
    num_cols = cache.block_table.shape[1]
    wide_table = torch.zeros_like(cache.block_table).repeat(1, 2)
    wide_table[:, :num_cols] = cache.block_table

    got = run_decode(
        replace(cache, block_table=wide_table), q_view, k_view, v_view, old_lens
    )
    assert torch.equal(got, want)
