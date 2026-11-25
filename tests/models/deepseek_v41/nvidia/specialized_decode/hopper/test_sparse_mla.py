# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM90 decode attention's sparse MLA over V4 records written by vLLM's
own V4 writer (`quantize_and_insert_k_cache`):

* the reference decoder reads those records back within E4M3 rounding, the
  BF16 dims exactly;
* the CuTe DSL kernel matches the reference over them;
* on SM90, it matches FlashMLA's decode on the same caches, with the queries
  padded to 64 heads as the stock layer pads them.

The writer converts to FP8 in Triton, so SM89 or newer runs the first two.
"""

import pytest
import torch

from vllm.platforms import current_platform

from ..ampere import reference

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="needs an SM89-or-newer CUDA GPU",
)

DEVICE = "cuda"
BLOCK_SIZE = 64
WINDOW, TOP_K = 128, 512


def _stock_cache(blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A V4-record cache `[blocks, 64, 584]` (pages padded to a multiple of
    576 bytes) filled by vLLM's writer, and the BF16 keys it holds."""
    from vllm.models.deepseek_v41.common.ops import quantize_and_insert_k_cache

    page = BLOCK_SIZE * 584
    raw = torch.zeros(blocks, page + -page % 576, dtype=torch.uint8, device=DEVICE)
    cache = raw[:, :page].view(blocks, BLOCK_SIZE, 584)
    k = torch.randn(blocks * BLOCK_SIZE, 8, 64, device=DEVICE)
    k *= torch.exp2(torch.randint(-3, 4, (len(k), 8, 1), device=DEVICE))
    k = k.flatten(1).bfloat16()
    slots = torch.arange(len(k), device=DEVICE)
    quantize_and_insert_k_cache(k, raw, slots, block_size=BLOCK_SIZE)
    return cache, k


def _inputs(tokens: int, heads: int):
    """Caches, slot lists (some entries -1) and BF16 queries of `tokens`
    decode tokens."""
    swa, _ = _stock_cache(16)
    compressed, _ = _stock_cache(32)
    swa_indices = torch.randint(
        0, 16 * BLOCK_SIZE, (tokens, WINDOW), device=DEVICE, dtype=torch.int32
    )
    swa_lens = torch.randint(
        WINDOW // 2, WINDOW + 1, (tokens,), device=DEVICE, dtype=torch.int32
    )
    compressed_indices = torch.randint(
        0, 32 * BLOCK_SIZE, (tokens, TOP_K), device=DEVICE, dtype=torch.int32
    )
    compressed_indices[:, 5::40] = -1
    compressed_lens = torch.randint(
        TOP_K // 2, TOP_K + 1, (tokens,), device=DEVICE, dtype=torch.int32
    )
    q = torch.randn(tokens, heads, 512, device=DEVICE).bfloat16()
    caches = (swa, swa_indices, swa_lens, compressed, compressed_indices)
    return q, *caches, compressed_lens


def test_stock_records():
    torch.manual_seed(0)
    cache, k = _stock_cache(4)
    got = reference.decode_v4_records(cache, torch.arange(len(k), device=DEVICE))
    want = k.float()
    assert torch.equal(got[:, 448:], want[:, 448:])
    # E4M3 keeps 3 mantissa bits: within 1/16 of each 64-dim block's amax.
    error = (got[:, :448] - want[:, :448]).abs().unflatten(-1, (7, 64))
    amax = want[:, :448].abs().unflatten(-1, (7, 64)).amax(-1, keepdim=True)
    assert (error <= amax / 16).all()


@pytest.mark.parametrize("tokens", [6, 48])
def test_sparse_mla_stock_records(tokens):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import sparse_mla

    torch.manual_seed(0)
    q, swa, swa_indices, swa_lens, compressed, compressed_indices, compressed_lens = (
        _inputs(tokens, 8)
    )
    sink = torch.randn(8, device=DEVICE)
    caches = (swa, swa_indices, swa_lens, compressed, compressed_indices)
    args = (q, *caches, compressed_lens, sink, 512**-0.5, None, None)
    want = reference.sparse_mla(*args, v4_records=True)
    got = torch.full_like(q, float("nan"))
    sparse_mla.sparse_mla(*args, got, v4_records=True)
    assert (got.float() - want.float()).abs().max() < 2e-2 * want.float().abs().max()


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="FlashMLA's SM90 decode"
)
@pytest.mark.parametrize("tokens", [6, 48, 192])
def test_sparse_mla_matches_flashmla(tokens):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import sparse_mla
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
        is_flashmla_sparse_supported,
    )

    if not is_flashmla_sparse_supported()[0]:
        pytest.skip("FlashMLA sparse decode is not built")
    torch.manual_seed(0)
    heads, padded = 8, 64
    q, swa, swa_indices, swa_lens, compressed, compressed_indices, compressed_lens = (
        _inputs(tokens, heads)
    )
    q_padded = torch.zeros(tokens, padded, 512, dtype=q.dtype, device=DEVICE)
    q_padded[:, :heads] = q
    sink = torch.full((padded,), float("-inf"), device=DEVICE)
    sink[:heads] = torch.randn(heads, device=DEVICE)
    want = torch.empty_like(q_padded)
    flash_mla_with_kvcache(
        q=q_padded.unsqueeze(1),
        k_cache=swa.unsqueeze(-2),
        block_table=None,
        head_dim_v=512,
        tile_scheduler_metadata=get_mla_metadata()[0],
        cache_seqlens=None,
        is_fp8_kvcache=True,
        indices=swa_indices.unsqueeze(1),
        topk_length=swa_lens,
        softmax_scale=512**-0.5,
        attn_sink=sink,
        extra_k_cache=compressed.unsqueeze(-2),
        extra_indices_in_kvcache=compressed_indices.unsqueeze(1),
        extra_topk_length=compressed_lens,
        out=want.unsqueeze(1),
    )
    got = torch.full_like(q_padded, float("nan"))
    sparse_mla.sparse_mla(
        q_padded[:, :heads],
        swa,
        swa_indices,
        swa_lens,
        compressed,
        compressed_indices,
        compressed_lens,
        sink[:heads],
        512**-0.5,
        None,
        None,
        got[:, :heads],
        v4_records=True,
    )
    want, got = want[:, :heads].float(), got[:, :heads].float()
    assert (got - want).abs().max() < 2e-2 * want.abs().max()
