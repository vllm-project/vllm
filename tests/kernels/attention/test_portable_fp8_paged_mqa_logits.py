# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable (bf16-dequant) Triton paged-MQA-logits kernel vs torch reference.

The portable kernel backs the DeepSeek-V4 sparse indexer on platforms whose
Triton target cannot compile fp8 tl.dot (e.g. ROCm gfx1151 / RDNA3.5). It
must produce the same logits as fp8_paged_mqa_logits_torch.
"""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "batch_size,next_n,ctx_lens",
    [
        (3, 1, [37, 811, 5120]),
        (2, 6, [64, 1930]),
        (1, 1, [1]),
        (1, 6, [64]),
    ],
)
def test_portable_fp8_paged_mqa_logits_matches_torch(batch_size, next_n, ctx_lens):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
        portable_fp8_paged_mqa_logits,
    )

    torch.manual_seed(7)
    device = "cuda"
    fp8 = current_platform.fp8_dtype()
    num_heads, dim, block_size, num_blocks = 64, 128, 64, 300
    max_model_len = 8192

    vals = (torch.randn(num_blocks, block_size, dim, device=device) * 0.25).to(fp8)
    scales = torch.rand(num_blocks, block_size, device=device) * 1.5 + 0.25
    flat = torch.empty(
        num_blocks, block_size * (dim + 4), dtype=torch.uint8, device=device
    )
    flat[:, : block_size * dim] = vals.reshape(num_blocks, block_size * dim).view(
        torch.uint8
    )
    flat[:, block_size * dim :] = (
        scales.contiguous().view(torch.uint8).reshape(num_blocks, block_size * 4)
    )
    kv_cache = flat.view(num_blocks, block_size, 1, dim + 4)

    q = (torch.randn(batch_size, next_n, num_heads, dim, device=device) * 0.3).to(fp8)
    weights = torch.rand(batch_size * next_n, num_heads, device=device) * 0.2
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device=device)
    block_tables = torch.randint(
        0,
        num_blocks,
        (batch_size, max_model_len // block_size),
        dtype=torch.int32,
        device=device,
    )

    ref = fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )
    out = portable_fp8_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )

    ref_inf = torch.isinf(ref) & (ref < 0)
    assert torch.equal(ref_inf, torch.isinf(out) & (out < 0))
    finite = ~ref_inf
    if finite.any():
        r, o = ref[finite], out[finite]
        rel = ((o - r).abs() / r.abs().clamp_min(1e-3)).max().item()
        assert rel < 0.05
        for i in range(ref.shape[0]):
            k = min(64, int((~ref_inf[i]).sum().item()))
            if k == 0:
                continue
            ti = set(torch.topk(ref[i], k).indices.tolist())
            to = set(torch.topk(out[i], k).indices.tolist())
            assert len(ti & to) / k > 0.93


@pytest.mark.parametrize(
    "compress_ratio,expect_portable", [(4, True), (2, False), (1, False)]
)
def test_portable_kernel_only_serves_block_flat_pages(
    monkeypatch, compress_ratio, expect_portable
):
    """Only block-flat pages may reach the portable kernel.

    Ratio 4 (C4A) pages are written token-major; ratio 1 and 2 pages are
    written 16x16-tiled, so the shared no-AITER branch must not hand them to
    a kernel that reads them as if they were flat.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    if mod._ON_GFX942 or mod._ON_GFX950:
        pytest.skip("C4A is served by the upstream Triton kernel on gfx942/950")

    # Pretend the AITER module is unavailable so the fallback branch is taken.
    monkeypatch.setattr(mod, "paged_mqa_logits_module", lambda: None)
    used = []
    real_portable = mod.portable_fp8_paged_mqa_logits

    def spy(*args, **kwargs):
        used.append(True)
        return real_portable(*args, **kwargs)

    monkeypatch.setattr(mod, "portable_fp8_paged_mqa_logits", spy)

    torch.manual_seed(11)
    device = "cuda"
    fp8 = current_platform.fp8_dtype()
    num_heads, dim, block_size, num_blocks = 64, 128, 64, 300
    max_model_len = 8192

    vals = (torch.randn(num_blocks, block_size, dim, device=device) * 0.25).to(fp8)
    scales = torch.rand(num_blocks, block_size, device=device) * 1.5 + 0.25
    flat = torch.empty(
        num_blocks, block_size * (dim + 4), dtype=torch.uint8, device=device
    )
    flat[:, : block_size * dim] = vals.reshape(num_blocks, block_size * dim).view(
        torch.uint8
    )
    flat[:, block_size * dim :] = (
        scales.contiguous().view(torch.uint8).reshape(num_blocks, block_size * 4)
    )
    kv_cache = flat.view(num_blocks, block_size, 1, dim + 4)

    q = (torch.randn(1, 1, num_heads, dim, device=device) * 0.3).to(fp8)
    weights = torch.rand(1, num_heads, device=device) * 0.2
    context_lens = torch.tensor([512], dtype=torch.int32, device=device)
    block_tables = torch.randint(
        0,
        num_blocks,
        (1, max_model_len // block_size),
        dtype=torch.int32,
        device=device,
    )
    schedule_metadata = torch.zeros(8, 2, dtype=torch.int32, device=device)

    out = mod.rocm_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        compress_ratio=compress_ratio,
    )

    assert out.shape == (1, max_model_len)
    assert bool(used) is expect_portable, (
        f"compress_ratio={compress_ratio}: portable kernel used={bool(used)}, "
        f"expected {expect_portable}"
    )


def _block_flat_cache(num_blocks, block_size, dim, seed):
    """Packed block-flat page cache: values, then scales, token-major."""
    torch.manual_seed(seed)
    device = "cuda"
    fp8 = current_platform.fp8_dtype()
    vals = (torch.randn(num_blocks, block_size, dim, device=device) * 0.25).to(fp8)
    scales = torch.rand(num_blocks, block_size, device=device) * 1.5 + 0.25
    flat = torch.empty(
        num_blocks, block_size * (dim + 4), dtype=torch.uint8, device=device
    )
    flat[:, : block_size * dim] = vals.reshape(num_blocks, block_size * dim).view(
        torch.uint8
    )
    flat[:, block_size * dim :] = (
        scales.contiguous().view(torch.uint8).reshape(num_blocks, block_size * 4)
    )
    return flat.view(num_blocks, block_size, 1, dim + 4)


@pytest.mark.parametrize(
    "batch_size,next_n,ctx_lens",
    [(3, 1, [37, 811, 5120]), (2, 6, [64, 1930]), (2, 3, [129, 4097])],
)
def test_torch_fallback_chunking_is_exact(monkeypatch, batch_size, next_n, ctx_lens):
    """Chunking the page walk must not change a single logit."""
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    num_heads, dim, block_size, num_blocks = 64, 128, 64, 300
    max_model_len = 8192
    kv_cache = _block_flat_cache(num_blocks, block_size, dim, seed=23)
    q = (torch.randn(batch_size, next_n, num_heads, dim, device="cuda") * 0.3).to(
        current_platform.fp8_dtype()
    )
    weights = torch.rand(batch_size * next_n, num_heads, device="cuda") * 0.2
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    block_tables = torch.randint(
        0,
        num_blocks,
        (batch_size, max_model_len // block_size),
        dtype=torch.int32,
        device="cuda",
    )

    whole = mod.fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )
    # Two pages per chunk forces many chunks over the same range.
    monkeypatch.setattr(mod, "_TORCH_MQA_CHUNK_TOKENS", 2 * block_size)
    chunked = mod.fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )

    assert torch.equal(whole, chunked)


def test_torch_fallback_memory_is_bounded():
    """A long max_model_len with short contexts must not allocate GBs."""
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    batch_size, num_heads, dim, block_size = 32, 64, 128, 64
    max_model_len, num_blocks = 131072, 1024
    kv_cache = _block_flat_cache(num_blocks, block_size, dim, seed=29)
    q = (torch.randn(batch_size, 1, num_heads, dim, device="cuda") * 0.3).to(
        current_platform.fp8_dtype()
    )
    weights = torch.rand(batch_size, num_heads, device="cuda") * 0.2
    # Every request is short, but the page-table width follows max_model_len.
    context_lens = torch.full((batch_size,), 64, dtype=torch.int32, device="cuda")
    block_tables = torch.randint(
        0,
        num_blocks,
        (batch_size, max_model_len // block_size),
        dtype=torch.int32,
        device="cuda",
    )

    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    baseline = torch.accelerator.memory_allocated()
    mod.fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )
    torch.accelerator.synchronize()
    peak = torch.accelerator.max_memory_allocated() - baseline

    # Unchunked this shape allocates >3 GB (values + [B, N, H] scores).
    assert peak < 1024 * 1024 * 1024, f"peak allocation {peak / 2**20:.0f} MiB"
