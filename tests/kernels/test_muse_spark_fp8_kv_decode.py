# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark specialized decode's attention kernels on FP8 KV caches,
with the page sizes the hybrid KV cache manager gives them."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels run on SM100-family devices",
)

# Power-of-two scales keep every dequantized entry exact in BF16, so a
# quantized cache must reproduce the BF16 kernel on its dequantization bitwise.
K_SCALE, V_SCALE = 0.5, 2.0
PATHS = [
    (1, 2048, [8, 700, 2048], 0),  # one-cluster kernel
    (2, 16384, [33, 4000], 0),  # split kernel, short path
    (1, 32768, [100, 9000, 20000], 0),  # split kernel, long path
    (1, 32768, [3000, 20000], 512),  # sliding window
]


def _attn2(attn2, q, gate, kv, block_table, lens, kv_heads, max_len, window, quant):
    split = attn2.split_kernel(max_len, window)
    num_splits = 4 if split and not window else 1
    workspace = None
    if split:
        workspace = torch.zeros(
            attn2.workspace_bytes(q.shape[0] * attn2.HEADS * kv_heads, num_splits),
            dtype=torch.uint8,
        )
    out = torch.empty_like(q)
    attn2.attn2(
        q,
        kv,
        block_table,
        lens,
        gate,
        out,
        kv_heads,
        max_seq_len=max_len,
        window=window,
        num_splits=num_splits,
        workspace=workspace,
        kv_quant=quant,
    )
    return out


def _fp8_cache(shape):
    k = (torch.randn(shape) * 2 / K_SCALE).to(torch.float8_e4m3fn)
    v = (torch.randn(shape) / V_SCALE).to(torch.float8_e4m3fn)
    cache = torch.cat([k.view(torch.uint8), v.view(torch.uint8)], -1)
    bf16 = torch.cat([k.float() * K_SCALE, v.float() * V_SCALE], -1).bfloat16()
    return cache.contiguous(), bf16.contiguous()


def _case(kv_heads, seq_lens, slots):
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.attention import (
        attn2,
    )

    rows, width = 8 * len(seq_lens), kv_heads * attn2.HEADS * attn2.HEAD_DIM
    q = torch.randn(rows, width).bfloat16()
    gate = torch.randn(rows, width).bfloat16()
    pages = -(-max(seq_lens) // slots)
    num_pages = len(seq_lens) * pages + 3
    block_table = torch.randperm(num_pages)[: len(seq_lens) * pages]
    block_table = block_table.int().view(len(seq_lens), pages)
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    shape = (num_pages, kv_heads, slots, attn2.HEAD_DIM)
    return attn2, q, gate, block_table, lens, shape


@pytest.mark.parametrize("kv_heads,max_len,seq_lens,window", PATHS)
@torch.inference_mode()
def test_attn2_page_size_only_changes_addressing(kv_heads, max_len, seq_lens, window):
    """A 32-slot BF16 cache gives the 16-slot kernel's result on the same
    entries: the hybrid KV cache manager's larger pages read as sub-pages."""
    torch.manual_seed(0)
    with torch.device("cuda"):
        attn2, q, gate, bt32, lens, shape = _case(kv_heads, seq_lens, 32)
        cache32 = torch.randn(*shape[:3], 128).bfloat16()
        # Page p of 32 slots is pages 2 p and 2 p + 1 of 16.
        cache16 = cache32.view(shape[0], kv_heads, 2, 16, 128).transpose(1, 2)
        cache16 = cache16.reshape(2 * shape[0], kv_heads, 16, 128).contiguous()
        bt16 = torch.stack([2 * bt32, 2 * bt32 + 1], -1).flatten(1)
        args = (kv_heads, max_len, window, None)
        out32 = _attn2(attn2, q, gate, cache32, bt32, lens, *args)
        out16 = _attn2(attn2, q, gate, cache16, bt16, lens, *args)
    torch.testing.assert_close(out32, out16, atol=0, rtol=0)


@pytest.mark.parametrize("slots", [16, 32])
@pytest.mark.parametrize("kv_heads,max_len,seq_lens,window", PATHS)
@torch.inference_mode()
def test_attn2_fp8_cache_matches_bf16_on_its_dequantization(
    slots, kv_heads, max_len, seq_lens, window
):
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.attention import (
        kv_format,
    )

    torch.manual_seed(0)
    with torch.device("cuda"):
        attn2, q, gate, block_table, lens, shape = _case(kv_heads, seq_lens, slots)
        cache, bf16 = _fp8_cache(shape)
        args = (block_table, lens, kv_heads, max_len, window)
        ref = _attn2(attn2, q, gate, bf16, *args, None)
        quant = kv_format.KvQuant("fp8", K_SCALE, V_SCALE)
        out = _attn2(attn2, q, gate, cache, *args, quant)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
        # vLLM allocates uint8; a float8_e4m3fn view is the same cache.
        fp8_view = cache.view(torch.float8_e4m3fn)
        out_view = _attn2(attn2, q, gate, fp8_view, *args, quant)
        torch.testing.assert_close(out_view, ref, atol=0, rtol=0)


@pytest.mark.parametrize("slots", [16, 32])
@pytest.mark.parametrize("m", [1, 8, 16])
@torch.inference_mode()
def test_qkv_front_fp8_insert_is_the_stock_store(slots, m):
    """The QKV front stores, for an FP8 cache, what
    `reshape_and_cache_flash` stores of the k and v its BF16 variant writes;
    the projection output is the BF16 variant's."""
    import vllm._custom_ops  # noqa: F401  (registers _C_cache_ops)
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.attention import (
        kv_format,
        qkv_front,
    )

    torch.manual_seed(0)
    quant = kv_format.KvQuant("fp8", 0.37, 0.05)
    hq, hkv, head_dim = 16, 2, qkv_front.HEAD_DIM
    n = (2 * hq + 2 * hkv) * head_dim
    pages = 4
    with torch.device("cuda"):
        weight = (torch.randn(n, qkv_front.HIDDEN) * 0.02).bfloat16()
        x = torch.randn(m, qkv_front.HIDDEN).bfloat16()
        freqs = torch.arange(4096)[:, None] / (
            10000 ** (torch.arange(0, head_dim, 2) / head_dim)
        )
        cos_sin = torch.cat([freqs.cos(), freqs.sin()], -1).contiguous()
        positions = torch.randint(0, 4096, (m,))
        slot_mapping = torch.randperm(pages * slots)[:m]
        if m > 1:
            slot_mapping[0] = -1  # a padded token: no insert
        bf16_cache = torch.zeros(pages, hkv, slots, 2 * head_dim, dtype=torch.bfloat16)
        bf16_out = torch.empty(m, n, dtype=torch.bfloat16)
        qkv_front.qkv_front(
            x, weight, positions, cos_sin, slot_mapping, bf16_cache, bf16_out, 1e-5
        )
        cache = torch.zeros(pages, hkv, slots, 2 * head_dim, dtype=torch.uint8)
        expected = torch.zeros_like(cache)
        k_cache, v_cache = expected.transpose(1, 2).split(head_dim, dim=-1)
        out = torch.empty_like(bf16_out)
        qkv_front.qkv_front(
            x, weight, positions, cos_sin, slot_mapping, cache, out, 1e-5, quant
        )
        # The stock store of the BF16 variant's k and v.
        valid = slot_mapping >= 0
        slots_v = slot_mapping[valid]
        rows = bf16_cache[slots_v // slots, :, slots_v % slots]  # [t, hkv, 128]
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            rows[..., :head_dim].contiguous(),
            rows[..., head_dim:].contiguous(),
            k_cache,
            v_cache,
            slots_v,
            "fp8",
            torch.tensor([quant.k_scale]),
            torch.tensor([quant.v_scale]),
        )

    torch.testing.assert_close(out, bf16_out, atol=0, rtol=0)
    torch.testing.assert_close(cache, expected, atol=0, rtol=0)


def _padded(cache: torch.Tensor, pad: int) -> torch.Tensor:
    """`cache` in pages `pad` elements longer than dense, as the hybrid KV
    cache manager pads a group's pages to the largest group's page bytes."""
    pages, page = cache.shape[0], cache[0].numel()
    storage = torch.full((pages, page + pad), 0x55, dtype=cache.dtype)
    storage[:, :page] = cache.reshape(pages, page)
    return storage.as_strided(
        cache.shape, (page + pad, *cache.stride()[1:]), storage.storage_offset()
    )


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@pytest.mark.parametrize("kv_heads,max_len,seq_lens,window", PATHS[1:3])
@torch.inference_mode()
def test_attn2_reads_padded_pages(kind, kv_heads, max_len, seq_lens, window):
    """Padded pages read as dense."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.attention import (
        kv_format,
    )

    torch.manual_seed(0)
    with torch.device("cuda"):
        attn2, q, gate, block_table, lens, shape = _case(kv_heads, seq_lens, 16)
        if kind == "bf16":
            cache, quant = torch.randn(*shape[:3], 128).bfloat16(), None
        else:
            cache = _fp8_cache(shape)[0]
            quant = kv_format.KvQuant(kind, K_SCALE, V_SCALE)
        args = (block_table, lens, kv_heads, max_len, window, quant)
        dense = _attn2(attn2, q, gate, cache, *args)
        padded = _attn2(attn2, q, gate, _padded(cache, 1024), *args)
    torch.testing.assert_close(padded, dense, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@torch.inference_mode()
def test_qkv_front_writes_padded_pages(kind):
    """The insert lands in the same rows of padded pages and leaves the
    padding alone."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.attention import (
        kv_format,
        qkv_front,
    )

    torch.manual_seed(0)
    m, hq, hkv, pages, slots = 8, 16, 2, 4, 16
    n = (2 * hq + 2 * hkv) * qkv_front.HEAD_DIM
    quant = None if kind == "bf16" else kv_format.KvQuant(kind, 0.37, 0.05)
    shape = (pages, hkv, slots, 128)
    dtype = torch.bfloat16 if kind == "bf16" else torch.uint8
    with torch.device("cuda"):
        weight = (torch.randn(n, qkv_front.HIDDEN) * 0.02).bfloat16()
        x = torch.randn(m, qkv_front.HIDDEN).bfloat16()
        positions = torch.arange(m)
        slot_mapping = torch.randperm(pages * slots)[:m]
        dense = torch.zeros(shape, dtype=dtype)
        padded = _padded(torch.zeros(shape, dtype=dtype), 1024)
        out = torch.empty(m, n, dtype=torch.bfloat16)
        for cache in (dense, padded):
            qkv_front.qkv_front(
                x, weight, positions, None, slot_mapping, cache, out, 1e-5, quant
            )
        padding = padded.as_strided(
            (pages, 1024), (padded.stride(0), 1), padded[0].numel()
        )
    torch.testing.assert_close(padded.contiguous(), dense, atol=0, rtol=0)
    assert bool((padding == 0x55).all())
