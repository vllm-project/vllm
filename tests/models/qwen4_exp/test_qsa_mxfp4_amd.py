# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mxfp4_qsa main KV cache: writer, layout, and split-K reader."""

import pytest
import torch

from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.amd.ops.mxfp4_kv import (
    block_records,
    record_bytes,
    store_mxfp4_qsa,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not HAS_TRITON,
    reason="the AMD QSA kernels require ROCm and Triton",
)

HEAD_DIM = 256


def _on_gfx950() -> bool:
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., d]`` -> packed ``[..., d/2]`` uint8 and E8M0 ``[..., d/32]``."""
    groups = x.float().unflatten(-1, (-1, 32))
    absmax = groups.abs().amax(-1)
    mantissa, exponent = torch.frexp(absmax / 6.0)
    ceil_log2 = torch.where(mantissa == 0.5, exponent - 1, exponent)
    encoded = torch.clamp(ceil_log2 + 127, 1, 254)
    encoded = torch.where(absmax == 0, 127, encoded)
    scaled = groups * torch.exp2(-(encoded - 127).float())[..., None]
    magnitude = scaled.abs()
    index = sum(
        (magnitude >= m if i % 2 else magnitude > m).int()
        for i, m in enumerate(_MIDPOINTS)
    )
    codes = (index | ((scaled < 0).int() << 3)).flatten(-2)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return packed, encoded.to(torch.uint8)


def dequantize_reference(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    values = _E2M1.to(packed.device)[codes & 7]
    values = torch.where(codes >= 8, -values, values)
    scale = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
    return values * scale


def _block_bytes(cache: torch.Tensor) -> torch.Tensor:
    """The raw ``[blocks, records * record_bytes]`` bytes of an mxfp4 view."""
    num_blocks = cache.shape[0]
    return torch.as_strided(
        cache, (num_blocks, block_records(cache) * cache.shape[3]), (cache.stride(0), 1)
    )


def _regions(cache: torch.Tensor):
    """K data, V data, K scales, V scales as ``[blocks, records, bytes]``."""
    records = block_records(cache)
    raw = _block_bytes(cache)
    half, groups = HEAD_DIM // 2, HEAD_DIM // 32
    k_data = raw[:, : records * half].unflatten(1, (records, half))
    v_data = raw[:, records * half : records * HEAD_DIM].unflatten(1, (records, half))
    scale_start = records * HEAD_DIM
    k_scales = raw[:, scale_start : scale_start + records * groups]
    v_scales = raw[
        :, scale_start + records * groups : scale_start + 2 * records * groups
    ]
    return (
        k_data,
        v_data,
        k_scales.unflatten(1, (records, groups)),
        v_scales.unflatten(1, (records, groups)),
    )


def _record_index(cache: torch.Tensor) -> torch.Tensor:
    """``[tokens, heads]`` record index of each (token, head) within a block."""
    tokens, heads, width = cache.shape[1:]
    token = torch.arange(tokens, device=cache.device)[:, None]
    head = torch.arange(heads, device=cache.device)[None, :]
    return (token * cache.stride(1) + head * cache.stride(2)) // width


def dequantize_cache(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode an mxfp4_qsa view to fp32 K and V ``[blocks, tokens, heads, d]``."""
    k_data, v_data, k_scales, v_scales = _regions(cache)
    index = _record_index(cache)
    keys = dequantize_reference(k_data[:, index], k_scales[:, index])
    values = dequantize_reference(v_data[:, index], v_scales[:, index])
    return keys, values


def _make_cache(num_blocks: int, page_size: int, heads: int, head_major: bool):
    """A uint8 mxfp4_qsa layer view, ``[blocks, tokens, heads, record_bytes]``.

    vLLM views a layer as ``[B, H, N, C]``; head-major blocks are that view
    transposed, token-major blocks are ``[B, N, H, C]`` directly. A block
    stride padded past the dense page mimics other layers' pages between
    blocks.
    """
    width = record_bytes(HEAD_DIM)
    page = page_size * heads * width
    storage = torch.full(
        (num_blocks * (page + 128),), 0xEE, dtype=torch.uint8, device="cuda"
    )
    if head_major:
        strides = (page + 128, width, page_size * width, 1)
    else:
        strides = (page + 128, heads * width, width, 1)
    return torch.as_strided(storage, (num_blocks, page_size, heads, width), strides)


def _random_rows(num_tokens: int, heads: int) -> torch.Tensor:
    rows = torch.randn(num_tokens, heads, HEAD_DIM, device="cuda") * 3
    # Outliers, an all-zero group, tiny negatives, and an exact power of two.
    rows[0, 0, 5] = 500.0
    rows[1 % num_tokens, 0, 32:64] = 0.0
    rows[2 % num_tokens, 0, 64:96] = -1e-6
    rows[3 % num_tokens, 0, 96:128] = 0.0
    rows[3 % num_tokens, 0, 100] = 6.0
    return rows.to(torch.bfloat16)


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("head_major", [False, True])
def test_store_matches_reference_bitwise(heads: int, head_major: bool) -> None:
    torch.manual_seed(0)
    num_blocks, page_size, num_tokens = 5, 32, 97
    cache = _make_cache(num_blocks, page_size, heads, head_major)
    keys = _random_rows(num_tokens, heads)
    values = _random_rows(num_tokens, heads)
    slots = torch.randperm(num_blocks * page_size, device="cuda")[:num_tokens]
    slots[7] = -1
    store_mxfp4_qsa(keys, values, cache, slots)

    expected = _make_cache(num_blocks, page_size, heads, head_major)
    k_data, v_data, k_scales, v_scales = _regions(expected)
    index = _record_index(expected)
    valid = slots >= 0
    blocks = (slots[valid] // page_size).long()
    records = index[(slots[valid] % page_size).long()]
    for source, data, scales in ((keys, k_data, k_scales), (values, v_data, v_scales)):
        packed, encoded = quantize_reference(source[valid])
        data[blocks[:, None], records] = packed
        scales[blocks[:, None], records] = encoded
    assert torch.equal(_block_bytes(cache), _block_bytes(expected))
    # The pad between blocks is never touched.
    pad = torch.as_strided(
        cache,
        (num_blocks, 128),
        (cache.stride(0), 1),
        cache.storage_offset() + block_records(cache) * cache.shape[3],
    )
    assert bool((pad == 0xEE).all())

    decoded_keys, _ = dequantize_cache(cache)
    stored = decoded_keys[blocks, (slots[valid] % page_size).long()]
    error = (stored - keys[valid].float()).abs()
    group_max = keys[valid].float().abs().unflatten(-1, (-1, 32)).amax(-1)
    # The worst case is a peak at 5 * scale, which rounds a full scale step
    # to 4 or 6: a fifth of the peak.
    assert bool(
        (error.unflatten(-1, (-1, 32)) <= group_max[..., None] / 5 + 1e-6).all()
    )


def _reference_attention(q, keys, values, logical_indices, block_table, token_to_req):
    output = torch.zeros_like(q, dtype=torch.float32)
    page_size, num_kv_heads = keys.shape[1], keys.shape[2]
    repeats = q.shape[1] // num_kv_heads
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        pages = block_table[token_to_req[row].long(), logical // page_size].long()
        offsets = logical % page_size
        k = keys[pages, offsets].repeat_interleave(repeats, dim=1)
        v = values[pages, offsets].repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), k) * HEAD_DIM**-0.5
        output[row] = torch.einsum("hk,khd->hd", torch.softmax(scores, -1), v)
    return output


@pytest.mark.parametrize(
    "hw_cvt",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not current_platform.is_rocm() or not _on_gfx950(),
                reason="v_cvt_scalef32_pk_bf16_fp4 is gfx950-only",
            ),
        ),
    ],
)
@pytest.mark.parametrize("dot_scaled", [False, True])
@pytest.mark.parametrize(
    ("num_rows", "num_query_heads", "num_kv_heads", "head_major"),
    [
        pytest.param(1, 12, 1, True, id="tp2-decode"),
        pytest.param(16, 12, 1, True, id="tp2-split"),
        pytest.param(3, 24, 2, True, id="tp1-head-major"),
        pytest.param(3, 24, 2, False, id="tp1-token-major"),
        pytest.param(300, 12, 1, True, id="tp2-prefill"),
    ],
)
def test_sparse_attention_matches_dequantized_reference(
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_major: bool,
    dot_scaled: bool,
    hw_cvt: bool,
) -> None:
    torch.manual_seed(1)
    page_size, num_requests, pages_per_request = 256, 2, 12
    width = 2048 + 3
    num_blocks = num_requests * pages_per_request
    cache = _make_cache(num_blocks, page_size, num_kv_heads, head_major)
    num_slots = num_blocks * page_size
    store_mxfp4_qsa(
        _random_rows(num_slots, num_kv_heads),
        _random_rows(num_slots, num_kv_heads),
        cache,
        torch.arange(num_slots, device="cuda"),
    )
    block_table = (
        torch.randperm(num_blocks, device="cuda")
        .reshape(num_requests, pages_per_request)
        .to(torch.int32)
    )
    context = pages_per_request * page_size
    token_to_req = (torch.arange(num_rows, device="cuda") % num_requests).to(
        torch.int32
    )
    logical_indices = torch.stack(
        [torch.randperm(context, device="cuda")[:width] for _ in range(num_rows)]
    ).to(torch.int32)
    # Ragged selections: the tail of some rows is unused.
    logical_indices[::3, width // 2 :] = -1
    q = torch.randn(
        num_rows, num_query_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        cache,
        cache,
        logical_indices,
        block_table,
        token_to_req,
        mxfp4_dot_scaled=dot_scaled,
        mxfp4_hw_cvt=hw_cvt,
    )
    keys, values = dequantize_cache(cache)
    expected = _reference_attention(
        q, keys, values, logical_indices, block_table, token_to_req
    )
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
