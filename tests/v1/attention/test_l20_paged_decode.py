# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


def _reference(query, key_cache, value_cache, block_tables, seq_lens):
    scale = query.shape[-1] ** -0.5
    outputs = []
    for batch in range(query.shape[0]):
        length = int(seq_lens[batch])
        logical = torch.arange(length, device=query.device)
        physical = block_tables[batch, logical // 16]
        offsets = logical % 16
        keys = key_cache[physical, offsets]
        values = value_cache[physical, offsets]
        group = query.shape[1] // keys.shape[1]
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,thd->ht", query[batch].float(), keys.float()) * scale
        outputs.append(
            torch.einsum("ht,thd->hd", scores.softmax(dim=-1), values.float())
        )
    return torch.stack(outputs).to(query.dtype)


@pytest.mark.parametrize(
    ("batch", "context", "q_heads", "kv_heads", "packed_kv_cache"),
    [
        (1, 129, 12, 2, False),
        (1, 129, 12, 2, True),
        (1, 2304, 16, 8, False),
        (4, 513, 12, 2, False),
        (4, 640, 16, 8, False),
    ],
)
def test_l20_paged_decode_matches_reference(
    batch, context, q_heads, kv_heads, packed_kv_cache
):
    if current_platform.get_device_capability() != DeviceCapability(8, 9):
        pytest.skip("L20 paged decode is specialized for SM89")

    torch.manual_seed(17)
    pages_per_request = (context + 15) // 16
    pages = batch * pages_per_request
    block_tables = torch.randperm(pages, device="cuda", dtype=torch.int32).reshape(
        batch, pages_per_request
    )
    seq_lens = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    query = torch.randn(batch, q_heads, 128, device="cuda", dtype=torch.float16)
    if packed_kv_cache:
        kv_cache = torch.randn(
            pages, 2, 16, kv_heads, 128, device="cuda", dtype=torch.float16
        )
        key_cache, value_cache = kv_cache.unbind(1)
        assert not key_cache.is_contiguous()
        assert not value_cache.is_contiguous()
    else:
        key_cache = torch.randn(
            pages, 16, kv_heads, 128, device="cuda", dtype=torch.float16
        )
        value_cache = torch.randn_like(key_cache)
    splits = (context + 63) // 64
    partial = torch.empty(
        batch, q_heads, splits, 128, device="cuda", dtype=torch.float16
    )
    maxima = torch.empty(batch, q_heads, splits, device="cuda", dtype=torch.float32)
    sums = torch.empty_like(maxima)
    output = torch.empty_like(query)

    ops.l20_paged_decode_split_out(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        partial,
        maxima,
        sums,
        output,
        context,
        64,
    )
    expected = _reference(query, key_cache, value_cache, block_tables, seq_lens)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_l20_paged_decode_rejects_too_many_splits():
    if current_platform.get_device_capability() != DeviceCapability(8, 9):
        pytest.skip("L20 paged decode is specialized for SM89")

    query = torch.randn(1, 12, 128, device="cuda", dtype=torch.float16)
    cache = torch.randn(257, 16, 2, 128, device="cuda", dtype=torch.float16)
    block_tables = torch.arange(257, device="cuda", dtype=torch.int32).reshape(1, 257)
    seq_lens = torch.full((1,), 4097, device="cuda", dtype=torch.int32)
    partial = torch.empty(1, 12, 65, 128, device="cuda", dtype=torch.float16)
    maxima = torch.empty(1, 12, 65, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="at most 64 splits"):
        ops.l20_paged_decode_split_out(
            query,
            cache,
            cache,
            block_tables,
            seq_lens,
            partial,
            maxima,
            torch.empty_like(maxima),
            torch.empty_like(query),
            4097,
            64,
        )


def test_l20_paged_decode_fake_tensor():
    if not hasattr(torch.ops._C, "l20_paged_decode_split_out"):
        pytest.skip("CUDA extension is not available")

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        query = torch.empty(1, 12, 128, device="cuda", dtype=torch.float16)
        cache = torch.empty(9, 16, 2, 128, device="cuda", dtype=torch.float16)
        block_tables = torch.empty(1, 9, device="cuda", dtype=torch.int32)
        seq_lens = torch.empty(1, device="cuda", dtype=torch.int32)
        partial = torch.empty(1, 12, 3, 128, device="cuda", dtype=torch.float16)
        maxima = torch.empty(1, 12, 3, device="cuda", dtype=torch.float32)
        ops.l20_paged_decode_split_out(
            query,
            cache,
            cache,
            block_tables,
            seq_lens,
            partial,
            maxima,
            torch.empty_like(maxima),
            torch.empty_like(query),
            129,
            64,
        )


def test_l20_paged_decode_is_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_L20_PAGED_DECODE", raising=False)
    assert not envs.VLLM_ENABLE_L20_PAGED_DECODE
    monkeypatch.setenv("VLLM_ENABLE_L20_PAGED_DECODE", "1")
    assert envs.VLLM_ENABLE_L20_PAGED_DECODE
