# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-selection, ownership and attention contracts for prefill expansion."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.fp8_prefill_cache import (
    PrefillCacheExpansion,
    expand_cache,
    expansion_bytes,
    plan_expansion,
    prefill_page_range,
)


def test_prefill_metadata_excludes_decode_pages():
    indptr = torch.tensor([2, 5, 7], dtype=torch.int32)
    start, end, compact = prefill_page_range(indptr, 9)
    assert (start, end) == (2, 7)
    torch.testing.assert_close(compact, torch.tensor([0, 3, 5], dtype=torch.int32))
    torch.testing.assert_close(indptr, torch.tensor([2, 5, 7], dtype=torch.int32))


@pytest.mark.parametrize("offsets", [[-1, 2], [3, 2], [0, 10], [1, 5, 4]])
def test_prefill_metadata_rejects_invalid_page_ranges(offsets):
    with pytest.raises(ValueError):
        prefill_page_range(torch.tensor(offsets, dtype=torch.int32), 9)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("query_tokens,budget", [(1023, 1 << 30), (8192, 0)])
def test_prefill_metadata_falls_back_before_cuda_allocation(
    layout, query_tokens, budget
):
    assert (
        plan_expansion(
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([1, 0], dtype=torch.int32),
            num_query_tokens=query_tokens,
            num_kv_heads=2,
            page_size=32,
            head_dim=256,
            budget=budget,
            layout=layout,
        )
        is None
    )


def test_prefill_metadata_budget_includes_keys_and_values():
    geometry = dict(num_pages=3, num_kv_heads=2, page_size=32, head_dim=256)
    required = 3 * 2 * 2 * 32 * 256 * 2
    assert (
        expansion_bytes(num_query_tokens=1024, budget=required, **geometry) == required
    )
    assert expansion_bytes(num_query_tokens=1024, budget=required - 1, **geometry) == 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA kernel")
@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_prefill_expansion_keeps_exact_values_and_independent_buffers(layout):
    generator = torch.Generator(device="cuda").manual_seed(1729)
    storage = (
        torch.randn((12, 2, 32, 512), generator=generator, device="cuda") * 6
    ).to(torch.float8_e4m3fn)
    if layout == "NHD":
        storage = storage.transpose(1, 2).contiguous()
    keys, values = storage.split(256, dim=-1)
    original = storage.view(torch.uint8).clone()
    plans: list[PrefillCacheExpansion] = []
    expected_first = None
    budget = 1 << 20
    for pages in [[7, 0, 7], [3, 8, 1, 3, 9]]:
        raw_indices = torch.tensor(
            [11, 0] + [item for page in pages for item in (page, 0)],
            dtype=torch.int32,
            device="cuda",
        )
        plan = plan_expansion(
            torch.tensor([1, 1 + len(pages)], dtype=torch.int32),
            raw_indices[::2],
            num_query_tokens=1024,
            num_kv_heads=2,
            page_size=32,
            head_dim=256,
            budget=budget,
            layout=layout,
        )
        assert plan is not None
        assert plan.cache.untyped_storage().nbytes() == budget
        expanded = expand_cache(keys, values, plan)
        for index, page in enumerate(pages):
            torch.testing.assert_close(
                expanded[index, 0], keys[page].bfloat16(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                expanded[index, 1], values[page].bfloat16(), rtol=0, atol=0
            )
        if plans:
            assert plan.cache.data_ptr() != plans[0].cache.data_ptr()
            torch.testing.assert_close(plans[0].cache, expected_first, rtol=0, atol=0)
        else:
            expected_first = expanded.clone()
        plans.append(plan)
    torch.testing.assert_close(storage.view(torch.uint8), original, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="FlashInfer requires CUDA")
@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_prefill_expansion_preserves_scaled_ragged_attention(layout):
    import flashinfer

    generator = torch.Generator(device="cuda").manual_seed(1730)
    storage = (
        torch.randn((10, 2, 32, 512), generator=generator, device="cuda") * 6
    ).to(torch.float8_e4m3fn)
    if layout == "NHD":
        storage = storage.transpose(1, 2).contiguous()
    keys, values = storage.split(256, dim=-1)
    indices = torch.tensor([8, 9, 4, 0, 4, 6], dtype=torch.int32, device="cuda")
    indptr = torch.tensor([2, 4, 6], dtype=torch.int32)
    plan = plan_expansion(
        indptr,
        indices,
        num_query_tokens=1024,
        num_kv_heads=2,
        page_size=32,
        head_dim=256,
        budget=1 << 20,
        layout=layout,
    )
    assert plan is not None
    expanded = expand_cache(keys, values, plan)
    query_lengths, cache_lengths = [19, 11], [57, 43]
    query = torch.randn(
        (30, 4, 256), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    results = []
    for dtype, ptr, ids, cache in [
        (storage.dtype, indptr, indices, (keys, values)),
        (torch.bfloat16, plan.indptr, plan.indices, expanded),
    ]:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(32 << 20, dtype=torch.uint8, device="cuda"),
            layout,
            backend="fa2",
        )
        wrapper.plan(
            torch.tensor([0, 19, 30], dtype=torch.int32),
            ptr,
            ids,
            torch.tensor([25, 11], dtype=torch.int32),
            4,
            2,
            256,
            32,
            causal=True,
            q_data_type=query.dtype,
            kv_data_type=dtype,
        )
        results.append(
            wrapper.run(query, cache, q_scale=1.0, k_scale=0.07, v_scale=0.11)
        )

    references = []
    nhd_keys = keys if layout == "NHD" else keys.transpose(1, 2)
    nhd_values = values if layout == "NHD" else values.transpose(1, 2)
    offset = 0
    for pages, qlen, kvlen in zip([[4, 0], [4, 6]], query_lengths, cache_lengths):
        k = torch.cat([nhd_keys[page].float() for page in pages])[:kvlen] * 0.07
        v = torch.cat([nhd_values[page].float() for page in pages])[:kvlen] * 0.11
        q = query[offset : offset + qlen].float()
        logits = torch.einsum("qhd,khd->hqk", q, k.repeat_interleave(2, dim=1)) / 16
        mask = torch.arange(kvlen, device="cuda")[None, :] > (
            torch.arange(qlen, device="cuda")[:, None] + kvlen - qlen
        )
        probabilities = logits.masked_fill(mask, -torch.inf).softmax(-1)
        references.append(
            torch.einsum("hqk,khd->qhd", probabilities, v.repeat_interleave(2, dim=1))
        )
        offset += qlen
    reference = torch.cat(references)
    for result in results:
        relative_error = (result.float() - reference).norm() / reference.norm()
        assert relative_error < 0.005
    relative_difference = (
        results[0].float() - results[1].float()
    ).norm() / reference.norm()
    assert relative_difference < 0.005
