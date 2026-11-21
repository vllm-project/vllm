# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 DSpark's paged KV writer, used where reshape_and_cache_flash is
unavailable."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

from vllm.models.inkling2.nvidia.ops.dspark import write_kv_paged  # noqa: E402


@pytest.mark.parametrize("head_dim", [64, 80])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dspark_write_kv_paged_matches_scatter(head_dim, dtype):
    """Valid slots are bit-exact and padding slots are left untouched."""
    g = torch.Generator(device="cuda").manual_seed(931)
    tokens, heads, page, blocks = 6, 3, 8, 4
    slots = torch.tensor([13, -1, 7, 0, -1, 25], device="cuda", dtype=torch.int32)

    k = torch.randn(tokens, heads, head_dim, generator=g, device="cuda", dtype=dtype)
    v = torch.randn(tokens, heads, head_dim, generator=g, device="cuda", dtype=dtype)
    kv_cache = torch.randn(
        blocks, heads, page, 2 * head_dim, generator=g, device="cuda", dtype=dtype
    )
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)

    ref_kv = kv_cache.clone()
    ref_key_cache, ref_value_cache = ref_kv.transpose(1, 2).split(head_dim, dim=-1)
    valid = slots >= 0
    block, offset = slots[valid] // page, slots[valid] % page
    ref_key_cache[block, offset] = k[valid]
    ref_value_cache[block, offset] = v[valid]

    write_kv_paged(k, v, key_cache, value_cache, slots)

    assert torch.equal(kv_cache, ref_kv)
