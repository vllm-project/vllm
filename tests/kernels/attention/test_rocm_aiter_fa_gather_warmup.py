# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER FA gather warmup.", allow_module_level=True)


def test_gather_cache_warmup_keys_cover_layouts():
    from vllm.v1.attention.backends.rocm_aiter_fa import gather_cache_warmup_keys

    common = dict(
        head_dim=128,
        page_size=16,
        dequant=False,
        cache_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    nhd = gather_cache_warmup_keys(**common, shuffle_enabled=False)
    assert [key.layout for key in nhd] == ["NHD"]
    assert nhd[0].head_dim == 128
    assert nhd[0].page_size == 16

    both = gather_cache_warmup_keys(**common, shuffle_enabled=True)
    assert [key.layout for key in both] == ["NHD", "SHUFFLE"]


def test_gather_cache_kernel_does_not_specialize_shape_ints():
    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache_kernel

    unspecialized = cp_mha_gather_cache_kernel.do_not_specialize
    for name in (
        "num_heads",
        "head_size",
        "x",
        "max_block_num",
        "k_cache_stride0",
        "k_cache_stride1",
        "k_cache_stride2",
        "v_cache_stride0",
        "v_cache_stride1",
        "v_cache_stride2",
    ):
        assert name in unspecialized
