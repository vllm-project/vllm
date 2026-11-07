# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter must not inherit an MLA-only KV cache dtype from the target.

A target's sparse MLA layers canonicalize the shared cache_config to
fp8_ds_mla/nvfp4_ds_mla while they are built (mla_attention writes it back).
Loaded afterwards, a non-MLA drafter has no attention backend for those
formats and startup dies in backend selection (#58733).
"""

from types import SimpleNamespace

import pytest

from vllm.config import CacheConfig
from vllm.v1.worker.gpu.spec_decode.dflash.utils import _draft_cache_config

pytestmark = pytest.mark.cpu_test


def _vllm_config(cache_dtype, draft_dtype=None, use_mla=False):
    return SimpleNamespace(
        cache_config=CacheConfig(cache_dtype=cache_dtype),
        speculative_config=SimpleNamespace(
            kv_cache_dtype=draft_dtype,
            draft_model_config=SimpleNamespace(use_mla=use_mla),
        ),
    )


def test_explicit_draft_dtype_wins():
    config = _draft_cache_config(_vllm_config("fp8_ds_mla", draft_dtype="float16"))
    assert config.cache_dtype == "float16"


@pytest.mark.parametrize("dtype", ["fp8_ds_mla", "nvfp4_ds_mla"])
def test_non_mla_drafter_drops_inherited_mla_only_dtype(dtype):
    config = _draft_cache_config(_vllm_config(dtype, use_mla=False))
    assert config.cache_dtype == "auto"


@pytest.mark.parametrize("dtype", ["fp8_ds_mla", "nvfp4_ds_mla"])
def test_mla_drafter_keeps_inherited_dtype(dtype):
    config = _draft_cache_config(_vllm_config(dtype, use_mla=True))
    assert config.cache_dtype == dtype


@pytest.mark.parametrize("dtype", ["auto", "fp8", "bfloat16"])
def test_non_mla_drafter_keeps_shareable_dtype(dtype):
    config = _draft_cache_config(_vllm_config(dtype, use_mla=False))
    assert config.cache_dtype == dtype
