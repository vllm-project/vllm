# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query/cache form selection for the SM100 DSA attention layer.

The fused kernels emit either a packed fp8 MQA query against an fp8 cache, or a
bf16 (ql_nope, q_pe) tuple against a cache used as-is. Picking the wrong one is
silent: an unquantized cache viewed as fp8 reinterprets its bytes.
"""

import pytest

from vllm.models.deepseek_v32.attention import select_query_and_cache_form


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3"])
def test_per_tensor_fp8_packs_the_query_and_views_the_cache(
    kv_cache_dtype: str,
) -> None:
    assert select_query_and_cache_form(kv_cache_dtype, True) == (True, True)


def test_ds_mla_uses_the_bf16_tuple_and_raw_bytes() -> None:
    """FlashMLA dequantizes internally, so the query stays bf16."""
    assert select_query_and_cache_form("fp8_ds_mla", False) == (False, False)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16", "float16"])
def test_unquantized_cache_is_never_viewed_as_fp8(kv_cache_dtype: str) -> None:
    """The regression this guards: viewing a bf16 cache as fp8 reads garbage."""
    assert select_query_and_cache_form(kv_cache_dtype, True) == (False, False)


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3"])
def test_fp8_cache_on_a_bf16_query_backend_is_rejected(kv_cache_dtype: str) -> None:
    """Only the fp8_ds_mla layout pairs an fp8 cache with a bf16 query."""
    with pytest.raises(AssertionError, match="fp8_ds_mla"):
        select_query_and_cache_form(kv_cache_dtype, False)


def test_unquantized_cache_does_not_depend_on_backend_query_support() -> None:
    both = {select_query_and_cache_form("auto", s) for s in (True, False)}
    assert both == {(False, False)}


@pytest.mark.parametrize("kv_cache_dtype", ["fp8_e5m2", "fp8_inc", "nvfp4"])
def test_fp8_layouts_this_layer_cannot_address_are_rejected(
    kv_cache_dtype: str,
) -> None:
    """These count as quantized but are not e4m3, so reading them as e4m3
    would be silent corruption. A backend that allowed one should fail loudly."""
    with pytest.raises(AssertionError, match="cannot address"):
        select_query_and_cache_form(kv_cache_dtype, True)
