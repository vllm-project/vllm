# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An explicit unquantized KV cache dtype must match the model dtype.

`get_fp8_kv_cache_data_type` maps "auto", "float16" and "bfloat16" alike to
`Fp8KVCacheDataType::kAuto`, and that branch of `DISPATCH_BY_KV_CACHE_DTYPE` only
instantiates the cache type equal to the source type. No instantiation exists in which the
two differ, so the cross-dtype copy the kernel implements is unreachable and the cache
receives the model dtype's bits, read back as the requested dtype.

Measured on a bfloat16 model with `--kv-cache-dtype float16`: the cache holds
0.1621 -> 1.537, 188 -> 3.617, 436 -> 3.926 -- each the bit reinterpretation, matching to
four significant figures -- and the model emits fluent nonsense with no error and no
non-finite value.

The quantized dtypes are unaffected: their branches instantiate `OutT != InT` and convert.
"""

import pytest

from vllm.engine.arg_utils import EngineArgs

MODEL = "hmellor/tiny-random-LlamaForCausalLM"


def _build(model_dtype: str, cache_dtype: str):
    return EngineArgs(
        model=MODEL, dtype=model_dtype, kv_cache_dtype=cache_dtype, max_model_len=128
    ).create_engine_config()


@pytest.mark.parametrize(
    "model_dtype, cache_dtype",
    [("bfloat16", "float16"), ("float16", "bfloat16")],
)
def test_mismatched_unquantized_cache_dtype_is_rejected(model_dtype, cache_dtype):
    """The combination that silently reinterprets bits must not start."""
    with pytest.raises(ValueError, match="cannot be used with a"):
        _build(model_dtype, cache_dtype)


@pytest.mark.parametrize(
    "model_dtype, cache_dtype",
    [
        ("bfloat16", "auto"),
        ("float16", "auto"),
        # naming the model's own dtype is just `auto` spelled out
        ("bfloat16", "bfloat16"),
        ("float16", "float16"),
        # quantized caches do convert, and must keep working
        ("bfloat16", "fp8"),
        ("bfloat16", "fp8_e4m3"),
    ],
)
def test_supported_combinations_still_build(model_dtype, cache_dtype):
    """Everything that has a real conversion, or needs none, is untouched.

    These are the cases a user actually reaches -- `auto` and the fp8 family -- so they
    carry the weight of showing the guard is narrow rather than a blanket refusal.
    """
    config = _build(model_dtype, cache_dtype)
    assert config.cache_config.cache_dtype == cache_dtype
