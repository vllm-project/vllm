# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An explicit unquantized KV cache dtype must match the model dtype.

The C++ cache writers have no float16 <-> bfloat16 conversion, so a mismatched pair
would store the model dtype's bits and read them back as the requested dtype.
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
    """Everything that has a real conversion, or needs none, is untouched."""
    config = _build(model_dtype, cache_dtype)
    assert config.cache_config.cache_dtype == cache_dtype
