# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether PTPC w8a8 FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_ptpc_fp8.py --forked`.
"""

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.quantization.fp8 import Fp8KVCacheMethod
from vllm.model_executor.layers.quantization.ptpc_fp8 import PTPCFp8LinearMethod
from vllm.platforms import current_platform

UNSUPPORTED_STR = (
    "Currently torch._scaled_mm (hipBLASLt) rowwise gemm only "
    "support output dtype of bfloat16. torch.float16 is specified."
)


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@pytest.mark.skipif(
    not is_quant_method_supported("ptpc_fp8"),
    reason="PTPC FP8 is not supported on this GPU type.",
)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="This test is for ROCm GPU.")
@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "float16"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_e4m3"])
def test_ptpc_fp8_rocm(vllm_runner, dtype: str, kv_cache_dtype: str) -> None:
    try:
        llm = vllm_runner(
            "facebook/opt-125m",
            dtype=dtype,
            quantization="ptpc_fp8",
            enforce_eager=True,
            kv_cache_dtype=kv_cache_dtype,
        )
    except AssertionError as e:
        if str(e) == UNSUPPORTED_STR:
            # If the error message matches, the test passes
            return
        else:
            # If the error message does not match, re-raise the exception
            raise

    with llm:

        def check_model(model):
            fc1 = model.model.decoder.layers[0].fc1
            assert isinstance(fc1.quant_method, PTPCFp8LinearMethod)
            if kv_cache_dtype == "ptpc_fp8":
                attn = model.model.decoder.layers[0].self_attn.attn
                assert isinstance(attn.quant_method, Fp8KVCacheMethod)
                assert attn._k_scale == 1.0
                assert attn._v_scale == 1.0

            if current_platform.has_device_capability(94):
                # For GPUs with hardware support, we keep weights in fp8
                assert fc1.weight.dtype == torch.float8_e4m3fnuz

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=4)
        assert output
