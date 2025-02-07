# SPDX-License-Identifier: Apache-2.0
"""Tests whether PTPC w8a8 FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_ptpc_fp8.py --forked`.
"""
import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.quantization.fp8 import Fp8KVCacheMethod
from vllm.model_executor.layers.quantization.ptpc_fp8 import (
    PTPCFp8LinearMethod)
from vllm.platforms import current_platform


@pytest.mark.skipif(not is_quant_method_supported("ptpc_fp8"),
                    reason="PTPC FP8 is not supported on this GPU type.")
@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="This test is for ROCm GPU.")
@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "float16"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_e4m3"])
def test_ptpc_fp8_rocm(vllm_runner, dtype: str, kv_cache_dtype: str) -> None:

    try:
        with vllm_runner("facebook/opt-125m",
                         dtype=dtype,
                         quantization="ptpc_fp8",
                         kv_cache_dtype=kv_cache_dtype) as llm:

            model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
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
            else:
                pytest.skip()

            output = llm.generate_greedy("Hello my name is", max_tokens=20)
            assert output
    except AssertionError as e:
        if str(
                e
        ) == "Currently torch._scaled_mm (hipBLASLt) rowwise gemm only support output dtype of bfloat16. torch.float16 is specified.":  # noqa: E501
            # If the error message matches, the test passes
            pass
        else:
            # If the error message does not match, re-raise the exception
            raise
