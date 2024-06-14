"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_fp8.py --forked`.
"""
import pytest
import torch

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

if should_skip_test_group(group_name="TEST_QUANTIZATION"):
    pytest.skip("TEST_QUANTIZATION=DISABLE, skipping quantization test group",
                allow_module_level=True)

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]


@pytest.mark.skipif(
    capability < QUANTIZATION_METHODS["fp8"].get_min_capability(),
    reason="FP8 is not supported on this GPU type.")
def test_load_fp16_model(vllm_runner) -> None:
    with vllm_runner("facebook/opt-125m", quantization="fp8") as llm:

        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        fc1 = model.model.decoder.layers[0].fc1
        assert isinstance(fc1.quant_method, Fp8LinearMethod)
        assert fc1.weight.dtype == torch.float8_e4m3fn
