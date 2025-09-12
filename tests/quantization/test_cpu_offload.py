# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Expanded quantized model tests for CPU offloading
# Base tests: tests/basic_correctness/test_cpu_offload.py

import pytest

from tests.quantization.utils import is_quant_method_supported

from ..utils import compare_two_settings


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="gptq_marlin is not supported on this GPU type.")
def test_cpu_offload_compressed_tensors(monkeypatch):
    # This quant method is sensitive to dummy weights, so we force real weights
    monkeypatch.setenv('VLLM_TEST_FORCE_LOAD_FORMAT', 'auto')
    # Test wNa16
    compare_two_settings("nm-testing/tinyllama-oneshot-w4a16-channel-v2",
                         ["--enforce-eager"],
                         ["--enforce-eager", "--cpu-offload-gb", "1"],
                         max_wait_seconds=480)
    # Test w4a16_marlin24
    compare_two_settings("nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t",
                         ["--enforce-eager"],
                         ["--enforce-eager", "--cpu-offload-gb", "1"],
                         max_wait_seconds=480)
    # Test w8a8
    compare_two_settings(
        "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change",
        ["--enforce-eager"], ["--enforce-eager", "--cpu-offload-gb", "1"],
        max_wait_seconds=480)
