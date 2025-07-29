# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Expanded quantized model tests for CPU offloading
# Base tests: tests/basic_correctness/test_cpu_offload.py

import pytest

from tests.quantization.utils import is_quant_method_supported

from ..utils import compare_two_settings


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="fp8 is not supported on this GPU type.")
def test_cpu_offload_fp8():
    # Test quantization of an unquantized checkpoint
    compare_two_settings("meta-llama/Llama-3.2-1B-Instruct",
                         ["--quantization", "fp8"],
                         ["--quantization", "fp8", "--cpu-offload-gb", "1"],
                         max_wait_seconds=480)
    # Test loading a quantized checkpoint
    compare_two_settings("neuralmagic/Qwen2-1.5B-Instruct-FP8", [],
                         ["--cpu-offload-gb", "1"],
                         max_wait_seconds=480)


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="gptq_marlin is not supported on this GPU type.")
def test_cpu_offload_gptq(monkeypatch):
    # This quant method is sensitive to dummy weights, so we force real weights
    monkeypatch.setenv('VLLM_TEST_FORCE_LOAD_FORMAT', 'auto')
    # Test GPTQ Marlin
    compare_two_settings("Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4", [],
                         ["--cpu-offload-gb", "1"],
                         max_wait_seconds=480)
    # Test GPTQ
    compare_two_settings("Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
                         ["--quantization", "gptq"],
                         ["--quantization", "gptq", "--cpu-offload-gb", "1"],
                         max_wait_seconds=480)


@pytest.mark.skipif(not is_quant_method_supported("awq_marlin"),
                    reason="awq_marlin is not supported on this GPU type.")
def test_cpu_offload_awq(monkeypatch):
    # This quant method is sensitive to dummy weights, so we force real weights
    monkeypatch.setenv('VLLM_TEST_FORCE_LOAD_FORMAT', 'auto')
    # Test AWQ Marlin
    compare_two_settings("Qwen/Qwen2-1.5B-Instruct-AWQ", [],
                         ["--cpu-offload-gb", "1"],
                         max_wait_seconds=480)
    # Test AWQ
    compare_two_settings("Qwen/Qwen2-1.5B-Instruct-AWQ",
                         ["--quantization", "awq"],
                         ["--quantization", "awq", "--cpu-offload-gb", "1"],
                         max_wait_seconds=480)


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="gptq_marlin is not supported on this GPU type.")
def test_cpu_offload_compressed_tensors(monkeypatch):
    # This quant method is sensitive to dummy weights, so we force real weights
    monkeypatch.setenv('VLLM_TEST_FORCE_LOAD_FORMAT', 'auto')
    # Test wNa16
    compare_two_settings("nm-testing/tinyllama-oneshot-w4a16-channel-v2", [],
                         ["--cpu-offload-gb", "1"],
                         max_wait_seconds=480)
    # Test w4a16_marlin24
    compare_two_settings("nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t",
                         [], ["--cpu-offload-gb", "1"],
                         max_wait_seconds=480)
    # Test w8a8
    compare_two_settings(
        "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change", [],
        ["--cpu-offload-gb", "1"],
        max_wait_seconds=480)
