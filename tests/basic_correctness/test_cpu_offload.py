import pytest

from tests.quantization.utils import is_quant_method_supported

from ..utils import compare_two_settings


def test_cpu_offload():
    compare_two_settings("meta-llama/Llama-2-7b-hf", [],
                         ["--cpu-offload-gb", "4"])


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="fp8 is not supported on this GPU type.")
def test_cpu_offload_fp8():
    compare_two_settings("meta-llama/Meta-Llama-3-8B-Instruct",
                         ["--quantization", "fp8"],
                         ["--quantization", "fp8", "--cpu-offload-gb", "2"])
    compare_two_settings("neuralmagic/Meta-Llama-3-8B-Instruct-FP8", [],
                         ["--cpu-offload-gb", "2"])


@pytest.mark.skipif(not is_quant_method_supported("awq"),
                    reason="awq is not supported on this GPU type.")
def test_cpu_offload_awq():
    compare_two_settings("casperhansen/llama-3-8b-instruct-awq", [],
                         ["--cpu-offload-gb", "2"])


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin_24"),
                    reason="gptq_marlin_24 is not supported on this GPU type.")
def test_cpu_offload_gptq_marlin_24():
    compare_two_settings("nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t",
                         [], ["--cpu-offload-gb", "1"])


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason="bitsandbytes is not supported on this GPU type.")
def test_cpu_offload_bitsandbytes():
    compare_two_settings("lllyasviel/omost-llama-3-8b-4bits", [],
                         ["--cpu-offload-gb", "2"])
