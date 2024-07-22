from vllm.utils import is_hip

from ..utils import compare_two_settings


def test_cpu_offload():
    compare_two_settings("meta-llama/Llama-2-7b-hf", [],
                         ["--cpu-offload-gb", "4"])
    if not is_hip():
        # compressed-tensors quantization is currently not supported in ROCm.
        compare_two_settings(
            "nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t", [],
            ["--cpu-offload-gb", "1"])
