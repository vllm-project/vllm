from ..utils import compare_two_settings


def test_cpu_offload():
    compare_two_settings("meta-llama/Llama-3.1-8B", [],
                         ["--cpu-offload-gb", "2"])


#
#
# def test_cpu_offload_gptq():
#     # Test GPTQ Marlin
#     compare_two_settings("Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4", [],
#                          ["--cpu-offload-gb", "1"],
#                          max_wait_seconds=480)
#     # Test GPTQ
#     compare_two_settings("Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
#                          ["--quantization", "gptq"],
#                          ["--quantization", "gptq", "--cpu-offload-gb", "1"],
#                          max_wait_seconds=480)
