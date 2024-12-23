from ..utils import compare_two_settings


def test_cpu_offload():
    compare_two_settings("meta-llama/Llama-3.2-1B", [],
                         ["--cpu-offload-gb", "1"])
