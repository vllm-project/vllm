from torch._dynamo.utils import compile_times

from vllm.config import CompilationLevel, VllmConfig


def start_monitoring_torch_compile(vllm_config: VllmConfig):
    pass


def end_monitoring_torch_compile(vllm_config: VllmConfig):
    if vllm_config.level != CompilationLevel.PIECEWISE:
        print(f"{compile_times()=}")
        return

    print(f"{vllm_config.compilation_config.compilation_time=}")
