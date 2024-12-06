from torch._dynamo.utils import compile_times

from vllm.config import CompilationConfig, CompilationLevel


def start_monitoring_torch_compile(compilation_config: CompilationConfig):
    pass


def end_monitoring_torch_compile(compilation_config: CompilationConfig):
    if compilation_config.level != CompilationLevel.PIECEWISE:
        print(f"{compile_times()=}")
        return

    print(f"{compilation_config.compilation_time=}")
