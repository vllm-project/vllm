from contextlib import contextmanager, nullcontext
from vllm.config import VllmConfig, CompilationLevel
from torch._dynamo.utils import cumulative_time_spent_ns

@contextmanager
def monitor_torch_compile(vllm_config: VllmConfig):
    if vllm_config.level != CompilationLevel.PIECEWISE:
        yield
        print(f"{cumulative_time_spent_ns=}")
        return
    
    yield
    print(f"{vllm_config.compilation_config.compilation_time=}")
