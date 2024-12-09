import time

from vllm.config import CompilationConfig, CompilationLevel
from vllm.logger import init_logger

logger = init_logger(__name__)

torch_compile_start_time: float = 0.0


def start_monitoring_torch_compile(compilation_config: CompilationConfig):
    global torch_compile_start_time
    torch_compile_start_time = time.time()


def end_monitoring_torch_compile(compilation_config: CompilationConfig):
    if compilation_config.level == CompilationLevel.PIECEWISE:
        logger.info("torch.compile takes %.2f s in total",
                    compilation_config.compilation_time)
