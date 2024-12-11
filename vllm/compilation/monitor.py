from vllm.config import CompilationConfig, CompilationLevel
from vllm.logger import init_logger

logger = init_logger(__name__)


def start_monitoring_torch_compile(compilation_config: CompilationConfig):
    pass


def end_monitoring_torch_compile(compilation_config: CompilationConfig):
    if compilation_config.level == CompilationLevel.PIECEWISE:
        logger.info("graph compilation takes %.2f s in total",
                    compilation_config.compilation_time)
