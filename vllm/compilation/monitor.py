# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

from vllm.config import CompilationConfig, CompilationLevel, VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

context_manager = None
torch_compile_start_time: float = 0.0


def start_monitoring_torch_compile(vllm_config: VllmConfig):
    global torch_compile_start_time
    torch_compile_start_time = time.time()

    compilation_config: CompilationConfig = vllm_config.compilation_config
    if compilation_config.level == CompilationLevel.PIECEWISE and \
        compilation_config.debug_dump_path:
        import depyf
        path = os.path.join(compilation_config.debug_dump_path,
                            f"rank_{vllm_config.parallel_config.rank}")
        global context_manager
        context_manager = depyf.prepare_debug(path)
        context_manager.__enter__()


def end_monitoring_torch_compile(vllm_config: VllmConfig):
    compilation_config: CompilationConfig = vllm_config.compilation_config
    if compilation_config.level == CompilationLevel.PIECEWISE:
        logger.info("torch.compile takes %.2f s in total",
                    compilation_config.compilation_time)
        global context_manager
        if context_manager is not None:
            context_manager.__exit__(None, None, None)
            context_manager = None


cudagraph_capturing_enabled: bool = True


def validate_cudagraph_capturing_enabled():
    # used to monitor whether an cudagraph capturing is legal at runtime.
    # should be called before any cudagraph capturing.
    # if an illegal cudagraph capturing happens, raise an error.
    global cudagraph_capturing_enabled
    if not cudagraph_capturing_enabled:
        raise RuntimeError("CUDA graph capturing detected at an inappropriate "
                           "time. This operation is currently disabled.")


def set_cudagraph_capturing_enabled(enabled: bool):
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled
