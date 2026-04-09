# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import time
from collections.abc import Generator

from vllm.config import CompilationMode, VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Shared global so backends.py can read the start time for Dynamo timing.
torch_compile_start_time: float = 0.0


@contextlib.contextmanager
def monitor_torch_compile(
    vllm_config: VllmConfig,
    message: str = "torch.compile took %.2f s in total",
) -> Generator[None, None, None]:
    """Context manager that times torch.compile and manages depyf debugging.

    On normal exit: logs the compile time and exits depyf.
    On exception: cleans up depyf without logging (compilation failed).
    """
    global torch_compile_start_time
    torch_compile_start_time = time.perf_counter()

    compilation_config = vllm_config.compilation_config
    depyf_cm = None
    path = vllm_config.compile_debug_dump_path()
    if compilation_config.mode == CompilationMode.VLLM_COMPILE and path:
        import depyf

        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Dumping depyf output to %s", path)
        depyf_cm = depyf.prepare_debug(path.as_posix())
        depyf_cm.__enter__()

    try:
        yield
    except Exception:
        raise
    else:
        total_compile_time = time.perf_counter() - torch_compile_start_time
        if compilation_config.mode == CompilationMode.VLLM_COMPILE:
            logger.info_once(message, total_compile_time, scope="local")
    finally:
        if depyf_cm is not None:
            try:
                depyf_cm.__exit__(None, None, None)
            except Exception:
                logger.warning("Exception during depyf cleanup.", exc_info=True)


@contextlib.contextmanager
def monitor_profiling_run() -> Generator[None, None, None]:
    """Context manager that times the initial profiling run.

    Asserts that no backend compilation occurs during the profiling run
    (all compilation should have completed before this point).
    """
    from vllm.compilation.counter import compilation_counter

    backend_compilations_before = compilation_counter.num_backend_compilations
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert (
        compilation_counter.num_backend_compilations == backend_compilations_before
    ), (
        "backend compilation occurred during the initial profiling run; "
        "all compilation should be complete before the profiling run starts."
    )
    logger.info_once(
        "Initial profiling/warmup run took %.2f s",
        elapsed,
        scope="local",
    )


cudagraph_capturing_enabled: bool = True


def validate_cudagraph_capturing_enabled() -> None:
    # used to monitor whether a cudagraph capturing is legal at runtime.
    # should be called before any cudagraph capturing.
    # if an illegal cudagraph capturing happens, raise an error.
    global cudagraph_capturing_enabled
    if not cudagraph_capturing_enabled:
        raise RuntimeError(
            "CUDA graph capturing detected at an inappropriate "
            "time. This operation is currently disabled."
        )


def set_cudagraph_capturing_enabled(enabled: bool) -> None:
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled
