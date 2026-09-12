# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import time
from collections.abc import Generator

import torch

from vllm.config import CompilationMode, VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Shared global so backends.py can read the start time for Dynamo timing.
torch_compile_start_time: float = 0.0

# Peak torch-allocated bytes recorded before compilation started. Compilation
# and Inductor autotuning allocate large temporaries inside profile_run(),
# which runs inside the memory-profiling window. They are freed before the
# first real forward, so leaving them in the allocator high-water mark charges
# them as activation headroom and shrinks the KV cache on a cold start.
# Clearing the counter alone would also drop a peak legitimately recorded
# earlier in the same window, because a multimodal encoder pass runs before the
# backbone compiles, so that value is kept here for memory_profiling() to fold
# back in.
peak_memory_before_compile: int = 0


def _peak_allocated_bytes() -> int:
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        return 0

    device = torch.device(current_platform.current_device())
    return torch.accelerator.memory_stats(device).get("allocated_bytes.all.peak", 0)


def _discard_compilation_peak(peak_before_compile: int) -> None:
    """Drop the high-water mark left by compilation, keeping any earlier peak."""
    global peak_memory_before_compile

    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        return

    peak_memory_before_compile = max(peak_memory_before_compile, peak_before_compile)
    torch.accelerator.reset_peak_memory_stats(
        torch.device(current_platform.current_device())
    )


@contextlib.contextmanager
def monitor_torch_compile(
    vllm_config: VllmConfig,
    message: str = "torch.compile took %.2f s in total",
    is_encoder: bool = False,
) -> Generator[None, None, None]:
    """Context manager that times torch.compile and manages depyf debugging.

    On normal exit: logs the compile time and exits depyf.
    On exception: cleans up depyf without logging (compilation failed).
    """
    global torch_compile_start_time
    torch_compile_start_time = time.perf_counter()
    peak_before_compile = _peak_allocated_bytes()

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
        _discard_compilation_peak(peak_before_compile)
        total_compile_time = time.perf_counter() - torch_compile_start_time
        if compilation_config.mode == CompilationMode.VLLM_COMPILE:
            if is_encoder:
                compilation_config.encoder_compilation_time += total_compile_time
            else:
                compilation_config.compilation_time += total_compile_time
            logger.info_once(message, total_compile_time)
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
