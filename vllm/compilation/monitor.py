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

# Allocator peak from before compilation. memory_profiling() adds it back
# after the compile peak is reset, so the KV cache size ignores the compile.
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
    message: str = "%storch.compile took %.2f s in total",
    is_encoder: bool = False,
    tag: str = "",
    announce_start: bool = True,
    discard_peak: bool = True,
) -> Generator[None, None, None]:
    """Context manager that times torch.compile and manages depyf debugging.

    On normal exit: logs the compile time and exits depyf.
    On exception: cleans up depyf without logging (compilation failed).

    `announce_start=False` suppresses the "Starting torch.compile" line;
    used by the AOT-load probe so a failed load attempt doesn't announce a
    compile it didn't perform. `discard_peak=False` keeps the allocator peak,
    for blocks that also run a real forward.
    """
    global torch_compile_start_time
    torch_compile_start_time = time.perf_counter()
    peak_before_compile = _peak_allocated_bytes()

    compilation_config = vllm_config.compilation_config
    log_prefix = f"[{tag}] " if tag else ""
    if announce_start and compilation_config.mode == CompilationMode.VLLM_COMPILE:
        logger.info("%sStarting torch.compile", log_prefix)
    depyf_cm = None
    path = vllm_config.compile_debug_dump_path()
    if compilation_config.mode == CompilationMode.VLLM_COMPILE and path:
        from vllm.compilation import depyf_compat

        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Dumping depyf output to %s", path)
        depyf_cm = depyf_compat.prepare_debug(path.as_posix())
        depyf_cm.__enter__()

    try:
        yield
    except Exception:
        raise
    else:
        if discard_peak:
            _discard_compilation_peak(peak_before_compile)
        total_compile_time = time.perf_counter() - torch_compile_start_time
        if compilation_config.mode == CompilationMode.VLLM_COMPILE:
            if is_encoder:
                compilation_config.encoder_compilation_time += total_compile_time
            else:
                compilation_config.compilation_time += total_compile_time
            logger.info(message, log_prefix, total_compile_time)
    finally:
        if depyf_cm is not None:
            try:
                depyf_cm.__exit__(None, None, None)
            except Exception:
                logger.warning("Exception during depyf cleanup.", exc_info=True)


@contextlib.contextmanager
def monitor_profiling_run(tag: str = "") -> Generator[None, None, None]:
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
    log_prefix = f"[{tag}] " if tag else ""
    logger.info(
        "%sInitial profiling/warmup run took %.2f s",
        log_prefix,
        elapsed,
    )
