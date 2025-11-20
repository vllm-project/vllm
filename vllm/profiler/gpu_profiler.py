# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
from typing_extensions import override

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


class WorkerProfiler(ABC):
    def __init__(self) -> None:
        self._delay_iters = envs.VLLM_PROFILER_DELAY_ITERS
        if self._delay_iters > 0:
            logger.info_once(
                "GPU profiling will start "
                f"{self._delay_iters} steps after start_profile."
            )

        self._max_iters = envs.VLLM_PROFILER_MAX_ITERS
        if self._max_iters > 0:
            logger.info_once(
                "GPU profiling will stop "
                f"after {self._max_iters} worker steps, "
                "or when stop_profile is received."
            )

        # Track when the profiler gets triggered by start_profile
        self._active_iteration_count = 0
        self._active = False

        # Track when the profiler is actually running
        self._profiling_for_iters = 0
        self._running = False

    @abstractmethod
    def _start(self) -> None:
        """Start the profiler."""
        pass

    @abstractmethod
    def _stop(self) -> None:
        """Stop the profiler."""
        pass

    def _call_start(self) -> None:
        """Call _start with error handling but no safeguards."""
        try:
            self._start()
            self._running = True  # Only mark as running if start succeeds
        except Exception as e:
            logger.warning("Failed to start profiler: %s", e)

    def _call_stop(self) -> None:
        """Call _stop with error handling but no safeguards."""
        try:
            self._stop()
            logger.info("Profiler stopped successfully.")
        except Exception as e:
            logger.warning("Failed to stop profiler: %s", e)
        self._running = False  # Always mark as not running, assume stop worked

    def start(self) -> None:
        """Attempt to start the profiler, accounting for delayed starts."""
        if self._active:
            logger.debug(
                "start_profile received when profiler is already active. "
                "Ignoring request."
            )
            return
        self._active = True
        if self._delay_iters == 0:
            self._call_start()

    def step(self) -> None:
        """Update the profiler state at each worker step,
        to handle delayed starts and max iteration limits."""
        if not self._active:
            return

        self._active_iteration_count += 1

        if (
            not self._running
            and self._delay_iters > 0
            and self._active_iteration_count == self._delay_iters
        ):
            logger.info("Starting profiler after delay...")
            self._call_start()

        if self._running:
            self._profiling_for_iters += 1

        if (
            self._max_iters > 0
            and self._running
            and self._profiling_for_iters > self._max_iters
        ):
            # Automatically stop the profiler after max iters
            # will be marked as not running, but leave as active so that stop
            # can clean up properly
            logger.info("Max profiling iterations reached. Stopping profiler...")
            self._call_stop()
            return

    def stop(self) -> None:
        """Attempt to stop the profiler, accounting for overlapped calls."""
        if not self._active:
            logger.debug(
                "stop_profile received when profiler is not active. Ignoring request."
            )
            return
        self._active = False
        self._active_iteration_count = 0
        self._profiling_for_iters = 0

        if self._running:
            self._call_stop()

    def shutdown(self) -> None:
        """Ensure profiler is stopped when shutting down."""
        logger.info_once("Shutting down profiler")
        if self._running:
            self.stop()

    def annotate_context_manager(self, name: str):
        """Return a context manager to annotate profiler traces."""
        return nullcontext()


class TorchProfilerWrapper(WorkerProfiler):
    def __init__(self, worker_name: str, local_rank: int) -> None:
        super().__init__()

        self.local_rank = local_rank
        torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
        logger.info(
            "Torch profiling enabled. Traces will be saved to: %s",
            torch_profiler_trace_dir,
        )
        logger.debug(
            "Profiler config: record_shapes=%s,"
            "profile_memory=%s,with_stack=%s,with_flops=%s",
            envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
            envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
            envs.VLLM_TORCH_PROFILER_WITH_STACK,
            envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
        )
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
            profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
            with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
            with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                torch_profiler_trace_dir, worker_name=worker_name, use_gzip=True
            ),
        )

    @override
    def _start(self) -> None:
        self.profiler.start()

    @override
    def _stop(self) -> None:
        self.profiler.stop()

        rank = self.local_rank
        profiler_dir = envs.VLLM_TORCH_PROFILER_DIR
        profiler_out_file = f"{profiler_dir}/profiler_out_{rank}.txt"
        sort_key = "self_cuda_time_total"
        table = self.profiler.key_averages().table(sort_by=sort_key)

        with open(profiler_out_file, "w") as f:
            print(table, file=f)

        # only print profiler results on rank 0
        if rank == 0:
            print(table)

    @override
    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)


class CudaProfilerWrapper(WorkerProfiler):
    def __init__(self) -> None:
        super().__init__()
        # Note: lazy import to avoid dependency issues if CUDA is not available.
        import torch.cuda.profiler as cuda_profiler

        self._cuda_profiler = cuda_profiler

    @override
    def _start(self) -> None:
        self._cuda_profiler.start()

    @override
    def _stop(self) -> None:
        self._cuda_profiler.stop()

    @override
    def annotate_context_manager(self, name: str):
        return torch.cuda.nvtx.range(name)
