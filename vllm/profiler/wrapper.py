# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from typing import Literal

import torch
from typing_extensions import override

from vllm.config import ProfilerConfig
from vllm.config.profiler import _is_uri_path
from vllm.logger import init_logger

logger = init_logger(__name__)


class WorkerProfiler(ABC):
    def __init__(self, profiler_config: ProfilerConfig) -> None:
        self._delay_iters = profiler_config.delay_iterations
        if self._delay_iters > 0:
            logger.info_once(
                "GPU profiling will start "
                f"{self._delay_iters} steps after start_profile."
            )

        self._max_iters = profiler_config.max_iterations
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
            logger.info_once("Profiler stopped successfully.", scope="local")
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
            logger.info_once("Starting profiler after delay...", scope="local")
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
            logger.info_once(
                "Max profiling iterations reached. Stopping profiler...", scope="local"
            )
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
        logger.info_once("Shutting down profiler", scope="local")
        if self._running:
            self.stop()

    def annotate_context_manager(self, name: str):
        """Return a context manager to annotate profiler traces."""
        return nullcontext()


TorchProfilerActivity = Literal["CPU", "CUDA", "XPU"]
TorchProfilerActivityMap = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "CUDA": torch.profiler.ProfilerActivity.CUDA,
    "XPU": torch.profiler.ProfilerActivity.XPU,
}


class TorchProfilerWrapper(WorkerProfiler):
    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity],
        on_trace_ready: Callable[[torch.profiler.profile], None] | None = None,
    ) -> None:
        super().__init__(profiler_config)

        self.local_rank = local_rank
        self.profiler_config = profiler_config
        torch_profiler_trace_dir = profiler_config.torch_profiler_dir
        if local_rank in (None, 0):
            logger.info_once(
                "Torch profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
                scope="local",
            )
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                profiler_config.torch_profiler_record_shapes,
                profiler_config.torch_profiler_with_memory,
                profiler_config.torch_profiler_with_stack,
                profiler_config.torch_profiler_with_flops,
            )

        # Determine trace handler: use custom handler if provided,
        # otherwise default to tensorboard trace handler
        if on_trace_ready is not None:
            trace_handler = on_trace_ready
        else:
            trace_handler = torch.profiler.tensorboard_trace_handler(
                torch_profiler_trace_dir,
                worker_name=worker_name,
                use_gzip=profiler_config.torch_profiler_use_gzip,
            )

        self.dump_cpu_time_total = "CPU" in activities and len(activities) == 1
        self.profiler = torch.profiler.profile(
            activities=[TorchProfilerActivityMap[activity] for activity in activities],
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=trace_handler,
        )

    @override
    def _start(self) -> None:
        self.profiler.start()

    @override
    def _stop(self) -> None:
        self.profiler.stop()

        profiler_config = self.profiler_config
        rank = self.local_rank
        if profiler_config.torch_profiler_dump_cuda_time_total:
            profiler_dir = profiler_config.torch_profiler_dir
            sort_key = "self_cuda_time_total"
            table = self.profiler.key_averages().table(sort_by=sort_key)

            # Skip file write for URI paths (gs://, s3://, etc.)
            # as standard file I/O doesn't work with URI schemes
            if not _is_uri_path(profiler_dir):
                profiler_out_file = f"{profiler_dir}/profiler_out_{rank}.txt"
                with open(profiler_out_file, "w") as f:
                    print(table, file=f)

            # only print profiler results on rank 0
            if rank == 0:
                print(table)
        if self.dump_cpu_time_total and rank == 0:
            logger.info(
                self.profiler.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=50
                )
            )

    @override
    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)


class CudaProfilerWrapper(WorkerProfiler):
    def __init__(self, profiler_config: ProfilerConfig) -> None:
        super().__init__(profiler_config)
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
