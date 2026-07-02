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
        self._windows: list[tuple[int, int]] = profiler_config.get_iteration_windows()
        logger.info_once(
            "Profiling configured for following windows (start_iter, n_iters): %s",
            str(self._windows),
        )

        self._current_window = 0

        # Track when the profiler gets triggered by start_profile
        self._active_iteration_count = 0
        self._active = False

        # Track active-profiling iters within the current window
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
            logger.info_once("Profiler stopped successfully.")
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
        if self._current_window < len(self._windows):
            delay, _ = self._windows[self._current_window]
            if delay == 0:
                self._call_start()

    def step(self) -> None:
        """Update the profiler state at each worker step,
        to handle delayed starts and max iteration limits across one or more
        profiling windows."""
        if not self._active or self._current_window >= len(self._windows):
            return

        self._active_iteration_count += 1

        delay, max_iters = self._windows[self._current_window]

        if not self._running and self._active_iteration_count == delay:
            logger.info_once(
                "Starting profiler for window %d (delay=%d, max_iters=%d)...",
                self._current_window,
                delay,
                max_iters,
            )
            self._call_start()

        # Call profiler step for schedule-based profiling.
        # Only count iterations where data is actually recorded (not warmup).
        if self._running and self._profiler_step():
            self._profiling_for_iters += 1

        if max_iters > 0 and self._running and self._profiling_for_iters == max_iters:
            logger.info_once(
                "Profiling window %d complete (%d iters). Stopping profiler...",
                self._current_window,
                max_iters,
            )
            self._call_stop()

            # Reset and move to next window
            self._profiling_for_iters = 0
            self._current_window += 1

    def _profiler_step(self) -> bool:
        """Called each step when profiler is running.
        Override in subclasses to handle schedule-based profiling.

        Returns:
            True if the step was an active profiling step (data recorded),
            False if the step was a warmup step (data discarded).
        """
        return True

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
            self._trace_handler = on_trace_ready
        else:
            self._trace_handler = torch.profiler.tensorboard_trace_handler(
                torch_profiler_trace_dir,
                worker_name=worker_name,
                use_gzip=profiler_config.torch_profiler_use_gzip,
            )

        self.dump_cpu_time_total = "CPU" in activities and len(activities) == 1
        self._activities = [
            TorchProfilerActivityMap[activity] for activity in activities
        ]

        # Create profiler schedule if warmup or wait iterations are configured
        self._uses_schedule = (
            profiler_config.warmup_iterations > 0 or profiler_config.wait_iterations > 0
        )
        if self._uses_schedule and local_rank in (None, 0):
            logger.info_once(
                "Profiler schedule configured: wait=%d, warmup=%d, active=%d",
                profiler_config.wait_iterations,
                profiler_config.warmup_iterations,
                profiler_config.active_iterations,
            )

        self._warmup_iterations = profiler_config.warmup_iterations
        # Subtract 1 because profiler.start() already consumes step 0
        # (WAIT or WARMUP), so only wait + warmup - 1 non-active steps
        # remain to be advanced through via profiler.step() calls.
        self._warmup_steps_per_window = max(
            profiler_config.wait_iterations + profiler_config.warmup_iterations - 1,
            0,
        )
        self._warmup_steps_remaining = self._warmup_steps_per_window
        self.profiler: torch.profiler.profile | None = None

    def _create_profiler_schedule(
        self,
    ) -> Callable[[int], torch.profiler.ProfilerAction] | None:
        if not self._uses_schedule:
            return None
        profiler_config = self.profiler_config
        return torch.profiler.schedule(
            skip_first=0,
            wait=profiler_config.wait_iterations,
            warmup=profiler_config.warmup_iterations,
            active=profiler_config.active_iterations,
            repeat=1,
        )

    def _create_profiler(self) -> torch.profiler.profile:
        profiler_config = self.profiler_config
        return torch.profiler.profile(
            activities=self._activities,
            schedule=self._create_profiler_schedule(),
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=self._trace_handler,
        )

    def _build_profiler_table(
        self,
        sort_key: str,
        row_limit: int | None = None,
    ) -> str:
        assert self.profiler is not None
        if row_limit is None:  # use profiler default row limit of 100
            return self.profiler.key_averages().table(sort_by=sort_key)
        return self.profiler.key_averages().table(
            sort_by=sort_key,
            row_limit=row_limit,
        )

    def _write_profiler_table(self, rank: int, table: str) -> None:
        profiler_dir = self.profiler_config.torch_profiler_dir

        # Skip file write for URI paths (gs://, s3://, etc.)
        # as standard file I/O doesn't work with URI schemes
        if not _is_uri_path(profiler_dir):
            window_suffix = ""
            if len(self._windows) > 1:
                window_suffix = f"_window_{self._current_window}"
            profiler_out_file = f"{profiler_dir}/profiler_out_{rank}"
            profiler_out_file += f"{window_suffix}.txt"
            with open(profiler_out_file, "w") as f:
                print(table, file=f)

    @override
    def _start(self) -> None:
        assert self.profiler is None
        profiler = self._create_profiler()
        self.profiler = profiler
        self._warmup_steps_remaining = self._warmup_steps_per_window
        try:
            profiler.start()
        except Exception:
            self.profiler = None
            raise

    @override
    def _stop(self) -> None:
        assert self.profiler is not None
        try:
            self.profiler.stop()

            profiler_config = self.profiler_config
            rank = self.local_rank
            if profiler_config.torch_profiler_dump_cuda_time_total:
                table = self._build_profiler_table(sort_key="self_cuda_time_total")
                self._write_profiler_table(rank, table)

                # only print profiler results on rank 0
                if rank == 0:
                    print(table)

            if self.dump_cpu_time_total:
                table = self._build_profiler_table(
                    sort_key="self_cpu_time_total", row_limit=50
                )
                self._write_profiler_table(rank, table)

                # only print profiler results on rank 0
                if rank == 0:
                    print(table)
        finally:
            self.profiler = None

    @override
    def _profiler_step(self) -> bool:
        """Call profiler.step() when using schedule-based profiling.

        Returns:
            True if the step was an active profiling step (data recorded),
            False if the step was a warmup step (data discarded).
        """
        if self._uses_schedule:
            assert self.profiler is not None
            self.profiler.step()
            # Track warmup steps - only count active steps toward max_iterations
            if self._warmup_steps_remaining > 0:
                self._warmup_steps_remaining -= 1
                return False
        return True

    @override
    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)


class CudaProfilerWrapper(WorkerProfiler):
    def __init__(
        self,
        profiler_config: ProfilerConfig,
        enable_cuda_profiler_api: bool = True,
    ) -> None:
        super().__init__(profiler_config)
        self.enable_cuda_profiler_api = enable_cuda_profiler_api
        # Note: lazy import to avoid dependency issues if CUDA is not available.
        import torch.cuda.profiler as cuda_profiler

        self._cuda_profiler = cuda_profiler

    @override
    def _start(self) -> None:
        if self.enable_cuda_profiler_api:
            self._cuda_profiler.start()

    @override
    def _stop(self) -> None:
        if self.enable_cuda_profiler_api:
            self._cuda_profiler.stop()

    @override
    def annotate_context_manager(self, name: str):
        return torch.cuda.nvtx.range(name)
