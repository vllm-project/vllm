# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from typing import Literal

import torch
from typing_extensions import override

import vllm.version
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
            logger.info_once("Starting profiler after delay...")
            self._call_start()

        # Call profiler step for schedule-based profiling
        # Only count iterations where data is actually recorded (not warmup)
        if self._running and self._profiler_step():
            self._profiling_for_iters += 1

        if (
            self._max_iters > 0
            and self._running
            and self._profiling_for_iters > self._max_iters
        ):
            # Automatically stop the profiler after max iters
            # will be marked as not running, but leave as active so that stop
            # can clean up properly
            logger.info_once("Max profiling iterations reached. Stopping profiler...")
            self._call_stop()
            return

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


TorchProfilerActivity = Literal["CPU", "CUDA", "PrivateUse1", "XPU"]
TorchProfilerActivityMap = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "CUDA": torch.profiler.ProfilerActivity.CUDA,
    "PrivateUse1": torch.profiler.ProfilerActivity.PrivateUse1,
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
            trace_handler = on_trace_ready
        else:
            trace_handler = torch.profiler.tensorboard_trace_handler(
                torch_profiler_trace_dir,
                worker_name=worker_name,
                use_gzip=profiler_config.torch_profiler_use_gzip,
            )

        def _async_trace_ready(prof: torch.profiler.profile) -> None:
            """Runs the trace export (JSON serialization + optional gzip) on a
            background thread instead of inline on whatever called
            profiler.step()/stop() -- normally this worker's own request-
            handling loop.

            By the time this fires, Kineto has already stopped collecting and
            holds a complete, immutable snapshot of the trace in memory (the
            CUDA-side stop happens synchronously just before this callback, as
            part of the same profiler.stop()/step() call) -- so none of this
            touches CUDA or the model-execution stream anymore, and it's safe
            to run concurrently with whatever this worker does next.

            This matters because export_chrome_trace() with gzip enabled
            writes the full uncompressed JSON to a temp file, rereads it, and
            recompresses it with single-threaded stdlib gzip -- for a
            multi-hundred-MB-to-GB trace this can take tens of seconds, during
            which (without this) the worker cannot schedule its next step.
            """

            def run() -> None:
                try:
                    trace_handler(prof)
                except Exception as e:
                    logger.warning("Failed to export profiler trace: %s", e)

            threading.Thread(
                target=run, name="vllm-profiler-trace-export", daemon=True
            ).start()

        self.dump_cpu_time_total = "CPU" in activities and len(activities) == 1

        # Create profiler schedule if warmup or wait iterations are configured
        profiler_schedule = None
        if profiler_config.warmup_iterations > 0 or profiler_config.wait_iterations > 0:
            profiler_schedule = torch.profiler.schedule(
                skip_first=0,
                wait=profiler_config.wait_iterations,
                warmup=profiler_config.warmup_iterations,
                active=profiler_config.active_iterations,
                repeat=1,
            )
            if local_rank in (None, 0):
                logger.info_once(
                    "Profiler schedule configured: wait=%d, warmup=%d, active=%d",
                    profiler_config.wait_iterations,
                    profiler_config.warmup_iterations,
                    profiler_config.active_iterations,
                )

        self.profiler = torch.profiler.profile(
            activities=[TorchProfilerActivityMap[activity] for activity in activities],
            schedule=profiler_schedule,
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=_async_trace_ready,
        )

        # Track if we're using a schedule (need to call step())
        self._uses_schedule = profiler_schedule is not None
        self._warmup_iterations = profiler_config.warmup_iterations
        # Subtract 1 because profiler.start() already consumes step 0
        # (WAIT or WARMUP), so only wait + warmup - 1 non-active steps
        # remain to be advanced through via profiler.step() calls.
        self._warmup_steps_remaining = max(
            profiler_config.wait_iterations + profiler_config.warmup_iterations - 1,
            0,
        )
        self._version_metadata_added = False

    def _build_profiler_table(
        self,
        sort_key: str,
        row_limit: int | None = None,
    ) -> str:
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
            profiler_out_file = f"{profiler_dir}/profiler_out_{rank}.txt"
            with open(profiler_out_file, "w") as f:
                print(table, file=f)

    def _maybe_add_version_metadata(self) -> None:
        """Stamp the vLLM version (which embeds the git commit) into the trace.

        add_metadata_json is a no-op until Kineto is initialized, which with a
        schedule only happens after the WAIT phase, so stamp once it's live.
        """
        if self._version_metadata_added:
            return
        # None while the schedule is still in the WAIT phase.
        if self.profiler.profiler is None:
            return
        try:
            self.profiler.add_metadata_json(
                "vllm_version", json.dumps(vllm.version.__version__)
            )
            self.profiler.add_metadata_json(
                "vllm_version_tuple",
                json.dumps([str(p) for p in vllm.version.__version_tuple__]),
            )
        except Exception as e:
            logger.warning("Failed to add vLLM version to profiler metadata: %s", e)
        # Mark done even on failure, to avoid retrying every step.
        self._version_metadata_added = True

    @override
    def _start(self) -> None:
        self.profiler.start()
        # No-schedule case: Kineto is live immediately. With a schedule this
        # no-ops and _profiler_step stamps it once WAIT ends.
        self._maybe_add_version_metadata()

    @override
    def _stop(self) -> None:
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

    @override
    def _profiler_step(self) -> bool:
        """Call profiler.step() when using schedule-based profiling.

        Returns:
            True if the step was an active profiling step (data recorded),
            False if the step was a warmup step (data discarded).
        """
        if self._uses_schedule:
            self.profiler.step()
            # Stamp once the schedule leaves WAIT and Kineto is live.
            self._maybe_add_version_metadata()
            # Track warmup steps - only count active steps toward max_iterations
            if self._warmup_steps_remaining > 0:
                self._warmup_steps_remaining -= 1
                return False
        return True

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
