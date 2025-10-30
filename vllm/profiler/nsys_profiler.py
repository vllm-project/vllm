# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch.cuda.profiler as cuda_profiler

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class NsysIterationProfiler:
    """Controls Nsight Systems capture via CUDA profiler API.

    This helper encapsulates iteration-based start/stop control so callers can
    simply call `maybe_profile_now()` in their main loop and `shutdown()` when
    exiting. All validation and logging are handled internally.

    Args:
        start_iter: Iteration index to start capture (inclusive). -1 disables.
        stop_iter: Iteration index to stop capture (inclusive). -1 disables.

    Behavior:
        - If `start_iter` and `stop_iter` are valid, capture starts exactly
          when `current_iter == start_iter` and stops when
          `current_iter == stop_iter`.
        - If invalid values are provided (e.g., negative start or
          stop <= start), the profiler remains disabled and logs a single
          warning.
    """

    start_iter: int = -1
    stop_iter: int = -1

    _profiler_running: bool = False
    _current_iter: int = 0

    @staticmethod
    def from_env_string(env_value: str) -> NsysIterationProfiler:
        """Construct from env value formatted as "<start>-<stop>".

        Any invalid value disables the profiler and logs a warning once.
        """
        if env_value == "None":
            profiler = NsysIterationProfiler()
            profiler.maybe_profile_now = profiler._noop
            return profiler

        try:
            env_value = env_value.strip()
            if "-" not in env_value:
                profiler = NsysIterationProfiler()
                profiler.maybe_profile_now = profiler._noop
                return profiler
            parts = env_value.split("-")
            if len(parts) != 2:
                raise ValueError("Expected format 'start-stop'.")
            start_iter, stop_iter = map(int, parts)
            if start_iter < 0 or stop_iter <= start_iter:
                raise ValueError(
                    "Start must be non-negative and stop must be greater than start."
                )
            logger.info_once(
                "NSYS profiling will start at iteration %d and stop at iteration %d",
                start_iter,
                stop_iter,
            )
            return NsysIterationProfiler(start_iter=start_iter, stop_iter=stop_iter)
        except Exception as exc:  # noqa: BLE001 - preserve original reason
            logger.warning_once(
                "Invalid VLLM_NSYS_PROFILE_START_STOP value: '%s'. "
                "Reason: %s. Disabling profiling.",
                env_value,
                exc,
            )
            profiler = NsysIterationProfiler()
            profiler.maybe_profile_now = profiler._noop
            return profiler

    def maybe_profile_now(self) -> None:
        """Start/stop CUDA profiler based on internal iteration counter."""
        if self._current_iter == self.start_iter:
            logger.info_once("Starting NSYS profiler")
            try:
                cuda_profiler.start()
            except Exception as exc:  # noqa: BLE001
                logger.warning_once("Failed to start NSYS profiler: %s", exc)
            else:
                self._profiler_running = True

        if self._current_iter == self.stop_iter:
            logger.info_once("Stopping NSYS profiler")
            try:
                cuda_profiler.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning_once("Failed to stop NSYS profiler: %s", exc)
            else:
                self._profiler_running = False

        # Increment at the end of the iteration check
        self._current_iter += 1

    # ---- internals ----
    def _noop(self) -> None:
        return None

    def shutdown(self) -> None:
        """Ensure profiler is stopped when shutting down."""
        if self._profiler_running:
            logger.info_once("Stopping NSYS profiler")
            try:
                cuda_profiler.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning_once(
                    "Failed to stop NSYS profiler during shutdown: %s", exc
                )
            finally:
                self._profiler_running = False
