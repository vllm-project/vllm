# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

logger = init_logger(__name__)

ProfilerKind = Literal["torch", "cuda"]


def _is_uri_path(path: str) -> bool:
    """Check if path is a URI (scheme://...), excluding Windows drive letters.

    Supports custom URI schemes like gs://, s3://, hdfs://, etc.
    These paths should not be converted to absolute paths.
    """
    if "://" in path:
        scheme = path.split("://")[0]
        # Windows drive letters are single characters (e.g., C://)
        # Valid URI schemes have more than one character
        return len(scheme) > 1
    return False


@config
class ProfilerConfig:
    """Dataclass which contains profiler config for the engine."""

    profiler: ProfilerKind | None = None
    """Which profiler to use. Defaults to None. Options are:

    - 'torch': Use PyTorch profiler.
    - 'cuda': Use CUDA profiler."""

    torch_profiler_dir: str = ""
    """Directory to save torch profiler traces. Both AsyncLLM's CPU traces and
    worker's traces (CPU & GPU) will be saved under this directory. Note that
    it must be an absolute path."""

    torch_profiler_with_stack: bool = True
    """If `True`, enables stack tracing in the torch profiler. Enabled by default
    as it is useful for debugging. Can be disabled via 
    --profiler-config.torch_profiler_with_stack=false CLI flag."""

    torch_profiler_with_flops: bool = False
    """If `True`, enables FLOPS counting in the torch profiler. Disabled by default."""

    torch_profiler_use_gzip: bool = True
    """If `True`, saves torch profiler traces in gzip format. Enabled by default"""

    torch_profiler_dump_cuda_time_total: bool = True
    """If `True`, dumps total CUDA time in torch profiler traces. Enabled by default."""

    cuda_profiler_control_dp_rank: int = Field(default=0, ge=-1)
    """Data-parallel rank that controls CUDA profiler API start/stop calls.
    Set to -1 to let every data-parallel rank call the CUDA profiler API."""

    cuda_profiler_control_worker_rank: int = Field(default=0, ge=-1)
    """Worker rank, within the selected data-parallel rank, that controls CUDA
    profiler API start/stop calls. Set to -1 to let every worker in the
    selected data-parallel rank call the CUDA profiler API."""

    torch_profiler_record_shapes: bool = False
    """If `True`, records tensor shapes in the torch profiler. Disabled by default."""

    torch_profiler_with_memory: bool = False
    """If `True`, enables memory profiling in the torch profiler.
    Disabled by default."""

    ignore_frontend: bool = False
    """If `True`, disables the front-end profiling of AsyncLLM when using the
    'torch' profiler. This is needed to reduce overhead when using delay/limit options,
    since the front-end profiling does not track iterations and will capture the
    entire range.
    """

    delay_iterations: int | str = Field(default=0)
    """Number of engine iterations to skip before starting profiling.
    Defaults to 0, meaning profiling starts immediately after receiving /start_profile.

    May also be a comma-separated string (e.g. `"30,100"`) to define multiple
    profiling windows. Each value is paired positionally with `max_iterations`,
    and each value is measured from `/start_profile`.
    Multi-window mode is supported for `profiler='cuda'` and `profiler='torch'`.
    """

    max_iterations: int | str = Field(default=0)
    """Maximum number of engine iterations to profile after starting profiling.
    Defaults to 0, meaning no limit.

    May also be a comma-separated string (e.g. `"10,20"`), paired positionally
    with `delay_iterations`. In multi-window mode, each entry except the final
    one must be > 0. A final value of 0 means profile until `/stop_profile`.
    """

    warmup_iterations: int = Field(default=0, ge=0)
    """Number of warmup iterations for PyTorch profiler schedule.
    During warmup, the profiler runs but data is discarded. This helps reduce
    noise from JIT compilation and other one-time costs in the profiled trace.
    Defaults to 0 (schedule-based profiling disabled, recording all iterations).
    Set to a positive value (e.g., 2) to enable schedule-based profiling.
    """

    active_iterations: int = Field(default=5, ge=1)
    """Number of active iterations for PyTorch profiler schedule.
    This is the number of iterations where profiling data is actually collected.
    Defaults to 5 active iterations.
    """

    wait_iterations: int = Field(default=0, ge=0)
    """Number of wait iterations for PyTorch profiler schedule.
    During wait, the profiler is completely off with zero overhead.
    This allows skipping initial iterations before warmup begins.
    Defaults to 0 (no wait period).
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def get_iteration_windows(self) -> list[tuple[int, int]]:
        """Return profiling windows as a list of (delay, max) pairs.

        Normalizes scalar `delay_iterations`/`max_iterations` to single-element
        lists so callers can iterate uniformly.
        """
        delays = self._parse_iteration_values("delay_iterations", self.delay_iterations)
        maxes = self._parse_iteration_values("max_iterations", self.max_iterations)
        if len(delays) != len(maxes):
            raise ValueError(
                "delay_iterations and max_iterations must have the same number "
                f"of values, got {len(delays)} and {len(maxes)}"
            )
        return list(zip(delays, maxes))

    @staticmethod
    def _parse_iteration_values(field_name: str, value: int | str) -> list[int]:
        if isinstance(value, int):
            return [value]
        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be an integer or comma-separated integers"
            )

        parts = [part.strip() for part in value.split(",")]
        if not parts or any(part == "" for part in parts):
            raise ValueError(
                f"{field_name} must be an integer or comma-separated integers"
            )

        try:
            return [int(part) for part in parts]
        except ValueError as e:
            raise ValueError(
                f"{field_name} must be an integer or comma-separated integers"
            ) from e

    def _validate_multi_window_config(self, windows: list[tuple[int, int]]) -> None:
        if self.profiler not in ("cuda", "torch"):
            raise ValueError(
                "Multiple profiling windows are only supported when "
                f"profiler is 'cuda' or 'torch', got profiler={self.profiler!r}"
            )

        for i, (delay, max_iters) in enumerate(windows):
            is_last_window = i == len(windows) - 1
            if max_iters == 0 and not is_last_window:
                raise ValueError(
                    f"max_iterations[{i}] must be > 0 in multi-window mode "
                    "(0 means unlimited, which is incompatible with later "
                    "windows)"
                )
            if i > 0:
                previous_delay, previous_max_iters = windows[i - 1]
                previous_end = previous_delay + previous_max_iters
                if delay < previous_end:
                    raise ValueError(
                        "Profiling windows must not overlap: window "
                        f"{i - 1} ends at iteration {previous_end} but window "
                        f"{i} starts at iteration {delay}"
                    )

    @model_validator(mode="after")
    def _validate_profiler_config(self) -> Self:
        windows = self.get_iteration_windows()

        if any(delay < 0 for delay, _ in windows):
            raise ValueError("delay_iterations values must be non-negative")
        if any(max_iters < 0 for _, max_iters in windows):
            raise ValueError("max_iterations values must be non-negative")

        if len(windows) > 1:
            self._validate_multi_window_config(windows)

        has_delay_or_limit = any(
            delay > 0 or max_iters > 0 for delay, max_iters in windows
        )
        if self.profiler == "torch" and has_delay_or_limit and not self.ignore_frontend:
            logger.warning_once(
                "Using 'torch' profiler with delay_iterations or max_iterations "
                "while ignore_frontend is False may result in high overhead."
            )

        profiler_dir = self.torch_profiler_dir
        if profiler_dir and self.profiler != "torch":
            raise ValueError(
                "torch_profiler_dir is only applicable when profiler is set to 'torch'"
            )
        if self.profiler == "torch" and not profiler_dir:
            raise ValueError("torch_profiler_dir must be set when profiler is 'torch'")

        # Support any URI scheme (gs://, s3://, hdfs://, etc.)
        # These paths should not be converted to absolute paths
        if profiler_dir and not _is_uri_path(profiler_dir):
            self.torch_profiler_dir = os.path.abspath(os.path.expanduser(profiler_dir))

        return self
