# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

import vllm.envs as envs
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
@dataclass
class ProfilerConfig:
    """Dataclass which contains profiler config for the engine."""

    profiler: ProfilerKind | None = None
    """Which profiler to use. Defaults to None. Options are:

    - 'torch': Use PyTorch profiler.\n
    - 'cuda': Use CUDA profiler."""

    torch_profiler_dir: str = ""
    """Directory to save torch profiler traces. Both AsyncLLM's CPU traces and
    worker's traces (CPU & GPU) will be saved under this directory. Note that
    it must be an absolute path."""

    torch_profiler_with_stack: bool = True
    """If `True`, enables stack tracing in the torch profiler. Enabled by default."""

    torch_profiler_with_flops: bool = False
    """If `True`, enables FLOPS counting in the torch profiler. Disabled by default."""

    torch_profiler_use_gzip: bool = True
    """If `True`, saves torch profiler traces in gzip format. Enabled by default"""

    torch_profiler_dump_cuda_time_total: bool = True
    """If `True`, dumps total CUDA time in torch profiler traces. Enabled by default."""

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

    delay_iterations: int = Field(default=0, ge=0)
    """Number of engine iterations to skip before starting profiling.
    Defaults to 0, meaning profiling starts immediately after receiving /start_profile.
    """

    max_iterations: int = Field(default=0, ge=0)
    """Maximum number of engine iterations to profile after starting profiling.
    Defaults to 0, meaning no limit.
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

    def _get_from_env_if_set(self, field_name: str, env_var_name: str) -> None:
        """Get field from env var if set, with deprecation warning."""

        if envs.is_set(env_var_name):
            value = getattr(envs, env_var_name)
            logger.warning_once(
                "Using %s environment variable is deprecated and will be removed in "
                "v0.14.0 or v1.0.0, whichever is soonest. Please use "
                "--profiler-config.%s command line argument or "
                "ProfilerConfig(%s=...) config field instead.",
                env_var_name,
                field_name,
                field_name,
            )
            return value
        return None

    def _set_from_env_if_set(
        self,
        field_name: str,
        env_var_name: str,
        to_bool: bool = True,
        to_int: bool = False,
    ) -> None:
        """Set field from env var if set, with deprecation warning."""
        value = self._get_from_env_if_set(field_name, env_var_name)
        if value is not None:
            if to_bool:
                value = value == "1"
            if to_int:
                value = int(value)
            setattr(self, field_name, value)

    @model_validator(mode="after")
    def _validate_profiler_config(self) -> Self:
        maybe_use_cuda_profiler = self._get_from_env_if_set(
            "profiler", "VLLM_TORCH_CUDA_PROFILE"
        )
        if maybe_use_cuda_profiler is not None:
            self.profiler = "cuda" if maybe_use_cuda_profiler == "1" else None
        else:
            self._set_from_env_if_set(
                "torch_profiler_dir", "VLLM_TORCH_PROFILER_DIR", to_bool=False
            )
            if self.torch_profiler_dir:
                self.profiler = "torch"
                self._set_from_env_if_set(
                    "torch_profiler_record_shapes",
                    "VLLM_TORCH_PROFILER_RECORD_SHAPES",
                )
                self._set_from_env_if_set(
                    "torch_profiler_with_memory",
                    "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY",
                )
                self._set_from_env_if_set(
                    "torch_profiler_with_stack",
                    "VLLM_TORCH_PROFILER_WITH_STACK",
                )
                self._set_from_env_if_set(
                    "torch_profiler_with_flops",
                    "VLLM_TORCH_PROFILER_WITH_FLOPS",
                )
                self._set_from_env_if_set(
                    "ignore_frontend",
                    "VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM",
                )
                self._set_from_env_if_set(
                    "torch_profiler_use_gzip",
                    "VLLM_TORCH_PROFILER_USE_GZIP",
                )
                self._set_from_env_if_set(
                    "torch_profiler_dump_cuda_time_total",
                    "VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL",
                )

        self._set_from_env_if_set(
            "delay_iterations", "VLLM_PROFILER_DELAY_ITERS", to_bool=False, to_int=True
        )
        self._set_from_env_if_set(
            "max_iterations", "VLLM_PROFILER_MAX_ITERS", to_bool=False, to_int=True
        )

        has_delay_or_limit = self.delay_iterations > 0 or self.max_iterations > 0
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
