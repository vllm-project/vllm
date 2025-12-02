# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Literal

from pydantic import model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

logger = init_logger(__name__)

ProfilerKind = Literal["none", "torch", "cuda"]


@config
@dataclass
class ProfilerConfig:
    """Dataclass which contains profiler config for the engine."""

    profiler: ProfilerKind = "none"
    """Which profiler to use. Defaults to 'none'. Options are:
    - 'none': No profiling.
    - 'torch': Use PyTorch profiler.
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

    delay_iterations: int = 0
    """Number of engine iterations to skip before starting profiling.
    Defaults to 0, meaning profiling starts immediately after receiving /start_profile.
    """

    max_iterations: int = 0
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

    @model_validator(mode="after")
    def _validate_profiler_config(self) -> Self:
        if envs.VLLM_TORCH_CUDA_PROFILE is not None:
            self.profiler = "cuda" if envs.VLLM_TORCH_CUDA_PROFILE == "1" else "none"
            logger.warning_once(
                "Environment variable VLLM_TORCH_CUDA_PROFILE is deprecated. "
                "Please use profiler_config.profiler instead."
            )
        elif envs.VLLM_TORCH_PROFILER_DIR is not None:
            self.profiler = "torch"
            self.torch_profiler_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.warning_once(
                "Environment variable VLLM_TORCH_PROFILER_DIR is deprecated. "
                "Please use profiler_config.torch_profiler_dir instead."
            )

            if envs.VLLM_TORCH_PROFILER_RECORD_SHAPES is not None:
                self.torch_profiler_record_shapes = (
                    envs.VLLM_TORCH_PROFILER_RECORD_SHAPES == "1"
                )
            if envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY is not None:
                self.torch_profiler_with_memory = (
                    envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY == "1"
                )
            if envs.VLLM_TORCH_PROFILER_WITH_STACK is not None:
                self.torch_profiler_with_stack = (
                    envs.VLLM_TORCH_PROFILER_WITH_STACK == "1"
                )
            if envs.VLLM_TORCH_PROFILER_WITH_FLOPS is not None:
                self.torch_profiler_with_flops = (
                    envs.VLLM_TORCH_PROFILER_WITH_FLOPS == "1"
                )
            if envs.VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM is not None:
                self.ignore_frontend = envs.VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM == "1"
            if envs.VLLM_TORCH_PROFILER_USE_GZIP is not None:
                self.torch_profiler_use_gzip = envs.VLLM_TORCH_PROFILER_USE_GZIP == "1"
            if envs.VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL is not None:
                self.torch_profiler_dump_cuda_time_total = (
                    envs.VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL == "1"
                )

        if envs.VLLM_PROFILER_DELAY_ITERS is not None:
            self.delay_iterations = int(envs.VLLM_PROFILER_DELAY_ITERS)
            logger.warning_once(
                "Environment variable VLLM_PROFILER_DELAY_ITERS is deprecated. "
                "Please use profiler_config.delay_iterations instead."
            )
        if envs.VLLM_PROFILER_MAX_ITERS is not None:
            self.max_iterations = int(envs.VLLM_PROFILER_MAX_ITERS)
            logger.warning_once(
                "Environment variable VLLM_PROFILER_MAX_ITERS is deprecated. "
                "Please use profiler_config.max_iterations instead."
            )

        if self.profiler not in ("none", "torch", "cuda"):
            raise ValueError(
                f"Invalid profiler: {self.profiler} "
                f"(choose from 'none', 'torch', 'cuda')"
            )
        if self.delay_iterations < 0:
            raise ValueError("Profiler delay_iterations must be >= 0")
        if self.max_iterations < 0:
            raise ValueError("Profiler max_iterations must be >= 0")

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

        if profiler_dir:
            is_gs_path = (
                profiler_dir.startswith("gs://")
                and profiler_dir[5:]
                and profiler_dir[5] != "/"
            )
            if not is_gs_path:
                self.torch_profiler_dir = os.path.abspath(
                    os.path.expanduser(profiler_dir)
                )

        return self
