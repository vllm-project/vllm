# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""Base classes for model parameter offloading."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch.nn as nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import OffloadConfig

logger = init_logger(__name__)

# Tracks deprecated offload env vars we have already warned about, so the
# deprecation warning is emitted at most once per process.
_warned_deprecated_offload_envs: set[str] = set()


def resolve_offload_flag(config_value: bool | None, env_name: str) -> bool:
    """Resolve a weight-offloading boolean from config, with env-var fallback.

    The config value takes precedence. When it is ``None`` (unset), fall back to
    the deprecated environment variable, warning once if the user enabled it.

    Args:
        config_value: The value from ``OffloadConfig`` (``None`` if unset).
        env_name: Name of the deprecated env var to fall back to.

    Returns:
        The resolved boolean flag.
    """
    if config_value is not None:
        return config_value
    env_value = bool(getattr(envs, env_name))
    if env_value and env_name not in _warned_deprecated_offload_envs:
        _warned_deprecated_offload_envs.add(env_name)
        logger.warning(
            "%s is deprecated and will be removed in a future release. Use the "
            "offload config (e.g. `--offload-config`) instead.",
            env_name,
        )
    return env_value


def should_pin_memory() -> bool:
    """Check if pinned memory should be used for weight offloading.

    Combines the platform capability check with the user override from
    ``OffloadConfig.disable_pin_memory`` (falling back to the deprecated
    ``VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY`` env var). On unified-memory
    systems (e.g. GH200) pinned memory eats into GPU memory.
    """
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    config_value = (
        vllm_config.offload_config.disable_pin_memory
        if vllm_config is not None
        else None
    )
    disable_pin_memory = resolve_offload_flag(
        config_value, "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY"
    )
    return is_pin_memory_available() and not disable_pin_memory


"""
class relation:

BaseOffloader (ABC)
  * implemented by: UVAOffloader
  * implemented by: PrefetchOffloader
    * uses: _ModuleOffloader
        * uses: _BaseParamOffloader (ABC)
            * implemented by: _CpuParamOffloader
"""


class BaseOffloader(ABC):
    """Base class for model parameter offloading strategies.

    Offloaders control how model parameters are stored and loaded during
    inference. Different strategies trade memory for compute/transfer time.
    """

    @abstractmethod
    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with offloading logic.

        Args:
            modules_generator: Generator yielding modules to potentially offload.

        Returns:
            List of modules, potentially with offloading hooks installed.
        """
        pass

    def post_init(self):
        """Called after model construction completes.

        Offloaders can use this to:
        - Finalize parameter storage
        - Start initial prefetching
        - Allocate shared resources
        """
        return

    def sync_prev_onload(self) -> None:  # noqa: B027
        """Sync previous onload operations. Override in subclasses."""
        pass

    def join_after_forward(self) -> None:  # noqa: B027
        """Join streams after forward. Override in subclasses."""
        pass

    def _wait_for_layer(self, layer_idx: int) -> None:  # noqa: B027
        """Wait for layer prefetch. Override in subclasses."""
        pass

    def _start_prefetch(self, layer_idx: int) -> None:  # noqa: B027
        """Start layer prefetch. Override in subclasses."""
        pass


class NoopOffloader(BaseOffloader):
    """No-op offloader that returns modules as-is without any offloading."""

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Return modules unchanged."""
        return list(modules_generator)


# Global singleton offloader instance (defaults to no-op).
_instance: BaseOffloader = NoopOffloader()


def get_offloader() -> BaseOffloader:
    """Get the global offloader instance."""
    return _instance


def set_offloader(instance: BaseOffloader) -> None:
    """Set the global offloader instance."""
    global _instance
    _instance = instance
    if isinstance(instance, NoopOffloader):
        logger.debug_once("Offloader set to NoopOffloader (no offloading).")
    else:
        logger.info_once("Offloader set to %s", type(instance).__name__)


def create_offloader(offload_config: "OffloadConfig") -> BaseOffloader:
    """Create an offloader based on the offload configuration.

    Uses the explicit ``offload_backend`` selector.  When set to ``"auto"``,
    selects prefetch if ``offload_group_size > 0``, UVA if
    ``cpu_offload_gb > 0``, otherwise noop.
    """
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    backend = offload_config.offload_backend
    uva = offload_config.uva
    prefetch = offload_config.prefetch

    if backend == "auto":
        if prefetch.offload_group_size > 0:
            backend = "prefetch"
        elif uva.cpu_offload_gb > 0:
            backend = "uva"
        else:
            return NoopOffloader()

    if backend == "prefetch":
        return PrefetchOffloader(
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            prefetch_step=prefetch.offload_prefetch_step,
            offload_params=prefetch.offload_params,
            mode="cpu",
        )
    elif backend == "uva":
        return UVAOffloader(
            cpu_offload_max_bytes=int(uva.cpu_offload_gb * 1024**3),
            cpu_offload_params=uva.cpu_offload_params,
        )
    else:
        return NoopOffloader()
