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


def should_pin_memory() -> bool:
    """Check if pinned memory should be used for weight offloading.

    Combines the platform capability check with the user override env var.
    On unified-memory systems (e.g. GH200) pinned memory eats into GPU
    memory, so users can disable it via VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY.
    """
    return (
        is_pin_memory_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
    )


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

    supports_tower_offload: bool = False
    """Whether `wrap_modules` also accepts modules routed by
    `SupportsMultiModal._mark_tower_model`, outside the `make_layers` call.

    Offloaders whose `wrap_modules` may only be called on the decoder layer
    stack (e.g. `PrefetchOffloader`, which schedules prefetches over a
    circular layer stack) must keep this `False`.
    """

    @abstractmethod
    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
    ) -> list[nn.Module]:
        """Wrap modules with offloading logic.

        Args:
            modules_generator: Generator yielding modules to potentially offload.
            prefix: Name prefix prepended to parameter names before matching
                them against the offloading parameter set. Used when the
                modules are not the full model, so that name segments stay
                fully qualified (e.g. `visual` for a tower module).

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
        prefix: str = "",
    ) -> list[nn.Module]:
        """Return modules unchanged."""
        return list(modules_generator)


# Global singleton offloader instance (defaults to no-op).
_instance: BaseOffloader = NoopOffloader()


def get_offloader() -> BaseOffloader:
    """Get the global offloader instance."""
    return _instance


def set_offloader(
    instance: BaseOffloader,
    offload_config: "OffloadConfig | None" = None,
) -> None:
    """Set the global offloader instance.

    Args:
        instance: The offloader to install as the global singleton.
        offload_config: The config `instance` was derived from. Only used to
            detect and warn about the case where a `NoopOffloader` is
            selected despite a non-zero offload budget having been
            requested, which can only mean the offload configuration was
            lost or reset somewhere before this call.
    """
    global _instance
    _instance = instance
    if isinstance(instance, NoopOffloader):
        requested_cpu_offload_gb = (
            offload_config.uva.cpu_offload_gb if offload_config is not None else 0
        )
        requested_group_size = (
            offload_config.prefetch.offload_group_size
            if offload_config is not None
            else 0
        )
        if requested_cpu_offload_gb > 0 or requested_group_size > 0:
            logger.warning_once(
                "Offloader set to NoopOffloader, but cpu_offload_gb=%s and "
                "offload_group_size=%s were requested (offload_backend=%r). "
                "No CPU offloading will occur. The offload configuration was "
                "likely lost or reset before reaching create_offloader().",
                requested_cpu_offload_gb,
                requested_group_size,
                offload_config.offload_backend,
            )
        else:
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

    requested_backend = offload_config.offload_backend
    backend = requested_backend
    uva = offload_config.uva
    prefetch = offload_config.prefetch

    # An explicitly requested (non-"auto") backend handed a zero budget for
    # that backend can only mean the offload configuration was lost or reset
    # before reaching this call: there is no legitimate reason to force a
    # specific backend and then configure it with nothing to offload.
    if requested_backend == "uva" and uva.cpu_offload_gb <= 0:
        logger.warning_once(
            "offload_backend=%r was requested but uva.cpu_offload_gb=%s. "
            "No CPU offloading will occur. This combination can only happen "
            "if the offload configuration was lost or reset before reaching "
            "create_offloader().",
            requested_backend,
            uva.cpu_offload_gb,
        )
    elif requested_backend == "prefetch" and prefetch.offload_group_size <= 0:
        logger.warning_once(
            "offload_backend=%r was requested but "
            "prefetch.offload_group_size=%s. No CPU offloading will occur. "
            "This combination can only happen if the offload configuration "
            "was lost or reset before reaching create_offloader().",
            requested_backend,
            prefetch.offload_group_size,
        )

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
