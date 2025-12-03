# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""Base classes for model parameter offloading."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import torch.nn as nn

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import CacheConfig

logger = init_logger(__name__)

_SubmoduleAccessor = Callable[[nn.Module], nn.Module]
_WhitelistParamNamesCreator = Callable[[nn.Module], list[str]]


"""
class relation:

BaseOffloader (ABC)
  * implemented by: UVAOffloader
  * implemented by: OffloaderV2
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
        submodule_accessor: _SubmoduleAccessor | None = None,
        whitelist_param_names_creator: _WhitelistParamNamesCreator | None = None,
    ) -> list[nn.Module]:
        """Wrap modules with offloading logic.

        Args:
            modules_generator: Generator yielding modules to potentially offload.
            submodule_accessor: Optional function to extract a submodule from
                each module (e.g., lambda layer: layer.mlp.experts).
            whitelist_param_names_creator: Optional function to get parameter
                names to offload from a submodule (e.g., ["w13_weight", "w2_weight"]).

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


class NoopOffloader(BaseOffloader):
    """No-op offloader that returns modules as-is without any offloading."""

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        submodule_accessor: _SubmoduleAccessor | None = None,
        whitelist_param_names_creator: _WhitelistParamNamesCreator | None = None,
    ) -> list[nn.Module]:
        """Return modules unchanged."""
        return list(modules_generator)


# Global singleton offloader instance
_instance: BaseOffloader | None = NoopOffloader()


def get_offloader() -> BaseOffloader:
    """Get the global offloader instance."""
    assert _instance is not None, "Offloader instance is None"
    return _instance


def set_offloader(instance: BaseOffloader) -> None:
    """Set the global offloader instance."""
    global _instance
    _instance = instance


def create_offloader(cache_config: "CacheConfig") -> BaseOffloader:
    """Create an offloader based on the cache configuration.

    Priority: V2 offloading if configured, else UVA, else noop.
    """
    from vllm.model_executor.offloader.uva import UVAOffloader
    from vllm.model_executor.offloader.v2 import OffloaderV2

    if cache_config.offload_group_size > 0:
        # Use V2 offloading
        return OffloaderV2(
            group_size=cache_config.offload_group_size,
            num_in_group=cache_config.offload_num_in_group,
            prefetch_step=cache_config.offload_prefetch_step,
            mode="cpu",
        )
    elif cache_config.cpu_offload_gb > 0:
        # Use UVA offloading (legacy)
        return UVAOffloader(
            cpu_offload_max_bytes=int(cache_config.cpu_offload_gb * 1024**3)
        )
    else:
        # No offloading
        return NoopOffloader()
