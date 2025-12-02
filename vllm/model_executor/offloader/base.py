# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base classes for model parameter offloading."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator

import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# Type aliases for clarity
_SubmoduleAccessor = Callable[[nn.Module], nn.Module]
_WhitelistParamNamesCreator = Callable[[nn.Module], list[str]]


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
        pass

    @property
    def forbid_copy_engine_usage(self) -> bool:
        """Whether copy engine can be used (affects NCCL operations).

        Some offloading modes may conflict with CUDA copy engine usage
        in distributed operations.
        """
        return False


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
    logger.debug(f"{_instance=}")
    return _instance


def set_offloader(instance: BaseOffloader) -> None:
    """Set the global offloader instance."""
    global _instance
    _instance = instance
