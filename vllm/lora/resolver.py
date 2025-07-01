# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Set
from dataclasses import dataclass, field
from typing import Optional

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


class LoRAResolver(ABC):
    """Base class for LoRA adapter resolvers.

    This class defines the interface for resolving and fetching LoRA adapters.
    Implementations of this class should handle the logic for locating and
    downloading LoRA adapters from various sources (e.g. S3, cloud storage,
    etc.).
    """

    @abstractmethod
    async def resolve_lora(self, base_model_name: str,
                           lora_name: str) -> Optional[LoRARequest]:
        """Abstract method to resolve and fetch a LoRA model adapter.

        Implements logic to locate and download LoRA adapter based on the name.
        Implementations might fetch from a blob storage or other sources.

        Args:
            base_model_name: The name/identifier of the base model to resolve.
            lora_name: The name/identifier of the LoRA model to resolve.

        Returns:
            Optional[LoRARequest]: The resolved LoRA model information, or None
            if the LoRA model cannot be found.
        """
        pass


@dataclass
class _LoRAResolverRegistry:
    resolvers: dict[str, LoRAResolver] = field(default_factory=dict)

    def get_supported_resolvers(self) -> Set[str]:
        """Get all registered resolver names."""
        return self.resolvers.keys()

    def register_resolver(
        self,
        resolver_name: str,
        resolver: LoRAResolver,
    ) -> None:
        """Register a LoRA resolver.
        Args:
            resolver_name: Name to register the resolver under.
            resolver: The LoRA resolver instance to register.
        """
        if resolver_name in self.resolvers:
            logger.warning(
                "LoRA resolver %s is already registered, and will be "
                "overwritten by the new resolver instance %s.", resolver_name,
                resolver)

        self.resolvers[resolver_name] = resolver

    def get_resolver(self, resolver_name: str) -> LoRAResolver:
        """Get a registered resolver instance by name.
        Args:
            resolver_name: Name of the resolver to get.
        Returns:
            The resolver instance.
        Raises:
            KeyError: If the resolver is not found in the registry.
        """
        if resolver_name not in self.resolvers:
            raise KeyError(
                f"LoRA resolver '{resolver_name}' not found. "
                f"Available resolvers: {list(self.resolvers.keys())}")
        return self.resolvers[resolver_name]


LoRAResolverRegistry = _LoRAResolverRegistry()
