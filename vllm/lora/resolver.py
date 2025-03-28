# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, AbstractSet, Type, Optional

from vllm.logger import init_logger
from vllm.entrypoints.openai.protocol import LoadLoRAAdapterRequest

logger = init_logger(__name__)

class LoRAResolver(ABC):
    """Base class for LoRA adapter resolvers.
    
    This class defines the interface for resolving and fetching LoRA adapters.
    Implementations of this class should handle the logic for locating and 
    downloading LoRA adapters from various sources (e.g., local filesystem, 
    cloud storage, etc.).
    """

    @abstractmethod
    async def resolve_lora(
        self,
        lora_name: str
    ) -> Optional[LoadLoRAAdapterRequest]:
        """Abstract method to resolve and fetch a LoRA model adapter.
        
        This method should implement the logic to locate and download LoRA 
        adapter based on the provided name. Implementations might fetch from 
        a blob storage or other sources.
         
        Args:
            lora_name: str - The name or identifier of the LoRA model to 
                resolve.
        
        Returns:
            Optional[LoadLoRAAdapterRequest]: A LoadLoRAAdapterRequest object
                containing the resolved LoRA model information, or None if 
                the LoRA model cannot be found.
        """
        pass


@dataclass
class _LoRAResolverRegistry:
    resolvers: Dict[str, Type[LoRAResolver]] = field(default_factory=dict)

    def get_supported_resolvers(self) -> AbstractSet[str]:
        """Get all registered resolver names."""
        return self.resolvers.keys()

    def register_resolver(
        self,
        resolver_name: str,
        resolver_cls: Type[LoRAResolver],
    ) -> None:
        """Register a LoRA resolver.
        Args:
            resolver_name: Name to register the resolver under.
            resolver_cls: The LoRA resolver class to register.
        """
        if not isinstance(resolver_name, str):
            raise TypeError(
                f"`resolver_name` should be a string, not a {type(resolver_name)}")

        if resolver_name in self.resolvers:
            logger.warning(
                "LoRA resolver %s is already registered, and will be "
                "overwritten by the new resolver class %s.",
                resolver_name,
                resolver_cls)

        if not (isinstance(resolver_cls, type) and 
                issubclass(resolver_cls, LoRAResolver)):
            raise TypeError(
                f"`resolver_cls` should be a LoRAResolver subclass, "
                f"not a {type(resolver_cls)}")

        self.resolvers[resolver_name] = resolver_cls


    def get_resolver(self, resolver_name: str) -> Type[LoRAResolver]:
        """Get a registered resolver class by name.
        Args:
            resolver_name: Name of the resolver to get.
        Returns:
            The resolver class.
        Raises:
            KeyError: If the resolver is not found in the registry.
        """
        if resolver_name not in self.resolvers:
            raise KeyError(f"LoRA resolver '{resolver_name}' not found. "
                         f"Available resolvers: {list(self.resolvers.keys())}")
        return self.resolvers[resolver_name]


LoRAResolverRegistry = _LoRAResolverRegistry()