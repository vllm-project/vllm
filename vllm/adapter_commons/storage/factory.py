"""Factory for creating storage providers."""

import re
from typing import Dict, Type, Optional, List, Union
from urllib.parse import urlparse

from .base import (
    BaseStorageProvider,
    AdapterStorageProvider,
    AdapterValidationError,
)
from .config import (
    StorageConfig,
    LocalStorageConfig,
    S3StorageConfig,
)
from .providers.local import LocalStorageProvider
from .providers.s3 import S3StorageProvider

class StorageProviderFactory:
    """Factory for creating storage provider instances."""
    
    # Registry of available storage providers
    _provider_registry: Dict[str, Type[BaseStorageProvider]] = {
        "local": LocalStorageProvider,
        "s3": S3StorageProvider,
    }
    
    # Registry of URI schemes to provider types
    _scheme_registry: Dict[str, str] = {
        "file": "local",  # file:// -> local
        "s3": "s3",      # s3:// -> s3
    }
    
    @classmethod
    def register_provider(
        cls,
        provider_type: str,
        provider_class: Type[BaseStorageProvider],
        uri_schemes: Optional[List[str]] = None,
    ) -> None:
        """Register a new storage provider type.
        
        Args:
            provider_type: Unique identifier for the provider type
            provider_class: Provider class implementing BaseStorageProvider
            uri_schemes: Optional list of URI schemes to associate with provider
            
        Raises:
            ValueError: If provider type already registered or invalid
        """
        if provider_type in cls._provider_registry:
            raise ValueError(f"Provider type already registered: {provider_type}")
            
        if not issubclass(provider_class, BaseStorageProvider):
            raise ValueError(
                f"Provider class must implement BaseStorageProvider: {provider_class}"
            )
            
        cls._provider_registry[provider_type] = provider_class
        
        if uri_schemes:
            for scheme in uri_schemes:
                if scheme in cls._scheme_registry:
                    raise ValueError(f"URI scheme already registered: {scheme}")
                cls._scheme_registry[scheme] = provider_type
    
    @classmethod
    def create_provider(
        cls,
        config: Union[StorageConfig, LocalStorageConfig, S3StorageConfig]
    ) -> AdapterStorageProvider:
        """Create a storage provider instance from configuration.
        
        Args:
            config: Storage configuration object
            
        Returns:
            Configured storage provider instance
            
        Raises:
            ValueError: If provider type not registered or config invalid
        """
        provider_type = config.provider_type
        if provider_type not in cls._provider_registry:
            raise ValueError(
                f"Unknown storage provider type: {provider_type}. "
                f"Available types: {list(cls._provider_registry.keys())}"
            )
            
        provider_class = cls._provider_registry[provider_type]
        
        # Validate config type matches provider
        if provider_type == "local" and not isinstance(config, LocalStorageConfig):
            raise ValueError(
                f"Local storage provider requires LocalStorageConfig, got: {type(config)}"
            )
        elif provider_type == "s3" and not isinstance(config, S3StorageConfig):
            raise ValueError(
                f"S3 storage provider requires S3StorageConfig, got: {type(config)}"
            )
            
        return provider_class(config)
    
    @classmethod
    def create_provider_for_uri(
        cls,
        uri: str,
        config: Optional[Union[StorageConfig, LocalStorageConfig, S3StorageConfig]] = None
    ) -> AdapterStorageProvider:
        """Create a storage provider instance appropriate for a URI.
        
        Args:
            uri: Storage URI to create provider for
            config: Optional configuration to use instead of defaults
            
        Returns:
            Configured storage provider instance
            
        Raises:
            ValueError: If no provider supports the URI or config invalid
        """
        # Parse URI scheme
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower() if parsed.scheme else "file"
        
        # Get provider type for scheme
        provider_type = cls._scheme_registry.get(scheme)
        if not provider_type:
            raise ValueError(
                f"No provider registered for URI scheme: {scheme}. "
                f"Supported schemes: {list(cls._scheme_registry.keys())}"
            )
            
        # Use provided config or create default
        if config is None:
            if provider_type == "local":
                config = LocalStorageConfig()
            elif provider_type == "s3":
                config = S3StorageConfig()
            else:
                config = StorageConfig(provider_type=provider_type)
        elif config.provider_type != provider_type:
            raise ValueError(
                f"Config provider type {config.provider_type} does not match "
                f"URI scheme provider type {provider_type}"
            )
            
        return cls.create_provider(config)
    
    @classmethod
    def get_registered_providers(cls) -> List[str]:
        """Get list of registered provider types.
        
        Returns:
            List of registered provider type identifiers
        """
        return list(cls._provider_registry.keys())
    
    @classmethod
    def get_supported_schemes(cls) -> List[str]:
        """Get list of supported URI schemes.
        
        Returns:
            List of supported URI schemes
        """
        return list(cls._scheme_registry.keys()) 