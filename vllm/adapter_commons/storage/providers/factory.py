"""Factory for creating storage providers."""

import re
from typing import Type, Dict, Union

from ..base import AdapterStorageProvider
from ..config import StorageConfig, LocalStorageConfig, S3StorageConfig
from .local import LocalStorageProvider
from .s3 import S3StorageProvider

class StorageProviderFactory:
    """Factory for creating storage provider instances."""
    
    # URI patterns for different storage types
    URI_PATTERNS = {
        "local": re.compile(r"^(/|\.|\~).*$"),
        "s3": re.compile(r"^s3://.*$"),
    }
    
    # Mapping of storage types to provider classes
    PROVIDERS: Dict[str, Type[AdapterStorageProvider]] = {
        "local": LocalStorageProvider,
        "s3": S3StorageProvider,
    }
    
    @classmethod
    def create_provider(
        cls,
        config: Union[StorageConfig, LocalStorageConfig, S3StorageConfig]
    ) -> AdapterStorageProvider:
        """Create a storage provider instance.
        
        Args:
            config: Storage configuration
            
        Returns:
            Configured storage provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        provider_type = cls._get_provider_type(config)
        provider_class = cls.PROVIDERS.get(provider_type)
        
        if not provider_class:
            raise ValueError(
                f"Unsupported storage provider type: {provider_type}. "
                f"Supported types: {list(cls.PROVIDERS.keys())}"
            )
            
        return provider_class(config)
    
    @classmethod
    def create_provider_for_uri(
        cls,
        uri: str,
        config: Union[StorageConfig, LocalStorageConfig, S3StorageConfig]
    ) -> AdapterStorageProvider:
        """Create a storage provider instance appropriate for a URI.
        
        Args:
            uri: Storage URI to handle
            config: Storage configuration
            
        Returns:
            Configured storage provider instance
            
        Raises:
            ValueError: If no provider supports the URI
        """
        for provider_type, pattern in cls.URI_PATTERNS.items():
            if pattern.match(uri):
                provider_class = cls.PROVIDERS[provider_type]
                return provider_class(config)
                
        raise ValueError(
            f"No storage provider found for URI: {uri}. "
            f"Supported patterns: {cls.URI_PATTERNS}"
        )
    
    @classmethod
    def _get_provider_type(
        cls,
        config: Union[StorageConfig, LocalStorageConfig, S3StorageConfig]
    ) -> str:
        """Get provider type from configuration.
        
        Args:
            config: Storage configuration
            
        Returns:
            Provider type identifier
            
        Raises:
            ValueError: If provider type cannot be determined
        """
        if isinstance(config, LocalStorageConfig):
            return "local"
        elif isinstance(config, S3StorageConfig):
            return "s3"
        else:
            raise ValueError(
                f"Cannot determine provider type from config: {type(config)}"
            ) 