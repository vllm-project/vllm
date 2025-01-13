"""Storage providers for vLLM adapters.

This package provides a flexible storage system for LoRA adapters,
supporting both local filesystem and cloud storage (S3) with a unified interface.
"""

from .base import (
    AdapterStorageProvider,
    BaseStorageProvider,
    AdapterStorageError,
    AdapterNotFoundError,
    AdapterDownloadError,
    AdapterValidationError,
)

from .config import (
    StorageConfig,
    LocalStorageConfig,
    S3StorageConfig,
)

from .providers.local import LocalStorageProvider
from .providers.s3 import S3StorageProvider
from .factory import StorageProviderFactory

__all__ = [
    # Base interfaces
    "AdapterStorageProvider",
    "BaseStorageProvider",
    
    # Exceptions
    "AdapterStorageError",
    "AdapterNotFoundError",
    "AdapterDownloadError",
    "AdapterValidationError",
    
    # Configuration
    "StorageConfig",
    "LocalStorageConfig",
    "S3StorageConfig",
    
    # Provider implementations
    "LocalStorageProvider",
    "S3StorageProvider",
    
    # Factory
    "StorageProviderFactory",
]

# Version info
__version__ = "1.0.0" 