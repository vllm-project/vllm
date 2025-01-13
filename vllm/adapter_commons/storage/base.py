"""Base interfaces for adapter storage providers."""

import abc
from typing import Protocol, Dict, Any, Optional, AsyncIterator
from pathlib import Path
import asyncio

class AdapterStorageError(Exception):
    """Base class for adapter storage errors."""
    pass

class AdapterNotFoundError(AdapterStorageError):
    """Raised when adapter is not found in storage."""
    pass

class AdapterDownloadError(AdapterStorageError):
    """Raised when adapter download fails."""
    pass

class AdapterValidationError(AdapterStorageError):
    """Raised when adapter validation fails."""
    pass

class AdapterStorageProvider(Protocol):
    """Protocol defining the interface for adapter storage providers."""
    
    @property
    def provider_type(self) -> str:
        """Get the type identifier for this storage provider."""
        ...
    
    async def download_adapter(
        self,
        uri: str,
        dest_path: Path,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """Download an adapter from storage to local filesystem.
        
        Args:
            uri: Storage URI for the adapter
            dest_path: Local path to download adapter to
            timeout: Optional timeout in seconds
            
        Raises:
            AdapterNotFoundError: If adapter not found
            AdapterDownloadError: If download fails
            AdapterValidationError: If adapter validation fails
        """
        ...
    
    async def validate_uri(self, uri: str) -> bool:
        """Check if a URI is valid for this storage provider.
        
        Args:
            uri: Storage URI to validate
            
        Returns:
            True if URI is valid and adapter exists
        """
        ...
    
    async def get_adapter_size(self, uri: str) -> int:
        """Get the size of an adapter in bytes.
        
        Args:
            uri: Storage URI for the adapter
            
        Returns:
            Size in bytes
            
        Raises:
            AdapterNotFoundError: If adapter not found
        """
        ...
    
    async def get_adapter_metadata(
        self,
        uri: str
    ) -> Dict[str, Any]:
        """Get metadata about an adapter.
        
        Args:
            uri: Storage URI for the adapter
            
        Returns:
            Dictionary of metadata
            
        Raises:
            AdapterNotFoundError: If adapter not found
        """
        ...
    
    async def cleanup(self) -> None:
        """Clean up any resources associated with this provider."""
        ...

class BaseStorageProvider(abc.ABC):
    """Abstract base class for adapter storage providers.
    
    Implements common functionality and enforces the AdapterStorageProvider
    protocol. Storage providers should inherit from this class.
    """
    
    def __init__(self):
        self._closed = False
    
    @property
    @abc.abstractmethod
    def provider_type(self) -> str:
        """Get the type identifier for this storage provider."""
        pass
    
    @abc.abstractmethod
    async def _download_adapter_impl(
        self,
        uri: str,
        dest_path: Path,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """Implementation of adapter download logic.
        
        Args:
            uri: Storage URI for the adapter
            dest_path: Local path to download adapter to
            timeout: Optional timeout in seconds
            
        Raises:
            AdapterNotFoundError: If adapter not found
            AdapterDownloadError: If download fails
            AdapterValidationError: If adapter validation fails
        """
        pass
    
    async def download_adapter(
        self,
        uri: str,
        dest_path: Path,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """Download an adapter from storage to local filesystem.
        
        Wraps the implementation with common validation and error handling.
        
        Args:
            uri: Storage URI for the adapter
            dest_path: Local path to download adapter to
            timeout: Optional timeout in seconds
            
        Raises:
            AdapterNotFoundError: If adapter not found
            AdapterDownloadError: If download fails
            AdapterValidationError: If adapter validation fails
            RuntimeError: If provider is closed
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
            
        if not await self.validate_uri(uri):
            raise AdapterNotFoundError(f"Adapter not found: {uri}")
            
        try:
            if timeout:
                async with asyncio.timeout(timeout):
                    await self._download_adapter_impl(
                        uri,
                        dest_path,
                        timeout=timeout
                    )
            else:
                await self._download_adapter_impl(
                    uri,
                    dest_path,
                    timeout=None
                )
        except asyncio.TimeoutError as e:
            raise AdapterDownloadError(
                f"Download timed out after {timeout}s: {uri}"
            ) from e
        except Exception as e:
            raise AdapterDownloadError(
                f"Failed to download adapter: {uri}"
            ) from e
    
    @abc.abstractmethod
    async def validate_uri(self, uri: str) -> bool:
        """Check if a URI is valid for this storage provider.
        
        Args:
            uri: Storage URI to validate
            
        Returns:
            True if URI is valid and adapter exists
            
        Raises:
            RuntimeError: If provider is closed
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
        return False
    
    @abc.abstractmethod
    async def get_adapter_size(self, uri: str) -> int:
        """Get the size of an adapter in bytes.
        
        Args:
            uri: Storage URI for the adapter
            
        Returns:
            Size in bytes
            
        Raises:
            AdapterNotFoundError: If adapter not found
            RuntimeError: If provider is closed
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
        raise NotImplementedError
    
    @abc.abstractmethod
    async def get_adapter_metadata(
        self,
        uri: str
    ) -> Dict[str, Any]:
        """Get metadata about an adapter.
        
        Args:
            uri: Storage URI for the adapter
            
        Returns:
            Dictionary of metadata
            
        Raises:
            AdapterNotFoundError: If adapter not found
            RuntimeError: If provider is closed
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
        raise NotImplementedError
    
    async def cleanup(self) -> None:
        """Clean up any resources associated with this provider."""
        self._closed = True 