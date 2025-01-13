"""Local filesystem storage provider for adapters."""

import os
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, Set
from pathlib import Path
import stat

from ..base import (
    BaseStorageProvider,
    AdapterNotFoundError,
    AdapterDownloadError,
    AdapterValidationError,
)
from ..config import LocalStorageConfig

class LocalStorageProvider(BaseStorageProvider):
    """Local filesystem storage provider with security and async support."""
    
    def __init__(self, config: LocalStorageConfig):
        super().__init__()
        self.config = config
        
        # Initialize thread pool for file operations
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_ops,
            thread_name_prefix="local_storage_"
        )
        
        # Track allowed base directories
        self._allowed_dirs: Set[Path] = set()
        if config.allowed_paths:
            for path in config.allowed_paths:
                resolved = Path(path).expanduser().resolve()
                if not resolved.exists():
                    if config.create_dirs:
                        resolved.mkdir(parents=True, exist_ok=True)
                    else:
                        raise ValueError(f"Allowed path does not exist: {path}")
                self._allowed_dirs.add(resolved)
    
    @property
    def provider_type(self) -> str:
        """Get the type identifier for this storage provider."""
        return "local"
    
    def _is_path_allowed(self, path: Path) -> bool:
        """Check if a path is allowed based on configuration.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is allowed
        """
        if not self._allowed_dirs:
            return True
            
        try:
            resolved = path.resolve()
            return any(
                resolved.is_relative_to(allowed)
                for allowed in self._allowed_dirs
            )
        except (OSError, RuntimeError):
            return False
    
    def _validate_path_security(self, path: Path) -> None:
        """Validate path security requirements.
        
        Args:
            path: Path to validate
            
        Raises:
            AdapterValidationError: If path fails security checks
        """
        try:
            if not self._is_path_allowed(path):
                raise AdapterValidationError(
                    f"Path not in allowed directories: {path}"
                )
                
            # Check for symlink attacks
            if path.is_symlink() and not self.config.allow_symlinks:
                raise AdapterValidationError(
                    f"Symlinks not allowed: {path}"
                )
                
            # Verify permissions if needed
            if self.config.verify_permissions:
                mode = path.stat().st_mode
                if mode & (stat.S_IWGRP | stat.S_IWOTH):
                    raise AdapterValidationError(
                        f"Path has unsafe permissions: {path}"
                    )
                    
        except OSError as e:
            raise AdapterValidationError(f"Path validation failed: {e}")
    
    async def _download_adapter_impl(
        self,
        uri: str,
        dest_path: Path,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """Implementation of adapter download logic.
        
        For local storage, this is a copy operation to the destination.
        
        Args:
            uri: Local filesystem path
            dest_path: Destination path to copy to
            timeout: Optional timeout in seconds
            
        Raises:
            AdapterNotFoundError: If source path not found
            AdapterDownloadError: If copy fails
            AdapterValidationError: If security checks fail
        """
        src_path = Path(uri).expanduser()
        
        # Validate paths
        self._validate_path_security(src_path)
        self._validate_path_security(dest_path)
        
        if not src_path.exists():
            raise AdapterNotFoundError(f"Source path not found: {uri}")
            
        if not src_path.is_file():
            raise AdapterValidationError(f"Source is not a file: {uri}")
            
        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy file in thread pool
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                shutil.copy2 if self.config.preserve_permissions else shutil.copy,
                src_path,
                dest_path
            )
        except OSError as e:
            raise AdapterDownloadError(f"Failed to copy adapter: {e}")
    
    async def validate_uri(self, uri: str) -> bool:
        """Check if a URI is valid for this storage provider.
        
        Args:
            uri: Local filesystem path
            
        Returns:
            True if path exists and is valid
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
            
        try:
            path = Path(uri).expanduser()
            self._validate_path_security(path)
            return path.is_file()
        except (AdapterValidationError, OSError):
            return False
    
    async def get_adapter_size(self, uri: str) -> int:
        """Get the size of an adapter in bytes.
        
        Args:
            uri: Local filesystem path
            
        Returns:
            Size in bytes
            
        Raises:
            AdapterNotFoundError: If path not found
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
            
        try:
            path = Path(uri).expanduser()
            self._validate_path_security(path)
            
            if not path.exists():
                raise AdapterNotFoundError(f"Path not found: {uri}")
                
            return path.stat().st_size
        except AdapterValidationError:
            raise
        except OSError as e:
            raise AdapterNotFoundError(f"Failed to get size: {e}")
    
    async def get_adapter_metadata(
        self,
        uri: str
    ) -> Dict[str, Any]:
        """Get metadata about an adapter.
        
        Args:
            uri: Local filesystem path
            
        Returns:
            Dictionary of metadata
            
        Raises:
            AdapterNotFoundError: If path not found
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
            
        try:
            path = Path(uri).expanduser()
            self._validate_path_security(path)
            
            if not path.exists():
                raise AdapterNotFoundError(f"Path not found: {uri}")
                
            stat = path.stat()
            return {
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime,
                "mode": stat.st_mode,
            }
        except AdapterValidationError:
            raise
        except OSError as e:
            raise AdapterNotFoundError(f"Failed to get metadata: {e}")
    
    async def cleanup(self) -> None:
        """Clean up provider resources."""
        if not self._closed:
            self._executor.shutdown(wait=True)
            await super().cleanup() 