"""S3 storage provider for adapters."""

import os
import re
import boto3
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, Tuple, Match
from pathlib import Path
from urllib.parse import urlparse
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

from ..base import (
    BaseStorageProvider,
    AdapterNotFoundError,
    AdapterDownloadError,
    AdapterValidationError,
)
from ..config import S3StorageConfig

# S3 URI regex pattern
S3_URI_PATTERN = re.compile(r'^s3://([^/]+)/(.+)$')

class S3StorageProvider(BaseStorageProvider):
    """S3 storage provider with optimized downloads."""
    
    def __init__(self, config: S3StorageConfig):
        super().__init__()
        self.config = config
        
        # Initialize thread pool for downloads
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_ops,
            thread_name_prefix="s3_storage_"
        )
        
        # Configure boto3 client
        client_config = Config(
            region_name=config.region_name,
            max_pool_connections=config.max_connections,
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
            retries={"max_attempts": config.max_retries},
        )
        
        # Initialize S3 client
        self._client = boto3.client(
            's3',
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            aws_session_token=config.aws_session_token,
            config=client_config,
        )
        
        # Configure transfer settings
        self._transfer_config = boto3.s3.transfer.TransferConfig(
            max_concurrency=config.max_concurrent_downloads,
            multipart_threshold=config.multipart_threshold,
            multipart_chunksize=config.multipart_chunksize,
        )
    
    @property
    def provider_type(self) -> str:
        """Get the type identifier for this storage provider."""
        return "s3"
    
    def _parse_uri(self, uri: str) -> Tuple[str, str]:
        """Parse S3 URI into bucket and key.
        
        Args:
            uri: S3 URI (s3://bucket/key)
            
        Returns:
            Tuple of (bucket, key)
            
        Raises:
            AdapterValidationError: If URI is invalid
        """
        match: Optional[Match] = S3_URI_PATTERN.match(uri)
        if not match:
            raise AdapterValidationError(f"Invalid S3 URI: {uri}")
            
        bucket, key = match.groups()
        if not bucket or not key:
            raise AdapterValidationError(
                f"Invalid S3 URI - must have bucket and key: {uri}"
            )
            
        return bucket, key
    
    async def _download_adapter_impl(
        self,
        uri: str,
        dest_path: Path,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """Implementation of adapter download logic.
        
        Args:
            uri: S3 URI (s3://bucket/key)
            dest_path: Local path to download to
            timeout: Optional timeout in seconds
            
        Raises:
            AdapterNotFoundError: If object not found
            AdapterDownloadError: If download fails
            AdapterValidationError: If URI invalid
        """
        bucket, key = self._parse_uri(uri)
        
        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download in thread pool
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                self._client.download_file,
                bucket,
                key,
                str(dest_path),
                Config=self._transfer_config
            )
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise AdapterNotFoundError(f"Object not found: {uri}")
            elif error_code == 'NoSuchBucket':
                raise AdapterNotFoundError(f"Bucket not found: {bucket}")
            else:
                raise AdapterDownloadError(f"Failed to download: {e}")
        except (BotoCoreError, Exception) as e:
            raise AdapterDownloadError(f"Download failed: {e}")
    
    async def validate_uri(self, uri: str) -> bool:
        """Check if a URI is valid for this storage provider.
        
        Args:
            uri: S3 URI to validate
            
        Returns:
            True if object exists
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
            
        try:
            bucket, key = self._parse_uri(uri)
            try:
                self._client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code in ('404', 'NoSuchKey', 'NoSuchBucket'):
                    return False
                raise
        except (AdapterValidationError, BotoCoreError):
            return False
    
    async def get_adapter_size(self, uri: str) -> int:
        """Get the size of an adapter in bytes.
        
        Args:
            uri: S3 URI
            
        Returns:
            Size in bytes
            
        Raises:
            AdapterNotFoundError: If object not found
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
            
        bucket, key = self._parse_uri(uri)
        
        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
            return response['ContentLength']
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ('404', 'NoSuchKey', 'NoSuchBucket'):
                raise AdapterNotFoundError(f"Object not found: {uri}")
            raise AdapterDownloadError(f"Failed to get size: {e}")
        except BotoCoreError as e:
            raise AdapterDownloadError(f"Failed to get size: {e}")
    
    async def get_adapter_metadata(
        self,
        uri: str
    ) -> Dict[str, Any]:
        """Get metadata about an adapter.
        
        Args:
            uri: S3 URI
            
        Returns:
            Dictionary of metadata
            
        Raises:
            AdapterNotFoundError: If object not found
        """
        if self._closed:
            raise RuntimeError("Storage provider is closed")
            
        bucket, key = self._parse_uri(uri)
        
        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
            return {
                "size": response['ContentLength'],
                "etag": response.get('ETag', '').strip('"'),
                "modified": response['LastModified'].timestamp(),
                "storage_class": response.get('StorageClass'),
                "metadata": response.get('Metadata', {}),
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ('404', 'NoSuchKey', 'NoSuchBucket'):
                raise AdapterNotFoundError(f"Object not found: {uri}")
            raise AdapterDownloadError(f"Failed to get metadata: {e}")
        except BotoCoreError as e:
            raise AdapterDownloadError(f"Failed to get metadata: {e}")
    
    async def cleanup(self) -> None:
        """Clean up provider resources."""
        if not self._closed:
            self._executor.shutdown(wait=True)
            await super().cleanup() 