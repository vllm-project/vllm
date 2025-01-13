"""Unit tests for storage providers."""

import asyncio
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional
from unittest.mock import Mock, patch

import boto3
import pytest
import pytest_asyncio
from botocore.exceptions import ClientError
from moto import mock_aws

from vllm.adapter_commons.storage import (
    StorageConfig,
    S3StorageConfig,
    LocalStorageConfig,
    StorageProviderFactory,
)
from vllm.adapter_commons.storage.base import (
    AdapterStorageError,
    AdapterNotFoundError,
    AdapterDownloadError,
    AdapterValidationError,
)
from vllm.adapter_commons.storage.providers import (
    LocalStorageProvider,
    S3StorageProvider,
)

# Test data
TEST_ADAPTER_CONTENT = b"mock adapter content"
TEST_ADAPTER_SIZE = len(TEST_ADAPTER_CONTENT)


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="function")
def local_provider(temp_dir):
    """Create a local storage provider instance."""
    config = LocalStorageConfig(
        provider_type="local",
        allowed_paths=[temp_dir],
        create_dirs=True,
    )
    return LocalStorageProvider(config)


@pytest.fixture(scope="function")
def s3_provider():
    """Create an S3 storage provider instance."""
    config = S3StorageConfig(
        provider_type="s3",
        region_name="us-east-1",
        max_retries=1,
        connect_timeout=1,
        read_timeout=1,
    )
    return S3StorageProvider(config)


# Local Provider Tests
@pytest.mark.asyncio
async def test_local_provider_validate_uri(local_provider, temp_dir):
    """Test URI validation for local provider."""
    # Valid cases
    valid_path = Path(temp_dir) / "valid.bin"
    assert await local_provider.validate_uri(str(valid_path))
    
    # Invalid cases
    with pytest.raises(AdapterValidationError):
        await local_provider.validate_uri("/not/allowed/path.bin")
    
    with pytest.raises(AdapterValidationError):
        await local_provider.validate_uri("s3://wrong-scheme/path.bin")


@pytest.mark.asyncio
async def test_local_provider_download(local_provider, temp_dir):
    """Test download functionality for local provider."""
    # Create source file
    source_path = Path(temp_dir) / "source.bin"
    source_path.write_bytes(TEST_ADAPTER_CONTENT)
    
    # Test successful download
    target_path = Path(temp_dir) / "target.bin"
    await local_provider.download_adapter(str(source_path), str(target_path))
    assert target_path.exists()
    assert target_path.read_bytes() == TEST_ADAPTER_CONTENT
    
    # Test download with missing source
    with pytest.raises(AdapterNotFoundError):
        await local_provider.download_adapter(
            str(Path(temp_dir) / "nonexistent.bin"),
            str(target_path)
        )
    
    # Test download with permission error
    with patch("pathlib.Path.open", side_effect=PermissionError):
        with pytest.raises(AdapterDownloadError):
            await local_provider.download_adapter(
                str(source_path),
                str(target_path)
            )


@pytest.mark.asyncio
async def test_local_provider_size(local_provider, temp_dir):
    """Test size calculation for local provider."""
    # Create test file
    test_path = Path(temp_dir) / "test.bin"
    test_path.write_bytes(TEST_ADAPTER_CONTENT)
    
    # Test size calculation
    size = await local_provider.get_adapter_size(str(test_path))
    assert size == TEST_ADAPTER_SIZE
    
    # Test with missing file
    with pytest.raises(AdapterNotFoundError):
        await local_provider.get_adapter_size(
            str(Path(temp_dir) / "nonexistent.bin")
        )


# S3 Provider Tests
@pytest.mark.asyncio
async def test_s3_provider_validate_uri(s3_provider):
    """Test URI validation for S3 provider."""
    # Valid cases
    assert await s3_provider.validate_uri("s3://bucket/path/to/adapter.bin")
    
    # Invalid cases
    with pytest.raises(AdapterValidationError):
        await s3_provider.validate_uri("not-a-uri")
    
    with pytest.raises(AdapterValidationError):
        await s3_provider.validate_uri("file:///local/path.bin")
    
    with pytest.raises(AdapterValidationError):
        await s3_provider.validate_uri("s3://")


@pytest.mark.asyncio
async def test_s3_provider_download(s3_provider, temp_dir, mock_s3):
    """Test download functionality for S3 provider."""
    target_path = Path(temp_dir) / "downloaded.bin"
    
    # Test successful download
    await s3_provider.download_adapter(
        "s3://test-bucket/adapters/test.bin",
        str(target_path)
    )
    assert target_path.exists()
    assert target_path.read_bytes() == TEST_ADAPTER_CONTENT
    
    # Test download with nonexistent bucket
    with pytest.raises(AdapterNotFoundError):
        await s3_provider.download_adapter(
            "s3://nonexistent-bucket/test.bin",
            str(target_path)
        )
    
    # Test download with nonexistent key
    with pytest.raises(AdapterNotFoundError):
        await s3_provider.download_adapter(
            "s3://test-bucket/nonexistent.bin",
            str(target_path)
        )


@pytest.mark.asyncio
async def test_s3_provider_size(s3_provider, mock_s3):
    """Test size calculation for S3 provider."""
    # Test size of existing object
    size = await s3_provider.get_adapter_size(
        "s3://test-bucket/adapters/test.bin"
    )
    assert size == TEST_ADAPTER_SIZE
    
    # Test with nonexistent object
    with pytest.raises(AdapterNotFoundError):
        await s3_provider.get_adapter_size(
            "s3://test-bucket/nonexistent.bin"
        )


@pytest.mark.asyncio
async def test_s3_provider_retries(s3_provider, temp_dir):
    """Test retry behavior for S3 provider."""
    target_path = Path(temp_dir) / "should-not-exist.bin"
    
    # Mock boto3 client to simulate transient errors
    with patch("boto3.client") as mock_boto3:
        mock_client = Mock()
        mock_boto3.return_value = mock_client
        
        # Simulate transient error
        mock_client.download_file.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Internal Error"}},
            "download_file"
        )
        
        with pytest.raises(AdapterDownloadError):
            await s3_provider.download_adapter(
                "s3://test-bucket/test.bin",
                str(target_path)
            )
        
        # Verify retry attempts
        assert mock_client.download_file.call_count > 1


@pytest.mark.asyncio
async def test_concurrent_downloads(s3_provider, temp_dir, mock_s3):
    """Test concurrent download operations."""
    async def download(index: int) -> None:
        target_path = Path(temp_dir) / f"concurrent_{index}.bin"
        await s3_provider.download_adapter(
            "s3://test-bucket/adapters/test.bin",
            str(target_path)
        )
        assert target_path.exists()
        assert target_path.read_bytes() == TEST_ADAPTER_CONTENT
    
    # Run multiple downloads concurrently
    tasks = [download(i) for i in range(5)]
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_provider_cleanup(s3_provider, local_provider):
    """Test cleanup behavior for providers."""
    # Test S3 provider cleanup
    await s3_provider.cleanup()
    
    # Test local provider cleanup
    await local_provider.cleanup()
    
    # Verify providers can still be used after cleanup
    assert await s3_provider.validate_uri("s3://bucket/test.bin")
    assert await local_provider.validate_uri("/allowed/path/test.bin") 