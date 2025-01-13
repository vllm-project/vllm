"""Unit tests for storage provider factory."""

import os
import pytest
from pathlib import Path
from typing import Type
from unittest.mock import Mock

from vllm.adapter_commons.storage import (
    StorageProviderFactory,
    BaseStorageProvider,
    LocalStorageProvider,
    S3StorageProvider,
    StorageConfig,
    LocalStorageConfig,
    S3StorageConfig,
)

# Test data
TEST_PROVIDER_TYPE = "test_provider"
TEST_URI_SCHEME = "test"


class MockStorageProvider(BaseStorageProvider):
    """Mock storage provider for testing."""
    async def download_adapter(self, uri: str, target_path: str) -> None:
        pass
    
    async def validate_uri(self, uri: str) -> bool:
        return True
    
    async def get_adapter_size(self, uri: str) -> int:
        return 0
    
    async def get_adapter_metadata(self, uri: str) -> dict:
        return {}


@pytest.fixture(scope="function")
def mock_provider():
    """Create a mock storage provider class."""
    return MockStorageProvider


@pytest.fixture(scope="function")
def factory():
    """Create a fresh factory instance for each test."""
    # Reset class registries
    StorageProviderFactory._provider_registry = {
        "local": LocalStorageProvider,
        "s3": S3StorageProvider,
    }
    StorageProviderFactory._scheme_registry = {
        "file": "local",
        "s3": "s3",
    }
    return StorageProviderFactory()


def test_default_providers(factory):
    """Test default provider registration."""
    providers = factory.get_registered_providers()
    assert "local" in providers
    assert "s3" in providers
    assert len(providers) == 2


def test_default_schemes(factory):
    """Test default URI scheme registration."""
    schemes = factory.get_supported_schemes()
    assert "file" in schemes
    assert "s3" in schemes
    assert len(schemes) == 2


def test_register_provider(factory, mock_provider):
    """Test provider registration."""
    # Register new provider
    StorageProviderFactory.register_provider(
        TEST_PROVIDER_TYPE,
        mock_provider,
        [TEST_URI_SCHEME]
    )
    
    # Verify registration
    providers = factory.get_registered_providers()
    assert TEST_PROVIDER_TYPE in providers
    
    schemes = factory.get_supported_schemes()
    assert TEST_URI_SCHEME in schemes


def test_register_invalid_provider(factory):
    """Test registration with invalid provider class."""
    class InvalidProvider:
        pass
    
    with pytest.raises(ValueError):
        StorageProviderFactory.register_provider(
            TEST_PROVIDER_TYPE,
            InvalidProvider
        )


def test_register_duplicate_provider(factory, mock_provider):
    """Test registration with duplicate provider type."""
    StorageProviderFactory.register_provider(
        TEST_PROVIDER_TYPE,
        mock_provider
    )
    
    with pytest.raises(ValueError):
        StorageProviderFactory.register_provider(
            TEST_PROVIDER_TYPE,
            mock_provider
        )


def test_register_duplicate_scheme(factory, mock_provider):
    """Test registration with duplicate URI scheme."""
    with pytest.raises(ValueError):
        StorageProviderFactory.register_provider(
            TEST_PROVIDER_TYPE,
            mock_provider,
            ["file"]  # Already registered for local provider
        )


def test_create_provider_local(factory):
    """Test creating local storage provider."""
    config = LocalStorageConfig(
        provider_type="local",
        allowed_paths=["/tmp"]
    )
    
    provider = factory.create_provider(config)
    assert isinstance(provider, LocalStorageProvider)


def test_create_provider_s3(factory):
    """Test creating S3 storage provider."""
    config = S3StorageConfig(
        provider_type="s3",
        region_name="us-east-1"
    )
    
    provider = factory.create_provider(config)
    assert isinstance(provider, S3StorageProvider)


def test_create_provider_unknown_type(factory):
    """Test creating provider with unknown type."""
    config = StorageConfig(provider_type="unknown")
    
    with pytest.raises(ValueError):
        factory.create_provider(config)


def test_create_provider_mismatched_config(factory):
    """Test creating provider with mismatched config type."""
    # Try to create local provider with S3 config
    config = S3StorageConfig(
        provider_type="local",
        region_name="us-east-1"
    )
    
    with pytest.raises(ValueError):
        factory.create_provider(config)


def test_create_provider_for_uri_local(factory):
    """Test creating provider from local URI."""
    provider = factory.create_provider_for_uri("/tmp/adapter.bin")
    assert isinstance(provider, LocalStorageProvider)


def test_create_provider_for_uri_s3(factory):
    """Test creating provider from S3 URI."""
    provider = factory.create_provider_for_uri("s3://bucket/adapter.bin")
    assert isinstance(provider, S3StorageProvider)


def test_create_provider_for_uri_unknown(factory):
    """Test creating provider from unknown URI scheme."""
    with pytest.raises(ValueError):
        factory.create_provider_for_uri("unknown://adapter.bin")


def test_create_provider_for_uri_with_config(factory):
    """Test creating provider from URI with custom config."""
    config = S3StorageConfig(
        provider_type="s3",
        region_name="us-east-1"
    )
    
    provider = factory.create_provider_for_uri(
        "s3://bucket/adapter.bin",
        config=config
    )
    assert isinstance(provider, S3StorageProvider) 