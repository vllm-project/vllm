# SPDX-License-Identifier: Apache-2.0
"""
Test cases for remote storage plugin functionality.

This module tests the remote storage  plugin loading mechanism in ConnectorManager.
It creates mock connector adapters extending ConnectorAdapter and verifies that:
1. Connector adapters are properly loaded via the configuration
2. The connector adapter's create_connector method is called for matching URLs
3. Invalid configurations are handled gracefully
"""

# Standard
from typing import List, Optional
import asyncio

# Third Party
import pytest

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    ConnectorManager,
    CreateConnector,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector


class MockTestRemoteConnector(RemoteConnector):
    """
    A mock remote connector for testing the remote storage plugin functionality.
    This connector tracks method calls for verification.
    """

    def __init__(self, config, metadata):
        # Skip parent __init__ since we're mocking
        self.call_history: List[str] = []
        self.storage: dict = {}

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists."""
        self.call_history.append(f"exists:{key.chunk_hash}")
        return key.chunk_hash in self.storage

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Check if key exists synchronously."""
        self.call_history.append(f"exists_sync:{key.chunk_hash}")
        return key.chunk_hash in self.storage

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get a value."""
        self.call_history.append(f"get:{key.chunk_hash}")
        return self.storage.get(key.chunk_hash)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Put a value."""
        self.call_history.append(f"put:{key.chunk_hash}")
        self.storage[key.chunk_hash] = memory_obj

    async def list(self) -> List[str]:
        """List all keys."""
        self.call_history.append("list")
        return list(self.storage.keys())

    async def close(self):
        """Close the connector."""
        self.call_history.append("close")

    def __repr__(self) -> str:
        return "MockTestRemoteConnector"


class MockTestConnectorAdapter(ConnectorAdapter):
    """
    A mock connector adapter for testing the remote storage functionality.
    This adapter extends ConnectorAdapter and creates MockTestRemoteConnector instances.

    This adapter uses instance-level storage for call history and connector storage
    to enable proper testing of dynamically loaded connector adapters.
    """

    def __init__(self) -> None:
        super().__init__("mockplugin://")
        self.call_history: List[str] = []

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """Create a mock remote connector."""
        self.call_history.append(f"create_connector:{context.url}")
        connector = MockTestRemoteConnector(context.config, context.metadata)
        return connector


class NotAConnectorAdapter:
    """
    A class that does not implement ConnectorAdapter.
    Used to test that non-ConnectorAdapter classes are rejected.
    """

    def __init__(self) -> None:
        pass


# Module-level constants for connector plugin configuration
MOCK_CONNECTOR_EXTRA_CONFIG = {
    "remote_storage_plugin.mock_connector.module_path": (
        "tests.v1.storage_backend.test_remote_storage_plugin"
    ),
    "remote_storage_plugin.mock_connector.class_name": "MockTestConnectorAdapter",
}
MOCK_CONNECTOR_REMOTE_STORAGE_PLUGINS = ["mock_connector"]


def create_test_config(
    extra_config: Optional[dict] = None,
    remote_storage_plugins: Optional[List[str]] = None,
) -> LMCacheEngineConfig:
    """Create a test configuration for connector plugin testing."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="test_remote_storage_plugin_instance",
    )
    if extra_config:
        config.extra_config = extra_config
    if remote_storage_plugins:
        config.remote_storage_plugins = remote_storage_plugins
    return config


def isinstance_mock_adapter(x) -> bool:
    """Tests MockTestConnectorAdapter as isinstance() not identifying the child class"""
    return (
        isinstance(x, ConnectorAdapter)
        and x.__class__.__name__ == "MockTestConnectorAdapter"
    )


def isinstance_mock_connector(x) -> bool:
    """Tests MockTestRemoteConnector as isinstance() not identifying the child class"""
    return (
        isinstance(x, RemoteConnector)
        and x.__class__.__name__ == "MockTestRemoteConnector"
    )


def isinstance_mock_instrumented_connector(x) -> bool:
    """Tests InstrumentedRemoteConnector as isinstance() not identifying the child
    class"""
    return (
        isinstance(x, RemoteConnector)
        and x.__class__.__name__ == "InstrumentedRemoteConnector"
    )


class TestConnectorPluginLauncher:
    """Test cases for connector plugin launcher functionality."""

    @pytest.fixture
    def async_loop(self):
        """Create an async event loop for testing."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def test_remote_storage_plugin_adapter_is_loaded(self, async_loop):
        """
        Test that a connector adapter plugin is properly loaded via ConnectorManager.
        """
        config = create_test_config(
            extra_config=MOCK_CONNECTOR_EXTRA_CONFIG,
            remote_storage_plugins=MOCK_CONNECTOR_REMOTE_STORAGE_PLUGINS,
        )

        # Create ConnectorManager with the mock connector URL
        manager = ConnectorManager(
            url="mockplugin://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Verify the mock adapter is loaded
        mock_adapters = [a for a in manager.adapters if isinstance_mock_adapter(a)]
        assert len(mock_adapters) == 1, (
            f"Expected 1 MockTestConnectorAdapter, got: {len(mock_adapters)}"
        )

        # Verify the adapter can parse the URL
        mock_adapter = mock_adapters[0]
        assert mock_adapter.can_parse("mockplugin://localhost:1234"), (
            "Expected adapter to be able to parse 'mockplugin://' URLs"
        )

    def test_remote_storage_plugin_creates_connector(self, async_loop):
        """
        Test that ConnectorManager creates a connector using the plugin adapter.
        """
        config = create_test_config(
            extra_config=MOCK_CONNECTOR_EXTRA_CONFIG,
            remote_storage_plugins=MOCK_CONNECTOR_REMOTE_STORAGE_PLUGINS,
        )

        url = "mockplugin://localhost:1234"

        # Create ConnectorManager and get the connector
        manager = ConnectorManager(
            url=url,
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        connector = manager.create_connector()

        # Verify a connector was created
        assert connector is not None, "Expected a connector to be created"
        assert isinstance_mock_connector(connector), (
            "Expected a MockTestRemoteConnector connector to be created"
        )

    def test_remote_storage_plugin_via_create_connector_function(self, async_loop):
        """
        Test that CreateConnector function properly loads and uses connector plugins.
        """
        config = create_test_config(
            extra_config=MOCK_CONNECTOR_EXTRA_CONFIG,
            remote_storage_plugins=MOCK_CONNECTOR_REMOTE_STORAGE_PLUGINS,
        )

        # Create connector using the CreateConnector function
        connector = CreateConnector(
            url="mockplugin://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Verify a connector was created (wrapped in InstrumentedRemoteConnector)
        assert connector is not None, "Expected a connector to be created"
        assert isinstance_mock_instrumented_connector(connector), (
            "Expected a InstrumentedRemoteConnector connector to be created"
        )

    def test_remote_storage_plugin_without_remote_storage_plugins_config(
        self, async_loop
    ):
        """
        Test that no connector plugin is loaded when remote_storage_plugins
        is not configured.
        """
        # Configure extra_config but don't set remote_storage_plugins
        config = create_test_config(
            extra_config=MOCK_CONNECTOR_EXTRA_CONFIG,
            remote_storage_plugins=None,
        )

        manager = ConnectorManager(
            url="mockplugin://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Verify no MockTestConnectorAdapter is loaded
        mock_adapters = [a for a in manager.adapters if isinstance_mock_adapter(a)]
        assert len(mock_adapters) == 0, (
            f"Expected 0 MockTestConnectorAdapter, got: {len(mock_adapters)}"
        )

    def test_remote_storage_plugin_with_invalid_module_path(self, async_loop):
        """
        Test that invalid module path is handled gracefully.
        """
        extra_config = {
            "remote_storage_plugin.invalid_connector.module_path": "nonexistent.module.path",  # noqa: E501
            "remote_storage_plugin.invalid_connector.class_name": "NonexistentClass",
        }
        remote_storage_plugins = ["invalid_connector"]

        config = create_test_config(
            extra_config=extra_config,
            remote_storage_plugins=remote_storage_plugins,
        )

        # Should not raise an exception
        manager = ConnectorManager(
            url="lm://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Invalid connector adapter should not be loaded
        # (only builtin adapters should be present)
        adapter_class_names = [type(a).__name__ for a in manager.adapters]
        assert "NonexistentClass" not in adapter_class_names, (
            f"Expected 'NonexistentClass' not in adapters, got: {adapter_class_names}"
        )

    def test_remote_storage_plugin_with_invalid_class_name(self, async_loop):
        """
        Test that invalid class name is handled gracefully.
        """
        extra_config = {
            "remote_storage_plugin.invalid_connector.module_path": (
                "tests.v1.storage_backend.test_remote_storage_plugin"
            ),
            "remote_storage_plugin.invalid_connector.class_name": "NonexistentClass",
        }
        remote_storage_plugins = ["invalid_connector"]

        config = create_test_config(
            extra_config=extra_config,
            remote_storage_plugins=remote_storage_plugins,
        )

        # Should not raise an exception
        manager = ConnectorManager(
            url="lm://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Invalid connector adapter should not be loaded
        adapter_class_names = [type(a).__name__ for a in manager.adapters]
        assert "NonexistentClass" not in adapter_class_names, (
            f"Expected 'NonexistentClass' not in adapters, got: {adapter_class_names}"
        )

    def test_remote_storage_plugin_with_non_connector_adapter_class(self, async_loop):
        """
        Test that classes not implementing ConnectorAdapter are rejected.
        """
        extra_config = {
            "remote_storage_plugin.not_adapter.module_path": (
                "tests.v1.storage_backend.test_remote_storage_plugin"
            ),
            "remote_storage_plugin.not_adapter.class_name": "NotAConnectorAdapter",
        }
        remote_storage_plugins = ["not_adapter"]

        config = create_test_config(
            extra_config=extra_config,
            remote_storage_plugins=remote_storage_plugins,
        )

        # Should not raise an exception
        manager = ConnectorManager(
            url="lm://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Non-ConnectorAdapter class should not be loaded
        adapter_class_names = [type(a).__name__ for a in manager.adapters]
        assert "NotAConnectorAdapter" not in adapter_class_names, (
            f"Expected 'NotAConnectorAdapter' not in adapters, "
            f"got: {adapter_class_names}"
        )

    def test_remote_storage_plugin_with_missing_module_path(self, async_loop):
        """
        Test that missing module_path in extra_config is handled gracefully.
        """
        extra_config = {
            # Missing module_path
            "remote_storage_plugin.missing_path.class_name": "MockTestConnectorAdapter",
        }
        remote_storage_plugins = ["missing_path"]

        config = create_test_config(
            extra_config=extra_config,
            remote_storage_plugins=remote_storage_plugins,
        )

        # Should not raise an exception
        manager = ConnectorManager(
            url="lm://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Adapter with missing module_path should not be loaded
        mock_adapters = [a for a in manager.adapters if isinstance_mock_adapter(a)]
        assert len(mock_adapters) == 0, (
            f"Expected 0 MockTestConnectorAdapter, got: {len(mock_adapters)}"
        )

    def test_remote_storage_plugin_with_missing_class_name(self, async_loop):
        """
        Test that missing class_name in extra_config is handled gracefully.
        """
        extra_config = {
            "remote_storage_plugin.missing_class.module_path": (
                "tests.v1.storage_backend.test_remote_storage_plugin"
            ),
            # Missing class_name
        }
        remote_storage_plugins = ["missing_class"]

        config = create_test_config(
            extra_config=extra_config,
            remote_storage_plugins=remote_storage_plugins,
        )

        # Should not raise an exception
        manager = ConnectorManager(
            url="lm://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Adapter with missing class_name should not be loaded
        mock_adapters = [a for a in manager.adapters if isinstance_mock_adapter(a)]
        assert len(mock_adapters) == 0, (
            f"Expected 0 MockTestConnectorAdapter, got: {len(mock_adapters)}"
        )

    def test_remote_storage_plugin_with_missing_extra_config(self, async_loop):
        """
        Test that missing extra_config is handled gracefully.
        """
        config = create_test_config(
            extra_config=None,  # No extra_config
            remote_storage_plugins=["some_connector"],
        )

        # Should not raise an exception
        manager = ConnectorManager(
            url="lm://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # No plugin adapter should be loaded when extra_config is None
        mock_adapters = [a for a in manager.adapters if isinstance_mock_adapter(a)]
        assert len(mock_adapters) == 0, (
            f"Expected 0 MockTestConnectorAdapter, got: {len(mock_adapters)}"
        )

    def test_remote_storage_plugin_with_none_config(self, async_loop):
        """
        Test that None config is handled gracefully.
        """
        # Should not raise an exception
        manager = ConnectorManager(
            url="lm://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=None,  # No config at all
            metadata=None,
        )

        # Only builtin adapters should be loaded
        mock_adapters = [a for a in manager.adapters if isinstance_mock_adapter(a)]
        assert len(mock_adapters) == 0, (
            f"Expected 0 MockTestConnectorAdapter, got: {len(mock_adapters)}"
        )

    def test_remote_storage_plugin_multiple_plugins(self, async_loop):
        """
        Test that multiple connector plugins can be loaded simultaneously.
        """
        extra_config = {
            "remote_storage_plugin.mock_connector_1.module_path": (
                "tests.v1.storage_backend.test_remote_storage_plugin"
            ),
            "remote_storage_plugin.mock_connector_1.class_name": "MockTestConnectorAdapter",  # noqa: E501
            "remote_storage_plugin.mock_connector_2.module_path": (
                "tests.v1.storage_backend.test_remote_storage_plugin"
            ),
            "remote_storage_plugin.mock_connector_2.class_name": "MockTestConnectorAdapter",  # noqa: E501
        }
        remote_storage_plugins = ["mock_connector_1", "mock_connector_2"]

        config = create_test_config(
            extra_config=extra_config,
            remote_storage_plugins=remote_storage_plugins,
        )

        manager = ConnectorManager(
            url="mockplugin://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Verify both adapters are loaded
        mock_adapters = [a for a in manager.adapters if isinstance_mock_adapter(a)]
        assert len(mock_adapters) == 2, (
            f"Expected 2 MockTestConnectorAdapter instances, got: {len(mock_adapters)}"
        )

    def test_remote_storage_plugin_no_matching_url_raises_error(self, async_loop):
        """
        Test that ValueError is raised when no adapter matches the URL.
        """
        config = create_test_config(
            extra_config=MOCK_CONNECTOR_EXTRA_CONFIG,
            remote_storage_plugins=MOCK_CONNECTOR_REMOTE_STORAGE_PLUGINS,
        )

        # Create ConnectorManager and get the connector
        manager = ConnectorManager(
            url="unknownscheme://localhost:1234",
            loop=async_loop,
            local_cpu_backend=None,
            config=config,
            metadata=None,
        )

        # Should raise ValueError when no adapter matches
        with pytest.raises(ValueError, match="No adapter found for URL"):
            manager.create_connector()
