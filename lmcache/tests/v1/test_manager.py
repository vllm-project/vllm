# SPDX-License-Identifier: Apache-2.0
"""
Tests for LMCacheManager.
"""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.integration.base_service_factory import BaseServiceFactory
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.manager import LMCacheManager


def _make_mock_factory(**overrides):
    """Create a mock BaseServiceFactory with sensible defaults."""
    factory = MagicMock(spec=BaseServiceFactory)
    factory.get_or_create_metadata.return_value = overrides.get("metadata", None)
    factory.get_or_create_lmcache_engine.return_value = overrides.get("engine", None)
    factory.maybe_create_lookup_client.return_value = overrides.get(
        "lookup_client", None
    )
    factory.maybe_create_lookup_server.return_value = overrides.get(
        "lookup_server", None
    )
    factory.maybe_create_offload_server.return_value = overrides.get(
        "offload_server", None
    )
    factory.maybe_create_runtime_plugin_launcher.return_value = overrides.get(
        "plugin_launcher", None
    )
    factory.maybe_create_internal_api_server.return_value = overrides.get(
        "api_server", None
    )
    factory.maybe_create_health_monitor.return_value = overrides.get(
        "health_monitor", None
    )
    return factory


class TestLMCacheManagerInit:
    """Tests for LMCacheManager initialization."""

    def test_init_stores_config(self):
        """Test that __init__ stores config correctly."""
        config = LMCacheEngineConfig.from_defaults()
        connector = MagicMock()
        factory = _make_mock_factory()

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
            connector=connector,
        )

        assert manager._config is config
        assert manager._connector is connector

    def test_init_calls_factory_methods(self):
        """Test that __init__ calls all factory creation methods."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        LMCacheManager(
            config=config,
            service_factory=factory,
        )

        factory.get_or_create_metadata.assert_called_once()
        factory.get_or_create_lmcache_engine.assert_called_once()
        factory.maybe_create_lookup_client.assert_called_once()
        factory.maybe_create_lookup_server.assert_called_once()
        factory.maybe_create_offload_server.assert_called_once()
        factory.maybe_create_runtime_plugin_launcher.assert_called_once()
        factory.maybe_create_internal_api_server.assert_called_once()

    def test_init_does_not_call_health_monitor(self):
        """Test that __init__ does NOT create health monitor (deferred to post_init)."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        LMCacheManager(
            config=config,
            service_factory=factory,
        )

        factory.maybe_create_health_monitor.assert_not_called()


class TestLMCacheManagerProperties:
    """Tests for LMCacheManager property accessors."""

    @pytest.fixture
    def manager_with_mocked_init(self):
        """Create a manager with mocked factory."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
            connector=MagicMock(),
        )
        return manager

    def test_lmcache_engine_property(self, manager_with_mocked_init):
        """Test lmcache_engine property returns the engine."""
        manager = manager_with_mocked_init
        mock_engine = MagicMock()
        manager._lmcache_engine = mock_engine

        assert manager.lmcache_engine is mock_engine

    def test_lmcache_engine_metadata_property(self, manager_with_mocked_init):
        """Test lmcache_engine_metadata property returns the metadata."""
        manager = manager_with_mocked_init
        mock_metadata = MagicMock()
        manager._lmcache_engine_metadata = mock_metadata

        assert manager.lmcache_engine_metadata is mock_metadata

    def test_lookup_client_property(self, manager_with_mocked_init):
        """Test lookup_client property returns the client."""
        manager = manager_with_mocked_init
        mock_client = MagicMock()
        manager._lookup_client = mock_client

        assert manager.lookup_client is mock_client

    def test_lookup_server_property(self, manager_with_mocked_init):
        """Test lookup_server property returns the server."""
        manager = manager_with_mocked_init
        mock_server = MagicMock()
        manager._lookup_server = mock_server

        assert manager.lookup_server is mock_server

    def test_offload_server_property(self, manager_with_mocked_init):
        """Test offload_server property returns the server."""
        manager = manager_with_mocked_init
        mock_server = MagicMock()
        manager._offload_server = mock_server

        assert manager.offload_server is mock_server

    def test_config_property(self, manager_with_mocked_init):
        """Test config property returns the config."""
        manager = manager_with_mocked_init
        assert manager.config is manager._config


class TestLMCacheManagerStart:
    """Tests for LMCacheManager start method."""

    def test_start_calls_api_server_start(self):
        """Test start_services() calls api_server.start() when api_server exists."""
        config = LMCacheEngineConfig.from_defaults()
        mock_api_server = MagicMock()
        mock_plugin_launcher = MagicMock()
        factory = _make_mock_factory(
            api_server=mock_api_server,
            plugin_launcher=mock_plugin_launcher,
        )

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.start_services()

        mock_api_server.start.assert_called_once()
        mock_plugin_launcher.launch_plugins.assert_called_once()

    def test_start_handles_none_api_server(self):
        """Test start_services() handles None api_server gracefully."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        # Should not raise any exception
        manager.start_services()


class TestLMCacheManagerPostInit:
    """Tests for LMCacheManager post_init method."""

    def test_post_init_without_engine(self):
        """Test post_init initializes health monitor when engine is None."""
        config = LMCacheEngineConfig.from_defaults()
        mock_health_monitor = MagicMock()
        factory = _make_mock_factory(health_monitor=mock_health_monitor)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.post_init()

        # Health monitor should still be created even without engine
        factory.maybe_create_health_monitor.assert_called_once_with(
            lmcache_manager=manager
        )

    def test_post_init_with_engine_and_async_loading(self):
        """Test post_init calls engine.post_init with async_lookup_server."""
        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        mock_engine = MagicMock()
        factory = _make_mock_factory(engine=mock_engine)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.post_init()

        # When lookup_server is None, engine.post_init should be called
        # with async_lookup_server=None
        mock_engine.post_init.assert_called_once_with(async_lookup_server=None)

    def test_post_init_with_engine_and_async_server(self):
        """Test post_init calls engine.post_init when async lookup server exists."""
        # First Party
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupServer,
        )

        config = LMCacheEngineConfig.from_defaults()
        config.enable_async_loading = True
        mock_engine = MagicMock()
        mock_lookup_server = MagicMock(spec=LMCacheAsyncLookupServer)
        factory = _make_mock_factory(
            engine=mock_engine,
            lookup_server=mock_lookup_server,
        )

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.post_init()

        # When lookup_server is LMCacheAsyncLookupServer, it should be passed
        mock_engine.post_init.assert_called_once_with(
            async_lookup_server=mock_lookup_server
        )


class TestLMCacheManagerShutdown:
    """Tests for LMCacheManager shutdown method."""

    def test_shutdown_closes_all_components(self):
        """Test stop_services() closes all components."""
        config = LMCacheEngineConfig.from_defaults()
        mock_offload = MagicMock()
        mock_plugin = MagicMock()
        mock_api = MagicMock()
        mock_lookup_server = MagicMock()
        mock_lookup_client = MagicMock()
        factory = _make_mock_factory(
            offload_server=mock_offload,
            plugin_launcher=mock_plugin,
            api_server=mock_api,
            lookup_server=mock_lookup_server,
            lookup_client=mock_lookup_client,
        )

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        with patch("lmcache.v1.manager.LMCacheEngineBuilder") as mock_builder:
            manager.stop_services()

            # Verify all components were closed
            mock_offload.close.assert_called_once()
            mock_plugin.stop_plugins.assert_called_once()
            mock_api.stop.assert_called_once()
            mock_lookup_server.close.assert_called_once()
            mock_lookup_client.close.assert_called_once()
            mock_builder.destroy.assert_called_once()

    def test_shutdown_handles_none_components(self):
        """Test stop_services() handles None components gracefully."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        with patch("lmcache.v1.manager.LMCacheEngineBuilder"):
            # Should not raise any exception
            manager.stop_services()

    def test_shutdown_handles_component_errors(self):
        """Test stop_services() handles errors from components gracefully."""
        config = LMCacheEngineConfig.from_defaults()
        mock_offload = MagicMock()
        mock_offload.close.side_effect = RuntimeError("Test error")
        mock_lookup_client = MagicMock()
        factory = _make_mock_factory(
            offload_server=mock_offload,
            lookup_client=mock_lookup_client,
        )

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        with patch("lmcache.v1.manager.LMCacheEngineBuilder"):
            # Should not raise exception, but should continue shutdown
            manager.stop_services()

            # lookup_client should still be closed even if offload_server failed
            mock_lookup_client.close.assert_called_once()


class TestLMCacheManagerHelpers:
    """Tests for LMCacheManager helper methods."""

    def test_need_gpu_interm_buffer_returns_not_enable_pd(self):
        """Test _need_gpu_interm_buffer returns opposite of enable_pd."""
        config = LMCacheEngineConfig.from_defaults()
        config.enable_pd = False
        # First Party
        from lmcache.v1.gpu_connector.utils import need_gpu_interm_buffer

        assert need_gpu_interm_buffer(config) is True

        config.enable_pd = True
        assert need_gpu_interm_buffer(config) is False


class TestLMCacheManagerValidation:
    """Tests for validate_mla_config (now in utils)."""

    def test_validate_mla_config_raises_on_wrong_serde(self):
        """Test validate_mla_config raises error for non-naive serde with MLA."""
        # First Party
        from lmcache.integration.vllm.utils import validate_mla_config

        config = LMCacheEngineConfig.from_defaults()
        config.remote_serde = "cachegen"

        with pytest.raises(ValueError, match="MLA only works with naive serde mode"):
            validate_mla_config(config, use_mla=True)

    def test_validate_mla_config_raises_on_layerwise_with_blending(self):
        """Test validate_mla_config raises with MLA + layerwise + blending."""
        # First Party
        from lmcache.integration.vllm.utils import validate_mla_config

        config = LMCacheEngineConfig.from_defaults()
        config.remote_serde = "naive"
        config.use_layerwise = True
        config.enable_blending = True

        with pytest.raises(ValueError, match="MLA with Cacheblend"):
            validate_mla_config(config, use_mla=True)


class TestLMCacheManagerCalculateDraftLayers:
    """Tests for calculate_draft_layers (now in utils)."""

    def test_calculate_draft_layers_no_speculative_config(self):
        """Test returns 0 when no speculative_config."""
        # First Party
        from lmcache.integration.vllm.utils import calculate_draft_layers

        vllm_config = MagicMock()
        vllm_config.speculative_config = None

        assert calculate_draft_layers(vllm_config) == 0

    def test_calculate_draft_layers_deepseek_mtp(self):
        """Test returns correct layers for deepseek_mtp method."""
        # First Party
        from lmcache.integration.vllm.utils import calculate_draft_layers

        vllm_config = MagicMock()
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = "deepseek_mtp"
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.hf_config = MagicMock()
        vllm_config.model_config.hf_config.num_nextn_predict_layers = 3

        assert calculate_draft_layers(vllm_config) == 3


class TestLMCacheManagerInitFailure:
    """Tests for LMCacheManager initialization failure handling."""

    def test_init_components_exception_makes_unhealthy(self):
        """Test that exception during factory calls makes manager unhealthy."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()
        factory.get_or_create_metadata.side_effect = RuntimeError("Test init failure")

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        # Verify through is_healthy API
        assert manager.is_healthy() is False

    def test_post_init_skipped_when_init_failed(self):
        """Test that post_init marks engine unhealthy when init failed."""
        config = LMCacheEngineConfig.from_defaults()
        factory = _make_mock_factory()
        factory.get_or_create_metadata.side_effect = RuntimeError("Test init failure")

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        mock_engine = MagicMock()
        manager._lmcache_engine = mock_engine

        manager.post_init()

        # Engine should be marked as init failed
        mock_engine.mark_init_failed.assert_called_once()
        # Manager should report unhealthy
        assert manager.is_healthy() is False

    def test_post_init_exception_makes_unhealthy(self):
        """Test that exception during post_init makes system unhealthy."""
        config = LMCacheEngineConfig.from_defaults()
        mock_engine = MagicMock()
        mock_engine.post_init.side_effect = RuntimeError("Test post_init failure")
        factory = _make_mock_factory(engine=mock_engine)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        manager.post_init()

        # Verify engine was marked as failed
        mock_engine.mark_init_failed.assert_called_once()
        # Manager should report unhealthy
        assert manager.is_healthy() is False

    def test_unhealthy_engine_makes_manager_unhealthy(self):
        """Test that manager is unhealthy when engine is unhealthy."""
        config = LMCacheEngineConfig.from_defaults()
        mock_engine = MagicMock()
        mock_engine.is_healthy.return_value = False
        factory = _make_mock_factory(engine=mock_engine)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        assert manager.is_healthy() is False

    def test_healthy_when_all_components_healthy(self):
        """Test that manager is healthy when all components are healthy."""
        config = LMCacheEngineConfig.from_defaults()
        mock_engine = MagicMock()
        mock_engine.is_healthy.return_value = True
        factory = _make_mock_factory(engine=mock_engine)

        manager = LMCacheManager(
            config=config,
            service_factory=factory,
        )

        mock_health_monitor = MagicMock()
        mock_health_monitor.is_healthy.return_value = True
        manager._health_monitor = mock_health_monitor

        assert manager.is_healthy() is True


class TestLMCacheEngineInitFailure:
    """Tests for LMCacheEngine init failure behavior on lookup/retrieve."""

    def test_lookup_returns_zero_when_init_failed(self):
        """Test that lookup returns 0 when engine is marked as init failed."""
        # First Party
        from lmcache.v1.cache_engine import LMCacheEngine

        engine = MagicMock(spec=LMCacheEngine)
        engine._init_failed = False
        engine._health_monitor = None

        # Attach real methods to the mock
        engine.mark_init_failed = LMCacheEngine.mark_init_failed.__get__(
            engine, LMCacheEngine
        )
        engine.is_healthy = LMCacheEngine.is_healthy.__get__(engine, LMCacheEngine)

        # Call mark_init_failed and verify is_healthy returns False
        engine.mark_init_failed("Test reason")
        assert engine.is_healthy() is False

    def test_retrieve_returns_empty_mask_when_init_failed(self):
        """Test retrieve returns all-False mask when engine init failed."""
        # First Party
        from lmcache.v1.cache_engine import LMCacheEngine

        engine = MagicMock(spec=LMCacheEngine)
        engine._init_failed = False
        engine._health_monitor = None

        # Attach real methods to the mock
        engine.mark_init_failed = LMCacheEngine.mark_init_failed.__get__(
            engine, LMCacheEngine
        )
        engine.is_healthy = LMCacheEngine.is_healthy.__get__(engine, LMCacheEngine)

        # Mark as init failed and verify is_healthy returns False
        # (which would cause retrieve to return all-False mask)
        engine.mark_init_failed("Test reason")
        assert engine.is_healthy() is False

    def test_mark_init_failed_makes_engine_unhealthy(self):
        """Test that mark_init_failed makes engine report unhealthy."""
        # First Party
        from lmcache.v1.cache_engine import LMCacheEngine

        engine = MagicMock(spec=LMCacheEngine)
        engine._init_failed = False
        engine._health_monitor = None

        # Attach real methods to the mock
        engine.mark_init_failed = LMCacheEngine.mark_init_failed.__get__(
            engine, LMCacheEngine
        )
        engine.is_healthy = LMCacheEngine.is_healthy.__get__(engine, LMCacheEngine)

        # Verify initially healthy
        assert engine.is_healthy() is True

        # Call the method under test
        engine.mark_init_failed("Test reason")

        # Assert on the outcome
        assert engine.is_healthy() is False
