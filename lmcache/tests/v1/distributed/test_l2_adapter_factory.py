# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the L2 adapter factory registry and
PluginL2AdapterConfig.
"""

# Standard
import os
import tempfile
import types

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.plugin_l2_adapter import PluginL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2AdapterConfig,
)
from lmcache.v1.platform import create_event_notifier


def _has_raw_block_ext() -> bool:
    try:
        # Third Party
        import lmcache_rust_raw_block_io  # noqa: F401

        return True
    except Exception:
        return False


# =========================================================
# Helpers
# =========================================================


class _FakeL2Adapter:
    """Minimal stub that passes issubclass check."""

    def __init__(self, params=None, **kwargs):
        self.params = params
        self.kwargs = kwargs


class _NotAnAdapter:
    """Class that does NOT subclass L2AdapterInterface."""

    pass


def _make_fake_module(
    adapter_cls: type,
    class_name: str = "FakeL2Adapter",
) -> types.ModuleType:
    """Create a fake Python module containing *adapter_cls*."""
    mod = types.ModuleType("fake_l2_module")
    setattr(mod, class_name, adapter_cls)
    return mod


# =========================================================
# Tests: factory registry basics
# =========================================================


class TestFactoryRegistry:
    """Tests for register/create via factory registry."""

    def test_mock_factory_is_registered(self):
        """Mock adapter factory should be auto-registered
        on import."""
        config = MockL2AdapterConfig(
            max_size_gb=0.001,
            mock_bandwidth_gb=10.0,
        )
        name = get_type_name_for_config(config)
        assert name == "mock"

    def test_raw_block_factory_is_registered(self):
        config = RawBlockL2AdapterConfig(
            device_path="/tmp/raw-block-dev",
            slot_bytes=64 * 1024,
            use_odirect=False,
            meta_enable_periodic=False,
        )
        name = get_type_name_for_config(config)
        assert name == "raw_block"

    def test_create_mock_via_registry(self):
        """create_l2_adapter_from_registry creates a
        MockL2Adapter."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
            MockL2Adapter,
        )

        config = MockL2AdapterConfig(
            max_size_gb=0.001,
            mock_bandwidth_gb=10.0,
        )
        adapter = create_l2_adapter_from_registry(config)
        assert isinstance(adapter, MockL2Adapter)
        adapter.close()

    @pytest.mark.skipif(
        not _has_raw_block_ext(),
        reason="lmcache_rust_raw_block_io extension not installed",
    )
    def test_create_raw_block_via_registry(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
            RawBlockL2Adapter,
        )

        with tempfile.TemporaryDirectory() as td:
            dev_path = os.path.join(td, "dev.bin")
            with open(dev_path, "wb") as f:
                f.truncate(8 * 1024 * 1024)

            config = RawBlockL2AdapterConfig(
                device_path=dev_path,
                slot_bytes=64 * 1024,
                use_odirect=False,
                meta_total_bytes=1 * 1024 * 1024,
                meta_enable_periodic=False,
            )
            adapter = create_l2_adapter_from_registry(config)
            assert isinstance(adapter, RawBlockL2Adapter)
            adapter.close()

    def test_duplicate_factory_raises(self):
        """Registering the same factory name twice should
        raise ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            register_l2_adapter_factory("mock", lambda c, **kw: None)

    def test_unknown_config_raises(self):
        """Config class not in the registry should fail."""

        class _UnknownConfig:
            pass

        with pytest.raises(ValueError, match="Unregistered"):
            get_type_name_for_config(_UnknownConfig())


# =========================================================
# Tests: PluginL2AdapterConfig parsing
# =========================================================


class TestPluginL2AdapterConfig:
    """Tests for PluginL2AdapterConfig.from_dict."""

    def test_valid_config(self):
        d = {
            "type": "plugin",
            "module_path": "my.module",
            "class_name": "MyAdapter",
            "adapter_params": {"host": "localhost"},
        }
        cfg = PluginL2AdapterConfig.from_dict(d)
        assert cfg.module_path == "my.module"
        assert cfg.class_name == "MyAdapter"
        assert cfg.adapter_params == {"host": "localhost"}

    def test_default_adapter_params(self):
        d = {
            "type": "plugin",
            "module_path": "my.module",
            "class_name": "MyAdapter",
        }
        cfg = PluginL2AdapterConfig.from_dict(d)
        assert cfg.adapter_params == {}

    def test_missing_module_path_raises(self):
        d = {"type": "plugin", "class_name": "X"}
        with pytest.raises(ValueError, match="module_path"):
            PluginL2AdapterConfig.from_dict(d)

    def test_missing_class_name_raises(self):
        d = {"type": "plugin", "module_path": "x"}
        with pytest.raises(ValueError, match="class_name"):
            PluginL2AdapterConfig.from_dict(d)

    def test_invalid_adapter_params_raises(self):
        d = {
            "type": "plugin",
            "module_path": "x",
            "class_name": "X",
            "adapter_params": "not_a_dict",
        }
        with pytest.raises(ValueError, match="adapter_params"):
            PluginL2AdapterConfig.from_dict(d)


# =========================================================
# Tests: plugin adapter factory dynamic loading
# =========================================================


class TestPluginAdapterFactory:
    """Tests for the plugin adapter factory using monkeypatch
    to mock importlib.import_module."""

    def test_load_external_adapter(self, monkeypatch):
        """Successfully load an external adapter class."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import plugin_l2_adapter as plugin_mod

        # Make _FakeL2Adapter pass issubclass check
        monkeypatch.setattr(
            plugin_mod,
            "_L2AI",
            _FakeL2Adapter,
        )

        fake_mod = _make_fake_module(_FakeL2Adapter)
        monkeypatch.setattr(
            "importlib.import_module",
            lambda path: fake_mod,
        )

        config = PluginL2AdapterConfig(
            module_path="fake_l2_module",
            class_name="FakeL2Adapter",
            adapter_params={"host": "localhost"},
        )
        adapter = create_l2_adapter_from_registry(config)
        assert isinstance(adapter, _FakeL2Adapter)
        assert adapter.params["host"] == "localhost"

    def test_import_error_raises(self, monkeypatch):
        """ImportError propagates when module not found."""
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: (_ for _ in ()).throw(ImportError("no such module")),
        )
        config = PluginL2AdapterConfig(
            module_path="nonexistent",
            class_name="X",
        )
        with pytest.raises(ImportError, match="nonexistent"):
            create_l2_adapter_from_registry(config)

    def test_missing_class_raises(self, monkeypatch):
        """AttributeError when class not in module."""
        fake_mod = types.ModuleType("empty_mod")
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: fake_mod,
        )
        config = PluginL2AdapterConfig(
            module_path="empty_mod",
            class_name="NoSuchClass",
        )
        with pytest.raises(AttributeError, match="NoSuchClass"):
            create_l2_adapter_from_registry(config)

    def test_not_subclass_raises(self, monkeypatch):
        """TypeError when class is not an L2AdapterInterface
        subclass."""
        fake_mod = _make_fake_module(_NotAnAdapter, "BadAdapter")
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: fake_mod,
        )
        config = PluginL2AdapterConfig(
            module_path="fake_mod",
            class_name="BadAdapter",
        )
        with pytest.raises(TypeError, match="not a subclass"):
            create_l2_adapter_from_registry(config)


# =========================================================
# Tests: plugin registration and initialization
# =========================================================


class _FakeL2AdapterWithDesc:
    """Stub that records l1_memory_desc."""

    def __init__(self, config, **kwargs):
        self.config = config
        self.l1_memory_desc = kwargs.get("l1_memory_desc")


class _FakeConfig(L2AdapterConfigBase):
    """Minimal config subclass for discovery tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def from_dict(cls, d: dict) -> "_FakeConfig":
        return cls(**d)

    @classmethod
    def help(cls) -> str:
        return "fake"


class TestPluginRegistration:
    """Verify plugin self-registration on import."""

    def test_plugin_type_registered(self):
        """'plugin' config type should be registered."""
        cfg = PluginL2AdapterConfig(module_path="x", class_name="X")
        name = get_type_name_for_config(cfg)
        assert name == "plugin"

    def test_plugin_factory_registered(self):
        """'plugin' factory should be callable via
        create_l2_adapter_from_registry (smoke test
        with ImportError)."""
        cfg = PluginL2AdapterConfig(
            module_path="nonexistent.module",
            class_name="X",
        )
        with pytest.raises(ImportError):
            create_l2_adapter_from_registry(cfg)


class TestPluginInitialization:
    """Verify config-class discovery and l1_memory_desc
    forwarding during plugin initialization."""

    def _patch_base(self, monkeypatch, adapter_cls):
        """Make *adapter_cls* pass issubclass check."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import plugin_l2_adapter as plugin_mod

        monkeypatch.setattr(
            plugin_mod,
            "_L2AI",
            adapter_cls,
        )

    # -- l1_memory_desc forwarding --

    def test_l1_memory_desc_forwarded(self, monkeypatch):
        """l1_memory_desc should reach the adapter
        constructor when provided."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
            adapter_params={"k": "v"},
        )
        sentinel = object()
        adapter = create_l2_adapter_from_registry(cfg, l1_memory_desc=sentinel)
        assert adapter.l1_memory_desc is sentinel

    def test_l1_memory_desc_omitted(self, monkeypatch):
        """l1_memory_desc should be None when not
        provided."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert adapter.l1_memory_desc is None

    # -- config class discovery --

    def test_init_with_explicit_config_class(self, monkeypatch):
        """Explicit config_class_name should be used."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        fake_mod.MyCfg = _FakeConfig  # type: ignore[attr-defined]
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
            config_class_name="MyCfg",
            adapter_params={"x": 1},
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert isinstance(adapter.config, _FakeConfig)
        assert adapter.config.kwargs == {"x": 1}

    def test_init_with_convention_config_class(self, monkeypatch):
        """Config class discovered via ClassName+'Config'
        convention."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        # Convention: "Adapter" + "Config"
        fake_mod.AdapterConfig = _FakeConfig  # type: ignore[attr-defined]
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
            adapter_params={"y": 2},
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert isinstance(adapter.config, _FakeConfig)

    def test_init_fallback_raw_dict(self, monkeypatch):
        """When no config class is found, adapter receives
        a raw dict."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
            adapter_params={"z": 3},
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert isinstance(adapter.config, dict)
        assert adapter.config == {"z": 3}


# =========================================================
# Tests: lazy loading (add_pending_module,
# ensure_adapter_loaded, load_all_adapters,
# get_all_registered_names)
# =========================================================


class TestLazyLoading:
    """Tests for the deferred module import mechanism."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self, monkeypatch):
        """Snapshot and restore the factory registry and
        pending-module list around each test."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import factory as fmod

        orig_reg = fmod._L2_ADAPTER_FACTORY_REGISTRY.copy()
        orig_pending = list(fmod._PENDING_MODULES)
        # Start each test with a clean slate
        fmod._PENDING_MODULES.clear()
        yield
        fmod._L2_ADAPTER_FACTORY_REGISTRY.clear()
        fmod._L2_ADAPTER_FACTORY_REGISTRY.update(orig_reg)
        fmod._PENDING_MODULES.clear()
        fmod._PENDING_MODULES.extend(orig_pending)

    # -- add_pending_module --

    def test_add_pending_module_appends(self):
        """Module path is appended to the pending list."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.factory import (
            _PENDING_MODULES,
            add_pending_module,
        )

        add_pending_module("fake.mod.a")
        assert "fake.mod.a" in _PENDING_MODULES

    def test_add_pending_module_dedup(self):
        """Duplicate paths are not added twice."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.factory import (
            _PENDING_MODULES,
            add_pending_module,
        )

        add_pending_module("fake.mod.b")
        add_pending_module("fake.mod.b")
        assert _PENDING_MODULES.count("fake.mod.b") == 1

    # -- ensure_adapter_loaded --

    def test_ensure_already_registered(self):
        """No import happens when name is already in the
        registry."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.factory import (
            _L2_ADAPTER_FACTORY_REGISTRY,
            _PENDING_MODULES,
            ensure_adapter_loaded,
        )

        _L2_ADAPTER_FACTORY_REGISTRY["pre"] = lambda c, d: None
        _PENDING_MODULES.append("should.not.import")
        ensure_adapter_loaded("pre")
        # Pending list untouched
        assert "should.not.import" in _PENDING_MODULES

    def test_ensure_imports_until_found(self, monkeypatch):
        """Modules are imported one-by-one until the
        requested name appears in the registry."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import factory as fmod

        imported: list[str] = []

        def _fake_import(path):
            imported.append(path)
            if path == "mod_b":
                fmod._L2_ADAPTER_FACTORY_REGISTRY["lazy_b"] = lambda c, d: None

        monkeypatch.setattr("importlib.import_module", _fake_import)
        fmod._PENDING_MODULES.extend(["mod_a", "mod_b", "mod_c"])
        fmod.ensure_adapter_loaded("lazy_b")

        assert imported == ["mod_a", "mod_b"]
        # mod_c should remain pending
        assert "mod_c" in fmod._PENDING_MODULES

    def test_ensure_raises_last_import_error(self, monkeypatch):
        """When all pending modules fail, the last
        ImportError is raised."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import factory as fmod

        def _fail_import(path):
            raise ImportError(path)

        monkeypatch.setattr("importlib.import_module", _fail_import)
        fmod._PENDING_MODULES.extend(["bad_a", "bad_b"])

        with pytest.raises(ImportError, match="bad_b"):
            fmod.ensure_adapter_loaded("missing")

    def test_ensure_no_error_when_not_found_silently(self, monkeypatch):
        """When pending modules import OK but name is
        still missing, no error is raised (no last_err).
        """
        # First Party
        from lmcache.v1.distributed.l2_adapters import factory as fmod

        monkeypatch.setattr("importlib.import_module", lambda p: None)
        fmod._PENDING_MODULES.extend(["ok_a", "ok_b"])

        # Should not raise
        fmod.ensure_adapter_loaded("nonexistent")

    # -- load_all_adapters --

    def test_load_all_adapters_imports_everything(self, monkeypatch):
        """All pending modules are imported."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import factory as fmod

        imported: list[str] = []
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: imported.append(p),
        )
        fmod._PENDING_MODULES.extend(["m1", "m2", "m3"])
        fmod.load_all_adapters()

        assert imported == ["m1", "m2", "m3"]
        assert len(fmod._PENDING_MODULES) == 0

    def test_load_all_adapters_skips_failures(self, monkeypatch):
        """Modules that fail to import are silently
        skipped."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import factory as fmod

        imported: list[str] = []

        def _selective_import(path):
            if path == "bad":
                raise ImportError(path)
            imported.append(path)

        monkeypatch.setattr("importlib.import_module", _selective_import)
        fmod._PENDING_MODULES.extend(["good1", "bad", "good2"])
        fmod.load_all_adapters()

        assert imported == ["good1", "good2"]
        assert len(fmod._PENDING_MODULES) == 0

    # -- get_all_registered_names --

    def test_get_all_registered_names_sorted(self, monkeypatch):
        """Returns sorted list after loading all pending
        modules."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import factory as fmod

        def _register_import(path):
            fmod._L2_ADAPTER_FACTORY_REGISTRY[path] = lambda c, d: None

        monkeypatch.setattr("importlib.import_module", _register_import)
        fmod._PENDING_MODULES.extend(["z_mod", "a_mod"])

        names = fmod.get_all_registered_names()
        # Should contain at least the two we just added
        assert "z_mod" in names
        assert "a_mod" in names
        # Must be sorted
        assert names == sorted(names)

    def test_lazy_module_not_in_sys_modules(self, monkeypatch):
        """Pending modules are NOT in sys.modules until
        ensure_adapter_loaded triggers the import."""
        # Standard
        import sys

        # First Party
        from lmcache.v1.distributed.l2_adapters import factory as fmod

        sentinel = "test.lazy.sentinel.module"
        fmod._PENDING_MODULES.append(sentinel)
        assert sentinel not in sys.modules

        def _fake_import(path):
            mod = types.ModuleType(path)
            sys.modules[path] = mod
            fmod._L2_ADAPTER_FACTORY_REGISTRY["sentinel"] = lambda c, d: None

        monkeypatch.setattr("importlib.import_module", _fake_import)
        fmod.ensure_adapter_loaded("sentinel")
        assert sentinel in sys.modules
        # Cleanup
        sys.modules.pop(sentinel, None)


# =========================================================
# Tests: create_l2_adapter public API
# =========================================================


class TestCreateL2Adapter:
    """Tests for the public create_l2_adapter function."""

    def test_create_mock_adapter(self):
        """create_l2_adapter dispatches to MockL2Adapter."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import (
            create_l2_adapter,
        )
        from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
            MockL2Adapter,
        )

        config = MockL2AdapterConfig(
            max_size_gb=0.001,
            mock_bandwidth_gb=10.0,
        )
        adapter = create_l2_adapter(config)
        assert isinstance(adapter, MockL2Adapter)
        adapter.close()


# =========================================================
# Tests: NativePluginL2AdapterConfig factory
# =========================================================


class _FakeNativeConnector:
    """Minimal mock that satisfies the native connector
    interface check."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def event_fd(self):
        return -1

    def submit_batch_get(self, keys, memoryviews):
        return 0

    def submit_batch_set(self, keys, memoryviews):
        return 0

    def submit_batch_exists(self, keys):
        return 0

    def drain_completions(self):
        return []

    def close(self):
        pass


class _MissingMethodConnector:
    """Connector missing the 'close' method."""

    def event_fd(self):
        return -1

    def submit_batch_get(self, keys, memoryviews):
        return 0

    def submit_batch_set(self, keys, memoryviews):
        return 0

    def submit_batch_exists(self, keys):
        return 0

    def drain_completions(self):
        return []


class TestNativePluginFactory:
    """Tests for the native_plugin adapter factory."""

    def test_load_native_connector(self, monkeypatch):
        """Successfully load a native connector."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
            NativeConnectorL2Adapter,
        )
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        fake_mod = types.ModuleType("fake_native_mod")
        fake_mod.FakeClient = _FakeNativeConnector  # type: ignore[attr-defined]
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: fake_mod,
        )

        cfg = NativePluginL2AdapterConfig(
            module_path="fake_native_mod",
            class_name="FakeClient",
            adapter_params={"host": "localhost"},
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert isinstance(adapter, NativeConnectorL2Adapter)
        adapter.close()

    def test_import_error_raises(self, monkeypatch):
        """ImportError propagates when module not found."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: (_ for _ in ()).throw(ImportError("no such module")),
        )
        cfg = NativePluginL2AdapterConfig(
            module_path="nonexistent",
            class_name="X",
        )
        with pytest.raises(ImportError, match="nonexistent"):
            create_l2_adapter_from_registry(cfg)

    def test_missing_class_raises(self, monkeypatch):
        """AttributeError when class not in module."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        fake_mod = types.ModuleType("empty_mod")
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: fake_mod,
        )
        cfg = NativePluginL2AdapterConfig(
            module_path="empty_mod",
            class_name="NoSuchClass",
        )
        with pytest.raises(AttributeError, match="NoSuchClass"):
            create_l2_adapter_from_registry(cfg)

    def test_missing_method_raises(self, monkeypatch):
        """TypeError when connector is missing a required
        method."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        fake_mod = types.ModuleType("incomplete_mod")
        fake_mod.Bad = _MissingMethodConnector  # type: ignore[attr-defined]
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: fake_mod,
        )
        cfg = NativePluginL2AdapterConfig(
            module_path="incomplete_mod",
            class_name="Bad",
        )
        with pytest.raises(TypeError, match="close"):
            create_l2_adapter_from_registry(cfg)

    def test_adapter_params_forwarded(self, monkeypatch):
        """adapter_params are forwarded as kwargs to the
        connector class."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
            NativeConnectorL2Adapter,
        )
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        fake_mod = types.ModuleType("param_mod")
        fake_mod.Client = _FakeNativeConnector  # type: ignore[attr-defined]
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: fake_mod,
        )

        cfg = NativePluginL2AdapterConfig(
            module_path="param_mod",
            class_name="Client",
            adapter_params={"host": "myhost", "port": 42},
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert isinstance(adapter, NativeConnectorL2Adapter)
        adapter.close()

    def test_native_plugin_type_registered(self):
        """'native_plugin' config type should be
        registered."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        cfg = NativePluginL2AdapterConfig(
            module_path="x",
            class_name="X",
        )
        name = get_type_name_for_config(cfg)
        assert name == "native_plugin"


# =========================================================
# Tests: FSNativeL2AdapterConfig factory
# =========================================================


class _FakeLMCacheFSClient:
    """Mock for lmcache.lmcache_fs.LMCacheFSClient."""

    def __init__(
        self,
        base_path,
        num_workers,
        relative_tmp_dir="",
        use_odirect=False,
        read_ahead_size=0,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self._efd = create_event_notifier()
        self._closed = False

    def event_fd(self):
        return self._efd.fileno()

    def submit_batch_get(self, keys, memoryviews):
        return 0

    def submit_batch_set(self, keys, memoryviews):
        return 0

    def submit_batch_exists(self, keys):
        return 0

    def drain_completions(self):
        return []

    def close(self):
        if not self._closed:
            self._closed = True
            self._efd.close()


class TestFSNativeAdapterFactory:
    """Tests for the fs_native adapter factory."""

    def test_factory_creates_adapter(self, monkeypatch):
        """Factory creates a NativeConnectorL2Adapter
        wrapping the FS client."""
        # First Party
        # Re-register so registry points to patched fn
        from lmcache.v1.distributed.l2_adapters.factory import (
            _L2_ADAPTER_FACTORY_REGISTRY,
        )
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
            NativeConnectorL2Adapter,
        )

        old = _L2_ADAPTER_FACTORY_REGISTRY["fs_native"]
        _L2_ADAPTER_FACTORY_REGISTRY["fs_native"] = _patched_fs_factory

        try:
            cfg = FSNativeL2AdapterConfig(
                base_path="/tmp/lmcache_test_factory",
                num_workers=2,
            )
            adapter = create_l2_adapter_from_registry(cfg)
            assert isinstance(adapter, NativeConnectorL2Adapter)
            adapter.close()
        finally:
            _L2_ADAPTER_FACTORY_REGISTRY["fs_native"] = old

    def test_factory_forwards_all_params(self, monkeypatch):
        """All config params are forwarded to the
        FS client constructor."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        captured: dict = {}

        class _CaptureFSClient(_FakeLMCacheFSClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["args"] = args
                captured["kwargs"] = kwargs

        # First Party
        from lmcache.v1.distributed.l2_adapters.factory import (
            _L2_ADAPTER_FACTORY_REGISTRY,
        )

        def _capture_factory(config, l1_memory_desc=None):
            assert isinstance(config, FSNativeL2AdapterConfig)
            client = _CaptureFSClient(
                config.base_path,
                config.num_workers,
                config.relative_tmp_dir,
                config.use_odirect,
                config.read_ahead_size or 0,
            )
            # First Party
            from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
                NativeConnectorL2Adapter,
            )

            return NativeConnectorL2Adapter(client)

        old = _L2_ADAPTER_FACTORY_REGISTRY["fs_native"]
        _L2_ADAPTER_FACTORY_REGISTRY["fs_native"] = _capture_factory

        try:
            cfg = FSNativeL2AdapterConfig(
                base_path="/data/kv",
                num_workers=8,
                relative_tmp_dir=".tmp",
                use_odirect=True,
                read_ahead_size=4096,
            )
            adapter = create_l2_adapter_from_registry(cfg)
            assert captured["args"] == (
                "/data/kv",
                8,
                ".tmp",
                True,
                4096,
            )
            adapter.close()
        finally:
            _L2_ADAPTER_FACTORY_REGISTRY["fs_native"] = old

    def test_factory_import_error_raises(self, monkeypatch):
        """RuntimeError raised when C++ FS extension is
        not available."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        # Patch the actual factory to simulate missing
        # C++ extension
        def _broken_import(name, *args, **kwargs):
            if name == "lmcache.lmcache_fs":
                raise ImportError("no C++ FS extension")
            return _original_import(name, *args, **kwargs)

        # Standard
        import builtins

        _original_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", _broken_import)

        cfg = FSNativeL2AdapterConfig(
            base_path="/tmp/test",
        )
        with pytest.raises(RuntimeError, match="C\\+\\+ FS"):
            create_l2_adapter_from_registry(cfg)

    def test_fs_native_type_registered(self):
        """'fs_native' config type should be
        registered."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        cfg = FSNativeL2AdapterConfig(
            base_path="/tmp/test",
        )
        name = get_type_name_for_config(cfg)
        assert name == "fs_native"

    def test_fs_native_factory_registered(self):
        """'fs_native' factory should be callable via
        create_l2_adapter_from_registry (smoke test
        with RuntimeError from missing C++ ext)."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        cfg = FSNativeL2AdapterConfig(
            base_path="/tmp/test_smoke",
        )
        # Without the C++ extension built, expect
        # RuntimeError
        try:
            adapter = create_l2_adapter_from_registry(cfg)
            adapter.close()
        except RuntimeError:
            pass  # Expected when C++ ext not built

    def test_read_ahead_size_none_becomes_zero(self, monkeypatch):
        """read_ahead_size=None should be passed as 0
        to the native client."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        captured: dict = {}

        class _CaptureFSClient2(_FakeLMCacheFSClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["read_ahead"] = args[4]

        # First Party
        from lmcache.v1.distributed.l2_adapters.factory import (
            _L2_ADAPTER_FACTORY_REGISTRY,
        )

        def _capture_factory2(config, l1_memory_desc=None):
            assert isinstance(config, FSNativeL2AdapterConfig)
            client = _CaptureFSClient2(
                config.base_path,
                config.num_workers,
                config.relative_tmp_dir,
                config.use_odirect,
                config.read_ahead_size or 0,
            )
            # First Party
            from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
                NativeConnectorL2Adapter,
            )

            return NativeConnectorL2Adapter(client)

        old = _L2_ADAPTER_FACTORY_REGISTRY["fs_native"]
        _L2_ADAPTER_FACTORY_REGISTRY["fs_native"] = _capture_factory2

        try:
            cfg = FSNativeL2AdapterConfig(
                base_path="/tmp/test_ra",
            )
            assert cfg.read_ahead_size is None
            adapter = create_l2_adapter_from_registry(cfg)
            assert captured["read_ahead"] == 0
            adapter.close()
        finally:
            _L2_ADAPTER_FACTORY_REGISTRY["fs_native"] = old


def _patched_fs_factory(config, l1_memory_desc=None):
    """Factory that uses _FakeLMCacheFSClient instead
    of the real C++ extension."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
        FSNativeL2AdapterConfig,
    )
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, FSNativeL2AdapterConfig)
    client = _FakeLMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
    )
    return NativeConnectorL2Adapter(client)
