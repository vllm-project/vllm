# SPDX-License-Identifier: Apache-2.0
# Standard
from types import ModuleType
import asyncio
import importlib
import pkgutil

# First Party
from lmcache.v1.storage_backend.connector import ConnectorAdapter, ConnectorManager


def test_discover_adapters(monkeypatch):
    """Validate adapter discovery filters modules and tolerates failures.

    Args:
        monkeypatch: Pytest fixture for patching module discovery.
    """

    class GoodAdapter(ConnectorAdapter):
        """Connector adapter used to verify discovery success."""

        def __init__(self) -> None:
            super().__init__("good://")

        def create_connector(self, context):
            raise AssertionError("not used")

    class BrokenAdapter(ConnectorAdapter):
        """Connector adapter that fails during instantiation."""

        def __init__(self) -> None:
            raise RuntimeError("boom")

        def create_connector(self, context):
            raise AssertionError("not used")

    good_module = ModuleType("good_adapter")
    good_module.GoodAdapter = GoodAdapter
    good_module.NotAnAdapter = object

    broken_module = ModuleType("broken_adapter")
    broken_module.BrokenAdapter = BrokenAdapter

    non_adapter_module = ModuleType("non_adapter")
    non_adapter_module.NotAnAdapter = object

    fake_package = ModuleType("lmcache.v1.storage_backend.connector")
    fake_package.__path__ = []  # type: ignore[attr-defined]

    def fake_iter_modules(_path):
        """Yield a mix of adapter and non-adapter module names."""
        yield None, "good_adapter", False
        yield None, "_private_adapter", False
        yield None, "non_adapter", False
        yield None, "broken_adapter", False
        yield None, "missing_adapter", False

    def fake_import_module(name):
        """Return fake modules or simulate import failure."""
        if name == "lmcache.v1.storage_backend.connector":
            return fake_package
        if name.endswith(".good_adapter"):
            return good_module
        if name.endswith(".broken_adapter"):
            return broken_module
        if name.endswith(".non_adapter"):
            return non_adapter_module
        if name.endswith(".missing_adapter"):
            raise ImportError("missing module")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    loop = asyncio.new_event_loop()
    try:
        manager = ConnectorManager("lm://localhost:1234", loop, None)
    finally:
        loop.close()

    assert len(manager.adapters) == 1
    assert isinstance(manager.adapters[0], GoodAdapter)
