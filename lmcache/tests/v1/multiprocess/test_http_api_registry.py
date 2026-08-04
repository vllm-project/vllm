# SPDX-License-Identifier: Apache-2.0
"""
Tests for HTTPAPIRegistry auto-discovery and the shared
``discover_api_routers`` utility.

Covers:
- Module filtering (only ``*_api`` modules are loaded)
- Router type-checking (non-APIRouter ``router`` attrs skipped)
- Nested router inclusion into the FastAPI app
- Missing directory gracefully handled
"""

# Standard
import sys

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.multiprocess.http_api_registry import (
    HTTPAPIRegistry,
)
from lmcache.v1.utils.router_discovery import discover_api_routers

# ------------------------------------------------------------------ #
#  discover_api_routers
# ------------------------------------------------------------------ #


class TestDiscoverApiRouters:
    """Unit tests for the shared discovery function."""

    def test_discovers_api_modules(self, tmp_path):
        """Modules ending with ``_api`` that expose an APIRouter
        are discovered."""
        # Create two valid _api modules
        (tmp_path / "foo_api.py").write_text(
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/foo')\n"
            "async def foo(): return 'ok'\n"
        )
        (tmp_path / "bar_api.py").write_text(
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/bar')\n"
            "async def bar(): return 'ok'\n"
        )
        # __init__.py to make it a package
        (tmp_path / "__init__.py").write_text("")

        sys.path.insert(0, str(tmp_path.parent))
        try:
            routers = discover_api_routers(tmp_path, pkg_name := tmp_path.name)
        finally:
            sys.path.pop(0)
            # cleanup imported modules
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

        assert len(routers) == 2

    def test_skips_non_api_modules(self, tmp_path):
        """Modules without the ``_api`` suffix are ignored."""
        (tmp_path / "helpers.py").write_text(
            "from fastapi import APIRouter\nrouter = APIRouter()\n"
        )
        (tmp_path / "__init__.py").write_text("")

        sys.path.insert(0, str(tmp_path.parent))
        try:
            routers = discover_api_routers(tmp_path, pkg_name := tmp_path.name)
        finally:
            sys.path.pop(0)
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

        assert len(routers) == 0

    def test_skips_module_without_router(self, tmp_path):
        """Modules that lack a ``router`` attribute are skipped."""
        (tmp_path / "empty_api.py").write_text("# no router here\nvalue = 42\n")
        (tmp_path / "__init__.py").write_text("")

        sys.path.insert(0, str(tmp_path.parent))
        try:
            routers = discover_api_routers(tmp_path, pkg_name := tmp_path.name)
        finally:
            sys.path.pop(0)
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

        assert len(routers) == 0

    def test_skips_non_apirouter_router(self, tmp_path):
        """A ``router`` attr that is not an APIRouter is skipped."""
        (tmp_path / "bad_api.py").write_text("router = 'not a router'\n")
        (tmp_path / "__init__.py").write_text("")

        sys.path.insert(0, str(tmp_path.parent))
        try:
            routers = discover_api_routers(tmp_path, pkg_name := tmp_path.name)
        finally:
            sys.path.pop(0)
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

        assert len(routers) == 0

    def test_exclude_filters_named_modules(self, tmp_path):
        """Modules whose base name is listed in ``exclude`` are skipped."""
        (tmp_path / "keep_api.py").write_text(
            "from fastapi import APIRouter\nrouter = APIRouter()\n"
        )
        (tmp_path / "drop_api.py").write_text(
            "from fastapi import APIRouter\nrouter = APIRouter()\n"
        )
        (tmp_path / "__init__.py").write_text("")

        sys.path.insert(0, str(tmp_path.parent))
        try:
            routers = discover_api_routers(
                tmp_path,
                pkg_name := tmp_path.name,
                exclude={"drop_api"},
            )
        finally:
            sys.path.pop(0)
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

        assert len(routers) == 1

    def test_exclude_none_is_default(self, tmp_path):
        """Passing ``exclude=None`` keeps all discoverable modules."""
        (tmp_path / "a_api.py").write_text(
            "from fastapi import APIRouter\nrouter = APIRouter()\n"
        )
        (tmp_path / "b_api.py").write_text(
            "from fastapi import APIRouter\nrouter = APIRouter()\n"
        )
        (tmp_path / "__init__.py").write_text("")

        sys.path.insert(0, str(tmp_path.parent))
        try:
            routers = discover_api_routers(
                tmp_path,
                pkg_name := tmp_path.name,
                exclude=None,
            )
        finally:
            sys.path.pop(0)
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

        assert len(routers) == 2


# ------------------------------------------------------------------ #
#  HTTPAPIRegistry integration
# ------------------------------------------------------------------ #


class TestHTTPAPIRegistry:
    """Integration tests using the real http_apis/ modules."""

    @pytest.fixture
    def app_with_registry(self):
        """Create a fresh FastAPI app and register all HTTP APIs."""
        test_app = FastAPI()
        registry = HTTPAPIRegistry(test_app)
        registry.register_all_apis()
        return test_app

    def test_root_endpoint_registered(self, app_with_registry):
        """The ``/`` endpoint from info_api is reachable."""
        client = TestClient(app_with_registry)
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_all_expected_routes_present(self, app_with_registry):
        """All four expected routes are registered."""
        routes = set(app_with_registry.openapi()["paths"])
        expected = {"/", "/healthcheck", "/cache/clear", "/status"}
        assert expected.issubset(routes)
