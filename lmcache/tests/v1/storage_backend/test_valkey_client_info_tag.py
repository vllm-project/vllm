# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the version-gated ``client_info_tag`` support in the shared
``ValkeyWorkerPool`` (used by both ``ValkeyConnector`` and the MP-mode
``ValkeyL2Adapter``).

The pool tags LMCache's GLIDE connections as
``GlidePySync(lmcache:<version>)`` via GLIDE's ``client_info_tag`` client
config option, but only when the installed ``valkey-glide`` build supports
it (>= 2.5.0, valkey-io/valkey-glide#6389). Older builds would raise
``TypeError`` on the unknown kwarg, so the pool feature-detects support
from the config constructor signature and applies the tag behind that gate.

These tests inject a fake ``glide_sync`` module so the real
``ValkeyWorkerPool._get_client`` config-building path is exercised without
``glide_sync`` or a real Valkey server.
"""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.valkey.worker_pool import (
    CLIENT_INFO_TAG_MIN_GLIDE_VERSION,
    ValkeyWorkerPool,
    _glide_config_supports_client_info_tag,
    _lmcache_client_info_tag,
)

# ── Fake glide_sync building blocks ─────────────────────────────────────
#
# Config classes come in two flavours whose constructor signatures differ:
# the *Tagged* variants accept ``client_info_tag`` (glide >= 2.5.0), the
# *Untagged* ones do not (older builds). Defining them as distinct top-level
# classes — rather than redefining one name under if/else — keeps mypy happy
# and lets ``_make_fake_glide_sync`` pick the right pair. Each records the
# kwargs it cares about so tests can assert what the connector threaded in.


class _FakeNodeAddress:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port


class _FakeServerCredentials:
    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password


class _FakeAdvancedClientConfig:
    def __init__(self, connection_timeout: Any = None) -> None:
        self.connection_timeout = connection_timeout


class _FakeAdvancedClusterConfig:
    def __init__(self, connection_timeout: Any = None) -> None:
        self.connection_timeout = connection_timeout


class _TaggedClientConfiguration:
    def __init__(
        self,
        addresses: Any,
        request_timeout: Any = None,
        use_tls: bool = False,
        advanced_config: Any = None,
        credentials: Any = None,
        database_id: Any = None,
        client_info_tag: Any = None,
    ) -> None:
        self.kwargs = {"database_id": database_id, "client_info_tag": client_info_tag}


class _UntaggedClientConfiguration:
    def __init__(
        self,
        addresses: Any,
        request_timeout: Any = None,
        use_tls: bool = False,
        advanced_config: Any = None,
        credentials: Any = None,
        database_id: Any = None,
    ) -> None:
        self.kwargs = {"database_id": database_id}


class _TaggedClusterConfiguration:
    def __init__(
        self,
        addresses: Any,
        request_timeout: Any = None,
        use_tls: bool = False,
        advanced_config: Any = None,
        credentials: Any = None,
        client_info_tag: Any = None,
    ) -> None:
        self.kwargs = {"client_info_tag": client_info_tag}


class _UntaggedClusterConfiguration:
    def __init__(
        self,
        addresses: Any,
        request_timeout: Any = None,
        use_tls: bool = False,
        advanced_config: Any = None,
        credentials: Any = None,
    ) -> None:
        self.kwargs: dict = {}


def _make_fake_glide_sync(supports_tag: bool) -> SimpleNamespace:
    """Build a fake ``glide_sync`` namespace for ``_get_client``.

    When ``supports_tag`` is True the config classes accept
    ``client_info_tag`` (glide >= 2.5.0); otherwise they don't, mirroring
    older builds. Fresh client classes are created per call so the recorded
    ``last_config`` never leaks between tests.
    """

    class GlideClient:
        last_config: Any = None

        def __init__(self, config: Any) -> None:
            self.config = config

        def get(self, key: Any, buffer: Any = None) -> Any:
            # ``buffer`` param presence drives has_buffer_get detection.
            return None

        def close(self) -> None:
            pass

        @classmethod
        def create(cls, config: Any) -> "GlideClient":
            cls.last_config = config
            return cls(config)

    class GlideClusterClient:
        last_config: Any = None

        def __init__(self, config: Any) -> None:
            self.config = config

        def get(self, key: Any, buffer: Any = None) -> Any:
            return None

        def close(self) -> None:
            pass

        @classmethod
        def create(cls, config: Any) -> "GlideClusterClient":
            cls.last_config = config
            return cls(config)

    client_cfg = (
        _TaggedClientConfiguration if supports_tag else _UntaggedClientConfiguration
    )
    cluster_cfg = (
        _TaggedClusterConfiguration if supports_tag else _UntaggedClusterConfiguration
    )

    return SimpleNamespace(
        NodeAddress=_FakeNodeAddress,
        ServerCredentials=_FakeServerCredentials,
        AdvancedGlideClientConfiguration=_FakeAdvancedClientConfig,
        AdvancedGlideClusterClientConfiguration=_FakeAdvancedClusterConfig,
        GlideClientConfiguration=client_cfg,
        GlideClusterClientConfiguration=cluster_cfg,
        GlideClient=GlideClient,
        GlideClusterClient=GlideClusterClient,
    )


# ── _lmcache_client_info_tag ────────────────────────────────────────────


def test_client_info_tag_value_format():
    """The tag is ``lmcache:<version>`` and contains no whitespace (a
    whitespace tag would be rejected by GLIDE's config validation)."""
    tag = _lmcache_client_info_tag()
    assert tag.startswith("lmcache:")
    assert not any(ch.isspace() for ch in tag)


def test_client_info_tag_whitespace_fallback(monkeypatch):
    """If the resolved version string somehow contains whitespace, fall back
    to a bare ``lmcache`` tag rather than emit an invalid tag."""
    # First Party
    import lmcache

    monkeypatch.setattr(lmcache, "__version__", "1.0 dev build", raising=False)
    assert _lmcache_client_info_tag() == "lmcache"


# ── _glide_config_supports_client_info_tag ──────────────────────────────


def test_supports_detection_true_and_false():
    """Support is feature-detected from the constructor signature."""

    class WithTag:
        def __init__(self, addresses, client_info_tag=None):
            pass

    class WithoutTag:
        def __init__(self, addresses):
            pass

    assert _glide_config_supports_client_info_tag(WithTag) is True
    assert _glide_config_supports_client_info_tag(WithoutTag) is False


def test_supports_detection_uninspectable_returns_false():
    """A type whose signature cannot be introspected is treated as
    unsupported rather than raising."""
    # ``object`` has no introspectable __init__ signature.
    assert _glide_config_supports_client_info_tag(object) is False


def test_min_version_constant():
    """The documented minimum glide version is 2.5.0."""
    assert CLIENT_INFO_TAG_MIN_GLIDE_VERSION == "2.5.0"


# ── _get_client config passthrough (real ValkeyWorkerPool) ──────────────


@pytest.mark.parametrize("cluster_mode", [False, True])
def test_get_client_sets_tag_when_supported(cluster_mode):
    """When the glide config supports it, the connector threads the LMCache
    tag through to the client config (standalone and cluster)."""
    fake = _make_fake_glide_sync(supports_tag=True)
    with patch.dict(sys.modules, {"glide_sync": fake}):
        pool = ValkeyWorkerPool(
            addresses=[("h", 1)],
            num_workers=1,
            username="",
            password="",
            cluster_mode=cluster_mode,
        )
        try:
            client_cls = fake.GlideClusterClient if cluster_mode else fake.GlideClient
            config = client_cls.last_config
            assert config is not None
            assert config.kwargs["client_info_tag"] == _lmcache_client_info_tag()
        finally:
            pool.close()


@pytest.mark.parametrize("cluster_mode", [False, True])
def test_get_client_omits_tag_when_unsupported(cluster_mode):
    """On an older glide build (no ``client_info_tag`` param) the connector
    must NOT pass the kwarg — doing so would raise ``TypeError``."""
    fake = _make_fake_glide_sync(supports_tag=False)
    with patch.dict(sys.modules, {"glide_sync": fake}):
        pool = ValkeyWorkerPool(
            addresses=[("h", 1)],
            num_workers=1,
            username="",
            password="",
            cluster_mode=cluster_mode,
        )
        try:
            client_cls = fake.GlideClusterClient if cluster_mode else fake.GlideClient
            config = client_cls.last_config
            assert config is not None
            assert "client_info_tag" not in config.kwargs
        finally:
            pool.close()


def test_get_client_standalone_still_passes_database_id_with_tag():
    """The tag is additive: standalone still forwards database_id alongside
    the new client_info_tag."""
    fake = _make_fake_glide_sync(supports_tag=True)
    with patch.dict(sys.modules, {"glide_sync": fake}):
        pool = ValkeyWorkerPool(
            addresses=[("h", 1)],
            num_workers=1,
            username="",
            password="",
            cluster_mode=False,
            database_id=7,
        )
        try:
            config = fake.GlideClient.last_config
            assert config.kwargs["database_id"] == 7
            assert config.kwargs["client_info_tag"] == _lmcache_client_info_tag()
        finally:
            pool.close()
