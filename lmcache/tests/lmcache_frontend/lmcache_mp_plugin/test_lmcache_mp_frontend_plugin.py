# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MP frontend plugin's argv builder.

These tests exercise the pure ``build_argv`` function to make sure
the aggregated ``LMCACHE_RUNTIME_PLUGIN_CONFIG`` payload is translated
into the right CLI argv for ``lmcache.lmcache_frontend.app.main()``.
"""

# Standard
import json

# First Party
from lmcache.lmcache_frontend.lmcache_mp_plugin.lmcache_mp_frontend_plugin import (
    build_argv,
)


def _find_arg(argv: list, name: str) -> str | None:
    """Return the value following ``name`` in ``argv``, or ``None``."""
    if name not in argv:
        return None
    idx = argv.index(name)
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def test_build_argv_with_tcp_http_config():
    config = {
        "http_frontend_config": {
            "http_host": "192.168.1.5",
            "http_port": 8085,
            "http_socket_path": None,
        },
        "runtime_plugin_extra_config": {
            "plugin.frontend.heartbeat-url": "http://disc.example/heartbeat",
            "plugin.frontend.heartbeat-interval": 30,
            "unrelated.key": "ignored",
        },
    }

    argv = build_argv(config, prog="plugin")

    assert argv[0] == "plugin"
    assert "--no-http" in argv
    assert _find_arg(argv, "--heartbeat-url") == "http://disc.example/heartbeat"
    assert _find_arg(argv, "--heartbeat-interval") == "30"
    # unrelated keys must not leak through
    assert "--unrelated-key" not in argv

    # --port is forwarded as the connectable lmcache HTTP port
    assert _find_arg(argv, "--port") == "8085"

    nodes_json = _find_arg(argv, "--nodes")
    assert nodes_json is not None
    nodes = json.loads(nodes_json)
    assert nodes == [
        {"name": "lmcache_server", "host": "192.168.1.5", "port": "8085"},
    ]


def test_build_argv_rewrites_wildcard_host():
    config = {
        "http_frontend_config": {
            "http_host": "0.0.0.0",
            "http_port": 9000,
        },
    }
    argv = build_argv(config)
    nodes = json.loads(_find_arg(argv, "--nodes"))
    assert nodes[0]["host"] == "localhost"
    assert nodes[0]["port"] == "9000"


def test_build_argv_prefers_socket_path_when_set():
    config = {
        "http_frontend_config": {
            "http_host": "0.0.0.0",
            "http_port": 9000,
            "http_socket_path": "/tmp/lmc.sock",
        },
    }
    argv = build_argv(config)
    nodes = json.loads(_find_arg(argv, "--nodes"))
    assert nodes == [
        {"name": "lmcache_server", "host": "localhost", "port": "/tmp/lmc.sock"},
    ]


def test_build_argv_accepts_http_config_alias():
    """``MPRuntimePluginLauncher`` passes the http config under the key
    ``http_config`` (see ``http_server.py``). Ensure we accept both
    ``http_config`` and ``http_frontend_config`` at the top level.
    """
    config = {
        "http_config": {
            "http_host": "10.0.0.1",
            "http_port": 7000,
        },
    }
    argv = build_argv(config)
    nodes = json.loads(_find_arg(argv, "--nodes"))
    assert nodes[0]["host"] == "10.0.0.1"
    assert nodes[0]["port"] == "7000"


def test_build_argv_with_empty_config_has_defaults():
    argv = build_argv({})
    assert "--no-http" in argv
    nodes = json.loads(_find_arg(argv, "--nodes"))
    # falls back to localhost:8080 when http config is missing
    assert nodes == [
        {"name": "lmcache_server", "host": "localhost", "port": "8080"},
    ]
    # no --port should be appended when http_port is missing
    assert _find_arg(argv, "--port") is None
