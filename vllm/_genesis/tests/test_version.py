# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for Genesis version.

Tests pin that the version constant exists, has a sane format, and
is referenced consistently across modules that previously hardcoded
"v7.63.x"-style strings.
"""
from __future__ import annotations

import re



class TestVersionConstant:
    def test_version_module_importable(self):
        from vllm._genesis import __version__ as v
        assert v is not None

    def test_version_string_format(self):
        from vllm._genesis import __version__
        assert isinstance(__version__, str)
        # Sander uses "v7.63.x" style — major.minor.patch with optional 'x'
        # for in-development. Match that or PEP 440-style.
        assert re.match(
            r"^v?\d+\.\d+(\.\w+)?(\.dev\d+)?(rc\d+)?(\+\w+)?$",
            __version__,
        ), f"version string {__version__!r} doesn't match expected format"

    def test_version_imported_from_top_level(self):
        """Ensure `from vllm._genesis import __version__` works."""
        import vllm._genesis as gen
        assert hasattr(gen, "__version__")


class TestVersionConsistency:
    """Modules that previously hardcoded version should now derive it."""

    def test_telemetry_uses_version(self):
        from vllm._genesis.compat import telemetry
        from vllm._genesis import __version__
        # telemetry._detect_genesis_version should return the canonical version
        info = telemetry._detect_genesis_version()
        assert info["version"] == __version__

    def test_update_channel_does_not_hardcode(self):
        """update_channel module shouldn't have a stale version string
        hardcoded — it should derive from __version__ if anywhere."""
        from vllm._genesis.compat import update_channel
        import inspect
        src = inspect.getsource(update_channel)
        # Permitted: v7.63.x (current) or VERSION pulled from constant
        # NOT permitted: stale older versions hardcoded
        for stale in ("v7.62.x", "v7.61", "v7.60"):
            assert stale not in src, (
                f"update_channel.py has stale hardcoded {stale!r} — "
                "should reference __version__"
            )
