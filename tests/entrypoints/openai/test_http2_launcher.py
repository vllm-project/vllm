# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HTTP/2 launcher additions:

- _HypercornAdapter  (vllm/entrypoints/launcher.py)
- serve_http() dispatch: enable_http2=False → Uvicorn path (default),
                          enable_http2=True  → _serve_hypercorn()
"""

import asyncio
import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────


def _get_adapter():
    """Import _HypercornAdapter freshly (it is a private name)."""
    from vllm.entrypoints.launcher import _HypercornAdapter

    return _HypercornAdapter


# ── _HypercornAdapter ─────────────────────────────────────────────────────────


class TestHypercornAdapter:
    def test_initial_should_exit_is_false(self):
        _HypercornAdapter = _get_adapter()
        event = asyncio.Event()
        adapter = _HypercornAdapter(event)
        assert adapter.should_exit is False

    def test_should_exit_true_when_event_set(self):
        _HypercornAdapter = _get_adapter()
        event = asyncio.Event()
        adapter = _HypercornAdapter(event)
        event.set()
        assert adapter.should_exit is True

    def test_setting_should_exit_true_sets_event(self):
        _HypercornAdapter = _get_adapter()
        event = asyncio.Event()
        adapter = _HypercornAdapter(event)
        adapter.should_exit = True
        assert event.is_set()

    def test_setting_should_exit_false_does_not_clear_event(self):
        """Assigning False should be a no-op (events can't be un-set here)."""
        _HypercornAdapter = _get_adapter()
        event = asyncio.Event()
        event.set()
        adapter = _HypercornAdapter(event)
        adapter.should_exit = False  # must not raise; event stays set
        assert event.is_set()

    def test_shutdown_sets_event(self):
        _HypercornAdapter = _get_adapter()
        event = asyncio.Event()
        adapter = _HypercornAdapter(event)

        async def _run():
            await adapter.shutdown()
            assert event.is_set()

        asyncio.run(_run())


# ── serve_http dispatch ───────────────────────────────────────────────────────


class TestServeHttpDispatch:
    """serve_http() must route to the correct backend without actually binding
    a socket.  We mock both _serve_hypercorn and uvicorn.Server to stay
    completely in-process."""

    def _make_app(self):
        app = MagicMock()
        app.routes = []
        app.state = MagicMock()
        return app

    def test_http1_does_not_call_serve_hypercorn(self):
        """Default (enable_http2=False) must never call _serve_hypercorn."""
        app = self._make_app()

        mock_uvicorn_server = MagicMock()
        mock_uvicorn_server.serve = AsyncMock(return_value=None)
        mock_uvicorn_server.should_exit = False

        with (
            patch(
                "vllm.entrypoints.launcher._serve_hypercorn", new_callable=AsyncMock
            ) as mock_h2,
            patch("uvicorn.Server", return_value=mock_uvicorn_server),
        ):

            async def _run():
                from vllm.entrypoints.launcher import serve_http

                await serve_http(
                    app, sock=None, enable_http2=False, host="127.0.0.1", port=8000
                )

            asyncio.run(_run())
            mock_h2.assert_not_called()

    def test_http2_calls_serve_hypercorn(self):
        """enable_http2=True must delegate to _serve_hypercorn."""
        app = self._make_app()

        with patch(
            "vllm.entrypoints.launcher._serve_hypercorn",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_h2:

            async def _run():
                from vllm.entrypoints.launcher import serve_http

                await serve_http(
                    app, sock=None, enable_http2=True, host="127.0.0.1", port=8000
                )

            asyncio.run(_run())
            mock_h2.assert_called_once()
            # Positional args: app, sock, enable_ssl_refresh, **kwargs
            call_args = mock_h2.call_args
            assert call_args.args[0] is app
            assert call_args.args[1] is None  # sock


# ── _serve_hypercorn import error ─────────────────────────────────────────────


class TestServeHypercornImportError:
    """If hypercorn is not installed _serve_hypercorn must raise RuntimeError
    with a helpful install hint."""

    def test_raises_runtime_error_when_hypercorn_missing(self):
        # Temporarily hide hypercorn from imports
        with patch.dict(
            sys.modules,
            {"hypercorn": None, "hypercorn.asyncio": None, "hypercorn.config": None},
        ):
            # Force reimport so the try/except inside _serve_hypercorn fires
            import vllm.entrypoints.launcher as _launcher

            importlib.reload(_launcher)

            async def _run():
                with pytest.raises(RuntimeError, match="hypercorn h2"):
                    await _launcher._serve_hypercorn(
                        MagicMock(),
                        None,
                        False,
                        host="127.0.0.1",
                        port=8000,
                    )

            asyncio.run(_run())
