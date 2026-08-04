# SPDX-License-Identifier: Apache-2.0
"""``lmcache kvcache`` — KV cache management.

Sub-commands:
    clear         Clear all cached KV data
"""

# Standard
from typing import Any, Optional
import argparse
import json
import sys
import urllib.error
import urllib.request

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _http_request(
    method: str, url: str, data: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Send an HTTP request and return the parsed JSON response.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE).
        url: Full URL to request.
        data: Optional JSON body to send.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        SystemExit: On connection error or non-2xx HTTP response.
    """
    body = None
    headers: dict[str, str] = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode())
            msg = error_body.get("message") or error_body.get("error") or str(e)
        except (json.JSONDecodeError, ValueError, OSError):
            msg = str(e)
        logger.error("Server error: %s", msg)
        sys.exit(1)
    except urllib.error.URLError as e:
        logger.error("Cannot reach %s — is the server running? (%s)", url, e.reason)
        sys.exit(1)


class KVCacheCommand(BaseCommand):
    """Manage KV cache state.

    This command provides sub-commands for managing KV cache data on the
    MP HTTP server. Currently supports clearing all L1 cache. Future
    sub-commands (pin, compress, info) are defined in the design doc.
    """

    def name(self) -> str:
        """Return the subcommand name.

        Returns:
            The string ``"kvcache"``.
        """
        return "kvcache"

    def help(self) -> str:
        """Return short help text.

        Returns:
            Help string shown by ``lmcache -h``.
        """
        return "Manage KV cache state."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add kvcache-specific sub-commands and their arguments.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        sub = parser.add_subparsers(dest="action")

        # clear
        p_clear = sub.add_parser("clear", help="Clear all cached KV data in L1 (CPU).")
        p_clear.add_argument(
            "--url",
            type=str,
            required=True,
            help="Target MP HTTP endpoint (e.g. http://localhost:8000).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch to the appropriate sub-command handler.

        Args:
            args: Parsed CLI arguments. Must contain ``action`` indicating
                which sub-command was invoked.
        """
        action = getattr(args, "action", None)
        if action is None:
            logger.error("No sub-command specified. Run: lmcache kvcache -h")
            sys.exit(1)

        dispatch = {
            "clear": self._clear,
        }
        dispatch[action](args)

    def _clear(self, args: argparse.Namespace) -> None:
        """Clear all cached KV data via the MP HTTP server."""
        url = args.url.rstrip("/")

        # MP HTTP server endpoint: POST /cache/clear
        _http_request("POST", f"{url}/cache/clear", {"tier": "l1"})

        quiet = getattr(args, "quiet", False)
        if quiet:
            return

        metrics = self.create_metrics("KV Cache Clear", args)
        metrics.add("status", "Status", "OK")
        metrics.emit()
