# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for ``lmcache quota`` subcommands."""

# Standard
from typing import Any, Optional
import json
import sys
import urllib.error
import urllib.request

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# The MP HTTP server uses "_default" as a sentinel for the empty-string
# cache_salt (anonymous / un-salted traffic).
DEFAULT_SALT_SENTINEL = "_default"


def normalize_url(url: str) -> str:
    """Ensure *url* has an ``http://`` or ``https://`` scheme."""
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def escape_salt(salt: str) -> str:
    """Translate the empty-string salt to the URL sentinel."""
    return DEFAULT_SALT_SENTINEL if salt == "" else salt


def unescape_salt(salt: str) -> str:
    """Translate the URL sentinel back to the empty-string salt."""
    return "" if salt == DEFAULT_SALT_SENTINEL else salt


def http_request(
    method: str,
    url: str,
    data: Optional[dict[str, Any]] = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """Send an HTTP request and return the parsed JSON response.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE).
        url: Full URL to request.
        data: Optional JSON body to send.
        timeout: HTTP timeout in seconds.

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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode())
            msg = error_body.get("error") or error_body.get("message") or str(e)
        except (json.JSONDecodeError, ValueError, OSError):
            msg = str(e)
        logger.error("Server error: %s", msg)
        sys.exit(1)
    except urllib.error.URLError as e:
        logger.error("Cannot reach %s — is the server running? (%s)", url, e.reason)
        sys.exit(1)
