# SPDX-License-Identifier: Apache-2.0
"""``lmcache ping`` — liveness check for LMCache and vLLM servers.

Usage::

    lmcache ping kvcache --url http://localhost:8080
    lmcache ping engine  --url http://localhost:8000
"""

# Standard
import argparse
import sys
import time
import urllib.error
import urllib.request

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.describe import normalize_url

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

HEALTH_ENDPOINTS: dict[str, str] = {
    "kvcache": "/healthcheck",
    "engine": "/health",
}

DEFAULT_URLS: dict[str, str] = {
    "kvcache": "http://localhost:8080",
    "engine": "http://localhost:8000",
}

TITLES: dict[str, str] = {
    "kvcache": "Ping KV Cache",
    "engine": "Ping Engine",
}

# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------


def ping(url: str, timeout: int = 10) -> tuple[str, float, str | None]:
    """GET *url* and return ``(status, rtt_ms, error_msg)``.

    Returns:
        ``("OK", rtt_ms, None)`` on HTTP 200.
        ``("FAIL", rtt_ms, detail)`` on any error.
    """
    start = time.monotonic()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url), timeout=timeout
        ) as resp:
            rtt_ms = (time.monotonic() - start) * 1000
            if resp.status == 200:
                return ("OK", rtt_ms, None)
            return ("FAIL", rtt_ms, f"HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        rtt_ms = (time.monotonic() - start) * 1000
        return ("FAIL", rtt_ms, f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        rtt_ms = (time.monotonic() - start) * 1000
        reason = getattr(exc, "reason", str(exc))
        return ("FAIL", rtt_ms, f"Cannot connect to {url}: {reason}")


# -------------------------------------------------------------------
# Command
# -------------------------------------------------------------------


class PingCommand(BaseCommand):
    """Ping LMCache or vLLM server (liveness check)."""

    def name(self) -> str:
        return "ping"

    def help(self) -> str:
        return "Ping LMCache or vLLM server (liveness check)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "target",
            choices=["kvcache", "engine"],
            help="What to ping.",
        )
        parser.add_argument(
            "--url",
            default=None,
            help=(
                "Server URL (default: http://localhost:8080 for kvcache, "
                "http://localhost:8000 for engine)."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        base_url = normalize_url(args.url or DEFAULT_URLS[args.target])
        endpoint = HEALTH_ENDPOINTS[args.target]
        title = TITLES[args.target]

        status, rtt_ms, error = ping(f"{base_url}{endpoint}")

        metrics = self.create_metrics(title, args, width=30)
        metrics.add("status", "Status", status)
        metrics.add("round_trip_time_ms", "Round trip time (ms)", round(rtt_ms, 2))
        metrics.emit()

        if error:
            print(error, file=sys.stderr)
            sys.exit(1)
