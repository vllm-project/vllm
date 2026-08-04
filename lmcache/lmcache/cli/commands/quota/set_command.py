# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota set`` — create or update a quota for a cache_salt."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.quota._helpers import (
    escape_salt,
    http_request,
    normalize_url,
)


class SetCommand(BaseCommand):
    """Create or update a quota for a cache_salt."""

    def name(self) -> str:
        return "set"

    def help(self) -> str:
        return "Create or update a quota for a cache_salt."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "salt",
            type=str,
            help=(
                "The cache_salt identifier. Use '_default' for anonymous "
                "(un-salted) traffic."
            ),
        )
        parser.add_argument(
            "--limit-gb",
            type=float,
            required=True,
            metavar="GB",
            help="Quota limit in gigabytes (non-negative).",
        )
        parser.add_argument(
            "--url",
            type=str,
            default="http://localhost:8080",
            help="LMCache HTTP server URL (default: http://localhost:8080).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        base_url = normalize_url(args.url)
        salt = escape_salt(args.salt)
        limit_gb = args.limit_gb

        result = http_request(
            "PUT",
            f"{base_url}/quota/{salt}",
            data={"limit_gb": limit_gb},
        )

        metrics = self.create_metrics("Quota Set", args)
        metrics.add("cache_salt", "Cache salt", result.get("cache_salt", salt))
        metrics.add("limit_gb", "Limit (GB)", result.get("limit_gb", limit_gb))
        metrics.add("status", "Status", result.get("status", "ok"))
        metrics.emit()
