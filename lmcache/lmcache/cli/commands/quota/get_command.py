# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota get`` — read the quota and usage for a cache_salt."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.quota._helpers import (
    escape_salt,
    http_request,
    normalize_url,
)


class GetCommand(BaseCommand):
    """Show the quota and current usage for a cache_salt."""

    def name(self) -> str:
        return "get"

    def help(self) -> str:
        return "Show the quota and current usage for a cache_salt."

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
            "--url",
            type=str,
            default="http://localhost:8080",
            help="LMCache HTTP server URL (default: http://localhost:8080).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        base_url = normalize_url(args.url)
        salt = escape_salt(args.salt)

        result = http_request("GET", f"{base_url}/quota/{salt}")

        metrics = self.create_metrics("Quota Info", args)
        metrics.add("cache_salt", "Cache salt", result.get("cache_salt", salt))
        metrics.add("limit_gb", "Limit (GB)", result.get("limit_gb"))
        metrics.add(
            "current_usage_gb", "Current usage (GB)", result.get("current_usage_gb")
        )
        metrics.add("exists", "Exists", result.get("exists"))
        metrics.emit()
