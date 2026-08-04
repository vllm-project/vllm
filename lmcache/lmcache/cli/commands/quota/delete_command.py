# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota delete`` — remove a quota for a cache_salt."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.quota._helpers import (
    escape_salt,
    http_request,
    normalize_url,
)


class DeleteCommand(BaseCommand):
    """Remove a quota for a cache_salt."""

    def name(self) -> str:
        return "delete"

    def help(self) -> str:
        return "Remove a quota for a cache_salt."

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

        result = http_request("DELETE", f"{base_url}/quota/{salt}")

        metrics = self.create_metrics("Quota Delete", args)
        metrics.add("cache_salt", "Cache salt", result.get("cache_salt", salt))
        metrics.add("status", "Status", result.get("status", "unknown"))
        metrics.emit()
