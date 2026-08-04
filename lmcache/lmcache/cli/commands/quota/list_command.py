# SPDX-License-Identifier: Apache-2.0
"""``lmcache quota list`` — list all registered quotas and their usage."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.quota._helpers import http_request, normalize_url


class ListCommand(BaseCommand):
    """List all registered quotas and their usage."""

    def name(self) -> str:
        return "list"

    def help(self) -> str:
        return "List all registered quotas and their usage."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--url",
            type=str,
            default="http://localhost:8080",
            help="LMCache HTTP server URL (default: http://localhost:8080).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        base_url = normalize_url(args.url)

        result = http_request("GET", f"{base_url}/quota")
        users = result.get("users", {})

        metrics = self.create_metrics("Quota List", args, width=55)

        if not users:
            metrics.add("info", "Info", "No quotas configured")
            metrics.emit()
            return

        for idx, (salt, info) in enumerate(users.items()):
            section_key = f"quota_{idx}"
            metrics.add_list_section("quotas", section_key, f"Salt: {salt}")
            sec = metrics[section_key]
            sec.add("cache_salt", "Cache salt", salt)
            sec.add("limit_gb", "Limit (GB)", info.get("limit_gb"))
            sec.add(
                "current_usage_gb", "Current usage (GB)", info.get("current_usage_gb")
            )

        metrics.emit()
