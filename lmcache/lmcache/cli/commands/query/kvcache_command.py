# SPDX-License-Identifier: Apache-2.0
"""``lmcache query kvcache`` — query KV-cache endpoints (placeholder)."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class KVCacheCommand(BaseCommand):
    """Query KV-cache endpoints (not implemented yet)."""

    def name(self) -> str:
        return "kvcache"

    def help(self) -> str:
        return "Query KV-cache endpoints (not implemented yet)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass  # TODO: add kvcache query arguments

    def execute(self, args: argparse.Namespace) -> None:
        # TODO: implement kvcache query logic
        pass
