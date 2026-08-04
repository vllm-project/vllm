# SPDX-License-Identifier: Apache-2.0
"""LMCache CLI entry point.

Subcommands are explicitly registered in
``lmcache.cli.commands.ALL_COMMANDS``.
"""

# Standard
import argparse
import sys

# First Party
from lmcache.banner import print_banner_once
from lmcache.cli.commands import ALL_COMMANDS
from lmcache.logging import init_logger

logger = init_logger(__name__)


def main() -> None:
    """CLI entry point registered as ``lmcache`` in *pyproject.toml*."""
    print_banner_once(sys.stderr)
    parser = argparse.ArgumentParser(
        prog="lmcache",
        description="LMCache — KV cache management for LLM serving",
    )
    subparsers = parser.add_subparsers(dest="command")

    for cmd in ALL_COMMANDS:
        cmd.register(subparsers)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001
        logger.exception("Command failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
