# SPDX-License-Identifier: Apache-2.0

import argparse

from vllm.entrypoints.cli.utils import FlexibleArgumentParser


class CLISubcommand:
    """Base class for CLI argument handlers."""

    name: str

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def validate(self, args: argparse.Namespace) -> None:
        # No validation by default
        pass

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        raise NotImplementedError("Subclasses should implement this method")

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        # No needed by default
        pass