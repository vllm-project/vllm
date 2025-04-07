# SPDX-License-Identifier: Apache-2.0
import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils import FlexibleArgumentParser


class BenchmarkSubcommandBase(CLISubcommand):
    """ The base class of subcommands for vllm bench. """

    @property
    def help(self) -> str:
        """The help message of the subcommand."""
        raise NotImplementedError

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Add the CLI arguments to the parser."""
        raise NotImplementedError

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Run the benchmark.

        Args:
            args: The arguments to the command.
        """
        raise NotImplementedError

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"vllm bench {self.name} [options]")
        self.add_cli_args(parser)
        return parser
