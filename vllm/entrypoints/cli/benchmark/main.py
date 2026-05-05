# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import sys
import typing

from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


def _import_bench_subcommand_modules() -> None:
    # Imported lazily so `BenchmarkSubcommandBase` subclasses register only
    # when `vllm bench` is actually invoked.
    import vllm.entrypoints.cli.benchmark.latency  # noqa: F401
    import vllm.entrypoints.cli.benchmark.mm_processor  # noqa: F401
    import vllm.entrypoints.cli.benchmark.serve  # noqa: F401
    import vllm.entrypoints.cli.benchmark.startup  # noqa: F401
    import vllm.entrypoints.cli.benchmark.sweep  # noqa: F401
    import vllm.entrypoints.cli.benchmark.throughput  # noqa: F401


class BenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the vLLM CLI."""

    name = "bench"
    help = "vLLM bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"vllm {self.name} <bench_type> [options]",
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        # Only build the nested bench subparsers when the user is actually
        # invoking `bench`; otherwise we'd drag in imports
        # unnecessarily on every `vllm --help` and `vllm serve`.
        # Scan for the first positional arg so global flags (e.g. `-v`)
        # before the subcommand don't break detection.
        first_positional = next(
            (arg for arg in sys.argv[1:] if not arg.startswith("-")), None
        )
        if first_positional == self.name:
            _import_bench_subcommand_modules()
            for cmd_cls in BenchmarkSubcommandBase.__subclasses__():
                cmd_subparser = bench_subparsers.add_parser(
                    cmd_cls.name,
                    help=cmd_cls.help,
                    description=cmd_cls.help,
                    usage=f"vllm {self.name} {cmd_cls.name} [options]",
                )
                cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
                cmd_cls.add_cli_args(cmd_subparser)
                cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(
                    subcmd=f"{self.name} {cmd_cls.name}"
                )
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
