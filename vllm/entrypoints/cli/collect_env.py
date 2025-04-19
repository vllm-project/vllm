# SPDX-License-Identifier: Apache-2.0

import argparse

from vllm.collect_env import main as collect_env_main
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


class CollectEnvSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "collect-env"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Collect information about the environment."""
        collect_env_main()

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "collect-env",
            help="Start collecting environment information.",
            description="Start collecting environment information.",
            usage="vllm collect-env")
        return make_arg_parser(serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [CollectEnvSubcommand()]
