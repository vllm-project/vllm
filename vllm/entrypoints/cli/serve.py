# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import serve
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)

DESCRIPTION = """Launch a local OpenAI-compatible API server to serve LLM
completions via HTTP. Defaults to Qwen/Qwen3-0.6B if no model is specified.

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=ModelConfig, --help=Frontend)
  Use `--help=all` to show all available flags at once.
"""


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """
    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            args.model = args.model_tag

        serve(args)

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            description=DESCRIPTION,
            usage="vllm serve [model_tag] [options]")

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(
            subcmd=self.name)
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
