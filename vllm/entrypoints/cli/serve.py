# SPDX-License-Identifier: Apache-2.0

import argparse

import uvloop

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.cli.utils import FlexibleArgumentParser


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "serve"
        self.serve_parser = None
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # The default value of `--model`
        from vllm.engine.arg_utils import EngineArgs
        from vllm.entrypoints.openai.api_server import run_server

        if args.model != EngineArgs.model:
            raise ValueError(
                "With `vllm serve`, you should provide the model as a "
                "positional argument instead of via the `--model` option.")

        # EngineArgs expects the model name to be passed as --model.
        args.model = args.model_tag

        uvloop.run(run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args
        validate_parsed_serve_args(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        self.serve_parser = subparsers.add_parser(
            "serve",
            help="Start the vLLM OpenAI Compatible API server",
            usage="vllm serve <model_tag> [options]")
        self.serve_parser.add_argument("model_tag",
                                       type=str,
                                       help="The model tag to serve")
        self.serve_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=False,
            help="Read CLI options from a config file."
            "Must be a YAML with the following options:"
            "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#cli-reference"
        )

        return self.serve_parser

    def add_cli_args(self) -> FlexibleArgumentParser:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        print("innnnn")
        return make_arg_parser(self.serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
