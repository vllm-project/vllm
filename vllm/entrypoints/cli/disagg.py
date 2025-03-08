# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import (LoRAParserAction,
                                              PromptAdapterParserAction)
from vllm.entrypoints.openai.zmq_server import run_zmq_server
from vllm.utils import FlexibleArgumentParser


class DisaggSubcommand(CLISubcommand):
    """The `disagg` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "disagg"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # The default value of `--model`
        if not args.model_tag:
            raise ValueError(
                "With `vllm disagg`, you should provide the model as a "
                "positional argument instead of via the `--model` option.")

        # EngineArgs expects the model name to be passed as --model.
        args.model = args.model_tag

        asyncio.run(run_zmq_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_disagg_args(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        disagg_parser = subparsers.add_parser(
            "disagg",
            help="Start the vLLM OpenAI Compatible API zmq server",
            usage="vllm disagg <model_tag> [options]")

        return make_disagg_arg_parser(disagg_parser)


def cmd_init() -> list[CLISubcommand]:
    return [DisaggSubcommand()]


def make_disagg_arg_parser(
        parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument(
        "model_tag",
        type=str,
        help=
        "The model tag to use for the vLLM OpenAI Compatible API zmq server.")
    parser.add_argument('--zmq-server-addr',
                        type=str,
                        required=True,
                        help='The address to serve the zmq server on.')
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        help="If specified, will run the OpenAI frontend server in the same "
        "process as the model serving engine.")
    parser.add_argument(
        "--return-tokens-as-token-ids",
        action="store_true",
        help="When ``--max-logprobs`` is specified, represents single tokens "
        " as strings of the form 'token_id:{token_id}' so that tokens "
        "that are not JSON-encodable can be identified.")
    parser.add_argument('--max-log-len',
                        type=int,
                        default=None,
                        help='Max number of prompt characters or prompt '
                        'ID numbers being printed in log.'
                        '\n\nDefault: Unlimited')
    parser.add_argument(
        "--lora-modules",
        type=nullable_str,
        default=None,
        nargs='+',
        action=LoRAParserAction,
        help="LoRA module configurations in either 'name=path' format"
        "or JSON format. "
        "Example (old format): ``'name=path'`` "
        "Example (new format): "
        "``{\"name\": \"name\", \"path\": \"lora_path\", "
        "\"base_model_name\": \"id\"}``")

    parser.add_argument(
        "--prompt-adapters",
        type=nullable_str,
        default=None,
        nargs='+',
        action=PromptAdapterParserAction,
        help="Prompt adapter configurations in the format name=path. "
        "Multiple adapters can be specified.")

    AsyncEngineArgs.add_cli_args(parser)

    return parser


def validate_parsed_disagg_args(args: argparse.Namespace):
    """Quick checks for model disagg args that raise prior to loading."""
    if hasattr(args, "subparser") and args.subparser != "disagg":
        return

    # Enable reasoning needs a reasoning parser to be valid
    if args.enable_reasoning and not args.reasoning_parser:
        raise TypeError("Error: --enable-reasoning requires "
                        "--reasoning-parser")
