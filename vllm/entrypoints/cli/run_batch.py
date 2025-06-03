# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio

from prometheus_client import start_http_server

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.logger import logger
from vllm.entrypoints.openai.run_batch import main as run_batch_main
from vllm.entrypoints.openai.run_batch import make_arg_parser
from vllm.utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION


class RunBatchSubcommand(CLISubcommand):
    """The `run-batch` subcommand for vLLM CLI."""

    def __init__(self):
        self.name = "run-batch"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        logger.info("vLLM batch processing API version %s", VLLM_VERSION)
        logger.info("args: %s", args)

        # Start the Prometheus metrics server.
        # LLMEngine uses the Prometheus client
        # to publish metrics at the /metrics endpoint.
        if args.enable_metrics:
            logger.info("Prometheus metrics enabled")
            start_http_server(port=args.port, addr=args.url)
        else:
            logger.info("Prometheus metrics disabled")

        asyncio.run(run_batch_main(args))

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        run_batch_parser = subparsers.add_parser(
            "run-batch",
            help="Run batch prompts and write results to file.",
            description=(
                "Run batch prompts using vLLM's OpenAI-compatible API.\n"
                "Supports local or HTTP input/output files."),
            usage=
            "vllm run-batch -i INPUT.jsonl -o OUTPUT.jsonl --model <model>",
        )
        return make_arg_parser(run_batch_parser)


def cmd_init() -> list[CLISubcommand]:
    return [RunBatchSubcommand()]
