# SPDX-License-Identifier: Apache-2.0

import argparse
import signal

import uvloop

import vllm.envs as envs
from vllm import AsyncEngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.entrypoints.utils import (VLLM_SERVE_PARSER_EPILOG,
                                    show_filtered_argument_or_group_from_help)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_tcp_uri
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import CoreEngineProcManager
from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "serve"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            args.model = args.model_tag

        if args.headless:
            run_headless(args)
        else:
            uvloop.run(run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "serve",
            help="Start the vLLM OpenAI Compatible API server.",
            description="Start the vLLM OpenAI Compatible API server.",
            usage="vllm serve [model_tag] [options]")
        serve_parser.add_argument("model_tag",
                                  type=str,
                                  nargs='?',
                                  help="The model tag to serve "
                                  "(optional if specified in config)")
        serve_parser.add_argument(
            "--headless",
            action='store_true',
            default=False,
            help="Run in headless mode. See multi-node data parallel "
            "documentation for more details.")
        serve_parser.add_argument(
            '--data-parallel-start-rank',
            '-dpr',
            type=int,
            default=0,
            help='Starting data parallel rank for secondary nodes.')
        serve_parser.add_argument(
            "--config",
            type=str,
            default='',
            required=False,
            help="Read CLI options from a config file."
            "Must be a YAML with the following options:"
            "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#cli-reference"
        )

        serve_parser = make_arg_parser(serve_parser)
        show_filtered_argument_or_group_from_help(serve_parser)
        serve_parser.epilog = VLLM_SERVE_PARSER_EPILOG
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]


def run_headless(args: argparse.Namespace):

    # Create the EngineConfig.
    engine_args = AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if not envs.VLLM_USE_V1:
        raise RuntimeError("Headless mode is only supported for V1")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    host = parallel_config.data_parallel_master_ip
    port = engine_args.data_parallel_rpc_port  # add to config too
    input_address = get_tcp_uri(host, port)

    if local_engine_count <= 0:
        raise RuntimeError("data_parallel_size_local must be > 0 in "
                           "headless mode")

    # Catch SIGTERM and SIGINT to allow graceful shutdown.
    def signal_handler(signum, frame):
        logger.debug("Received %d signal.", signum)
        raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(
        "Launching %d data parallel engine(s) in headless mode, "
        "with head node address %s.", local_engine_count, input_address)

    # Create the engines.
    engine_manager = CoreEngineProcManager(
        target_fn=EngineCoreProc.run_engine_core,
        local_engine_count=local_engine_count,
        start_index=args.data_parallel_start_rank,
        local_start_index=0,
        vllm_config=vllm_config,
        on_head_node=False,
        input_address=input_address,
        executor_class=Executor.get_class(vllm_config),
        log_stats=not engine_args.disable_log_stats,
    )

    try:
        engine_manager.join_first()
    finally:
        logger.info("Shutting down.")
        engine_manager.close()
