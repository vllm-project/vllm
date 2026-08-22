# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import logging
import os
from typing import TypeVar

_ParserT = TypeVar("_ParserT", bound=argparse.ArgumentParser)

CLI_COMMANDS = {
    "chat": (
        "vllm.entrypoints.cli.openai",
        "Generate chat completions via the running API server.",
    ),
    "complete": (
        "vllm.entrypoints.cli.openai",
        "Generate text completions based on the given prompt via the running "
        "API server.",
    ),
    "serve": (
        "vllm.entrypoints.cli.serve",
        "Launch a local OpenAI-compatible API server to serve LLM completions "
        "via HTTP.",
    ),
    "launch": (
        "vllm.entrypoints.cli.launch",
        "Launch individual vLLM components.",
    ),
    "bench": (
        "vllm.entrypoints.cli.benchmark.main",
        "vLLM bench subcommand.",
    ),
    "collect-env": (
        "vllm.entrypoints.cli.collect_env",
        "Start collecting environment information.",
    ),
    "run-batch": (
        "vllm.entrypoints.cli.run_batch",
        "Run batch prompts and write results to file.",
    ),
}

VLLM_SUBCMD_PARSER_EPILOG = (
    "For full list:            vllm {subcmd} --help=all\n"
    "For a section:            vllm {subcmd} --help=ModelConfig    "
    "(case-insensitive)\n"
    "For a flag:               vllm {subcmd} --help=max-model-len  "
    "(_ or - accepted)\n"
    "Documentation:            https://docs.vllm.ai\n"
)

SERVE_DESCRIPTION = """Launch a local OpenAI-compatible API server to serve LLM
completions via HTTP. Defaults to Qwen/Qwen3-0.6B if no model is specified.

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=ModelConfig, --help=Frontend)
  Use `--help=all` to show all available flags at once.
"""


def cli_env_setup() -> None:
    """Set process defaults shared by all CLI runtime commands."""
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logging.getLogger(__name__).debug(
            "Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'"
        )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def add_serve_core_args(parser: _ParserT) -> _ParserT:
    parser.add_argument(
        "model_tag",
        type=str,
        nargs="?",
        help="The model tag to serve (optional if specified in config)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode. See multi-node data parallel "
        "documentation for more details.",
    )
    parser.add_argument(
        "--api-server-count",
        "-asc",
        type=int,
        default=None,
        help="How many API server processes to run. "
        "Defaults to data_parallel_size if not specified.",
    )
    parser.add_argument(
        "--config",
        help="Read CLI options from a config file. "
        "Must be a YAML with the following options: "
        "https://docs.vllm.ai/en/latest/configuration/serve_args.html",
    )
    parser.add_argument(
        "--grpc",
        action="store_true",
        default=False,
        help="Launch a gRPC server instead of the HTTP OpenAI-compatible "
        "server. Requires: pip install vllm[grpc].",
    )
    return parser


def is_plain_help(argv: list[str]) -> bool:
    return any(arg in ("-h", "--help") for arg in argv) and not any(
        arg.startswith("--help=") for arg in argv
    )
