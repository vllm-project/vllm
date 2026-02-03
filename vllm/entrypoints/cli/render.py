# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CLI subcommand for the vLLM Render gRPC server.

Usage:
    vllm render --model <model_path>
"""

import argparse

import uvloop

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.render_server import serve_render
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

DESCRIPTION = """Launch a gRPC server for tokenization and chat message rendering.

This is a lightweight server that only handles input preprocessing (tokenization,
chat template rendering) without any LLM inference capabilities. It is useful for
separating the rendering/tokenization layer from the inference layer.

Example:
    vllm render --model meta-llama/Llama-2-7b-hf --port 50052
"""


class RenderSubcommand(CLISubcommand):
    """The `render` subcommand for the vLLM CLI."""

    name = "render"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified as positional arg, use it
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        # Run the render server
        uvloop.run(serve_render(args))

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Start a gRPC server for tokenization and chat rendering",
            description=DESCRIPTION,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            usage="vllm render [options] [model]",
        )

        # Positional argument for model (optional)
        parser.add_argument(
            "model_tag",
            type=str,
            nargs="?",
            help="Model path or HuggingFace model ID",
        )

        # Server args
        parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to bind gRPC server to (default: 0.0.0.0)",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=50052,
            help="Port to bind gRPC server to (default: 50052)",
        )

        # Model args
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Model path or HuggingFace model ID",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=None,
            help="Tokenizer path (defaults to model path)",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default="auto",
            choices=["auto", "hf", "mistral"],
            help="Tokenizer mode (default: auto)",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code from HuggingFace",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="Model revision",
        )
        parser.add_argument(
            "--tokenizer-revision",
            type=str,
            default=None,
            help="Tokenizer revision",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Maximum model context length",
        )
        parser.add_argument(
            "--trust-request-chat-template",
            action="store_true",
            help="Trust chat templates from requests",
        )

        return parser

    def validate(self, args: argparse.Namespace) -> None:
        # Ensure model is specified
        model = getattr(args, "model", None) or getattr(args, "model_tag", None)
        if not model:
            raise ValueError(
                "Model is required. Specify via positional argument or --model flag."
            )


def cmd_init() -> list[CLISubcommand]:
    """Initialize and return the render subcommand."""
    return [RenderSubcommand()]
