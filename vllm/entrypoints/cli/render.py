# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
``vllm render`` — lightweight gRPC server for chat template rendering
and tokenization (no GPU / no inference).
"""

import argparse

import uvloop

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.utils.argparse_utils import FlexibleArgumentParser

DESCRIPTION = """Launch a lightweight render server that provides chat template
rendering and tokenization without GPU or LLM inference. Supports both gRPC
and HTTP protocols (selected via --server). This is useful for disaggregated
frontend architectures where input preprocessing is separated from model
serving.
"""


class RenderSubcommand(CLISubcommand):
    """The ``render`` subcommand for the vLLM CLI."""

    name = "render"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        from vllm.entrypoints.render_server import serve_render

        uvloop.run(serve_render(args))

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        render_parser = subparsers.add_parser(
            self.name,
            help="Launch a render server for chat template rendering "
            "and tokenization (no GPU required).",
            description=DESCRIPTION,
            usage="vllm render [model_tag] [options]",
        )

        render_parser.add_argument(
            "model_tag",
            type=str,
            nargs="?",
            help="The model tag to serve (optional, overrides --model).",
        )
        render_parser.add_argument(
            "--model",
            type=str,
            default="Qwen/Qwen3-0.6B",
            help="Name or path of the model to use.",
        )
        render_parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to listen on.",
        )
        render_parser.add_argument(
            "--port",
            type=int,
            default=50052,
            help="Port to listen on.",
        )
        render_parser.add_argument(
            "--server",
            type=str,
            default="grpc",
            choices=["grpc", "http"],
            help="Server protocol to use (default: grpc).",
        )
        render_parser.add_argument(
            "--tokenizer",
            type=str,
            default=None,
            help="Name or path of the tokenizer (defaults to --model).",
        )
        render_parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default="auto",
            choices=["auto", "slow", "mistral"],
            help="Tokenizer mode.",
        )
        render_parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=False,
            help="Trust remote code from HuggingFace.",
        )
        render_parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="The specific model version to use.",
        )
        render_parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Maximum model context length (auto-detected if omitted).",
        )
        render_parser.add_argument(
            "--chat-template",
            type=str,
            default=None,
            help="Path to a custom chat template (Jinja2).",
        )

        render_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return render_parser


def cmd_init() -> list[CLISubcommand]:
    return [RenderSubcommand()]
