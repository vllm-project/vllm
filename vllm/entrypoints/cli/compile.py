# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.utils.argparse_utils import FlexibleArgumentParser

DESCRIPTION = """[Experimental] Populate vLLM's torch.compile cache for a model.

This command is experimental and a work in progress. Not all models and
configurations are supported yet.

This runs compilation using fake weights (zero GPU memory) so that
vLLM's torch.compile cache is populated.  Subsequent ``vllm serve`` or
``LLM(...)`` calls for the same model will hit the warm cache and skip
compilation.
"""


class CompileSubcommand(CLISubcommand):
    """The ``compile`` subcommand for the vLLM CLI."""

    name = "compile"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.compile_only import run_compile_only

        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag
        run_compile_only(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        compile_parser = subparsers.add_parser(
            self.name,
            help="[Experimental] Populate vLLM's torch.compile cache for a model.",
            description=DESCRIPTION,
            usage="vllm compile [model_tag] [options]",
        )
        compile_parser = make_arg_parser(compile_parser)
        compile_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return compile_parser


def cmd_init() -> list[CLISubcommand]:
    return [CompileSubcommand()]
