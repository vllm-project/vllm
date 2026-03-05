# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.eval.runner import add_cli_args
from vllm.eval.runner import main as eval_main

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class EvalSubcommand(CLISubcommand):
    """Standalone eval subcommand that manages its own vLLM server.

    Note: the primary eval integration is via vllm bench serve --eval,
    which runs against an existing server. This subcommand is for cases
    where you want vllm to manage the server lifecycle automatically.
    """

    name = "eval"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        eval_main(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        help_text = (
            "Run accuracy (lm_eval) and performance benchmarks in a single "
            "pass, producing a unified JSONL report with environment info."
        )
        eval_parser = subparsers.add_parser(
            self.name,
            help=help_text,
            description=help_text,
            usage="vllm eval [options]",
        )
        add_cli_args(eval_parser)
        return eval_parser


def cmd_init() -> list[CLISubcommand]:
    return [EvalSubcommand()]
