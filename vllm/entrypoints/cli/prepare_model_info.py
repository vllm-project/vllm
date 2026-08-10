# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser


class PrepareModelInfoSubcommand(CLISubcommand):
    """The ``prepare-model-info`` subcommand for the vLLM CLI."""

    name = "prepare-model-info"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.model_executor.models import ModelRegistry

        ModelRegistry.prepare_model_info(args.architecture)
        print(f"prepared model info for {args.architecture}")

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Prepare cached metadata for a built-in model architecture.",
            description=(
                "Inspect a built-in model architecture in a subprocess and cache "
                "its metadata in VLLM_CACHE_ROOT for later vLLM starts."
            ),
            usage="vllm prepare-model-info ARCHITECTURE",
        )
        parser.add_argument(
            "architecture",
            help="A built-in architecture name, such as Qwen3ForCausalLM.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [PrepareModelInfoSubcommand()]
