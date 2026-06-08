# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class CLISubcommand:
    """Base class for CLI argument handlers."""

    SUBCMD_EPILOG: typing.ClassVar[str] = (
        "For full list:            vllm {subcmd} --help=all\n"
        "For a section:            vllm {subcmd} --help=ModelConfig    "
        "(case-insensitive)\n"
        "For a flag:               vllm {subcmd} --help=max-model-len  "
        "(_ or - accepted)\n"
        "Documentation:            https://docs.vllm.ai\n"
    )

    name: str
    help: str
    description: str
    usage: str

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def validate(self, args: argparse.Namespace) -> None:
        # No validation by default
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        raise NotImplementedError("Subclasses should implement this method")
