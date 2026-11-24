# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `download-kernels` subcommand for the vLLM CLI.

Installs the precompiled kernels that vLLM's dependencies would otherwise
download or compile at startup. Run it after installing or upgrading vLLM:

    vllm download-kernels
"""

import argparse
import os
import subprocess
import sys
import typing
from importlib.util import find_spec

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.logger import init_logger

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


def _download_flashinfer_kernels(dry_run: bool) -> int:
    if find_spec("flashinfer") is None:
        logger.info("FlashInfer is not installed; skipping its kernels.")
        return 0
    from vllm.utils.flashinfer import has_flashinfer_jit_cache_wheels

    cmd = [sys.executable, "-m", "flashinfer"]
    if has_flashinfer_jit_cache_wheels():
        cmd.append("download-kernels")
    else:
        logger.info(
            "FlashInfer does not publish flashinfer-jit-cache for this CUDA "
            "version; installing flashinfer-cubin only."
        )
        cmd.append("install-cubin-wheel")
    if dry_run:
        cmd.append("--dry-run")
    # FlashInfer refuses to import while kernels from another FlashInfer version
    # are installed, which would stop its own CLI from replacing them.
    env = {**os.environ, "FLASHINFER_DISABLE_VERSION_CHECK": "1"}
    return subprocess.run(cmd, env=env, check=False).returncode


class DownloadKernelsSubcommand(CLISubcommand):
    """The `download-kernels` subcommand for the vLLM CLI."""

    name = "download-kernels"

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the install commands without running them.",
        )
        return parser

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        sys.exit(_download_flashinfer_kernels(args.dry_run))

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Install precompiled kernels for faster startup.",
            description=(
                "Install the precompiled kernels that vLLM's dependencies would "
                "otherwise download or compile at startup. Run it after "
                "installing or upgrading vLLM."
            ),
            usage="vllm download-kernels [--dry-run]",
        )
        return DownloadKernelsSubcommand.add_cli_args(parser)


def cmd_init() -> list[CLISubcommand]:
    return [DownloadKernelsSubcommand()]
