# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoint of vLLM."""

import importlib
import importlib.metadata
import sys
from importlib.util import find_spec

from vllm.entrypoints.cli._utils import (
    CLI_COMMANDS,
    VLLM_SUBCMD_PARSER_EPILOG,
    cli_env_setup,
    is_serve_help,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _selected_command(argv: list[str]) -> str | None:
    return next((arg for arg in argv if not arg.startswith("-")), None)


def _load_command(name: str):
    module_name, _ = CLI_COMMANDS[name]
    module = importlib.import_module(module_name)
    for command in module.cmd_init():
        if command.name == name:
            return command
    raise RuntimeError(f"CLI module {module_name} did not register {name}")


def _vllm_version() -> str:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        # Source-tree runs have no installed distribution metadata.
        from vllm.version import __version__

        return __version__


def _build_parser(selected_name: str | None):
    selected_command = None
    if selected_name in CLI_COMMANDS:
        selected_command = _load_command(selected_name)

    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=_vllm_version(),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")

    for name, (_, help_text) in CLI_COMMANDS.items():
        if name == selected_name:
            assert selected_command is not None
            selected_command.subparser_init(subparsers).set_defaults(
                dispatch_function=selected_command.cmd
            )
        else:
            subparsers.add_parser(name, help=help_text, add_help=False)

    return parser, selected_command


def main():
    argv = sys.argv[1:]

    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" in argv:
        cli_env_setup()
        # NOTE: Check the spec instead of importing directly here, since things could
        # fail with ImportError due to mismatched versions if things are moved around.
        if find_spec("vllm_omni") is None:
            logger.error(
                "--omni flag requires a valid instance of vllm-omni to be installed."
            )
            raise SystemExit(1)

        from vllm_omni.entrypoints.cli.main import main as omni_main

        logger.info("Delegating entrypoint handling to vllm-omni")
        omni_main()
        return

    selected_name = _selected_command(argv)
    if selected_name in CLI_COMMANDS and not is_serve_help(argv):
        cli_env_setup()

        if selected_name == "bench":
            benchmark_module = importlib.import_module(CLI_COMMANDS["bench"][0])
            benchmark_module.maybe_exec_rust_bench()

            # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
            logger.debug(
                "Bench command detected, must ensure current platform is not "
                "UnspecifiedPlatform to avoid device type inference error"
            )
            from vllm import platforms

            if platforms.current_platform.is_unspecified():
                from vllm.platforms.cpu import CpuPlatform

                platforms.current_platform = CpuPlatform()
                logger.info(
                    "Unspecified platform detected, switching to CPU Platform instead."
                )

    parser, selected_command = _build_parser(selected_name)
    args = parser.parse_args()
    if selected_command is not None and args.subparser == selected_command.name:
        selected_command.validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
