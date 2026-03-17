# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM."""

import importlib
import importlib.metadata
import sys
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    module: str
    help: str


COMMAND_SPECS = (
    CommandSpec(
        name="pull",
        module="vllm.entrypoints.cli.local",
        help="Download a model by alias, Hugging Face repo, or local path.",
    ),
    CommandSpec(
        name="run",
        module="vllm.entrypoints.cli.local",
        help="Run a model locally in the terminal.",
    ),
    CommandSpec(
        name="ls",
        module="vllm.entrypoints.cli.local",
        help="List locally tracked models.",
    ),
    CommandSpec(
        name="list",
        module="vllm.entrypoints.cli.local",
        help="List locally tracked models. Alias for `ls`.",
    ),
    CommandSpec(
        name="aliases",
        module="vllm.entrypoints.cli.local",
        help="List built-in model aliases.",
    ),
    CommandSpec(
        name="inspect",
        module="vllm.entrypoints.cli.local",
        help="Show how a model reference resolves locally.",
    ),
    CommandSpec(
        name="serve",
        module="vllm.entrypoints.cli.serve",
        help="Launch a local OpenAI-compatible API server.",
    ),
    CommandSpec(
        name="ps",
        module="vllm.entrypoints.cli.local",
        help="List managed local services.",
    ),
    CommandSpec(
        name="stop",
        module="vllm.entrypoints.cli.local",
        help="Stop a managed local service.",
    ),
    CommandSpec(
        name="logs",
        module="vllm.entrypoints.cli.local",
        help="Show logs for a managed local service.",
    ),
    CommandSpec(
        name="rm",
        module="vllm.entrypoints.cli.local",
        help="Remove local model metadata or cached files.",
    ),
    CommandSpec(
        name="chat",
        module="vllm.entrypoints.cli.openai",
        help="Generate chat completions via the running API server.",
    ),
    CommandSpec(
        name="complete",
        module="vllm.entrypoints.cli.openai",
        help="Generate text completions via the running API server.",
    ),
    CommandSpec(
        name="bench",
        module="vllm.entrypoints.cli.benchmark.main",
        help="Run vLLM benchmarks.",
    ),
    CommandSpec(
        name="collect-env",
        module="vllm.entrypoints.cli.collect_env",
        help="Collect environment information.",
    ),
    CommandSpec(
        name="run-batch",
        module="vllm.entrypoints.cli.run_batch",
        help="Run batch prompts and write results to file.",
    ),
    CommandSpec(
        name="launch",
        module="vllm.entrypoints.cli.launch",
        help="Launch individual vLLM components.",
    ),
)
COMMAND_SPECS_BY_NAME = {spec.name: spec for spec in COMMAND_SPECS}


def _get_root_parser():
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    for spec in COMMAND_SPECS:
        subparsers.add_parser(
            spec.name,
            help=spec.help,
            description=spec.help,
        )
    return parser


def _requested_subcommand(argv: list[str]) -> str | None:
    for arg in argv:
        if arg == "--":
            break
        if arg.startswith("-"):
            continue
        return arg
    return None


def _load_subcommand(spec: CommandSpec):
    module = importlib.import_module(spec.module)
    for cmd in module.cmd_init():
        if cmd.name == spec.name:
            return cmd
    raise RuntimeError(
        f"Subcommand `{spec.name}` was not found in module `{spec.module}`."
    )


def _build_command_parser(spec: CommandSpec):
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmd = _load_subcommand(spec)
    cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
    return parser, cmd


def main():
    from vllm.entrypoints.utils import cli_env_setup

    argv = sys.argv[1:]
    requested_subcommand = _requested_subcommand(argv)
    if requested_subcommand is None:
        parser = _get_root_parser()
        args = parser.parse_args(argv)
        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            parser.print_help()
        return

    spec = COMMAND_SPECS_BY_NAME.get(requested_subcommand)
    if spec is None:
        _get_root_parser().parse_args(argv)
        return

    cli_env_setup()

    # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
    if requested_subcommand == "bench":
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

    parser, cmd = _build_command_parser(spec)
    args = parser.parse_args(argv)
    setattr(args, "_argv", sys.argv[1:])
    if args.subparser == cmd.name:
        cmd.validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
