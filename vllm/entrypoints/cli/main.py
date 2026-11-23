# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM.

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys
from importlib.util import find_spec

from vllm.logger import configure_logging_from_args, init_logger

logger = init_logger(__name__)

# Keep the order shown by `vllm --help`. A module may define multiple commands.
_SUBCOMMAND_MODULES = {
    "chat": "vllm.entrypoints.cli.openai",
    "complete": "vllm.entrypoints.cli.openai",
    "serve": "vllm.entrypoints.cli.serve",
    "launch": "vllm.entrypoints.cli.launch",
    "bench": "vllm.entrypoints.cli.benchmark.main",
    "collect-env": "vllm.entrypoints.cli.collect_env",
    "download-kernels": "vllm.entrypoints.cli.download_kernels",
    "preload": "vllm.entrypoints.cli.preload",
    "run-batch": "vllm.entrypoints.cli.run_batch",
    "snapshot": "vllm.entrypoints.cli.snapshot",
}


def maybe_run_omni() -> bool:
    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" not in sys.argv:
        return False

    # NOTE: Check the spec instead of importing directly here, since things could
    # fail with ImportError due to mismatched versions if things are moved around.
    spec = find_spec("vllm_omni")
    if spec is None:
        logger.error(
            "--omni flag requires a valid instance of vllm-omni to be installed."
        )
        sys.exit(1)

    from vllm_omni.entrypoints.cli.main import main as omni_main

    logger.info("Delegating entrypoint handling to vllm-omni")
    omni_main()

    return True


def main():
    if maybe_run_omni():
        return

    from vllm.utils.gc_utils import gc_paused_for_imports

    subcommand = sys.argv[1] if len(sys.argv) > 1 else None
    with gc_paused_for_imports():
        from vllm.entrypoints.common.startup import (
            VLLM_SUBCMD_PARSER_EPILOG,
            cli_env_setup,
        )
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        if subcommand in _SUBCOMMAND_MODULES:
            cmd_modules = [importlib.import_module(_SUBCOMMAND_MODULES[subcommand])]
        elif subcommand in ("-v", "--version"):
            cmd_modules = []
        else:
            cmd_modules = [
                importlib.import_module(name)
                for name in dict.fromkeys(_SUBCOMMAND_MODULES.values())
            ]

    if subcommand != "snapshot":
        cli_env_setup()

    if subcommand == "bench":
        from vllm.entrypoints.cli.benchmark.main import maybe_exec_rust_bench

        maybe_exec_rust_bench()

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
    if subcommand in _SUBCOMMAND_MODULES:
        # Preserve full usage metavar without changing `argument subparser:` errors.
        subparsers.metavar = "{" + ",".join(_SUBCOMMAND_MODULES) + "}"
    cmds = {}
    for cmd_module in cmd_modules:
        if cmd_module.__name__ == _SUBCOMMAND_MODULES["snapshot"]:
            new_cmds = cmd_module.cmd_init(
                create_requested=sys.argv[1:3] == ["snapshot", "create"]
            )
        else:
            new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmd = cmds[args.subparser]
        configure_logging_from_args(args)
        cmd.validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
