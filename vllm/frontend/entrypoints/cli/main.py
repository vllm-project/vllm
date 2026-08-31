# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys
from importlib.util import find_spec

from vllm.foundation.observability.logger import init_logger

logger = init_logger(__name__)


def main():
    import vllm.frontend.entrypoints.cli.benchmark.main
    import vllm.frontend.entrypoints.cli.collect_env
    import vllm.frontend.entrypoints.cli.launch
    import vllm.frontend.entrypoints.cli.openai
    import vllm.frontend.entrypoints.cli.run_batch
    import vllm.frontend.entrypoints.cli.serve
    from vllm.frontend.entrypoints.serve.utils.api_utils import (
        VLLM_SUBCMD_PARSER_EPILOG,
        cli_env_setup,
    )
    from vllm.foundation.utilities.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        vllm.frontend.entrypoints.cli.openai,
        vllm.frontend.entrypoints.cli.serve,
        vllm.frontend.entrypoints.cli.launch,
        vllm.frontend.entrypoints.cli.benchmark.main,
        vllm.frontend.entrypoints.cli.collect_env,
        vllm.frontend.entrypoints.cli.run_batch,
    ]

    cli_env_setup()

    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" in sys.argv:
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
    else:
        vllm.frontend.entrypoints.cli.benchmark.main.maybe_exec_rust_bench()

        # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
        if len(sys.argv) > 1 and sys.argv[1] == "bench":
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
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        if args.subparser in cmds:
            cmds[args.subparser].validate(args)

        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
