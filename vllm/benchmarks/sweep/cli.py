# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

from .plot import SweepPlotArgs
from .plot import main as plot_main
from .plot_pareto import SweepPlotParetoArgs
from .plot_pareto import main as plot_pareto_main
from .serve import SweepServeArgs
from .serve import main as serve_main
from .serve_sla import SweepServeSLAArgs
from .serve_sla import main as serve_sla_main
from .startup import SweepStartupArgs
from .startup import main as startup_main

SUBCOMMANDS = (
    (SweepServeArgs, serve_main),
    (SweepServeSLAArgs, serve_sla_main),
    (SweepStartupArgs, startup_main),
    (SweepPlotArgs, plot_main),
    (SweepPlotParetoArgs, plot_pareto_main),
)


def add_cli_args(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(required=True, dest="sweep_type")

    for cmd, entrypoint in SUBCOMMANDS:
        cmd_subparser = subparsers.add_parser(
            cmd.parser_name,
            description=cmd.parser_help,
            usage=f"vllm bench sweep {cmd.parser_name} [options]",
        )
        cmd_subparser.set_defaults(dispatch_function=entrypoint)
        cmd.add_cli_args(cmd_subparser)
        cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(
            subcmd=f"sweep {cmd.parser_name}"
        )


def main(args: argparse.Namespace):
    args.dispatch_function(args)
