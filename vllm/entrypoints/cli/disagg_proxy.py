# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`vllm disagg-proxy`: the front end of a disaggregated EPD deployment."""

import argparse

import uvicorn

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser


class DisaggProxySubcommand(CLISubcommand):
    """Run the encode/prefill/decode proxy."""

    name = "disagg-proxy"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.distributed.ec_transfer.proxy.epd_proxy import (
            EPDProxyConfig,
            build_app,
        )

        app = build_app(
            EPDProxyConfig(
                no_rewrite=args.no_rewrite,
                probe_interval=args.probe_interval,
                probe_timeout=args.probe_timeout,
                fail_threshold=args.fail_threshold,
                evicted_ttl=args.evicted_ttl,
            )
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    def validate(self, args: argparse.Namespace) -> None:
        if args.fail_threshold < 1:
            raise ValueError("--fail-threshold must be at least 1")

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "disagg-proxy",
            help="Run the disaggregated encode/prefill/decode proxy.",
            description=(
                "Routes multimodal requests across encode, prefill and decode "
                "instances. The proxy holds no topology of its own: start it "
                "first, and instances register with it as they come up."
            ),
            usage="vllm disagg-proxy [options]",
        )
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument(
            "--no-rewrite",
            action="store_true",
            help=(
                "Forward media to the decoder unchanged instead of replacing "
                "it with a reference to the encoder's embedding. For A/B "
                "timing the rewrite itself."
            ),
        )
        parser.add_argument(
            "--probe-interval",
            type=float,
            default=5.0,
            help="Seconds between health probes. 0 disables probing.",
        )
        parser.add_argument("--probe-timeout", type=float, default=2.0)
        parser.add_argument(
            "--fail-threshold",
            type=int,
            default=3,
            help=(
                "Consecutive failed probes before an instance stops being "
                "routed to. It rejoins on its own once it responds again."
            ),
        )
        parser.add_argument(
            "--evicted-ttl",
            type=float,
            default=900.0,
            help=(
                "Seconds to keep probing an instance that stopped responding "
                "before forgetting it. 0 probes forever."
            ),
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [DisaggProxySubcommand()]
