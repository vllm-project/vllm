# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI reference client for the vLLM admin control plane API."""

import argparse
import json
import sys

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

ADMIN_DEFAULT_HOST = "localhost"
ADMIN_DEFAULT_PORT = 8000


def _base_url(args: argparse.Namespace) -> str:
    host = getattr(args, "admin_host", ADMIN_DEFAULT_HOST)
    port = getattr(args, "admin_port", ADMIN_DEFAULT_PORT)
    return f"http://{host}:{port}/v1/admin"


def _request(method: str, url: str) -> dict:
    """Make an HTTP request and return the JSON response."""
    import httpx

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url) if method == "GET" else client.post(url)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
        except json.JSONDecodeError:
            body = exc.response.text
        print(
            f"Error: HTTP {exc.response.status_code}: {body}",
            file=sys.stderr,
        )
        sys.exit(1)


def _cmd_status(args: argparse.Namespace) -> None:
    data = _request("GET", f"{_base_url(args)}/health")
    print(json.dumps(data, indent=2))


def _cmd_models(args: argparse.Namespace) -> None:
    data = _request("GET", f"{_base_url(args)}/models")
    print(json.dumps(data, indent=2))


def _cmd_queue(args: argparse.Namespace) -> None:
    data = _request("GET", f"{_base_url(args)}/queue")
    print(json.dumps(data, indent=2))


def _cmd_drain(args: argparse.Namespace) -> None:
    data = _request("POST", f"{_base_url(args)}/drain")
    print(json.dumps(data, indent=2))


def _cmd_resume(args: argparse.Namespace) -> None:
    data = _request("POST", f"{_base_url(args)}/resume")
    print(json.dumps(data, indent=2))


def _cmd_config(args: argparse.Namespace) -> None:
    data = _request("GET", f"{_base_url(args)}/config")
    print(json.dumps(data, indent=2))


def _cmd_reload(args: argparse.Namespace) -> None:
    data = _request("POST", f"{_base_url(args)}/reload_model")
    print(json.dumps(data, indent=2))


class AdminSubcommand(CLISubcommand):
    """The ``admin`` subcommand for the vLLM CLI."""

    name = "admin"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        sub = getattr(args, "admin_command", None)
        dispatch = {
            "status": _cmd_status,
            "models": _cmd_models,
            "queue": _cmd_queue,
            "drain": _cmd_drain,
            "resume": _cmd_resume,
            "config": _cmd_config,
            "reload": _cmd_reload,
        }
        if sub in dispatch:
            dispatch[sub](args)
        else:
            print("Usage: vllm admin {status|models|queue|drain|resume|config|reload}")
            sys.exit(1)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        admin_parser = subparsers.add_parser(
            self.name,
            help="Interact with a running vLLM admin control plane.",
            description="Reference CLI client for the vLLM admin API. "
            "Requires --enable-admin-api on the server.",
            usage="vllm admin <command> [options]",
        )
        admin_parser.add_argument(
            "--host",
            dest="admin_host",
            type=str,
            default=ADMIN_DEFAULT_HOST,
            help=f"Admin API host (default: {ADMIN_DEFAULT_HOST})",
        )
        admin_parser.add_argument(
            "--port",
            dest="admin_port",
            type=int,
            default=ADMIN_DEFAULT_PORT,
            help=f"Admin API port (default: {ADMIN_DEFAULT_PORT})",
        )
        admin_subs = admin_parser.add_subparsers(
            dest="admin_command",
        )
        admin_subs.add_parser("status", help="Show health and readiness")
        admin_subs.add_parser("models", help="List loaded models")
        admin_subs.add_parser("queue", help="Show queue / concurrency stats")
        admin_subs.add_parser("drain", help="Drain in-flight requests")
        admin_subs.add_parser("resume", help="Resume generation after drain")
        admin_subs.add_parser("config", help="Show engine configuration")
        admin_subs.add_parser("reload", help="Reload model (not yet implemented)")
        admin_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return admin_parser


def cmd_init() -> list[CLISubcommand]:
    return [AdminSubcommand()]
