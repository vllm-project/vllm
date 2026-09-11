# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import multiprocessing
import os
import signal
from logging import Logger
from multiprocessing import forkserver
from string import Template

from fastapi import FastAPI

from vllm import EngineArgs, envs
from vllm.entrypoints.serve.utils.api_utils import (
    get_non_default_args,
    redact_sensitive_args,
)
from vllm.logger import current_formatter_type, init_logger

logger = init_logger(__name__)


def log_version_and_model(lgr: Logger, version: str, model_name: str) -> None:
    formatter = None if envs.VLLM_DISABLE_LOG_LOGO else current_formatter_type(lgr)
    if formatter is None:
        message = "vLLM server version %s, serving model %s"
    else:
        logo_template = Template(
            "\n       ${w}█     █     █▄   ▄█${r}\n"
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r}  version ${w}%s${r}\n"
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}  model   ${w}%s${r}\n"
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}\n"
        )
        colors = {
            "w": "\033[1m",  # bold, default foreground
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
        }
        if formatter != "color":
            # monochrome logo (no ansi escape codes)
            colors = dict.fromkeys(colors, "")

        message = logo_template.substitute(colors)

    lgr.info(message, version, model_name)


def log_non_default_args(args: argparse.Namespace | EngineArgs):
    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", redact_sensitive_args(non_default_args))


def setup_forkserver():
    """Pre-import heavy modules in the forkserver process (idempotent)."""
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")


def setup_interrupt_handler():
    """Install a pre-uvicorn SIGINT/SIGTERM handler that raises KeyboardInterrupt.

    This is a one-shot, process-level side effect.  It is replaced by
    ``loop.add_signal_handler`` once ``_install_signal_handlers`` runs
    (after uvicorn has been built), so it only matters for the window
    between socket bind and ``server.serve()`` starting.
    """

    def _interrupt_init(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGINT, _interrupt_init)
    signal.signal(signal.SIGTERM, _interrupt_init)


def init_parser_plugin(args: argparse.Namespace):
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)


def log_routes(app: FastAPI) -> None:
    """Log the routes exposed by ``app`` for operator convenience."""
    logger.info("Available routes are:")
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue

        methods = getattr(route, "methods", None)
        if methods:
            logger.info("Route: %s, Methods: %s", path, ", ".join(sorted(methods)))
            continue

        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            name = getattr(endpoint, "__name__", type(endpoint).__name__)
            logger.info("Route: %s, Endpoint: %s", path, name)
