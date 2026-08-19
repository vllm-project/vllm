# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from argparse import Namespace
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.utils.gc_utils import freeze_gc_heap

logger = init_logger(__name__)


def load_log_config(log_config_file: str | None) -> dict | None:
    if not log_config_file:
        return None
    try:
        with open(log_config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(
            "Failed to load log config from file %s: error %s", log_config_file, e
        )
        return None


def get_uvicorn_log_config(args: Namespace) -> dict | None:
    """
    Get the uvicorn log config based on the provided arguments.

    Priority:
    1. If log_config_file is specified, use it
    2. If disable_access_log_for_endpoints is specified, create a config with
       the access log filter
    3. Otherwise, return None (use uvicorn defaults)
    """
    # First, try to load from file if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        return log_config

    # If endpoints to filter are specified, create a config with the filter
    if args.disable_access_log_for_endpoints:
        from vllm.logging_utils import create_uvicorn_log_config

        # Parse comma-separated string into list
        excluded_paths = [
            p.strip()
            for p in args.disable_access_log_for_endpoints.split(",")
            if p.strip()
        ]
        return create_uvicorn_log_config(
            excluded_paths=excluded_paths,
            log_level=args.uvicorn_log_level,
        )

    return None


_running_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
            for attr_name in (
                "openai_serving_transcription",
                "openai_serving_translation",
            ):
                serving = getattr(app.state, attr_name, None)
                if serving is not None and hasattr(serving, "shutdown"):
                    serving.shutdown()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state
