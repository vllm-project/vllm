# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.utils.gc_utils import freeze_gc_heap

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
