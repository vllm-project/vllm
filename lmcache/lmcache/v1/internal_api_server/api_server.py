# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING
import asyncio
import os
import threading

# Third Party
from fastapi import FastAPI
import uvicorn

# First Party
from lmcache.logging import init_logger

# Local
from .api_registry import APIRegistry

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.manager import LMCacheManager

logger = init_logger(__name__)


def _build_app() -> FastAPI:
    """Create a fresh FastAPI app with all internal API routes registered."""
    new_app = FastAPI()
    APIRegistry(new_app).register_all_apis()
    return new_app


# Module-level app kept for backward compatibility with existing tests that
# import `app` directly. Production code no longer relies on this shared
# singleton; each InternalAPIServer owns its own FastAPI app so that
# multiple instances in the same process (e.g. scheduler + worker in TP=1
# non-MP mode) do not overwrite each other's app.state.lmcache_adapter.
app = _build_app()


class InternalAPIServer:
    def __init__(self, lmcache_manager: "LMCacheManager"):
        lmcache_engine = lmcache_manager.lmcache_engine

        # Check if lmcache_engine is None and handle accordingly
        if lmcache_engine is None:
            # Use manager's config directly when engine is not available
            config = lmcache_manager.config
            port_offset = 0  # Default for scheduler mode
        else:
            config = lmcache_engine.config
            # 0 for scheduler, 1 for worker 0, 2 for worker 1, ...
            port_offset = 1 + lmcache_engine.metadata.worker_id

        self.port = config.internal_api_server_port_start + port_offset
        self.socket_path_prefix = config.internal_api_server_socket_path_prefix
        self.socket_path = (
            f"{self.socket_path_prefix}_{self.port}"
            if self.socket_path_prefix
            else None
        )
        include_index_list = config.internal_api_server_include_index_list

        self.enable = True
        if not config.internal_api_server_enabled or (
            include_index_list and port_offset not in include_index_list
        ):
            logger.info(
                "Internal API server disabled. internal_api_server_enabled=%s, "
                "port_offset=%s, port=%s, socket_path=%s, include_index_list=%s",
                config.internal_api_server_enabled,
                port_offset,
                self.port,
                self.socket_path,
                include_index_list,
            )
            self.enable = False
            return

        self.app = _build_app()

        uvicorn_config = {
            "app": self.app,
            "host": config.internal_api_server_host,
            "loop": "uvloop",
            "http": "httptools",
            "access_log": config.get_extra_config_value(
                "internal_api_server_access_log", True
            ),
            "log_level": config.get_extra_config_value(
                "internal_api_server_log_level", "warning"
            ),
        }

        if self.socket_path:
            self.server_log_info = f"socket {self.socket_path}"
            logger.info("Init internal API server on %s", self.server_log_info)
            uvicorn_config["uds"] = self.socket_path
            # Ensure socket directory exists
            os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
            # Remove existing socket file if exists
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        else:
            self.server_log_info = f"port {self.port}"
            logger.info("Init internal API server on %s", self.server_log_info)
            uvicorn_config["port"] = self.port

        self.server = uvicorn.Server(uvicorn.Config(**uvicorn_config))
        self.app.state.lmcache_adapter = lmcache_manager

    async def run(self):
        logger.info("Running LMCache internal API server on %s", self.server_log_info)
        if self.server:
            await self.server.serve()

    def start(self):
        if not self.enable:
            return
        logger.info("Starting LMCache internal API server on %s", self.server_log_info)
        threading.Thread(
            target=asyncio.run,
            args=(self.run(),),
            daemon=True,
            name="api-server-thread",
        ).start()

    def stop(self):
        if not self.enable:
            return
        logger.info("Stopping LMCache internal API server")
        if self.server:
            self.server.should_exit = True
            if self.socket_path and os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
