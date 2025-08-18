# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class KVConnHandshakeServer:

    def __init__(self, vllm_config: VllmConfig, host: str, port: int):
        self.vllm_config = vllm_config
        self.host = host
        self.port = port
        self.app = FastAPI(title="vLLM KVConnector Handshake Server")
        self.server: Optional[uvicorn.Server] = None
        self._setup_routes()

    def _get_connector_name(self) -> str:
        if self.vllm_config.kv_transfer_config is None:
            return "Unknown"
        return self.vllm_config.kv_transfer_config.kv_connector or "Unknown"

    def _setup_routes(self):

        @self.app.get("/get_kv_connector_metadata")
        @self.app.get("/get_kv_connector_metadata/{dp_rank}")
        @self.app.get("/get_kv_connector_metadata/{dp_rank}/{tp_rank}")
        async def get_kv_connector_metadata(dp_rank: Optional[int] = None,
                                            tp_rank: Optional[int] = None):
            kv_meta = self.vllm_config.parallel_config.xfer_handshake_metadata

            if kv_meta is None:
                raise HTTPException(
                    status_code=404,
                    detail="KV connector handshake metadata is not available")
            if dp_rank is None:
                return kv_meta

            if not (dp_data := kv_meta.get(dp_rank)):
                raise HTTPException(
                    status_code=404,
                    detail=f"Data parallel rank {dp_rank} not found")

            if tp_rank is None:
                return {dp_rank: dp_data}

            if not (tp_data := dp_data.get(tp_rank)):
                raise HTTPException(
                    status_code=404,
                    detail=f"Tensor parallel rank {tp_rank} not found for data \
                        parallel rank {dp_rank}")

            return {dp_rank: {tp_rank: tp_data}}

    async def start_async(self):
        if self.server is not None:
            logger.warning("Side channel server is already running")
            return

        listen_address = f"http://{self.host}:{self.port}"
        logger.info("Starting %s handshake server on %s",
                    self._get_connector_name(), listen_address)

        # prepare uvicorn configuration
        config_kwargs: dict[str, Any] = {
            "app": self.app,
            "host": self.host,
            "port": self.port,
            "log_level": "info",
            "access_log": True,
        }

        config = uvicorn.Config(**config_kwargs)
        config.load()  # need to load config to get SSL context
        self.server = uvicorn.Server(config)

        # start the server in a background task
        if self.server is not None:
            asyncio.create_task(self.server.serve())
        logger.info("%s handshake server started successfully",
                    self._get_connector_name())

    async def stop_async(self):
        if self.server is not None:
            logger.info("Stopping %s handshake server",
                        self._get_connector_name())
            try:
                self.server.should_exit = True
                await asyncio.sleep(1)  # give it time to shutdown
            except Exception as e:
                logger.warning("Error during side channel server shutdown: %s",
                               e)
            self.server = None
            logger.info("%s handshake server stopped",
                        self._get_connector_name())


def should_start_kv_handshake_server(vllm_config: VllmConfig) -> bool:
    if vllm_config.kv_transfer_config is None:
        return False

    connector_name = vllm_config.kv_transfer_config.kv_connector
    if connector_name is None:
        return False

    # check connector-specific handshake method
    if connector_name == "NixlConnector":
        handshake_method = envs.VLLM_NIXL_HANDSHAKE_METHOD.lower()
        return handshake_method == "http"

    # other connectors can be added here with their own logic
    # for now, only nixl supports http handshake
    return False


def _get_handshake_server_config(vllm_config: VllmConfig) -> tuple[str, int]:
    """get host and port for the handshake server based on connector type."""
    if vllm_config.kv_transfer_config is None:
        raise RuntimeError(
            "KV transfer config is None but tried to start handshake server")
    connector_name = vllm_config.kv_transfer_config.kv_connector

    if connector_name == "NixlConnector":
        return (envs.VLLM_NIXL_SIDE_CHANNEL_HOST,
                envs.VLLM_NIXL_SIDE_CHANNEL_PORT)

    # default fallback values
    return "localhost", 5557


async def set_up_kv_handshake_server(
        vllm_config: VllmConfig) -> Optional[KVConnHandshakeServer]:
    if not should_start_kv_handshake_server(vllm_config):
        return None

    side_channel_host, side_channel_port = _get_handshake_server_config(
        vllm_config)

    assert vllm_config.kv_transfer_config is not None
    connector_name = vllm_config.kv_transfer_config.kv_connector

    logger.info("Starting %s handshake server on %s:%d", connector_name,
                side_channel_host, side_channel_port)

    server = KVConnHandshakeServer(vllm_config, side_channel_host,
                                   side_channel_port)
    await server.start_async()
    return server
