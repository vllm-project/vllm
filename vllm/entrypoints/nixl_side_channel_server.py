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


class NixlSideChannelServer:

    def __init__(self, vllm_config: VllmConfig, host: str, port: int):
        self.vllm_config = vllm_config
        self.host = host
        self.port = port
        self.app = FastAPI(title="vLLM NIXL Side Channel Server")
        self.server: Optional[uvicorn.Server] = None
        self._setup_routes()

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
            if dp_rank is not None and not (dp_data := kv_meta.get(dp_rank)):
                raise HTTPException(
                    status_code=404,
                    detail=f"Data parallel rank {dp_rank} not found")
            if tp_rank is not None and dp_data is not None and not (
                    tp_data := dp_data.get(tp_rank)):
                raise HTTPException(
                    status_code=404,
                    detail=f"Tensor parallel rank {tp_rank} not found for data \
                        parallel rank {dp_rank}")

            if dp_rank is None:
                return kv_meta

            if tp_rank is None:
                return {dp_rank: dp_data}

            return {dp_rank: {tp_rank: tp_data}}

    async def start_async(self):
        if self.server is not None:
            logger.warning("Side channel server is already running")
            return

        listen_address = f"http://{self.host}:{self.port}"
        logger.info("Starting NIXL side channel server on %s", listen_address)

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
        logger.info("NIXL side channel server started successfully")

    async def stop_async(self):
        if self.server is not None:
            logger.info("Stopping NIXL side channel server")
            try:
                self.server.should_exit = True
                await asyncio.sleep(1)  # give it time to shutdown
            except Exception as e:
                logger.warning("Error during side channel server shutdown: %s",
                               e)
            self.server = None
            logger.info("NIXL side channel server stopped")


def should_start_nixl_side_channel_server(vllm_config: VllmConfig) -> bool:
    if vllm_config.kv_transfer_config is None:
        return False

    if vllm_config.kv_transfer_config.kv_connector != "NixlConnector":
        return False

    handshake_method = envs.VLLM_NIXL_HANDSHAKE_METHOD.lower()
    return handshake_method == "http"


async def set_up_nixl_side_channel_server(
        vllm_config: VllmConfig) -> Optional[NixlSideChannelServer]:
    if not should_start_nixl_side_channel_server(vllm_config):
        return None

    side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
    side_channel_port = envs.VLLM_NIXL_SIDE_CHANNEL_PORT

    logger.info("Starting NIXL side channel metadata server on %s:%d",
                side_channel_host, side_channel_port)

    server = NixlSideChannelServer(vllm_config, side_channel_host,
                                   side_channel_port)
    await server.start_async()
    return server
