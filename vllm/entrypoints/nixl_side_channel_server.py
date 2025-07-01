# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


class NixlSideChannelServer:

    def __init__(self, vllm_config: VllmConfig, host: str, port: int):
        self.vllm_config = vllm_config
        self.host = host
        self.port = port
        self.app = FastAPI(title="vLLM NIXL Side Channel Server")
        self.server = None
        self.server_thread = None
        self._setup_routes()

    def _setup_routes(self):

        @self.app.get("/get_kv_connector_metadata")
        @self.app.get("/get_kv_connector_metadata/{dp_rank}")
        @self.app.get("/get_kv_connector_metadata/{dp_rank}/{tp_rank}")
        async def get_kv_connector_metadata(dp_rank: Optional[int] = None,
                                            tp_rank: Optional[int] = None):
            kv_meta: Optional[dict[int, dict[
                int, KVConnectorHandshakeMetadata]]] = (
                    self.vllm_config.cache_config.transfer_handshake_metadata)

            if kv_meta is None:
                return None

            if dp_rank is not None:
                if dp_rank not in kv_meta:
                    return {}
                dp_data = kv_meta[dp_rank]

                if tp_rank is not None:
                    if tp_rank not in dp_data:
                        return {}
                    return {dp_rank: {tp_rank: dp_data[tp_rank]}}
                else:
                    return {dp_rank: dp_data}

            return kv_meta

    async def start_async(self):
        if self.server is not None:
            logger.warning("Side channel server is already running")
            return

        logger.info("Starting NIXL side channel server on %s:%s", self.host,
                    self.port)

        # use uvicorn directly to avoid dependency on engine_client
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True,
        )
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


async def start_nixl_side_channel_server_if_needed(
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
