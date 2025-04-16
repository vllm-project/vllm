# SPDX-License-Identifier: Apache-2.0

import asyncio
import aiohttp
from collections import defaultdict
from fastapi import Request
import msgspec
import threading
from typing import Optional
from urllib.parse import urlparse

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.remote_prefill import (RemotePrefillParams,
                                 RemotePrefillRequest)

# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (NixlMetadataRequest,
                                              NixlMetadataResponse,
                                              RemotePrefillEpRequest,
                                              RemotePrefillGenerateRequest)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.inputs.data import TokensPrompt

from vllm.distributed.device_communicators.nixl import NixlMetadata

logger = init_logger(__name__)


class OpenAIServingRemotePrefill(OpenAIServing):
    """OpenAI API for remote prefill.

    Handles the routes:
    - /remote_nixl_metadata [POST]
    - /nixl_metadata [GET]
    - /remote_prefill [POST]
    """

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
    ):
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
        )
        
        self.remote_prefill_endpoint_map = defaultdict(int)
        self.remote_prefill_endpoints = []
        self.counter = 0
        
        self._request_queue = asyncio.Queue()

        loop = asyncio.get_event_loop()
        self._background_thread = threading.Thread(target=self.background_event_loop, daemon=True, args=(loop, ))
        self._background_thread.start()
        
    def background_event_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.create_task(self._process_requests())

    async def _process_requests(self):
        while True:
            request = await self._request_queue.get()

            sampling_params = request.sampling_params
            sampling_params.max_tokens = 1
            sampling_params.min_tokens = 1

            remote_prefill_params = RemotePrefillParams(
                is_remote_decode=True,
                decode_block_ids=request.block_ids,
                decode_engine_id=request.engine_id,
            )

            async for _ in self.engine_client.generate(
                request_id=request.request_id,
                prompt=TokensPrompt(prompt_token_ids=request.prompt_token_ids),
                sampling_params=sampling_params,
                remote_prefill_params=remote_prefill_params,
            ):
                pass

    def nixl_metadata(self) -> NixlMetadataResponse:
        """Get Nixl metadata"""

        metadata = str(
            msgspec.json.encode(self.engine_client.nixl_metadata), encoding="utf-8"
        )

        return NixlMetadataResponse(metadata=metadata)

    async def remote_nixl_metadata(
        self,
        request: NixlMetadataRequest,
    ):
        """Add remote Nixl metadata"""
        metadata = msgspec.json.decode(
            request.metadata.encode(encoding="utf-8"), type=NixlMetadata
        )

        await self.engine_client.add_remote_nixl_metadata(metadata)

    def remote_prefill(self, request: RemotePrefillGenerateRequest):
        request = msgspec.json.decode(
            request.content.encode(encoding="utf-8"),
            type=RemotePrefillRequest,
        )

        self._request_queue.put_nowait(request)

    def get_remote_prefill_request_callback(self):
        # TODO: integrate prefill_queue to dynamo endpoint
        async def callback(request: RemotePrefillRequest):
            endpoint = self.remote_prefill_endpoints[self.counter % len(self.remote_prefill_endpoints)]
            self.counter = (self.counter + 1) % (2** 31 - 1)
            remote_prefill_url = f"{endpoint}/remote_prefill"
            logger.debug(f"Remote prefill endpoint: {remote_prefill_url}")
            request = RemotePrefillGenerateRequest(
                content=str(msgspec.json.encode(request), encoding="utf-8"),
            )
            async with aiohttp.ClientSession() as session:
                await session.post(remote_prefill_url, json=request.model_dump())

        return callback

    def get_remote_prefill_params(self, request: Request):
        if len(self.remote_prefill_endpoints) == 0:
            return None

        remote_prefill_params = RemotePrefillParams(
                is_remote_prefill=True,
                remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
        )
        return remote_prefill_params

    def _update_remote_prefill_endpoints(self):
        """Calculate remote prefill endpoints"""
        if not self.remote_prefill_endpoint_map:
            self.remote_prefill_endpoints = []

        self.remote_prefill_endpoints = [ep for ep in self.remote_prefill_endpoint_map.keys() \
                                         if self.remote_prefill_endpoint_map[ep] == 1]
        #TODO: let's clean up the map with the value = 0
        # and we should remote the nixl connections
        logger.info(f"Remote prefill endpoints: {self.remote_prefill_endpoints}")
        return self.remote_prefill_endpoints

    async def add_remote_prefill_ep(self, ep: str):
        add_remote_nixl_metadata_url = f"{ep}/remote_nixl_metadata"
        metadata = NixlMetadataRequest(
            metadata=self.nixl_metadata().metadata,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(add_remote_nixl_metadata_url, json=metadata.model_dump()) as resp:
                if resp.status != 200:
                    raise ValueError(f"add local nixl metadata to remote failed with status: {resp.status}")

    async def add_remote_prefill_eps(self, request: RemotePrefillEpRequest):
        if not request.endpoints or len(request.endpoints) == 0:
            raise ValueError("Empty URL")
        endpoints = [parsed for parsed in map(urlparse, request.endpoints) \
                     if all([parsed.scheme, parsed.netloc]) and parsed.scheme in ["http", "https"]]
        endpoints = [f"{x.scheme}://{x.netloc}" for x in endpoints]
        if len(endpoints) == 0:
            raise ValueError(f"No valid endpoints: {request.endpoints}")
        for ep in endpoints:
            try:
                await self.add_remote_prefill_ep(ep)
                self.remote_prefill_endpoint_map[ep] = 1
            except ValueError as e:
                logger.error(f"Failed to add remote prefill endpoint {ep}: {e}")
                continue
        self._update_remote_prefill_endpoints()

    async def remove_remote_prefill_eps(self, request: RemotePrefillEpRequest):
        if not request.endpoints or len(request.endpoints) == 0:
            logger.error("No remote prefill endpoint to be removed")
            return
        for ep in request.endpoints:
            if ep in self.remote_prefill_endpoint_map:
                self.remote_prefill_endpoint_map[ep] = 0
        self._update_remote_prefill_endpoints()
