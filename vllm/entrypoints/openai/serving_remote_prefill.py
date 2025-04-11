# SPDX-License-Identifier: Apache-2.0

import asyncio
import aiohttp
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Optional, Union, cast
import msgspec
import threading

import jinja2
from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.remote_prefill import (RemotePrefillParams,
                                 RemotePrefillRequest)

# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (NixlMetadataRequest,
                                              NixlMetadataResponse,
                                              RemotePrefillGenerateRequest)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import OpenAIServing, clamp_prompt_logprobs
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import merge_async_iterators
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
        
        self.remote_prefill_endpoints = ["127.0.0.1:8090"]
        
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

    async def remote_prefill(self, request: RemotePrefillGenerateRequest):
        request = msgspec.json.decode(
            request.content.encode(encoding="utf-8"),
            type=RemotePrefillRequest,
        )

        await self._request_queue.put(request)

    def get_remote_prefill_request_callback(self):
        # TODO: integrate prefill_queue to dynamo endpoint
        async def callback(request: RemotePrefillRequest):
            remote_prefill_url = f"http://{self.remote_prefill_endpoints[0]}/remote_prefill"
            request = RemotePrefillGenerateRequest(
                content=str(msgspec.json.encode(request), encoding="utf-8"),
            )
            async with aiohttp.ClientSession() as session:
                await session.post(remote_prefill_url, json=request.model_dump())

        return callback

    def get_remote_prefill_params(self, request: Request):
        remote_prefill_params = RemotePrefillParams(
                is_remote_prefill=True,
                remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
        )
        return remote_prefill_params