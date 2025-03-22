# SPDX-License-Identifier: Apache-2.0

import asyncio
import msgspec
import os
from collections.abc import AsyncGenerator
from typing import Dict, List, Mapping, Optional

import zmq
import zmq.asyncio

from vllm import SamplingParams
from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.entrypoints.disaggregated.types import PDRequest, PDResponse
from vllm.inputs.data import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import (PoolingRequestOutput, RequestOutput,
                          CompletionOutput)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.utils import Device

logger = init_logger(__name__)

DEFAULT_MAX_TOKENS = 32000

class PDEngine:
    """
    PDEngine:
        Equiavlent of AsyncLLM for P/D. Assumes there is
        a Prefill and Decode service already running.

        * TODO: actually handle errors and failure.
        * TODO: support more than just text input.
        * TODO: move under vllm/v1/engine one past prototype.
    """

    def __init__(
        self,
        prefill_addr: str,
        decode_addr: str,
        connector_addr: str,
        model_name: str
    ):
        # Request queues.
        self.queues: Dict[str, asyncio.Queue] = {}

        # Serialization encoder.
        self.encoder = msgspec.msgpack.Encoder()

        # ZMQ communication.
        self.ctx = zmq.asyncio.Context()
        self.to_decode = self.ctx.socket(zmq.constants.PUSH)
        self.to_decode.bind(f"{decode_addr}")
        self.to_prefill = self.ctx.socket(zmq.constants.PUSH)
        self.to_prefill.bind(f"{prefill_addr}")
        self.connector_addr = connector_addr
        self.decode_addr = decode_addr
        self.prefill_addr = prefill_addr
        
        # Background loops (started on first generate()).
        self.output_handler: Optional[asyncio.Task] = None
        self.log_running: Optional[asyncio.Task] = None

        # Dummy: needed for EngineClient Protocol.
        # TODO: refactor EngineClient to avoid needing this.
        self.model_config = ModelConfig(
            model=model_name,
            tokenizer=model_name,
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="auto",
            task="generate",
            seed=42
        )

        # Dummy: needed for EngineClient Protocol.
        # TODO: refactor EngineClient to avoid needing this.
        init_kwargs = dict(
            tokenizer_id=self.model_config.tokenizer,
            enable_lora=False,
            max_num_seqs=1024,
            max_loras=0,
            max_input_length=None,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision,
            truncation_side=self.model_config.truncation_side)
        self.tokenizer = TokenizerGroup(**init_kwargs)

    def shutdown(self):
        if (ctx := self.ctx) is not None:
            ctx.destroy(linger=0)
        if (task := self.log_running) is not None:
            task.cancel()
        if (task := self.output_handler) is not None:
            task.cancel()

        ipc_paths = [
            self.connector_addr, self.decode_addr, self.prefill_addr
        ]
        for path in ipc_paths:
            socket_path = path.replace("ipc://", "")
            if os.path.exists(socket_path):
                os.remove(socket_path)

    async def _run_log_running(self):
        logger.info("Running requests: %d", len(self.queues))
        await asyncio.sleep(10.)

    async def _run_output_handler(self):
        """
        Pull responses from Decode + Prefill engines and
        distribute back to the generate() tasks.
        """
        decoder = msgspec.msgpack.Decoder(PDResponse)
        
        socket: Optional[zmq.asyncio.Socket] = None
        try:
            socket = self.ctx.socket(zmq.constants.PULL)
            socket.bind(self.connector_addr)

            while True:
                reponse_bytes = await socket.recv()
                response = decoder.decode(reponse_bytes)
                logger.debug("Got Response: %s", response.request_id)
                self.queues[response.request_id].put_nowait(response)
        except:
            # TODO: actually handle failure and shutdown.
            raise 
        finally:
            if socket is not None:
                socket.close(linger=0)
    
    async def _prefill(
        self,
        request: PDRequest,
        q: asyncio.Queue[PDResponse],
    ):
        # Send request to the prefill instance.
        req_bytes = self.encoder.encode(request)
        await self.to_prefill.send(req_bytes, copy=False)

        # Wait for the prefill to be done.
        response = await q.get()
        assert response.request_id == request.request_id
        if not response.success:
            # TODO: actual error handling and shutdown.
            raise Exception("Failed Prefill Request.")
    
    async def _decode(
        self,
        request: PDRequest,
        q: asyncio.Queue[PDResponse],
    ) -> AsyncGenerator[PDRequest]:

        # Send request to the decode instance.
        req_bytes = self.encoder.encode(request)
        await self.to_decode.send(req_bytes, copy=False)

        # Iterate response queue and yield each response to caller..
        finished = False
        while not finished:
            response = await q.get()
            logger.debug(f"{response}")
            if not response.success:
                # TODO: actual error handling and shutdown.
                raise Exception("Failed Decode Request.")
            finished = response.finish_reason is not None
            yield response
    
    def _to_request_output(
        self,
        pd_response: PDResponse,
        prompt_token_ids: List[int],
    ) -> RequestOutput:
        finished = pd_response.finish_reason is not None
        return RequestOutput(
            request_id=pd_response.request_id,
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            outputs=[CompletionOutput(
                index=0,
                text=pd_response.text,
                token_ids=pd_response.token_ids,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=pd_response.finish_reason,
                stop_reason=pd_response.stop_reason
            )],
            finished=finished,
        )

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput]:
        # Start loops on first request.
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(self._run_output_handler())
            self.log_running = asyncio.create_task(self._run_log_running())

        # TODO: Expand to support the full matrix.
        if not "prompt_token_ids" in prompt:
            raise NotImplementedError(
                "We currently only support TokensPrompt for P/D!")
        if lora_request is not None:
            raise NotImplementedError(
                "We currently do not suppport LoRA for P/D!")
        if trace_headers is not None:
            raise NotImplementedError(
                "We currently do not suppport tracing for P/D!")
        if prompt_adapter_request is not None:
            raise NotImplementedError(
                "We currently do not suppport prompt adapter for P/D!")
        if priority != 0:
            raise NotImplementedError(
                "We currently do not support priority for P/D!")
        if request_id in self.queues:
            raise ValueError(f"Found duplicate request_id: {request_id}!")
        
        # Queue to gather output from output_handler.
        q: asyncio.Queue[PDResponse] = asyncio.Queue()
        self.queues[request_id] = q

        # (1) Perform the Prefill.
        original_max_tokens = sampling_params.max_tokens
        prompt_token_ids = prompt["prompt_token_ids"]
        request = PDRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params)
        request.sampling_params.max_tokens = 1
        logger.debug("Sending Prefill: %s", request.request_id)
        pd_response = await self._prefill(request, q)

        # (2) Perform the Decodes.
        logger.debug("Sending Decode: %s", request.request_id)
        request.sampling_params.max_tokens = original_max_tokens
        async for pd_response in self._decode(request, q):
            logger.debug("Got Decode: %s", request.request_id)
            yield self._to_request_output(pd_response, prompt_token_ids)

    async def beam_search(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:
        raise NotImplementedError

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        raise NotImplementedError

    async def abort(self, request_id: str) -> None:
        raise NotImplementedError

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_decoding_config(self) -> DecodingConfig:
        raise NotImplementedError

    async def get_input_preprocessor(self) -> InputPreprocessor:
        raise NotImplementedError

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        if lora_request is not None:
            raise NotImplementedError(
                "LoRA is not yet supported in the PDEngine.")
        return self.tokenizer.get_lora_tokenizer(None)

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[List[SamplerOutput]] = None,
    ) -> None:
        pass

    async def check_health(self) -> None:
        pass

    async def start_profile(self) -> None:
        raise NotImplementedError

    async def stop_profile(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        raise NotImplementedError

    async def sleep(self, level: int = 1) -> None:
        raise NotImplementedError

    async def wake_up(self) -> None:
        raise NotImplementedError

    async def is_sleeping(self) -> bool:
        False

    async def add_lora(self, lora_request: LoRARequest) -> None:
        raise NotImplementedError

    @property
    def errored(self) -> bool:
        return False
