# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from collections.abc import AsyncGenerator, Mapping
from typing import Optional, Union

import msgspec
import zmq
import zmq.asyncio

from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.disaggregated.protocol import (PDGenerationRequest,
                                         PDGenerationResponse, PDRequestType,
                                         PDResponseType)
from vllm.engine.protocol import EngineClient
from vllm.inputs.data import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.utils import Device

logger = init_logger(__name__)

DEFAULT_MAX_TOKENS = 32000


class PDController(EngineClient):
    """
    Controller that schedules work on the PDWorkers.

    Conforms for the EngineClient protocol so it can
    be wrapped with the OpenAI Server.

    Two Phases:
    * Send request to prefill worker, await ack.
    * Send request to decode worker, stream responses.

    KVSync happens directly between Engines,
    handled by vLLM KVCacheTransfer.

            [ OpenAI Server ]
                    |
             [ PDController ]
                    |
                 [ zmq ]
                    |
    [ PDWorker ]          [ PDWorker ]
         |                     |
    [  Engine  ]  <-kv->  [  Engine  ]

    After PR #12957, we will support xPyD, so we will
    also need to implement a scheduler and service 
    discovery for the workers.

    This PDController may be implemented as a K8s
    controller. This is intended to be a prototype.

    * TODO: better error handling
    * TODO: support logprobs, multimodal, etc.
    """

    def __init__(self, prefill_addr: str, decode_addr: str,
                 controller_addr: str, model_name: str):
        # Request queues.
        self.queues: dict[str, asyncio.Queue] = {}

        # Serialization encoder.
        self.encoder = msgspec.msgpack.Encoder()

        # ZMQ communication.
        # TODO: once https://github.com/vllm-project/vllm/pull/12957
        # lands, do service discovery to scale out workers.
        self.ctx = zmq.asyncio.Context()
        self.to_prefill = self.ctx.socket(zmq.constants.PUSH)
        self.to_prefill.connect(prefill_addr)
        self.to_decode = self.ctx.socket(zmq.constants.PUSH)
        self.to_decode.connect(decode_addr)
        self.controller_addr = controller_addr
        self.ipc_paths = [prefill_addr, decode_addr, controller_addr]

        # Background loops (started on first generate()).
        self.output_handler: Optional[asyncio.Task] = None
        self.log_running: Optional[asyncio.Task] = None

        # Dummy: needed for EngineClient Protocol.
        # TODO: refactor OAI Server to avoid needing this.
        self.model_config = ModelConfig(model=model_name,
                                        tokenizer=model_name,
                                        tokenizer_mode="auto",
                                        trust_remote_code=False,
                                        dtype="auto",
                                        task="generate",
                                        seed=42)

        # Dummy: needed for EngineClient Protocol.
        # TODO: refactor OAI Server to avoid needing this.
        self.tokenizer = TokenizerGroup(
            **dict(tokenizer_id=self.model_config.tokenizer,
                   enable_lora=False,
                   max_num_seqs=1024,
                   max_loras=0,
                   max_input_length=None,
                   tokenizer_mode=self.model_config.tokenizer_mode,
                   trust_remote_code=self.model_config.trust_remote_code,
                   revision=self.model_config.tokenizer_revision,
                   truncation_side=self.model_config.truncation_side))

    def shutdown(self):
        if (ctx := self.ctx) is not None:
            ctx.destroy(linger=0)
        if (task := self.log_running) is not None:
            task.cancel()
        if (task := self.output_handler) is not None:
            task.cancel()

        for path in self.ipc_paths:
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
        decoder = msgspec.msgpack.Decoder(PDGenerationResponse)

        socket: Optional[zmq.asyncio.Socket] = None
        try:
            socket = self.ctx.socket(zmq.constants.PULL)
            socket.bind(self.controller_addr)

            while True:
                res_type, res_data = await socket.recv_multipart()
                if res_type == PDResponseType.FAILURE:
                    raise Exception("Failure Response from PDWorker.")
                elif res_type == PDResponseType.GENERATION:
                    response = decoder.decode(res_data)
                    logger.debug("Got Response: %s", response.request_id)
                    self.queues[response.request_id].put_nowait(response)
                else:
                    raise Exception("Unknown response type.")
        except Exception as e:
            # TODO: distinguish between fatal and non-fatal errors.
            for q in self.queues.values():
                q.put_nowait(e)
            raise e
        finally:
            if socket is not None:
                socket.close(linger=0)

    async def _run_prefill(
        self,
        request: PDGenerationRequest,
        q: asyncio.Queue[Union[Exception, PDGenerationResponse]],
    ):
        # Send request to the prefill instance.
        req_bytes = self.encoder.encode(request)
        msg = (PDRequestType.GENERATION, req_bytes)
        await self.to_prefill.send_multipart(msg, copy=False)

        # Await completion of the prefill.
        response = await q.get()
        if isinstance(response, Exception):
            raise response
        logger.debug("Prefill Response: %s", request.request_id)

    async def _run_decode(
        self,
        request: PDGenerationRequest,
        q: asyncio.Queue[Union[Exception, PDGenerationResponse]],
    ) -> AsyncGenerator[PDGenerationResponse]:
        # Send request to the decode instance.
        req_bytes = self.encoder.encode(request)
        msg = (PDRequestType.GENERATION, req_bytes)
        await self.to_decode.send_multipart(msg, copy=False)

        # Iterate response queue and yield each response to caller.
        finished = False
        while not finished:
            response = await q.get()
            if isinstance(response, Exception):
                raise response
            logger.debug("Decode Response: %s", request.request_id)
            finished = response.finish_reason is not None
            yield response

    def _to_request_output(
        self,
        response: PDGenerationResponse,
        prompt_token_ids: list[int],
    ) -> RequestOutput:
        finished = response.finish_reason is not None
        return RequestOutput(
            request_id=response.request_id,
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(index=0,
                                 text=response.text,
                                 token_ids=response.token_ids,
                                 cumulative_logprob=None,
                                 logprobs=None,
                                 finish_reason=response.finish_reason,
                                 stop_reason=response.stop_reason)
            ],
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
            self.output_handler = asyncio.create_task(
                self._run_output_handler())
            self.log_running = asyncio.create_task(self._run_log_running())

        # TODO: Expand to support the full matrix.
        if "prompt_token_ids" not in prompt:
            raise NotImplementedError(
                "We currently only support TokensPrompt for P/D!")
        if lora_request is not None:
            raise NotImplementedError(
                "We currently do not support LoRA for P/D!")
        if trace_headers is not None:
            raise NotImplementedError(
                "We currently do not support tracing for P/D!")
        if prompt_adapter_request is not None:
            raise NotImplementedError(
                "We currently do not support prompt adapter for P/D!")
        if priority != 0:
            raise NotImplementedError(
                "We currently do not support priority for P/D!")
        if request_id in self.queues:
            raise ValueError(f"Found duplicate request_id: {request_id}!")

        # Queue to gather output from output_handler.
        q = asyncio.Queue()
        self.queues[request_id] = q

        # (1) Perform the Prefill.
        original_max_tokens = sampling_params.max_tokens
        request = PDGenerationRequest(
            request_id=request_id,
            prompt_token_ids=prompt["prompt_token_ids"],
            sampling_params=sampling_params)
        request.sampling_params.max_tokens = 1
        pd_response = await self._run_prefill(request, q)

        # (2) Perform the Decodes.
        request.sampling_params.max_tokens = original_max_tokens
        async for pd_response in self._run_decode(request, q):
            yield self._to_request_output(pd_response,
                                          prompt["prompt_token_ids"])

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
        model_output: Optional[list[SamplerOutput]] = None,
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
        return False

    async def add_lora(self, lora_request: LoRARequest) -> None:
        raise NotImplementedError

    @property
    def errored(self) -> bool:
        return False

    def dead_error(self) -> Exception:
        return Exception("PDController has failed.")

    def is_running(self) -> bool:
        return True

    def is_stopped(self) -> bool:
        return False
