import asyncio
import copy
import pickle
from typing import (AsyncGenerator, Mapping, Optional, Union)
import threading
import socket
import uuid

import cloudpickle

from vllm import PoolingParams
from vllm.config import EngineConfig
from vllm.engine.async_llm_engine import (
    build_guided_decoding_logits_processor_async)
from vllm.engine.multiprocessing import (ENGINE_DEAD_ERROR, 
                                         RPCError, RPCProcessRequest)
# yapf: enable
from vllm.envs import VLLM_DISTRIBUTED_KV_ROLE
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.rabbitmq.rabbitmq import RabbitMQ

logger = init_logger(__name__)

GLOBAL_DECODE_REQ_QUEUE_NAME = "global_decode_req"

def encode_req_id(req_id, local_decode_res_mq_name):
    return req_id + "/" + local_decode_res_mq_name

def decode_req_id(req_id):
    req_infos = req_id.split("/")
    assert len(req_infos) == 2
    return req_infos

class PDMQLLMEngineClient(MQLLMEngineClient):
    IS_KV_PRODUCER: bool = (VLLM_DISTRIBUTED_KV_ROLE in ["producer"])
    IS_KV_CONSUMER: bool = (VLLM_DISTRIBUTED_KV_ROLE in ["consumer"])

    def __init__(self, ipc_path: str,
                 engine_config: EngineConfig,
                 mq_addr: str,
                 mq_port: int):
        super().__init__(ipc_path, engine_config)

        if self.IS_KV_PRODUCER:
            self.machine_name = socket.gethostname()
            self.global_decode_req_mq: RabbitMQ = RabbitMQ(mq_addr, mq_port, GLOBAL_DECODE_REQ_QUEUE_NAME)

            # TODO(Lu Changqi): Should the name of this message queue remain the same after a restart?
            self.local_decode_res_mq_name = self.machine_name + "-" + str(uuid.uuid4())
            self.local_decode_res_mq: RabbitMQ = RabbitMQ(mq_addr, mq_port, self.local_decode_res_mq_name)

            threading.Thread(target=self.run_output_handler_loop_from_consumer).start()


    def run_output_handler_loop_from_consumer(self):
        try:
            while True:
                frame, body = self.local_decode_res_mq.pull()
                if frame is None:
                    logger.debug("Waiting for output from MQLLMEngine.")
                    continue
                
                request_outputs = pickle.loads(body)
                is_error = isinstance(request_outputs,
                                        (BaseException, RPCError))
                if is_error:
                    if isinstance(request_outputs, RPCError):
                        rpc_error: RPCError = request_outputs
                        request_id = decode_req_id(rpc_error.request_id)[0]
                        exception = rpc_error.exception
                        is_engine_errored = rpc_error.is_engine_errored
                    else:
                        error: BaseException = request_outputs
                        logger.error(
                            "Received Exception %s rather than RPCError from "
                            "MPLLMEngine. This should never happen.", error)
                        request_id = None
                        exception = error
                        is_engine_errored = True

                    if is_engine_errored and not self._errored_with:
                        self._errored_with = exception

                    if request_id is None:
                        for queue_i in tuple(self.output_queues.values()):
                            queue_i.put_nowait(exception)
                    else:
                        queue = self.output_queues.get(request_id)
                        if queue is not None:
                            queue.put_nowait(exception)
                else:
                    for request_output in request_outputs:
                        request_id = decode_req_id(request_output.request_id)[0]
                        queue = self.output_queues.get(request_id)
                        if queue is not None:
                            queue.put_nowait(request_output)
                            
                self.local_decode_res_mq.ack(frame)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Exiting...")
        finally:
            self.close()

    async def consumer_generate(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:

        if self._errored_with is not None:
            raise ENGINE_DEAD_ERROR(self._errored_with)
        
        if isinstance(params, SamplingParams) and \
            params.guided_decoding is not None:
            params = await \
                build_guided_decoding_logits_processor_async(
                    sampling_params=params,
                    tokenizer=await self.get_tokenizer(lora_request),
                    default_guided_backend=self.decoding_config.guided_decoding_backend
                )

        queue: asyncio.Queue[Union[RequestOutput,
                                   BaseException]] = asyncio.Queue()
        self.output_queues[request_id] = queue

        try:
            if isinstance(params, SamplingParams) and params.logits_processors:
                params = copy.copy(params)
                logits_processors = params.logits_processors
                params.logits_processors = None
                lp_bytes = cloudpickle.dumps(logits_processors)
            else:
                lp_bytes = None

            request_bytes = pickle.dumps(
                RPCProcessRequest(
                    prompt=prompt,
                    params=params,
                    request_id=encode_req_id(request_id, self.local_decode_res_mq_name),
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority,
                ))

            parts = (request_bytes,
                     lp_bytes) if lp_bytes else (request_bytes, )
            message_body = pickle.dumps(parts)
            self.global_decode_req_mq.push(message_body)

            finished = False
            try:
                while not finished:
                    request_output = await queue.get()

                    if isinstance(request_output, BaseException):
                        raise request_output

                    finished = request_output.finished
                    yield request_output
            finally:
                if not finished and not self.errored:
                    await self.abort(request_id)
        finally:
            self.output_queues.pop(request_id)

    async def generate(
        self,
        prompt: Optional[PromptType] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        *,
        inputs: Optional[PromptType] = None  # DEPRECATED
    ) -> AsyncGenerator[RequestOutput, None]:
        if inputs is not None:
            prompt = inputs
        assert (prompt is not None and sampling_params is not None
                and request_id is not None)

        prefill_param = copy.deepcopy(sampling_params)
        prefill_param.max_tokens = 1

        # prefill generate
        async for _ in super().generate(
            prompt,
            prefill_param,
            request_id,
            lora_request,
            trace_headers,
            prompt_adapter_request
        ):
            continue

        # decode generate
        async for decode_res in self.consumer_generate(
            prompt,
            sampling_params,
            request_id,
            lora_request,
            trace_headers,
            prompt_adapter_request,
            priority,
        ):
            yield decode_res


    def close(self):
        self.context.destroy(linger=0)

        if self.health_loop is not None:
            self.health_loop.cancel()
        self.output_loop.cancel()

        if self.IS_KV_PRODUCER:
            self.global_decode_req_mq.close()
            self.local_decode_res_mq.close()