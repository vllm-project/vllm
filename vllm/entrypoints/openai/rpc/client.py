import pickle
from typing import AsyncIterator, Mapping, Optional

import zmq
import zmq.asyncio

from vllm.config import DecodingConfig, ModelConfig
from vllm.entrypoints.openai.rpc import (VLLM_GENERATE_RPC_PATH,
                                         VLLM_GET_DATA_RPC_PATH,
                                         VLLM_IS_READY_RPC_PATH,
                                         GenerateRequest, GetDataRequest)
from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams


class RPCClient:

    # TODO: check if opening all these sockets is an antipattern?
    def __init__(self, tokenizer):
        self.context = zmq.asyncio.Context()

        # TODO: do the tokenizer properly.
        self.tokenizer = tokenizer
        self.decoding_config = DecodingConfig()

        # Socket to check if the RPC server is ready.
        self.is_ready_socket = self.context.socket(zmq.constants.REP)
        self.is_ready_socket.connect(VLLM_IS_READY_RPC_PATH)

        # Socket to query data (e.g. get_model_config)
        self.get_data_socket = self.context.socket(zmq.constants.REQ)
        self.get_data_socket.connect(VLLM_GET_DATA_RPC_PATH)

    async def wait_for_server(self):
        await self.is_ready_socket.recv()

    def close(self):
        """Destroy the zmq context and close all sockets"""
        self.context.destroy()

    async def get_model_config(self) -> ModelConfig:
        self.get_data_socket.send(pickle.dumps(GetDataRequest.MODEL_CONFIG))
        model_config = await self.get_data_socket.recv()
        return pickle.loads(model_config)

    async def get_tokenizer(self, lora_request: LoRARequest):
        # TODO: handle this via get data? - or avoid doing via RPC
        return self.tokenizer

    async def get_decoding_config(self):
        # TODO: handle this via get data? -  or avoid doing via RPC
        return self.decoding_config

    async def abort(self, request_id: str):
        # TODO: actually handle this with a new socket.
        pass

    async def is_tracing_enabled(self):
        return False

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncIterator[RequestOutput]:

        # Connect to RPC socket for Request-Reply pattern,
        # Note that we use DEALER to enable asynchronous communication
        # to enable streaming.
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(VLLM_GENERATE_RPC_PATH)

        # Send GenerateRequest to the RPC Server.
        await socket.send_multipart([
            pickle.dumps(
                GenerateRequest(inputs=inputs,
                                sampling_params=sampling_params,
                                request_id=request_id,
                                lora_request=lora_request,
                                trace_headers=trace_headers,
                                prompt_adapter_request=prompt_adapter_request),
                pickle.HIGHEST_PROTOCOL)
        ])

        # Stream back the results from the RPC Server.
        while True:
            message = await socket.recv()
            request_output = pickle.loads(message)

            if isinstance(request_output, Exception):
                socket.close()
                raise request_output

            if request_output.finished:
                break
            yield request_output

        yield request_output
        socket.close()
