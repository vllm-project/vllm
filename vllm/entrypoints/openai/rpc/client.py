from typing import AsyncIterator, Optional, Mapping

from vllm.config import ModelConfig
from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.openai.rpc import (VLLM_GENERATE_RPC_PATH,
                                         VLLM_GET_DATA_RPC_PATH, 
                                         VLLM_IS_READY_RPC_PATH,
                                         GenerateRequest, GetDataRequest)

import zmq
import zmq.asyncio
import pickle


class RPCClient:
    def __init__(self):
        self.context = zmq.asyncio.Context()
        self.is_ready_socket = self.context.socket(zmq.PULL)
        self.is_ready_socket.connect(VLLM_GET_DATA_RPC_PATH)
        self.get_data_socket = self.context.socket(zmq.REQ)
        self.get_data_socket.connect(VLLM_GET_DATA_RPC_PATH)

    async def wait_for_server(self):
        await self.is_ready_socket.recv()


    async def get_model_config(self) -> ModelConfig:
        self.get_data_socket.send(pickle.dumps(GetDataRequest.MODEL_CONFIG))
        return pickle.loads(await self.get_data_socket.recv())


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
        socket = self.context.socket(zmq.DEALER)
        socket.connect(VLLM_GENERATE_RPC_PATH)

        # Send GenerateRequest to the RPC Server.
        await socket.send_multipart([
            pickle.dumps(
                GenerateRequest(
                    inputs=inputs,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request
                ), pickle.HIGHEST_PROTOCOL
            )
        ])

        # Stream back the results from the RPC Server.
        while True:
            message = await socket.recv()
            request_output = pickle.loads(message)

            if request_output.finished:
                break
            yield request_output

        socket.close()
        yield request_output
