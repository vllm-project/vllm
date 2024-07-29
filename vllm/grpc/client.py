from vllm import AsyncLLMEngine
from vllm.grpc.pb import generate_pb2
from typing import AsyncIterator, List, Optional, Mapping

from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.outputs import CompletionOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
from dataclasses import dataclass

import time
import zmq
import zmq.asyncio
import pickle

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

@dataclass
class RCPRequest:
    inputs: PromptInputs
    sampling_params: SamplingParams
    request_id: str


class RPCClient(AsyncLLMEngine):
    def __init__(self):
        self.engine_use_ray = False
        self.worker_use_ray = False
        self.log_requests = False
        self.engine = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

        self.context = zmq.asyncio.Context()
        

    @property
    def is_running(self) -> bool:
        return True
    
    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False
    
    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> "PreTrainedTokenizer":
        # TODO: what to return :/
        return self.tokenizer
    
    def start_background_loop(self):
        # TODO something lol
        pass

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncIterator[RequestOutput]:
        socket = self.context.socket(zmq.DEALER)
        socket.connect('tcp://localhost:5570')

        # socket.send_multipart([
        #     pickle.dumps(
        #         RCPRequest(
        #             inputs=inputs,
        #             sampling_params=sampling_params,
        #             request_id=request_id
        #         ), pickle.HIGHEST_PROTOCOL
        #     )
        # ])
        prompt: str = inputs.get('prompt', "")
        prompt_token_ids: List[int] = inputs.get('prompt_token_ids', [])
        proto = generate_pb2.GenerateRequest(
            prompt_inputs=generate_pb2.PromptInputs(
                prompt=prompt,
                prompt_token_ids=prompt_token_ids),
            request_id=request_id,
        )
        await socket.send_multipart([proto.SerializeToString()])

        while True:
            message = await socket.recv()
            # request_output = pickle.loads(message)
            generate_response = generate_pb2.GenerateResponse()
            generate_response.ParseFromString(message)
            
            completion_outputs = [
                CompletionOutput(
                    index=output.index,
                    text=output.text,
                    token_ids=output.token_ids,
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason=(None if output.finish_reason == "" else output.finish_reason),
                ) for output in generate_response.outputs
            ]
        
            request_output = RequestOutput(
                request_id=request_id,
                prompt_token_ids=[],
                outputs=completion_outputs,
                finished=(completion_outputs[0].finish_reason is not None),
                prompt_logprobs=None,
                prompt=prompt,
            )

            if request_output.finished:
                break

            yield request_output

        socket.close()
        yield request_output
