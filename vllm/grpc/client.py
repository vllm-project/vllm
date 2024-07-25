from vllm import AsyncLLMEngine
import grpc
from .pb import generate_pb2_grpc, generate_pb2
from typing import AsyncIterator, Optional, Mapping

from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.outputs import CompletionOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams



class TextGenerationClient(AsyncLLMEngine):
    def __init__(self):
        channel = grpc.insecure_channel("localhost:5543")
        self.stub = generate_pb2_grpc.TextGenerationServiceStub(channel)
    
    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncIterator[RequestOutput]:
        
        generate_stream = self.stub.Generate(
            generate_pb2.GenerateRequest(
                prompt_inputs=generate_pb2.PromptInputs(prompt=inputs.prompt),
                request_id=request_id,
            )
        )

        async for generate_response in generate_stream:
            completion_outputs = [
                CompletionOutput(
                    index=output.index,
                    text=output.text,
                    token_ids=output.token_ids,
                    cumulative_logprob=0.0,
                ) for output in generate_response.outputs
            ]

            yield RequestOutput(
                request_id=request_id,
                prompt_token_ids=[],
                outputs=completion_outputs
            )