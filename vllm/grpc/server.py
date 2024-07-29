from vllm.inputs.data import TextPrompt, TokensPrompt
from .pb import generate_pb2_grpc, generate_pb2
from .pb.generate_pb2 import DESCRIPTOR as _GENERATION_DESCRIPTOR
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from collections.abc import AsyncIterator
from grpc import aio
import asyncio
import time
# from grpc_reflection.v1alpha import reflection

# MODEL = "facebook/opt-125m"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_TOKENS = 150
TEMPERATURE = 0
UNIX_SOCKET = "unix:///tmp/ricky-bobby"

class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    SERVICE_NAME = _GENERATION_DESCRIPTOR.services_by_name[
        "TextGenerationService"
    ].full_name

    def __init__(self):
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=MODEL, 
                                                                      enable_chunked_prefill=True))

    async def Generate(
        self, request: generate_pb2.GenerateRequest, context
    # ) -> AsyncIterator[generate_pb2.GenerateResponse]:
    ) -> AsyncIterator:
        
        start = time.time()
        first = True
        ttft = 0
        tpots = []

        
        if len(request.prompt_inputs.prompt_token_ids) > 0:
            inputs = TokensPrompt(prompt_token_ids=request.prompt_inputs.prompt_token_ids)
        else:
            inputs = TextPrompt(prompt=request.prompt_inputs.prompt)

        results_generator = self.engine.generate(
            inputs, 
            sampling_params=SamplingParams(max_tokens=MAX_TOKENS, 
                                           temperature=TEMPERATURE,),
            request_id=request.request_id)
        
        async for request_output in results_generator:

            outputs = [ 
                generate_pb2.CompletionOutput(
                    index=output.index,
                    token_ids=output.token_ids,
                    text=output.text,
                    finish_reason=output.finish_reason)
                for output in request_output.outputs
            ]
            yield generate_pb2.GenerateResponse(outputs=outputs)

        # print(f"TTFT (backend): {ttft}")
        # print(f"TPOT (backend): {sum(tpots)/len(tpots)}")
 

async def start_grpc_server() -> aio.Server:
    server = aio.server()
    generation = TextGenerationService()
    generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(generation, server)

    # service_names = (
    #     generation.SERVICE_NAME,
    #     reflection.SERVICE_NAME,
    # )
    # reflection.enable_server_reflection(service_names, server)

    host = "0.0.0.0"
    grpc_port = 5543
    # server.add_insecure_port(f"{host}:{grpc_port}")
    server.add_insecure_port(UNIX_SOCKET)
    await server.start()
    print("ready")
    return server


async def run_grpc_server() -> None:
    server = await start_grpc_server()

    try:
        await server.wait_for_termination()

    except asyncio.CancelledError:
        print("Gracefully stopping gRPC server")  # noqa: T201
        await server.stop(30)  # TODO configurable grace
        await server.wait_for_termination()

if __name__ == "__main__":
    import uvloop

    # uvloop.run(run_grpc_server())
    asyncio.run(run_grpc_server())
