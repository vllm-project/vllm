from .pb import generate_pb2_grpc, generate_pb2
from .pb.generate_pb2 import DESCRIPTOR as _GENERATION_DESCRIPTOR
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from collections.abc import AsyncIterator
from grpc import aio
import asyncio
from grpc_reflection.v1alpha import reflection

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_TOKENS = 200
TEMPERATURE = 0


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    SERVICE_NAME = _GENERATION_DESCRIPTOR.services_by_name[
        "TextGenerationService"
    ].full_name

    def __init__(self):
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=MODEL, enforce_eager=True))

    async def Generate(
        self, request: generate_pb2.GenerateRequest, context
    ) -> AsyncIterator[generate_pb2.GenerateResponse]:

        results_generator = self.engine.generate(
            request.prompt_inputs.prompt, 
            sampling_params=SamplingParams(max_tokens=MAX_TOKENS, 
                                           temperature=TEMPERATURE),
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
 

async def start_grpc_server() -> aio.Server:
    server = aio.server()
    generation = TextGenerationService()
    generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(generation, server)

    service_names = (
        generation.SERVICE_NAME,
        reflection.SERVICE_NAME,
    )

    reflection.enable_server_reflection(service_names, server)

    host = "0.0.0.0"
    grpc_port = 5543
    server.add_insecure_port(f"{host}:{grpc_port}")
    await server.start()
    print("ready")
    return server


async def run_grpc_server() -> None:
    server = await start_grpc_server()

    try:
        while True:
            await asyncio.sleep(10)

    except asyncio.CancelledError:
        print("Gracefully stopping gRPC server")  # noqa: T201
        await server.stop(30)  # TODO configurable grace
        await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(run_grpc_server())
