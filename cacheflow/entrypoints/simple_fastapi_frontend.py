import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import ServerArgs
from cacheflow.server.async_llm_server import AsyncLLMServer
from cacheflow.server.ray_utils import initialize_cluster

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds
app = FastAPI()


@app.post("/generate")
async def generate_stream(request: Request) -> StreamingResponse:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    sampling_params = SamplingParams(**request_dict)
    results_generator = server.generate(prompt, sampling_params)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text
                for output in request_output.outputs
            ]
            ret = {
                "text": text_outputs,
                "error": 0,
            }
            yield (json.dumps(ret) + "\0").encode("utf-8")

    return StreamingResponse(stream_results())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser = ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    server = AsyncLLMServer.from_server_args(server_args)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
