import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import AsyncServerArgs
from cacheflow.server.async_llm_server import AsyncLLMServer
from cacheflow.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5 # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds
app = FastAPI()


@app.post("/generate")
async def generate_stream(request: Request) -> StreamingResponse:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = server.generate(prompt, sampling_params, request_id)

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

    async def abort_request() -> None:
        await server.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)
    return StreamingResponse(stream_results(), background=background_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser = AsyncServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = AsyncServerArgs.from_cli_args(args)
    server = AsyncLLMServer.from_server_args(server_args)

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
