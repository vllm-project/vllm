import argparse
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import (
    add_server_arguments, create_server_configs_from_args)
from cacheflow.server.async_llm_server import AsyncLLMServer
from cacheflow.server.ray_utils import initialize_cluster

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds
app = FastAPI()


@app.post("/generate")
async def generate_stream(request: Request):
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    sampling_params = SamplingParams(**request_dict)
    results_generator = server.generate(prompt, sampling_params)

    async def stream_results():
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
    parser.add_argument("--port", type=int, default=10002)
    parser = add_server_arguments(parser)
    args = parser.parse_args()

    server_configs = create_server_configs_from_args(args)
    parallel_config = server_configs[2]
    distributed_init_method, stage_devices = initialize_cluster(parallel_config)

    server = AsyncLLMServer(
        args.use_ray, *server_configs, distributed_init_method, stage_devices)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
