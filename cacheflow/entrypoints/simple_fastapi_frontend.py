import argparse
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from cacheflow.server.arg_utils import (
    add_server_arguments, create_server_configs_from_args)
from cacheflow.server.async_llm_server import AsyncLLMServer
from cacheflow.server.ray_utils import initialize_cluster

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds
app = FastAPI()


@app.post("/generate")
async def generate_stream(request: Request):
    request_dict = await request.json()
    return StreamingResponse(server.generate(request_dict))


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
