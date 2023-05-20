import argparse
import asyncio
import json
import time
from typing import Any, Dict
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import ray
import uvicorn

from cacheflow.outputs import RequestOutput
from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import (
    add_server_arguments, create_server_configs_from_args)
from cacheflow.server.llm_server import LLMServer
from cacheflow.server.ray_utils import initialize_cluster

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds
app = FastAPI()


class FastAPIServer:

    def __init__(self, server_use_ray: bool, *args, **kwargs) -> None:
        if server_use_ray:
            remote_server_class = ray.remote(num_cpus=0)(LLMServer)
        else:
            remote_server_class = ray.remote(num_gpus=1)(LLMServer)
        self.server = remote_server_class.remote(*args, **kwargs)

        # Request id -> request output.
        self.request_outputs: Dict[str, RequestOutput] = {}
        # Request id -> event to notify that there is new output.
        self.request_events: Dict[str, asyncio.Event] = {}
        self.is_server_running = False

    async def server_step(self):
        self.is_server_running = True
        request_outputs = await self.server.step.remote()
        self.is_server_running = False
        # Notify the waiting coroutines that there are new outputs ready.
        for request_output in request_outputs:
            request_id = request_output.request_id
            self.request_outputs[request_id] = request_output
            self.request_events[request_id].set()

    async def generate(self, request_dict: Dict[str, Any]):
        # Preprocess the request.
        arrival_time = time.time()
        prompt = request_dict.pop("prompt")
        sampling_params = SamplingParams(**request_dict)

        # Create an event to notify us that there is new output from the
        # cacheflow server.
        request_id = str(uuid.uuid4().hex[:8])
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        # Add the request into the cacheflow server's waiting queue.
        await self.server.add_request.remote(
            request_id, prompt, sampling_params, arrival_time=arrival_time)

        # The cacheflow server does not have a background loop that keeps
        # processing incoming requests. Therefore, we need to keep kicking
        # the server to process the requests.
        while True:
            # Kick the server if the server is not running.
            if not self.is_server_running:
                await self.server_step()

            # Wait for new output. The group_event will be set in server_step
            # when there is new output available for the sequence group.
            # Added a timeout to prevent deadlock.
            try:
                await asyncio.wait_for(request_event.wait(),
                                       timeout=TIMEOUT_TO_PREVENT_DEADLOCK)
            except asyncio.TimeoutError:
                continue
            # Reset the event to wait for the next output.
            request_event.clear()

            # Decode and return new outputs.
            request_output = self.request_outputs[request_id]
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

            # Once finished, release the resources of the sequence group.
            if request_output.done:
                del self.request_outputs[request_id]
                del self.request_events[request_id]
                # Kick the server if the server is not running. This is to
                # prevent that there are still requests in server's waiting
                # queue to be executed.
                if not self.is_server_running:
                    await self.server_step()
                break


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

    server = FastAPIServer(args.use_ray, *server_configs,
                           distributed_init_method, stage_devices,
                           log_stats=not args.disable_log_stats)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
