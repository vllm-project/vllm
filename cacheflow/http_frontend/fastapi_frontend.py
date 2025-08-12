import argparse
import asyncio
import time
from typing import List, Dict
import json

import ray
from transformers import AutoTokenizer
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from cacheflow.sampling_params import SamplingParams
from cacheflow.sequence import Sequence, SequenceGroup
from cacheflow.master.server import (Server, add_server_arguments,
                                     initialize_ray_cluster)
from cacheflow.worker.controller import DeviceID
from cacheflow.utils import Counter, get_gpu_memory, get_cpu_memory

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds
app = FastAPI()


class FastAPIFrontend:
    def __init__(
        self,
        model: str,
        model_path: str,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        block_size: int,
        dtype: str,
        seed: int,
        swap_space: int,
        max_num_batched_tokens: int,
        num_nodes: int,
        num_devices_per_node: int,
        distributed_init_method: str,
        all_stage_devices: List[List[DeviceID]],
    ):
        self.block_size = block_size

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.seq_group_counter = Counter()
        self.seq_counter = Counter()
        remote_server_class = ray.remote(num_cpus=0)(Server)
        self.server = remote_server_class.remote(
            model=model,
            model_path=model_path,
            use_dummy_weights=False,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            block_size=block_size,
            dtype=dtype,
            seed=seed,
            swap_space=swap_space,
            max_num_batched_tokens=max_num_batched_tokens,
            num_nodes=num_nodes,
            num_devices_per_node=num_devices_per_node,
            distributed_init_method=distributed_init_method,
            all_stage_devices=all_stage_devices,
            gpu_memory=get_gpu_memory(),
            cpu_memory=get_cpu_memory(),
        )

        self.running_seq_groups: Dict[int, SequenceGroup] = {}
        self.sequence_group_events: Dict[int, asyncio.Event] = {}
        self.is_server_running = False

    async def server_step(self):
        self.is_server_running = True
        updated_seq_groups = await self.server.step.remote()
        self.is_server_running = False
        # Notify the waiting coroutines that there new outputs ready.
        for seq_group in updated_seq_groups:
            group_id = seq_group.group_id
            self.running_seq_groups[group_id] = seq_group
            self.sequence_group_events[group_id].set()

    async def generate(self, request_dict: Dict):
        # Preprocess the request.
        prompt = request_dict["prompt"]
        sampling_params = SamplingParams.from_dict(request_dict)
        sampling_params.stop_token_ids.add(self.tokenizer.eos_token_id)
        token_ids = self.tokenizer.encode(prompt)
        seqs: List[Sequence] = []
        for _ in range(sampling_params.n):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, token_ids, block_size=self.block_size)
            seqs.append(seq)

        arrival_time = time.time()
        group_id = next(self.seq_group_counter)
        seq_group = SequenceGroup(group_id, seqs, arrival_time)
        # Create an event to notify us that there is new output from the
        # cacheflow server.
        group_event = asyncio.Event()
        self.running_seq_groups[group_id] = seq_group
        self.sequence_group_events[group_id] = group_event
        # Add the request into the cacheflow server's waiting queue.
        await self.server.add_sequence_groups.remote([(seq_group, sampling_params)])
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
            await asyncio.wait_for(group_event.wait(), timeout=TIMEOUT_TO_PREVENT_DEADLOCK)
            # Reset the event to wait for the next output.
            group_event.clear()
            # Decode and return new outputs
            seq_group = self.running_seq_groups[group_id]
            all_outputs = []
            for seq in seq_group.seqs:
                token_ids = seq.get_token_ids()
                output = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                all_outputs.append(output)
            ret = {
                "text": all_outputs,
                "error": 0,
            }
            yield (json.dumps(ret) + "\0").encode("utf-8")

            # Once finished, release the resources of the sequence group.
            if seq_group.is_finished():
                del self.running_seq_groups[group_id]
                del self.sequence_group_events[group_id]
                # Kick the server if the server is not running. This is to
                # prevent that there are still requests in server's waiting
                # queue to be executed.
                if not self.is_server_running:
                    await self.server_step()
                break


@app.post("/generate")
async def generate_stream(request: Request):
    request_dict = await request.json()
    return StreamingResponse(frontend.generate(request_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10002)
    parser = add_server_arguments(parser)
    args = parser.parse_args()

    # TODO(zhuohan): Support pipeline parallelism.
    assert args.pipeline_parallel_size == 1, (
        'Pipeline parallelism is not supported yet.')

    (num_nodes, num_devices_per_node, distributed_init_method,
    all_stage_devices) = (
        initialize_ray_cluster(
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size))

    frontend = FastAPIFrontend(
        model=args.model,
        model_path=args.model_path,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        dtype=args.dtype,
        seed=args.seed,
        swap_space=args.swap_space,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_nodes=num_nodes,
        num_devices_per_node=num_devices_per_node,
        distributed_init_method=distributed_init_method,
        all_stage_devices=all_stage_devices,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
