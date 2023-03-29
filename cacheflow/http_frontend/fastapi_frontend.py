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
        max_batch_size: int,
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
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            block_size=block_size,
            dtype=dtype,
            seed=seed,
            swap_space=swap_space,
            max_batch_size=max_batch_size,
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
        for seq_group in updated_seq_groups:
            group_id = seq_group.group_id
            self.running_seq_groups[group_id] = seq_group
            self.sequence_group_events[group_id].set()

    async def generate(self, request_dict: Dict):
        prompt = request_dict["prompt"]
        sampling_params = SamplingParams.from_dict(request_dict)
        sampling_params.stop_token_ids.add(self.tokenizer.eos_token_id)
        token_ids = self.tokenizer.encode(prompt)
        seqs: List[Sequence] = []
        for _ in range(sampling_params.n):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, token_ids, block_size=self.block_size)
            seqs.append(seq)

        group_id = next(self.seq_group_counter)
        seq_group = SequenceGroup(group_id, seqs)
        group_event = asyncio.Event()
        self.sequence_group_events[group_id] = group_event
        await self.server.add_sequence_groups.remote([(seq_group, sampling_params)])
        while True:
            if not self.is_server_running:
                await self.server_step()
            # Wait for new output. Add a 1s timeout to prevent dead lock.
            await asyncio.wait_for(group_event.wait(), timeout=1)
            group_event.clear()
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
            if seq_group.is_finished():
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
        max_batch_size=args.max_batch_size,
        num_nodes=num_nodes,
        num_devices_per_node=num_devices_per_node,
        distributed_init_method=distributed_init_method,
        all_stage_devices=all_stage_devices,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
