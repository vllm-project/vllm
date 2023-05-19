import argparse
import asyncio
import json
import time
from typing import List, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import ray
import uvicorn

from cacheflow.core.server import (Server, add_server_arguments,
                                   process_server_arguments,
                                   initialize_cluster)
from cacheflow.frontend.utils import get_tokenizer
from cacheflow.sampling_params import SamplingParams
from cacheflow.sequence import Sequence, SequenceGroup
from cacheflow.utils import Counter
from cacheflow.worker.controller import DeviceID

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds
app = FastAPI()


class FastAPIServer:
    def __init__(
        self,
        model: str,
        cache_dir: Optional[str],
        use_np_cache: bool,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        block_size: int,
        dtype: str,
        seed: int,
        swap_space: int,
        gpu_memory_utilization: float,
        max_num_batched_tokens: int,
        max_num_sequences: int,
        num_nodes: int,
        num_devices_per_node: int,
        distributed_init_method: str,
        all_stage_devices: List[List[DeviceID]],
        server_use_ray: bool,
        log_stats: bool,
    ):
        self.block_size = block_size

        self.tokenizer = get_tokenizer(model)
        self.seq_group_counter = Counter()
        self.seq_counter = Counter()
        if server_use_ray:
            remote_server_class = ray.remote(num_cpus=0)(Server)
        else:
            remote_server_class = ray.remote(num_gpus=1)(Server)
        self.server = remote_server_class.remote(
            model=model,
            cache_dir=cache_dir,
            use_dummy_weights=False,
            use_np_cache=use_np_cache,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            block_size=block_size,
            dtype=dtype,
            seed=seed,
            swap_space=swap_space,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_sequences=max_num_sequences,
            num_nodes=num_nodes,
            num_devices_per_node=num_devices_per_node,
            distributed_init_method=distributed_init_method,
            all_stage_devices=all_stage_devices,
            use_ray=server_use_ray,
            log_stats=log_stats,
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
        prompt = request_dict.pop("prompt")
        sampling_params = SamplingParams(**request_dict)
        sampling_params.stop_token_ids.add(self.tokenizer.eos_token_id)
        token_ids = self.tokenizer.encode(prompt)
        seqs: List[Sequence] = []
        for _ in range(sampling_params.n):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, token_ids, block_size=self.block_size)
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
    return StreamingResponse(server.generate(request_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10002)
    parser = add_server_arguments(parser)
    args = parser.parse_args()
    args = process_server_arguments(args)

    # TODO(zhuohan): Support pipeline parallelism.
    assert args.pipeline_parallel_size == 1, (
        'Pipeline parallelism is not supported yet.')

    (num_nodes, num_devices_per_node, distributed_init_method,
    all_stage_devices) = (
        initialize_cluster(
            use_ray=True,
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size))

    server = FastAPIServer(
        model=args.model,
        cache_dir=args.cache_dir,
        use_np_cache=args.use_np_cache,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        dtype=args.dtype,
        seed=args.seed,
        swap_space=args.swap_space,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_sequences=args.max_num_sequences,
        num_nodes=num_nodes,
        num_devices_per_node=num_devices_per_node,
        distributed_init_method=distributed_init_method,
        all_stage_devices=all_stage_devices,
        server_use_ray=args.use_ray,
        log_stats=args.log_stats,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
