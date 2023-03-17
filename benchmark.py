import argparse
from datetime import datetime
import os
import pickle
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from transformers import AutoTokenizer

from cacheflow.master.scheduler import Scheduler
from cacheflow.models import get_memory_analyzer
from cacheflow.sampling_params import SamplingParams
from cacheflow.sequence import Sequence
from cacheflow.sequence import SequenceGroup
from cacheflow.worker.controller import Controller
from cacheflow.utils import Counter

parser = argparse.ArgumentParser(description='CacheFlow server')
parser.add_argument('--model', type=str, default='facebook/opt-13b', help='model name')
parser.add_argument('--num-nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--num-workers', type=int, default=1, help='number of workers per node')
parser.add_argument('--block-size', type=int, default=8, choices=[8, 16], help='token block size')
# NOTE(woosuk): If FlashAttention is used, the float data type is not supported.
parser.add_argument('--dtype', type=str, default='half', choices=['half', 'float'], help='data type')
# TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--swap-space', type=int, default=20, help='CPU swap space size (GiB) per GPU')
parser.add_argument('--max-batch-size', type=int, default=2560, help='maximum number of batched tokens')

parser.add_argument('--len-estimator', type=str, choices=['oracle', 'power2', 'constant'], default='oracle')

parser.add_argument('--dataset', type=str, default='text_completion_opt.pkl', help='dataset path')
parser.add_argument('--request-rate', type=float, default=1.0, help='reqs/sec')
parser.add_argument('--duration', type=int, default=600, help='duration in seconds')
parser.add_argument('--n', type=int, default=1, help='number of output sequences per request')
parser.add_argument('--use-beam', action='store_true', help='use beam search')

args = parser.parse_args()
args.max_batch_size = max(2048 * args.n, args.max_batch_size)


def generate_requests(
    dataset: str,
    request_rate: float,
    duration: int,
    seed: int,
    max_seq_len: int = 2048,
    time_quantum: int = 100,
):
    random.seed(seed)
    np.random.seed(seed)

    # Generate timestamps for requests using Poisson distribution.
    lam = request_rate * (time_quantum / 1000)
    quantums_per_sec = 1000 / time_quantum
    arrival_times = np.random.poisson(
        lam=lam, size=int(duration * quantums_per_sec))
    timestamps = []
    for i, n in enumerate(arrival_times):
        timestamps += [i * (time_quantum / 1000)] * n

    # Load and shuffle the dataset.
    with open(dataset, 'rb') as f:
        data = pickle.load(f)
    random.shuffle(data)

    # Generate requests.
    num_requests = len(timestamps)
    requests = []
    for pair, timestamp in zip(data[:num_requests], timestamps):
        input_tokens, output_tokens = pair
        input_len = len(input_tokens)
        # Skip the data if the input length is too long.
        if input_len >= max_seq_len:
            continue

        output_len = len(output_tokens)
        # Truncate the output length if it is too long.
        output_len = min(output_len, max_seq_len - input_len)
        requests.append((timestamp, input_tokens, output_len))
    return requests


class FakeFrontend:

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seq_group_counter = Counter()
        self.seq_counter = Counter()
        self.requests = []
        self.timestamps = []
        self.sampling_params: Dict[int, SamplingParams] = {}
        self.results: Dict[int, Tuple[int, int, SamplingParams]] = {}

    def add_request(
        self,
        token_ids: List[int],
        timestamp: float,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_beam_search: bool = False,
        stop_token_ids: Set[int] = set(),
        max_num_steps: int = 16,
        num_logprobs: int = 0,
        context_window_size: Optional[int] = None,
    ) -> None:
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=use_beam_search,
            stop_token_ids=stop_token_ids,
            max_num_steps=max_num_steps,
            num_logprobs=num_logprobs,
            context_window_size=context_window_size,
        )

        seqs = []
        for _ in range(n):
            seq_id = next(self.seq_counter)
            seq = Sequence(
                seq_id=seq_id,
                token_ids=token_ids,
                block_size=args.block_size,
            )
            seqs.append(seq)
        group_id = next(self.seq_group_counter)
        seq_group = SequenceGroup(
            group_id=group_id,
            seqs=seqs,
            max_num_steps=max_num_steps,
        )
        self.requests.append((seq_group, sampling_params))
        self.timestamps.append(timestamp)
        self.sampling_params[group_id] = sampling_params

    def start_timer(self) -> None:
        self.start = time.time()

    def get_inputs(self):
        num_requests = self.get_num_requests()
        if num_requests == 0:
            return []

        requests = self.requests[:num_requests]
        self.requests = self.requests[num_requests:]
        self.timestamps = self.timestamps[num_requests:]

        now = time.time()
        for seq_group, _ in requests:
            seq_group.arrival = now
        return requests

    def get_num_requests(self):
        if not self.timestamps:
            return 0

        now = time.time()
        now = now - self.start

        for i, timestamp in enumerate(self.timestamps):
            if timestamp > now:
                break
        else:
            i = len(self.timestamps)
        return i

    def print_response(self, seq_group):
        now = time.time()
        sampling_params = self.sampling_params[seq_group.group_id]
        for seq in seq_group.seqs:
            self.results[seq.seq_id] = (seq_group.arrival, now, sampling_params)
            continue

            # This is for debugging.
            token_ids = seq.get_token_ids()
            output = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            print(f'Seq {seq.seq_id}: {output!r}')


def main():
    memory_analyzer = get_memory_analyzer(
        model_name=args.model,
        block_size=args.block_size,
        dtype=args.dtype,
    )
    num_gpu_blocks = memory_analyzer.get_max_num_gpu_blocks(
        max_num_batched_tokens=args.max_batch_size)
    num_cpu_blocks = memory_analyzer.get_max_num_cpu_blocks(
        swap_space=args.swap_space)
    print(f'# GPU blocks: {num_gpu_blocks}, # CPU blocks: {num_cpu_blocks}')

    # Create a controller for each node.
    controllers: List[Controller] = []
    for i in range(args.num_nodes):
        controller = Controller(
            node_id=i,
            num_workers=args.num_workers,
            model_name=args.model,
            block_size=args.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            dtype=args.dtype,
            seed=args.seed,
        )
        controllers.append(controller)

    # Create a frontend and add requests.
    frontend = FakeFrontend(args.model)

    # Create a scheduler.
    scheduler = Scheduler(
        frontend=frontend,
        controllers=controllers,
        block_size=args.block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        max_num_batched_tokens=args.max_batch_size,
        len_estimator=args.len_estimator,
    )
    # Connect the controllers.
    for i in range(len(controllers) - 1):
        controllers[i].set_next(controllers[i + 1])
    controllers[-1].set_next(scheduler)

    # Generate and add requests.
    requests = generate_requests(
        dataset=args.dataset,
        request_rate=args.request_rate,
        duration=args.duration,
        seed=args.seed,
    )
    for request in requests:
        timestamp, input_tokens, output_len = request
        frontend.add_request(
            token_ids=input_tokens,
            timestamp=timestamp,
            max_num_steps=output_len,
            n=args.n,
            use_beam_search=args.use_beam,
            temperature=1.0 if not args.use_beam else 0.0,
        )

    start = datetime.now()
    print('Start at', start)

    frontend.start_timer()
    while True:
        scheduler.step()
        if not (scheduler.pending or scheduler.running or frontend.requests):
            break
    end = datetime.now()
    print('End at', end)
    print('Duration', end - start)

    # Save the results.
    model_name = args.model.replace('/', '_')
    beam = 'beam' if args.use_beam and args.n > 1 else 'no_beam'
    save_dir = (f'orca/{model_name}/bs{args.max_batch_size}/'
                f'n{args.n}/{beam}/r{args.request_rate}/s{args.seed}/d{args.duration}/')
    os.makedirs(save_dir, exist_ok=True)

    # Save the latency results.
    with open(f'{save_dir}/results.pkl', 'wb') as f:
        pickle.dump(frontend.results, f)
    # Save other data.
    with open(f'{save_dir}/input_lens.pkl', 'wb') as f:
        pickle.dump(scheduler.input_lens, f)
    with open(f'{save_dir}/swap_out_lens.pkl', 'wb') as f:
        pickle.dump(scheduler.swap_out_lens, f)
    with open(f'{save_dir}/swap_in_lens.pkl', 'wb') as f:
        pickle.dump(scheduler.swap_in_lens, f)
    with open(f'{save_dir}/num_pendings.pkl', 'wb') as f:
        pickle.dump(scheduler.num_pendings, f)
    with open(f'{save_dir}/next_seq_lens.pkl', 'wb') as f:
        pickle.dump(scheduler.next_seq_lens, f)
    with open(f'{save_dir}/gpu_cache_usage.pkl', 'wb') as f:
        pickle.dump(scheduler.gpu_blocks_usage, f)
    with open(f'{save_dir}/cpu_cache_usage.pkl', 'wb') as f:
        pickle.dump(scheduler.cpu_blocks_usage, f)
    with open(f'{save_dir}/requests_received.pkl', 'wb') as f:
        pickle.dump(scheduler.requests_received, f)
    print(f'Results saved to {save_dir}')


if __name__ == '__main__':
    main()
