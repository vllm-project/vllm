import argparse
from typing import List

from cacheflow.master.frontend import Frontend
from cacheflow.master.scheduler import Scheduler
from cacheflow.models import compute_max_num_cpu_blocks
from cacheflow.models import compute_max_num_gpu_blocks
from cacheflow.worker.controller import Controller

parser = argparse.ArgumentParser(description='CacheFlow server')
parser.add_argument('--model', type=str, default='facebook/opt-125m', help='model name')
parser.add_argument('--num-nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--num-workers', type=int, default=1, help='number of workers per node')
parser.add_argument('--block-size', type=int, default=8, choices=[8, 16], help='token block size')
# TODO(woosuk): Add an analytical model to determine the maximum number of GPU/CPU blocks.
parser.add_argument('--swap-space', type=int, default=16,
                    help='The CPU memory space in GiB pinned for swapping (per GPU)')
# NOTE(woosuk): If FlashAttention is used, the float data type is not supported.
parser.add_argument('--dtype', type=str, default='half', choices=['half', 'float'], help='data type')
args = parser.parse_args()


def main():
    num_gpu_blocks = compute_max_num_gpu_blocks(
        model_name=args.model,
        max_num_batched_tokens=2048,
        block_size=args.block_size,
        dtype=args.dtype,
    )
    num_cpu_blocks = compute_max_num_cpu_blocks(
        swap_space=args.swap_space,
        model_name=args.model,
        block_size=args.block_size,
        dtype=args.dtype,
    )
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
        )
        controllers.append(controller)

    # Create a frontend.
    frontend = Frontend(
        model_name=args.model,
        block_size=args.block_size,
    )

    # Create a scheduler.
    scheduler = Scheduler(
        frontend=frontend,
        controllers=controllers,
        block_size=args.block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
    )
    # Connect the controllers.
    for i in range(len(controllers) - 1):
        controllers[i].set_next(controllers[i + 1])
    controllers[-1].set_next(scheduler)

    test_inputs = [
        'Ion Stoica is a',
        'UC Berkeley is',
        'The future of cloud computing is',
    ]
    for prompt in test_inputs:
        frontend.query(prompt)

    # FIXME
    while True:
        scheduler.step()
        if not scheduler.pending and not scheduler.running:
            break


if __name__ == '__main__':
    main()
