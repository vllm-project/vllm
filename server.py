import argparse
from typing import List

from cacheflow.master.frontend import Frontend
from cacheflow.master.scheduler import Scheduler
from cacheflow.worker.controller import Controller

parser = argparse.ArgumentParser(description='CacheFlow server')
parser.add_argument('--model', type=str, default='facebook/opt-125m', help='model name')
parser.add_argument('--num-nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--num-workers', type=int, default=1, help='number of workers per node')
parser.add_argument('--block-size', type=int, default=8, help='token block size')
# TODO(woosuk): Add an analytical model to determine the maximum number of GPU/CPU blocks.
parser.add_argument('--num-gpu-blocks', type=int, default=1024, help='number of GPU blocks (per GPU)')
parser.add_argument('--num-cpu-blocks', type=int, default=256, help='number of CPU blocks (per GPU)')
args = parser.parse_args()


def main():
    # Create a controller for each node.
    controllers: List[Controller] = []
    for i in range(args.num_nodes):
        controller = Controller(
            node_id=i,
            num_workers=args.num_workers,
            model_name=args.model,
            block_size=args.block_size,
            num_gpu_blocks=args.num_gpu_blocks,
            num_cpu_blocks=args.num_cpu_blocks,
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
        num_gpu_blocks=args.num_gpu_blocks,
        num_cpu_blocks=args.num_cpu_blocks,
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
