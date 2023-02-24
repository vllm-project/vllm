import argparse
from typing import List

from cacheflow.master.scheduler import Scheduler
from cacheflow.worker.controller import Controller

parser = argparse.ArgumentParser(description='CacheFlow server')
parser.add_argument('--model', type=str, default='facebook/opt-125m', help='model name')
parser.add_argument('--num-nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--num-workers', type=int, default=1, help='number of workers per node')
parser.add_argument('--block-size', type=int, default=8, help='block size')
parser.add_argument('--num-gpu-blocks', type=int, default=1024, help='number of GPU blocks')
parser.add_argument('--num-cpu-blocks', type=int, default=256, help='number of CPU blocks')


def main():
    args = parser.parse_args()

    # Create controllers.
    controllers: List[Controller] = []
    for i in range(args.num_nodes):
        controller = Controller(
            node_id=i,
            num_workers=args.num_workers,
            model_name=args.model,
            block_size=args.block_size,
            num_gpu_blocks=args.num_gpu_blocks,
            num_cpu_blocks=args.num_cpu_blocks,
            dtype='float',
        )
        controllers.append(controller)

    # Create a scheduler.
    scheduler = Scheduler(
        controllers=controllers,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        num_cpu_blocks=args.num_cpu_blocks,
    )
    # Connect the controllers.
    for i in range(len(controllers) - 1):
        controllers[i].set_next(controllers[i + 1])
    controllers[-1].set_next(scheduler)

    # seq_groups, max_num_steps, stop_token_ids = generate_inputs(1000, args.block_size)
    seq_groups, max_num_steps, stop_token_ids = test_inputs(args.block_size)
    scheduler.pending.extend(seq_groups)
    scheduler.max_num_steps.update(max_num_steps)
    scheduler.stop_token_ids.update(stop_token_ids)

    while scheduler.pending or scheduler.running:
        scheduler.prepare()
        scheduler.step()


def test_inputs(block_size):
    from cacheflow.sequence import Sequence
    from cacheflow.sequence import SequenceGroup
    from cacheflow.utils import Counter

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    prompt = "Hello, I'm am conscious and"
    prompt_tokens = tokenizer.encode(prompt)

    seq = Sequence(0, prompt_tokens, block_size=block_size)
    seq_group = SequenceGroup(0, [seq])
    seq_groups = [seq_group]
    max_num_steps = {0: 8}
    stop_token_ids = {0: []}
    return seq_groups, max_num_steps, stop_token_ids


def generate_inputs(num_inputs, block_size):
    import random
    random.seed(0)

    from cacheflow.sequence import Sequence
    from cacheflow.sequence import SequenceGroup
    from cacheflow.utils import Counter

    seq_group_counter = Counter()
    seq_counter = Counter()

    max_num_steps = {}
    stop_token_ids = {}
    seq_groups = []
    for _ in range(num_inputs):
        seq_group_id = next(seq_group_counter)

        prompt_len = random.randint(16, 128)
        max_num_steps[seq_group_id] = random.randint(32, 1024)
        stop_token_ids[seq_group_id] = []

        seqs = []
        for _ in range(2):
            seq_id = next(seq_counter)
            seq = Sequence(seq_id, [0] * prompt_len, block_size=block_size)
            seqs.append(seq)
        seq_group = SequenceGroup(seq_group_id, seqs)
        seq_groups.append(seq_group)

    return seq_groups, max_num_steps, stop_token_ids


if __name__ == '__main__':
    main()
