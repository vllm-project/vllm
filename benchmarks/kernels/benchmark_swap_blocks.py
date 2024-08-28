import torch
import time

from vllm.utils import FlexibleArgumentParser
from vllm import _custom_ops as ops


def benchmark_swap_blocks(src_shape, num_blocks):
    src = torch.randn(src_shape, dtype=torch.float16).cuda()
    dst = torch.zeros_like(src).cpu()
    block_mapping = [(i, i) for i in range(num_blocks)]
    blocks_to_swap = torch.tensor(block_mapping,
                                  device="cpu",
                                  dtype=torch.int64).view(-1, 2)

    num_iterations = 100
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.time()
        ops.swap_blocks(src, dst, blocks_to_swap)
        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / num_iterations
    print(
        f"Avg. GPU->CPU time taken for swapping blocks: {average_time} seconds"
    )


def benchmark_swap_out_to_local_file(src_shape, dst, num_blocks):
    src = torch.randn(src_shape, dtype=torch.float16).cuda()
    num_elements = src.numel()
    element_size = src.element_size()
    total_bytes = num_elements * element_size
    with open(dst, 'wb') as file:
        file.write(b'0' * total_bytes)

    block_mapping = [(i, i) for i in range(num_blocks)]
    blocks_to_swap = torch.tensor(block_mapping,
                                  device="cpu",
                                  dtype=torch.int64).view(-1, 2)
    num_iterations = 100
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.time()
        ops.swap_out_to_local_file(src, dst, blocks_to_swap)
        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / num_iterations
    print(
        f"Avg. GPU->File time taken for swapping blocks: {average_time} seconds"
    )


def benchmark_swap_in_from_local_file(src_shape, src, num_blocks):
    dst = torch.zeros(src_shape, dtype=torch.float16).cuda()
    block_mapping = [(i, i) for i in range(num_blocks)]
    blocks_to_swap = torch.tensor(block_mapping,
                                  device="cpu",
                                  dtype=torch.int64).view(-1, 2)
    num_iterations = 100
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.time()
        ops.swap_in_from_local_file(src, dst, blocks_to_swap)
        torch.cuda.synchronize()
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / num_iterations
    print(
        f"Avg. File->GPU time taken for swapping blocks: {average_time} seconds"
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--num-blocks", type=int, default="1024")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=32)
    parser.add_argument("--filename", type=str, default="./test.txt")
    args = parser.parse_args()
    print(args)

    src_shape = (args.num_blocks, args.block_size, args.num_kv_heads,
                 args.head_size)

    benchmark_swap_blocks(src_shape, args.num_blocks)
    benchmark_swap_out_to_local_file(src_shape, args.filename, args.num_blocks)
    benchmark_swap_in_from_local_file(src_shape, args.filename,
                                      args.num_blocks)
