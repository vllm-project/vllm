"""
Benchmark the efficiency of prefix caching.

This script allows you to benchmark the performance of
a model with prefix-caching or cpu-offloading using fixed prompts

Fixed example usage:
    # This command run the vllm with 50GB CPU memory for offloading
    # The workload samples 8 different prompts with a default input
    # length of 20010 tokens, then replicates each prompt 2 times.
    python benchmark_long_document_qa.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --block-allocator CpuOffloadingBlockAllocator \
        --num-documents 8 \
        --repeat-count 2 \
        --cpu-memory-gb 50

Commandline arguments:

    # Basic arguments
    --model: The model to use for the benchmark.

    --enable-prefix-caching: Enable prefix caching or not.

    --block-allocator: The block allocator that vLLM uses.
        - CpuGpuBlockAllocator: The default block allocator.
        - CpuOffloadingBlockAllocator: The block allocator that supports
          cpu offloading

    --gpu-memory-utilization: GPU memory utilization for vLLM.

    --cpu-memory-gb: The amount of CPU memory (GB) that is used by vLLM.
        NOTE: CPU memory should be larger than GPU KV cache size when
        using CpuOffloadingBlockAllocator.  

    # Workload-related arguments 
    --num-documents: The number of documents to sample prompts from.

    --repeat-count: The number of times to repeat each prompt.

    # Other functionality
    --seed: Random seed for reproducibility.

    --profile-swap-blocks: Profile the swap_blocks function in the custom ops.
"""

import random
import time

import torch

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

execution_times = {}


def build_result_dict(start_time, end_time, *args):
    total_time = end_time - start_time
    length = -1
    if len(args) > 1 and isinstance(args[1], torch.Tensor):
        length = len(args[1])

    return {
        "start_time": start_time,
        "total_time": total_time,
        "swap_len": length
    }


def timing_decorator(func):

    def wrapper(*args, **kwargs):
        global execution_times
        torch.cuda.synchronize()
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the wrapped function
        torch.cuda.synchronize()
        end_time = time.time()  # Record the end time
        if func.__name__ not in execution_times:
            execution_times[func.__name__] = []

        res = build_result_dict(start_time, end_time, *args)
        execution_times[func.__name__].append(res)
        return result  # Return the result of the original function

    return wrapper


def process_timing_results():
    global execution_times
    for key in execution_times:
        len_to_time = {}
        len_to_count = {}
        for item in execution_times[key]:
            swap_len = item["swap_len"]
            if swap_len not in len_to_time:
                len_to_time[swap_len] = 0
            len_to_time[swap_len] += item["total_time"]

            if swap_len not in len_to_count:
                len_to_count[swap_len] = 0
            len_to_count[swap_len] += 1

        for swap_len in len_to_time:
            total_time = len_to_time[swap_len]
            count = len_to_count[swap_len]
            print(f"{key} on {swap_len} pages: "
                  f"{(count * swap_len) / total_time} pages per second")


def test_long_document_qa(llm=None, sampling_params=None, prompts=None):

    start_time = time.time()
    llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    print(f"cost time {end_time - start_time}")


def repeat_prompts(prompts, repeat_count):
    repeated_prompts = prompts * repeat_count
    random.shuffle(repeated_prompts)
    return repeated_prompts


def main(args):
    if args.profile_swap_blocks:
        from vllm.worker.cache_engine import CacheEngine
        CacheEngine.swap_out = timing_decorator(CacheEngine.swap_out)
        CacheEngine.swap_in = timing_decorator(CacheEngine.swap_in)

    random.seed(args.seed)

    # append the document id at the beginning to avoid any of the document
    # being the prefix of other documents
    prompts = [
        str(i) + ' '.join(['hi'] * args.document_length)
        for i in range(args.num_documents)
    ]

    preemption_mode = ""
    if args.block_allocator == "CpuOffloadingBlockAllocator":
        preemption_mode = "recompute"
    else:
        preemption_mode = "swap"

    llm = LLM(model=args.model,
              tokenizer_mode='auto',
              trust_remote_code=True,
              enforce_eager=True,
              tensor_parallel_size=args.tensor_parallel_size,
              enable_prefix_caching=args.enable_prefix_caching,
              block_allocator=args.block_allocator,
              preemption_mode=preemption_mode,
              swap_space=args.cpu_memory_gb,
              enable_chunked_prefill=False,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=30000)

    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)

    prompts = repeat_prompts(prompts, args.repeat_count)

    print("------warm up------")
    test_long_document_qa(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )

    random.shuffle(prompts)

    print("------start generating------")
    test_long_document_qa(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )

    if args.profile_swap_blocks:
        process_timing_results()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=
        'Benchmark the performance with or without automatic prefix caching.')
    parser.add_argument(
        '--model',
        type=str,
        # this test aims to test long document QA capability,
        # so we use llama 3.1 8B as it can process long context
        default='meta-llama/Llama-3.1-8B')
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--output-len', type=int, default=10)
    parser.add_argument('--enable-prefix-caching',
                        action='store_true',
                        help='enable prefix caching')
    parser.add_argument('--repeat-count',
                        type=int,
                        default=2,
                        help='Number of times to repeat each prompt')
    parser.add_argument(
        '--document-length',
        type=int,
        # Roughly the number of tokens for a system paper,
        # excluding images
        default=20010,
        help='Range of input lengths for sampling prompts,'
        'specified as "min:max" (e.g., "128:256").')
    parser.add_argument('--num-documents',
                        type=int,
                        default=8,
                        help='Range of input lengths for sampling prompts,'
                        'specified as "min:max" (e.g., "128:256").')
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='GPU memory utilization for vLLM. Should be a '
                        'float point number ranging from 0 to 1. For this '
                        'test please use a small value so that the GPU '
                        'cannot hold all KV caches of all documents, '
                        'and the effect of CPU offloading can be tested.')
    parser.add_argument(
        '--cpu-memory-gb',
        type=float,
        default=1,
        help="The amount of CPU memory (GB) that is used by vLLM. Not very "
        "useful for CpuGpuBlockAllocator, but useful for "
        "CpuOffloadingBlockAllocator to have more CPU KV cache space")
    parser.add_argument(
        '--block-allocator',
        type=str,
        default='CpuGpuBlockAllocator',
        choices=['CpuGpuBlockAllocator', 'CpuOffloadingBlockAllocator'],
        help='The block allocator that vLLM uses. Currently'
        ' can be CpuGpuBlockAllocator (the default) and '
        'CpuOffloadingBlockAllocator (experimental) that '
        'supports offloading the KV cache to CPU . '
        'When using CpuOffloadingBlockAllocator, the '
        'preemption mode must be recompute.')

    parser.add_argument(
        '--profile-swap-blocks',
        action='store_true',
        help='Profile the swap_blocks function in the custom ops')

    args = parser.parse_args()
    main(args)
