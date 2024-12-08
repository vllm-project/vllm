"""
Benchmark the efficiency of prefix caching.

This script allows you to benchmark the performance of
a model with and without prefix caching using either fixed prompts
or prompts sampled from the ShareGPT dataset.

Fixed example usage:
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --num-prompts 1 \
        --repeat-count 100

ShareGPT example usage:
    # This command samples 20 prompts with input lengths
    # between 128 and 256 tokens from the ShareGPT dataset,
    # then replicates each prompt 5 times.
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --enable-prefix-caching \
        --num-prompts 20 \
        --repeat-count 5 \
        --input-length-range 128:256
"""

import random
import time

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser


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

    print("------start generating------")
    test_long_document_qa(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )


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
                        default=0.5,
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
    args = parser.parse_args()
    main(args)
