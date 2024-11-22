"""
Benchmark the efficiency of prefix caching.

This script allows you to benchmark the performance of
a model with and without prefix caching using either fixed prompts
or prompts sampled from the ShareGPT dataset.

Fixed example usage:
    python benchmark_multi_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --num-prompts 1 \
        --repeat-count 100

ShareGPT example usage:
    # This command samples 20 prompts with input lengths
    # between 128 and 256 tokens from the ShareGPT dataset,
    # then replicates each prompt 5 times.
    python benchmark_multi_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --enable-prefix-caching \
        --num-prompts 20 \
        --input-length-range 128:256
"""

import json
import random
import time
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from vllm import LLM, CachingParams, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
import torch.distributed as dist
import torch
try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

# This two steps will prevent the nccl softlockup
import os
import signal
os.environ['NCCL_TIMEOUT'] = '20'
def signal_handler(signum, frame):
    print("Received signal to terminate")
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def test_prefix(llm=None, sampling_params=None, prompts=None):
    start_time = time.time()

    llm.generate(prompts, sampling_params=sampling_params)

    end_time = time.time()
    print(f"cost time {end_time - start_time}")


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: Tuple[int, int],
    fixed_output_len: Optional[int],
    ) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    min_len, max_len = input_length_range

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if min_len <= prompt_len <= max_len:
            filtered_dataset.append((prompt, prompt_len, output_len))

    cnt = len(filtered_dataset)
    return filtered_dataset[0: int(cnt / 2)], filtered_dataset[int(cnt / 2): ]


def select_requests(requests: List[Tuple[str, int, int]],
                             repeat_count: int,
                             sort: bool = False,
                             ratio: float = 1) -> List[str]:
    repeated_requests = requests * repeat_count
    if sort:
        repeated_requests.sort(key=lambda x: x[1])
    else:
        random.shuffle(repeated_requests)
    cnt = int(len(repeated_requests) * ratio)
    return [req[0] for req in repeated_requests[0: cnt]]


def main(args):
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    input_length_range = tuple(map(int, args.input_length_range.split(':')))
    if args.dataset_path is not None:
        print(f"Start to sample {args.num_prompts} prompts"
              "from {args.dataset_path}")
        cache_hit_datasets, cache_miss_datasets = sample_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
        )
    else:
        raise ValueError("Should support the caching docs to args.dataset_path")
    llm = LLM(model=args.model,
              tokenizer_mode='auto',
              trust_remote_code=True,
              enforce_eager=True,
              use_v2_block_manager=False,
              tensor_parallel_size=1,
              # dtype='float16',
              enable_prefix_caching=True,
              enable_memory_tiering=True,
              disable_custom_all_reduce=True)
    
    print("------ build prefix caching ------")
    start_time = time.time()
    caching_params = CachingParams(ttl=args.ttl)
    cache_output = llm.caching(cache_hit_datasets, caching_params=caching_params)
    end_time = time.time()

    print(
        f"Caching output: {cache_output} \ncost time {end_time - start_time}\n"
    )

    print("------ start generating cached prompts ------")
    start_time = time.time()
    prompts = [req[0] for req in cache_hit_datasets][:20]
    # print("The first cache hit prompt is {}".format(prompts[0]))
    output = llm.generate(prompts,
                 SamplingParams(temperature=0, max_tokens=args.output_len))
    end_time = time.time()
    print(f"Cost time {end_time - start_time}, output is {output[0].outputs[0].text}\n")

    print("------ start generating uncached prompts ------")
    start_time = time.time()
    prompts = [req[0] for req in cache_miss_datasets][0:20]
    # print("The first cache miss prompt is {}".format(prompts[0]))
    output = llm.generate(prompts,
                 SamplingParams(temperature=0, max_tokens=args.output_len))
    end_time = time.time()
    print(f"Cost time {end_time - start_time}, output is {output[0].outputs[0].text}\n")

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Benchmark the performance with context caching.')
    parser.add_argument('--model', type=str, default='01-ai/Yi-6B')
    parser.add_argument('--output-len', type=int, default=5)
    parser.add_argument('--ttl', type=int, default=300)
    parser.add_argument("--dataset-path", type=str, default=None, 
            help="Path to the dataset.")
    parser.add_argument('--input-length-range',
                        type=str,
                        default='1000:1200',
                        help='Range of input lengths for sampling prompts,'
                        'specified as "min:max" (e.g., "1000:2000").')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=60,
                        help="Number of the prompts sampled from dataset")
    parser.add_argument('--sort',
                        action='store_true',
                        help='Sort prompts by input length')

    args = parser.parse_args()
    main(args)
