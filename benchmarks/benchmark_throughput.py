import argparse
import json
import random
import time
from typing import List, Tuple

from cacheflow import LLM, SamplingParams
from transformers import PreTrainedTokenizerBase


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[List[int], int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompt_token_ids[i], output_len))
    # Filter out if the prompt length + output length is greater than 2048.
    tokenized_dataset = [
        (prompt_token_ids, output_len)
        for prompt_token_ids, output_len in tokenized_dataset
        if len(prompt_token_ids) + output_len <= 2048
    ]

    # Sample the requests.
    sampled_requests = random.sample(tokenized_dataset, num_requests)
    return sampled_requests


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    # Add the requests to the server.
    for prompt_token_ids, output_len in requests:
        sampling_params = SamplingParams(
            n=args.n,
            temperature=0.0 if args.use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=args.use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt="",
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )

    start = time.time()
    # FIXME(woosuk): Do use internal method.
    llm._run_server(use_tqdm=True)
    end = time.time()
    total_num_tokens = sum(
        len(prompt_token_ids) + output_len
        for prompt_token_ids, output_len in requests
    )
    print(f"Throughput: {total_num_tokens / (end - start):.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
