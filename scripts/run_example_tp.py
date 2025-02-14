from vllm import LLM, SamplingParams

import argparse
import os
from typing import Any, List, Tuple
from transformers import (PreTrainedTokenizerBase, AutoTokenizer)
import random
import datasets
import time
# get file location
file_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(file_path), "../benchmarks")

model_path = "/data/models/DeepSeek-R1/"
# model_path = "deepseek-ai/DeepSeek-V2-Lite"

# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=model_path, help="The model path.")
parser.add_argument("--tokenizer", type=str, default=model_path, help="The model path.")
parser.add_argument("--tp_size", type=int, default=8, help="The number of threads.")
parser.add_argument("--ep_size", type=int, default=1, help="The number of threads.")
parser.add_argument("--dataset", type=str, default=None, help="The dataset.")
parser.add_argument("--isl", type=int, default=1024, help="input sequence length.")
parser.add_argument("--osl", type=int, default=1024, help="output sequence length.")
parser.add_argument("--nprompts", type=int, default=4, help="The number of prompts.")
parser.add_argument("--random", action="store_true", help="Randomly sample prompts.")
args = parser.parse_args()

os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
os.environ["VLLM_MOE_N_SLICE"] = "1" if args.ep_size > 1 else "4"
os.environ["VLLM_EP_SIZE"] = f"{args.ep_size}"
os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"
os.environ["PT_HPU_WEIGHT_SHARING"] = "0"

def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int, None]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List = []
    for _ in range(num_requests):
        num_lines_needed = num_input_lines - num_prefix_lines
        sampled_lines = "".join(prefix_lines +
                                random.choices(poem_lines, k=num_lines_needed))
        

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        sampled_requests.append(prompt_formatted)

    return sampled_requests, None

def sample_gsm8k_requests(
    num_requests: int, tokenizer: PreTrainedTokenizerBase, do_random: bool = False
) -> List[Tuple[str, str]]:
    # Load the dataset from huggingface.
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    prompts = dataset["train"]["question"]
    expected_responses = dataset["train"]["answer"]
    few_shots = 5
    base_prompt = [f"Question: {prompts[i]}\nAnswer: {expected_responses[i]}\n" for i in range(few_shots)]
    base_prompt = "\n".join(base_prompt)
    base_prompt = f"{base_prompt}\n"
    
    # Sample the requests.
    sampled_requests: List = []
    sampled_response: List = []
    for j in range(num_requests):
        i = random.choice(range(len(prompts[few_shots:]))) if do_random else j + few_shots
        prompt = f"{base_prompt}Question: {prompts[i]}\nAnswer: "
        # message = [
        #     {
        #         "role": "user",
        #         "content": prompt,
        #     },
        # ]
        # prompt = tokenizer.apply_chat_template(
        #     message, add_generation_prompt=True, tokenize=False)
        expected_response = expected_responses[i]
        sampled_requests.append(prompt)
        sampled_response.append(expected_response)

    return sampled_requests, sampled_response

if __name__ == "__main__":

    # Sample prompts.
    
    if args.dataset == "sonnet":
        # Sample sonnet requests.
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        prompts, gt = sample_sonnet_requests(
            dataset_path=f"{dataset_path}/sonnet.txt",
            num_requests=args.nprompts,
            input_len=args.isl,
            prefix_len=200,
            tokenizer=tokenizer,
        )
    elif args.dataset == "gsm8k":
        # Sample GSM8K requests.
        args.osl=128
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        prompts, gt = sample_gsm8k_requests(
            num_requests=args.nprompts,
            tokenizer=tokenizer,
            do_random=args.random,
        )
    else:
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        if args.nprompts > 4:
            prompts += random.choices(prompts, k=args.nprompts - 4)
        elif args.nprompts < 4:
            prompts = prompts[: args.nprompts]
        gt = None
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=args.osl, ignore_eos=True)
    model = args.model
    if args.tp_size == 1:
        llm = LLM(
            model=model, 
            tokenizer=args.tokenizer,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=16384,
            gpu_memory_utilization=0.8,
        )
    else:
        llm = LLM(
            model=model, 
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tp_size,
            distributed_executor_backend='mp',
            trust_remote_code=True,
            max_model_len=16384,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
        )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = time.perf_counter()
    # Print the outputs.
    print(f"e2e took {end - start} seconds")
    for output_i in range(len(outputs)):
        output = outputs[output_i]
        gt_i = None if gt is None else gt[output_i]
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("====================================")
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Ground truth: {gt_i!r}")
        print("====================================")
    del llm