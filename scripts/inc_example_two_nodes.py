from vllm import LLM, SamplingParams

import argparse
import os
from typing import Any, List, Tuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import random
import datasets
from vllm.utils import reset_seed
reset_seed()
# get file location
file_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(file_path), "../benchmarks")

model_path = "/data/models/DeepSeek-R1/"
model_path = "/hf/hf_models/DeepSeek-R1"
model_path = "/mnt/disk5/hf_models/DeepSeek-R1-BF16"
# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=model_path, help="The model path.")
parser.add_argument("--tokenizer", type=str, default=model_path, help="The model path.")
parser.add_argument("--tp_size", type=int, default=16, help="The number of threads.")
parser.add_argument("--ep_size", type=int, default=16, help="The number of threads.")
parser.add_argument("--dataset", type=str, default=None, help="The dataset.")
parser.add_argument("--isl", type=int, default=1024, help="input sequence length.")
parser.add_argument("--osl", type=int, default=128, help="output sequence length.")
parser.add_argument("--nprompts", type=int, default=4, help="The number of prompts.")
parser.add_argument("--mode", type=str, default=None, choices=["quant", "prepare"], help="The mode.")
parser.add_argument("--random", action="store_true", help="Randomly sample prompts.")
parser.add_argument("--smoke", action="store_true", help="Smoke test")
parser.add_argument("--fp8_kvcache", action="store_true", help="Using FP8 KV cache.")
args = parser.parse_args()

max_num_seqs = 4

# ==-------------------------------------------------------------------------==
# Calibration parameters
least_tokens = 1024
num_samples = 512
max_new_tokens = 32
seed = 42
# https://github.com/deepseek-ai/DeepSeek-R1/blob/main/README.md#deepseek-r1-evaluation
# ... benchmarks requiring sampling, we use a temperature of 0.6, a top-p value of 0.95...
temperature = 0.6
temperature = 0 # greedy sample
top_p = 0.95
# ==-------------------------------------------------------------------------==

def _apply_inc():
    return os.getenv("QUANT_CONFIG", None) is not None

def _apply_inc_quant():
    INC_QUANT_CONFIG = os.getenv("QUANT_CONFIG", None)
    assert INC_QUANT_CONFIG is not None, "Please set the environment variable QUANT_CONFIG."
    from neural_compressor.torch.quantization import FP8Config
    config = FP8Config.from_json_file(INC_QUANT_CONFIG)
    return config.quantize

if __name__ == "__main__":

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    from utils import get_prompts, get_prompt_token_ids, get_pile_prompts
    if args.smoke:
        prompts = get_prompts()
    else:
        prompts = get_pile_prompts(args.model, num_samples)
    prompt_token_ids = get_prompt_token_ids(
        args.model, prompts, least_tokens
    )
    gt = None
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        truncate_prompt_tokens=least_tokens,
    )
    model = args.model
    quant = True
    if args.mode is not None:
        print(f"Mode is {args.mode}.")
        if args.mode == "quant":
            # quantize
            if args.fp8_kvcache:
                print(f">>>>>>>>>>>>>> Using FP8 KV cache.")
                llm = LLM(
                    model=model, 
                    tokenizer=args.tokenizer,
                    tensor_parallel_size=args.tp_size,
                    distributed_executor_backend='ray',
                    trust_remote_code=True,
                    quantization='inc',
                    weights_load_device="cpu",
                    kv_cache_dtype="fp8_inc",
                    max_model_len=16384,
                    max_num_seqs=max_num_seqs,
                    dtype="bfloat16",
                )
            else:
                llm = LLM(
                    model=model, 
                    tokenizer=args.tokenizer,
                    tensor_parallel_size=args.tp_size,
                    distributed_executor_backend='ray',
                    trust_remote_code=True,
                    quantization='inc',
                    weights_load_device="cpu",
                    max_num_seqs=max_num_seqs,
                    max_model_len=16384,
                    dtype="bfloat16",
                )
        else:
            # prepare
            llm = LLM(
                model=model, 
                tokenizer=args.tokenizer,
                tensor_parallel_size=args.tp_size,
                distributed_executor_backend='ray',
                trust_remote_code=True,
                quantization='inc',
                max_model_len=16384,
                max_num_seqs=max_num_seqs,
                dtype="bfloat16",
            )
    else:
        llm = LLM(
            model=model, 
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tp_size,
            distributed_executor_backend='ray',
            trust_remote_code=True,
            max_model_len=16384,
            dtype="bfloat16",
        )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(
        prompts=None, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids
    )
    # Print the outputs.
    for output_i in range(len(outputs)):
        output = outputs[output_i]
        gt_i = None if gt is None else gt[output_i]
        prompt_token_ids = output.prompt_token_ids
        generated_text = output.outputs[0].text
        print("====================================")
        print(f"Prompt[:10]: {prompt_token_ids[:10]!r}")
        print(f"Prompt[-10:]: {prompt_token_ids[-10:]!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Ground truth: {gt_i!r}")
        print("====================================")

    llm.llm_engine.model_executor.shutdown()
