# python inc_prepare.py 2>&1 | tee _log.inc_prepare

from vllm import LLM, SamplingParams

import argparse
import os
from typing import Any, List, Tuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import random
import datasets

import os
from vllm import LLM, SamplingParams
from vllm.utils import reset_seed
from datasets import load_dataset
from tqdm import tqdm
import transformers

"""
这个是Heng这边autoround calibration static model的方法：
“数据集NeelNanda/pile-10k,随机采样 seed=42, iters=1相当于使用rtn, nsamples=512 
采样512条数据，seqlen=1024选取大于长度大于1024的数据” 
"""

# ==-------------------------------------------------------------------------==
# Calibration parameters
least_tokens = 1024
num_samples = 512
max_new_tokens = 32
seed = 42
# https://github.com/deepseek-ai/DeepSeek-R1/blob/main/README.md#deepseek-r1-evaluation
"""
... benchmarks requiring sampling, we use a temperature of 0.6, a top-p value of 0.95...
"""
temperature = 0.6
top_p = 0.95
# ==-------------------------------------------------------------------------==


reset_seed(seed)

# get file location
file_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(file_path), "../benchmarks")

model_path = "/data/models/DeepSeek-R1/"
model_path = "/hf/hf_models/DeepSeek-R1"
# model_path = "deepseek-ai/DeepSeek-V2-Lite"
model_path = "/mnt/disk5/hf_models/DeepSeek-R1-BF16"
# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default=model_path, help="The model path."
)
parser.add_argument(
    "--tokenizer", type=str, default=model_path, help="The model path."
)
parser.add_argument(
    "--tp_size", type=int, default=16, help="The number of threads."
)
parser.add_argument(
    "--ep_size", type=int, default=16, help="The number of threads."
)
parser.add_argument("--dataset", type=str, default=None, help="The dataset.")
parser.add_argument(
    "--isl", type=int, default=1024, help="input sequence length."
)
parser.add_argument(
    "--osl", type=int, default=128, help="output sequence length."
)
parser.add_argument(
    "--nprompts", type=int, default=4, help="The number of prompts."
)
parser.add_argument(
    "--random", action="store_true", help="Randomly sample prompts."
)
args = parser.parse_args()

# os.environ["VLLM_SKIP_WARMUP"] = "true"
# os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
# os.environ['HABANA_VISIBLE_MODULES'] ='0,1,2,3,4,5,6,7'
# os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
# os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
# os.environ['PT_HPUGRAPH_DISABLE_TENSOR_CACHE']='1'
# os.environ['GLOO_SOCKET_IFNAME']='eth0'

# os.environ["VLLM_MOE_N_SLICE"] = "1" if args.ep_size > 1 else "4"
# os.environ["VLLM_EP_SIZE"] = f"{args.ep_size}"
# os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"

# os.environ["VLLM_RAY_DISABLE_LOG_TO_DRIVER"] = "0"
# os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "0"
# os.environ["RAY_DEDUP_LOGS"] = "1"
# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"


if __name__ == "__main__":
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        truncate_prompt_tokens=least_tokens,
    )
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    dataset = dataset.shuffle(seed=seed)

    model = args.model
    model_name = model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    llm = LLM(
        model=model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tp_size,
        distributed_executor_backend="ray",
        trust_remote_code=True,
        quantization="inc_p",
        max_model_len=16384,
        weights_load_device="cpu",
        dtype="bfloat16",
    )

    num_sample = 0
    for data in tqdm(dataset):
        prompt = data["text"]
        tokens = tokenizer(prompt, return_tensors="pt")
        if len(tokens.input_ids[0]) < least_tokens:
            continue
        num_sample += 1
        if num_sample > num_samples:
            break
        outputs = llm.generate(prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text
        print(
            f"Prompt: {prompt!r} ({len(tokens.input_ids[0])})\nGenerated: {generated_text!r}\n"
        )

    llm.llm_engine.model_executor.shutdown()
