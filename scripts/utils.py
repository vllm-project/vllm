from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import List, Dict, Any

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import List, Dict, Any
import json

import random

def get_prompts():
    filename = "pile.txt"
    with open(filename, "r") as f:
        prompts = f.readlines()
        print(f"Number of prompts: {len(prompts)}")
    return prompts


def get_prompt_token_ids(model_path, prompts, max_length=1024):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_token_ids = []
    for prompt in prompts:
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        if len(tokens.input_ids[0]) < max_length:
            continue
        prompt_token_ids.append([x.item() for x in tokens.input_ids[0]])
    return prompt_token_ids


def get_pile_prompts(model_name, num_samples=512):
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
    seed = 42
    # ==-------------------------------------------------------------------------==

    reset_seed(seed)

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    dataset = dataset.shuffle(seed=seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    num_sample = 0
    samples_lst = []
    for data in tqdm(dataset):
        prompt = data["text"]
        tokens = tokenizer(prompt, return_tensors="pt")
        if len(tokens.input_ids[0]) < least_tokens:
            continue
        num_sample += 1
        samples_lst.append(prompt)
        if num_sample >= num_samples:
            break
    return samples_lst

#==-------------------------------------------------------------------------==
# Load custom dataset
#==-------------------------------------------------------------------------==

def get_dataset(filepath: str) -> List[List[Dict[str, str]]]:
    """
    [
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "query prompt"},
        ],
        [
            {"role": "system", "content": "1. 角色设定：- 你是...."},
            {"role": "user", "content": "搜索关键词】\n梁斌是谁，做什么"},
        ],
        ...
    ]

    """
    with open(filepath) as f:
        dataset: List[List[Dict[str, str]]] = [json.loads(line) for line in f]
    return dataset


def sample_tc_requests(
    filepath: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    do_random: bool = False,
) -> List[str]:
    dataset = get_dataset(filepath)
    prompts = dataset
    few_shots = 0
    sampled_requests: List[str] = []
    for j in range(num_requests):
        i = (
            random.choice(range(len(prompts[few_shots:])))
            if do_random
            else j + few_shots
        )
        # message demo:
        # [
        #     {"role": "system", "content": "1. 角色设定：- 你是...."},
        #     {"role": "user", "content": "搜索关键词】\n梁斌是谁，做什么"},
        # ],
        message: List[Dict[str, str]] = prompts[i]
        prompt_with_template = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        sampled_requests.append(prompt_with_template)

    return sampled_requests

def get_tokenizer(model_path) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

