import json
import random
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset
from typing import List, Tuple, Optional
from pathlib import Path
import subprocess


@dataclass
class DatasetArgs:
    num_samples: int
    max_len: int = 4096
    seed: int = 42
    fixed_output_len: Optional[int] = None


DatasetTriple = List[Tuple[str, int, int]]


def make_dataset_triples(prompts: List[str], completions: List[str],
                         tokenizer: PreTrainedTokenizerBase,
                         dataset_args: DatasetArgs) -> DatasetTriple:
    assert len(prompts) == len(completions)
    dataset = []
    for prompt, completion in zip(prompts, completions):
        # Get length.
        prompt_len = len(tokenizer(prompt).input_ids)
        output_len = len(tokenizer(completion).input_ids)
        if dataset_args.fixed_output_len is not None:
            output_len = dataset_args.fixed_output_len

        # Prune too short or long sequences
        if (prompt_len < 4 or output_len < 4
                or prompt_len + output_len > dataset_args.max_len):
            continue

        # Make into dataset tripe.
        dataset.append((prompt, prompt_len, output_len))
        if (len(dataset) >= dataset_args.num_samples * 2):
            break

    # Sample num_requests from the list.
    assert dataset_args.num_samples <= len(dataset)
    random.seed(dataset_args.seed)
    return random.sample(dataset, dataset_args.num_samples)


# ultrachat
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
def get_ultrachat(tokenizer: PreTrainedTokenizerBase,
                  dataset_args: DatasetArgs) -> DatasetTriple:
    # Load dataset.
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft[:10%]").shuffle(seed=dataset_args.seed).select(
            range(dataset_args.num_samples))

    # Extract prompt, completion pairs (after adding system prompt to each.)
    prompts = []
    completions = []
    system_message = {
        "content": "You are a chatbot with the explicit goal of "
        "helping the user as best as possible",
        "role": "system",
    }
    for messages in ds["messages"]:
        convo = [system_message]
        convo.extend(messages)

        for i in range(2, len(convo), 2):
            prompts.append(
                tokenizer.apply_chat_template(convo[:i],
                                              tokenize=False,
                                              add_generation_prompt=True))
            completions.append(convo[i]["content"])

    # Convert to dataset triples for consumption by the benchmark scripts.
    return make_dataset_triples(
        prompts=prompts,
        completions=completions,
        tokenizer=tokenizer,
        dataset_args=dataset_args,
    )


# sharegpt
# https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
SHAREGPT_DOWNLOAD_STR = "wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_PATH = "ShareGPT_V3_unfiltered_cleaned_split.json"


def get_sharegpt(tokenizer: PreTrainedTokenizerBase,
                 dataset_args: DatasetArgs) -> DatasetTriple:
    # Load data (possibly downloading first).
    share_gpt_path = Path(SHAREGPT_PATH)
    if not share_gpt_path.exists():
        raise ValueError(f"sharegpt not found. To download, run: \n\n\t{SHAREGPT_DOWNLOAD_STR}")
    assert share_gpt_path.exists()
    with open(share_gpt_path) as f:
        dataset = json.load(f)

    # Extract Prompt / Completion pairs.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]
    prompts = [prompt for prompt, _ in dataset]
    completions = [completion for _, completion in dataset]

    # Convert to dataset triples for consumption by the benchmark scripts.
    return make_dataset_triples(
        prompts=prompts,
        completions=completions,
        tokenizer=tokenizer,
        dataset_args=dataset_args,
    )

# sonnet
SONNET_PATH = "sonnet.txt"
SONNET_CHARS = int(1024 * 3.415) # 3.415 char per token

def get_sonnet(tokenizer: PreTrainedTokenizerBase,
               dataset_args: DatasetArgs) -> DatasetTriple:
     # Load data (possibly downloading first).
    sonnet_path = Path(SONNET_PATH)
    if not sonnet_path.exists():
        raise ValueError(f"Sonnet not found. This should be in your `vllm/benchmarks directory.")
    with open(sonnet_path) as f:
        poem_lines = f.readlines()

    poem = ""
    for poem_line in poem_lines:
        poem += poem_line
    poem_start = poem[:SONNET_CHARS]

    # format into chat convo
    system_message = {
        "content": "You are a chatbot with the explicit goal of "
        "helping the user as best as possible",
        "role": "system",
    }
    convo = [system_message]
    convo.append({
        "content": f"Continue the following poem: \n\n{poem_start}",
        "role": "user",
    }) 
    
    prompt = tokenizer.apply_chat_template(
        convo,
        tokenize=False,
        add_generation_prompt=True)
            
    prompts = [prompt] * dataset_args.num_samples
    completions = [""] * dataset_args.num_samples
    dataset_args.fixed_output_len = 256

    return make_dataset_triples(
        prompts=prompts,
        completions=completions,
        tokenizer=tokenizer,
        dataset_args=dataset_args,
    )


_DATASET_REGISTRY = {
    "sharegpt": get_sharegpt,
    "ultrachat": get_ultrachat,
    "sonnet": get_sonnet,
}


def get_dataset(name: str, tokenizer: PreTrainedTokenizerBase,
                dataset_args: DatasetArgs) -> DatasetTriple:
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"{name} not found in dataset registry: {_DATASET_REGISTRY.keys()}"
        )
    else:
        return _DATASET_REGISTRY[name](tokenizer=tokenizer,
                                       dataset_args=dataset_args)
