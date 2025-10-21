import json
import torch
from transformers import AutoTokenizer, set_seed
from datasets import load_dataset
from torch.utils.data import RandomSampler


BASE_MODEL_PATH = "/home/girfan/models/Llama-3.2-1B-Instruct"
DATASET_NAME = "tatsu-lab/alpaca"
DATASET_SIZE = 500
MAX_LENGTH = 512


def format_instruction(example):
    """Format data in Alpaca instruction format"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

    return prompt


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset(DATASET_NAME, split=f"train[:{DATASET_SIZE}]")
split_idx = int(len(dataset) * 0.8)
train_dataset_raw = dataset.select(range(split_idx))
eval_dataset_raw = dataset.select(range(split_idx, len(dataset)))

set_seed(42)
sampler = RandomSampler(train_dataset_raw, generator=torch.Generator().manual_seed(42))
train_indices = list(sampler)

train_data = []
for sampled_idx in train_indices:
    ex = train_dataset_raw[int(sampled_idx)]
    text = format_instruction(ex)
    token_ids = tokenizer.encode(
        text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH
    )
    train_data.append(
        {
            "index": int(sampled_idx),
            "text": text,
            "input_ids": token_ids,
            "length": len(token_ids),
        }
    )

eval_data = []
for idx, ex in enumerate(eval_dataset_raw):
    text = format_instruction(ex)
    token_ids = tokenizer.encode(
        text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH
    )
    eval_data.append(
        {"index": idx, "text": text, "input_ids": token_ids, "length": len(token_ids)}
    )

all_data = train_data + eval_data

print("\nDataset:")
print(f"  Name: ({DATASET_NAME})")
print(f"  Total samples: {len(all_data)}")
print(f"  Training samples: {len(train_data)}")
print(f"  Eval samples: {len(eval_data)}")

output_file = "fixed_test_data.json"
output_data = {
    "config": {
        "base_model": BASE_MODEL_PATH,
        "dataset_name": DATASET_NAME,
        "dataset_size": DATASET_SIZE,
        "max_length": MAX_LENGTH,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    },
    "train": train_data,
    "eval": eval_data,
}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"\nSaved dataset to: {output_file}")