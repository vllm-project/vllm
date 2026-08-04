# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import json
import os

# Third Party
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np

_tokenizer: AutoTokenizer | None = None


def estimate_num_tokens(text: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        _tokenizer = AutoTokenizer.from_pretrained(args.model)
    return len(_tokenizer.tokenize(text))


def is_human(human):
    return human in ["human", "user"]


def is_gpt(gpt):
    return gpt in ["gpt", "chatgpt", "bing", "bard"]


def is_system(system):
    return system in ["system"]


def invalid_conversations(conversations):
    # pair does not match
    if len(conversations) < 2:
        return True

    # starting from gpt or systems
    entry = conversations[0]
    if is_gpt(entry["from"]) or is_system(entry["from"]):
        return True

    # ending with human
    entry = conversations[-1]
    if is_human(entry["from"]):
        return True

    prev_from = None
    total_tokens = 0
    for conv in conversations:
        _from = conv["from"]
        total_tokens += estimate_num_tokens(conv["value"])

        # consecutive rounds (gpt followed by gpt, human followed by human, ..)
        if prev_from == _from:
            return True

        # too long conversations
        if total_tokens > (128 * 1024):
            return True

        # unknown from
        if not is_human(_from) and not is_gpt(_from) and not is_system(_from):
            return True

        prev_from = _from

    return False


parser = argparse.ArgumentParser(description="Process data percentage.")
parser.add_argument(
    "--parse",
    type=float,
    default=1,
    help="The percentage of data to process (0 to 1). Default is 1 (100%).",
)
parser.add_argument(
    "--model",
    type=str,
    default="mistralai/Mistral-7B-Instruct-v0.2",
    help="Model for tokenizer. Default is mistralai/Mistral-7B-Instruct-v0.2.",
)
parser.add_argument(
    "--trace",
    type=str,
    default="ShareGPT_V3_unfiltered_cleaned_split.json",
    help="Trace file. Default is ShareGPT_V3_unfiltered_cleaned_split.json",
)

args = parser.parse_args()

print("Loading trace file..")
with open(args.trace, "r", encoding="utf-8") as file:
    data = json.load(file)

num_of_ids = len(data)
print(f"Number of IDs: {num_of_ids}")

# exclude invalid data
print("Veryfing trace..")
data = [d for d in tqdm(data) if not invalid_conversations(d["conversations"])]
excluded_ids = len(data)
num_of_ids -= excluded_ids
print(f"Excluded number of IDs: {excluded_ids}")

data_to_process = int(num_of_ids * args.parse)
data = data[:data_to_process]
print(f"Data to process: {data_to_process}")

for d in tqdm(data):
    conversations = d["conversations"]
    d["num_round"] = len(conversations)  # human is one round, gpt is another round
    human_tokens = []
    gpt_tokens = []
    for conv in conversations:
        num_tokens = estimate_num_tokens(conv["value"])

        if is_human(conv["from"]):
            human_tokens.append(num_tokens)
        elif is_gpt(conv["from"]):
            conv["num_tokens"] = num_tokens
            gpt_tokens.append(num_tokens)
        else:
            print("Invalid _from_")

    if len(human_tokens) == 0:
        d["average_human_token"] = 0
        d["max_human_token"] = 0
    else:
        d["average_human_token"] = float(np.mean(human_tokens))
        d["max_human_token"] = float(np.max(human_tokens))
    if len(gpt_tokens) == 0:
        d["average_gpt_token"] = 0
        d["max_gpt_token"] = 0
    else:
        d["average_gpt_token"] = float(np.mean(gpt_tokens))
        d["max_gpt_token"] = float(np.max(gpt_tokens))

with open("ShareGPT.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
