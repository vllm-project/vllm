# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import json
import os
import time

# Third Party
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

# vLLM
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import numpy as np
import pandas as pd

# setting PYTHONHASHSEED derandomizes token chunking
os.environ["PYTHONHASHSEED"] = "0"

global tokenizer
choices = ["A", "B", "C", "D"]


# grab the idx'th row of the df and generate a prompt string
# format of the MMLU csvs:
# question,option_A,option_B,option_C,option_D,answer
def prompt_string(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2  # number of columns - 2 (question and answer)
    for i in range(k):
        prompt += f"\n{choices[i]}. {df.iloc[idx, i + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k]}\n\n"
    return prompt


def evaluate(args, llm, subject, dev_df, test_df):
    prompts, labels = [], []

    shared_multi_shot_prefix = [
        f'The following are multiple choice questions (with answers) \
                    about {subject}. \
                    You must respond with a single letter only. \
                    Either "A", "B", "C", or "D". \
                    Do not include any other text in your response. \n\n'
    ]
    shared_multi_shot_prefix_length = 0
    for i in range(dev_df.shape[0]):
        # the multi-shot examples should contain answers
        shared_multi_shot_prefix.append(prompt_string(dev_df, i))

        # Use plain list of token IDs, no torch tensors
        token_ids = tokenizer(shared_multi_shot_prefix[-1], add_special_tokens=True)[
            "input_ids"
        ]
        shared_multi_shot_prefix_length += len(token_ids)

        if shared_multi_shot_prefix_length > 4000:
            break

    # all already have double newlines at the end
    shared_multi_shot_prefix = "".join(shared_multi_shot_prefix)

    for i in range(test_df.shape[0]):
        # do NOT include the answer for the actual question
        # we want the LLM to answer
        query_prompt = prompt_string(test_df, i, include_answer=False)
        prompt = f"{shared_multi_shot_prefix}\n\n{query_prompt}"
        prompts.append(prompt)
        label = test_df.iloc[i, test_df.shape[1] - 1]
        labels.append(label)

    # Create sampling params with deterministic settings
    # (temperature=0, seed=42)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2,
        seed=42,
        n=1,
        stop=None,
    )

    # even though offline serving can batch all the prompts, we do them one at a time
    # to keep the vllm scheduler deterministic
    outputs = []
    for prompt in prompts:
        # if we use lmcache, we need to first populate the kv cache
        if args.use_lmcache:
            llm.generate(prompt, sampling_params)
            time.sleep(0.5)
        outputs.append(llm.generate(prompt, sampling_params))
        time.sleep(0.5)

    predictions = []
    for output in outputs:
        prediction = output.outputs[0].text
        prediction_stripped = prediction.strip()
        if prediction_stripped and prediction_stripped[0] in choices:
            predictions.append(prediction_stripped[0])
        else:
            # Fallback: look for any A, B, C, D in the response
            for char in prediction_stripped:
                if char in ["A", "B", "C", "D"]:
                    predictions.append(char)
                    break
            else:
                predictions.append("A")  # Default fallback

    accuracy = np.mean(np.array(predictions) == np.array(labels))
    return accuracy


def main(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ktc = None
    if args.use_lmcache:
        ktc = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
        )
    llm = LLM(
        model=args.model,
        max_model_len=8000,
        gpu_memory_utilization=0.9,
        kv_transfer_config=ktc,
        tensor_parallel_size=2,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
    )

    mmlu_files = os.listdir("data/test")
    test_files = [f for f in mmlu_files if f.endswith("_test.csv")]
    subjects = sorted([f.split("_test.csv")[0] for f in test_files])

    accuracies = []
    num_questions = []
    output_dict = {}

    for subject_raw in tqdm(
        subjects[: args.number_of_subjects],
        desc="Processing subjects",
    ):
        subject = " ".join(subject_raw.split("_"))
        dev_df = pd.read_csv(
            os.path.join("data/dev", subject_raw + "_dev.csv"),
            header=None,
        )
        test_df = pd.read_csv(
            os.path.join("data/test", subject_raw + "_test.csv"),
            header=None,
        )
        accuracy = evaluate(args, llm, subject, dev_df, test_df)
        accuracies.append(accuracy)
        num_questions.append(len(test_df))
        output_dict[subject_raw] = {
            "accuracy": accuracy,
            "num_questions": len(test_df),
        }

    total_accuracy = np.mean(accuracies)
    total_num_questions = sum(num_questions)
    output_dict["total"] = {
        "accuracy": total_accuracy,
        "num_questions": total_num_questions,
    }

    with open(args.result_file, "w") as f:
        # output will be a jsonl file
        for subject, value in output_dict.items():
            f.write(json.dumps({subject: value}) + "\n")


if __name__ == "__main__":
    set_seed(42)  # some tokenizers may have randomness
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--number-of-subjects", type=int, default=25)
    parser.add_argument("--use-lmcache", action="store_true", default=False)
    args = parser.parse_args()
    if args.use_lmcache:
        args.result_file = args.model.split("/")[-1] + "-lmcache.jsonl"
    else:
        args.result_file = args.model.split("/")[-1] + "-baseline.jsonl"
    main(args)
