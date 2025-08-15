# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/MMMU-Benchmark/MMMU

"""Response Parsing and Evaluation for various models"""

import gc
import json
import os
import random
import re
from typing import Any, Callable

import numpy as np
from data_utils import (
    construct_prompt,
    load_mmmu_dataset,
    load_yaml,
    process_single_sample,
)
from tqdm import tqdm


# ----------- Default Configuration -------------
class BenchmarkDefaults:
    """Default values for benchmark parameters"""

    # Dataset parameters
    SPLIT = "validation"
    SUBJECT = None
    MAX_SAMPLES = -1
    CONFIG_PATH = "eval_config.yaml"

    # Generation parameters
    SEED = 42
    TEMPERATURE = 0.01
    TOP_P = 0.9
    TOP_K = None
    MAX_TOKENS = 512
    DO_SAMPLE = True

    # Benchmark parameters
    BATCH_SIZE = 1
    OUTPUT_PATH_HF = "benchmark_results_hf.json"
    OUTPUT_PATH_VLLM = "benchmark_results_vllm.json"

    # vLLM specific defaults
    MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

    @classmethod
    def get_common_args_dict(cls):
        """Get common arguments as a dictionary"""
        return {
            "split": cls.SPLIT,
            "subject": cls.SUBJECT,
            "max_samples": cls.MAX_SAMPLES,
            "config_path": cls.CONFIG_PATH,
            "seed": cls.SEED,
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P,
            "top_k": cls.TOP_K,
            "max_tokens": cls.MAX_TOKENS,
            "do_sample": cls.DO_SAMPLE,
            "batch_size": cls.BATCH_SIZE,
        }

    @classmethod
    def get_hf_args_dict(cls):
        """Get HuggingFace specific arguments"""
        args = cls.get_common_args_dict()
        args["output_path"] = cls.OUTPUT_PATH_HF
        return args

    @classmethod
    def get_vllm_args_dict(cls):
        """Get vLLM specific arguments"""
        args = cls.get_common_args_dict()
        args["model"] = cls.MODEL
        args["top_p"] = cls.TOP_P
        args["output_path"] = cls.OUTPUT_PATH_VLLM
        return args


# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content
    # is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes =
                # [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


# ----------- Process Open -------------
def check_is_number(string):
    """
    Check if the given string a number.
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """

    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses: list[str] = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation
            # (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            # the shortest response that may contain
            # the answer (tail part of the response)
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(
                            shortest_key_response
                        ):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())

            if shortest_key_response and shortest_key_response.strip() not in [
                ":",
                ",",
                ".",
                "!",
                "?",
                ";",
                ":",
                "'",
            ]:
                key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


# ----------- Evaluation -------------


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


# ----------- Batch Evaluation -------------
def evaluate(samples):
    """
    Batch evaluation for multiple choice and open questions.
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        if sample["question_type"] == "multiple-choice":
            correct = eval_multi_choice(gold_i, pred_i)
        else:  # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


# ----------- Calculate Accuracy -------------
def calculate_ins_level_acc(results: dict):
    """Calculate the instruction level accuracy for given Subject results"""
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


# ----------- Common Benchmark Logic -------------
def run_benchmark(
    samples: list[dict],
    config: dict,
    args: Any,
    generate_func: Callable[[list[str]], list[str]],
    batch_size: int = 1,
    subject: str | None = None,
    output_path: str = "benchmark_results.json",
    model_info: dict | None = None,
) -> dict:
    """
    Common benchmark logic for processing samples and evaluating results.

    Args:
        samples: List of dataset samples
        config: Evaluation configuration
        args: Arguments object containing generation parameters
        generate_func: Function that takes (prompts) and returns responses
        batch_size: Batch size for processing
        subject: Subject name for filtering results
        output_path: Path to save results
        model_info: Additional model information to save

    Returns:
        dictionary containing results, metrics, and other information
    """
    results = []

    # Set fixed seed for reproducibility
    if hasattr(args, "seed"):
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Process samples in batches
    batch_count = (len(samples) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(samples), batch_size),
        desc="Processing batches",
        total=batch_count,
        unit="batch",
    ):
        batch_samples = samples[i : i + batch_size]
        batch_prompts = []

        # Prepare batch prompts and images
        batch_prompts = []
        batch_images = []
        for sample in batch_samples:
            prompt_data = construct_prompt(sample, config)
            prompt = prompt_data["final_input_prompt"]
            batch_prompts.append(prompt)
            batch_images.append(sample.get("image"))  # Get image data if available

            # Store prompt data for later use
            sample["_prompt_data"] = prompt_data
            sample["_prompt"] = prompt

        # Generate responses using the provided function
        # Check if generate_func accepts images parameter (for vLLM) or not (for HF)
        try:
            responses = generate_func(batch_prompts, batch_images)
        except TypeError:
            # Fallback for functions that only accept prompts
            responses = generate_func(batch_prompts)

        # Process outputs
        for j, response in enumerate(responses):
            sample = batch_samples[j]
            prompt_data = sample["_prompt_data"]

            # Parse response based on question type
            if sample["question_type"] == "multiple-choice":
                parsed_pred = parse_multi_choice_response(
                    response, prompt_data["all_choices"], prompt_data["index2ans"]
                )
            else:
                parsed_pred = parse_open_response(response)

            # Store results
            result = {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "question_type": sample["question_type"],
                "response": response,
                "parsed_pred": parsed_pred,
                "prompt": sample["_prompt"],
                "subject": sample.get("subject", "unknown"),
            }
            results.append(result)

        # Clean up memory periodically
        if i % (batch_size * 10) == 0:
            gc.collect()

    # Evaluate results
    judge_dict, metrics = evaluate(results)

    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['acc']:.4f}")

    # Group results by subject if multiple subjects
    if subject is None:
        subject_results: dict[str, list[dict]] = {}
        for result in results:
            subj = result.get("subject", "unknown")
            if subj not in subject_results:
                subject_results[subj] = []
            subject_results[subj].append(result)

        print("\nResults by Subject:")
        for subj, subject_samples in subject_results.items():
            subject_judge_dict, subject_metrics = evaluate(subject_samples)
            print(
                f"{subj}: {subject_metrics['acc']:.4f} ({len(subject_samples)} samples)"
            )

    # Prepare final results
    final_results = {
        "results": results,
        "metrics": metrics,
        "judge_dict": judge_dict,
        "args": {},
    }

    # Add model info and args
    if model_info:
        final_results["args"].update(model_info)

    if hasattr(args, "__dict__"):
        # Add relevant args
        for attr in [
            "seed",
            "max_samples",
            "temperature",
            "top_p",
            "max_tokens",
            "max_new_tokens",
        ]:
            if hasattr(args, attr):
                final_results["args"][attr] = getattr(args, attr)

    # Save results
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved to {output_path}")

    return final_results


def load_benchmark_dataset(
    split: str = "validation", subject: str | None = None, max_samples: int = -1
):
    """
    Load and prepare MMMU dataset for benchmarking.

    Args:
        split: Dataset split to use
        subject: Specific subject to evaluate
        max_samples: Maximum number of samples to process (-1 for all)

    Returns:
        List of processed samples
    """
    print("Loading MMMU dataset from HuggingFace Hub...")
    print(f"Split: {split}, Subject: {subject}")

    dataset = load_mmmu_dataset(subset=split, subject=subject)

    # Convert dataset samples to our format
    samples = []
    for sample in dataset:
        samples.append(process_single_sample(sample))

    # Limit number of samples if specified
    if max_samples > 0:
        samples = samples[:max_samples]

    print(f"Processing {len(samples)} samples...")
    return samples


def get_message(prompt, image):
    split_prompt = prompt.split("<image 1>")
    content = [{"type": "text", "text": s} for s in split_prompt]
    content.insert(1, {"type": "image", "image": image} if image is not None else None)
    messages = [{"role": "user", "content": content}]
    if image is None:
        messages[0]["content"] = [{"type": "text", "text": prompt}]
    return messages


def load_benchmark_config(config_path: str = "eval_config.yaml"):
    """
    Load evaluation configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        config = load_yaml(config_path)
    else:
        # Default config
        config = {
            "multi_choice_example_format": "Question: {}\nOptions:\n{}\nAnswer:",
            "short_ans_example_format": "Question: {}\nAnswer:",
            "task_instructions": "Please answer the following\
                question based on the given information.",
        }
    for key, value in config.items():
        if key != "eval_params" and isinstance(value, list):
            assert len(value) == 1, "key {} has more than one value".format(key)
            config[key] = value[0]
    return config


def add_common_benchmark_args(parser, framework: str = "common"):
    """
    Add common benchmark arguments to a parser.

    Args:
        parser: ArgumentParser instance
        framework: "hf", "vllm", or "common"
    """
    defaults = BenchmarkDefaults()

    # Dataset arguments
    benchmark_group = parser.add_argument_group("Benchmark parameters")
    benchmark_group.add_argument(
        "--model", type=str, default=defaults.MODEL, help="model name"
    )
    benchmark_group.add_argument(
        "--split",
        type=str,
        default=defaults.SPLIT,
        choices=["validation", "test", "dev"],
        help="Dataset split to use",
    )
    benchmark_group.add_argument(
        "--subject",
        type=str,
        default=defaults.SUBJECT,
        help="Specific subject to evaluate (e.g., 'Art', 'Biology')."
        "If None, evaluates all subjects",
    )
    benchmark_group.add_argument(
        "--max-samples",
        type=int,
        default=defaults.MAX_SAMPLES,
        help="Maximum number of samples to process (-1 for all)",
    )
    benchmark_group.add_argument(
        "--config-path",
        type=str,
        default=defaults.CONFIG_PATH,
        help="Path to evaluation config file",
    )
    benchmark_group.add_argument(
        "--seed",
        type=int,
        default=defaults.SEED,
        help="Random seed for reproducibility",
    )

    # Generation arguments
    sampling_group = parser.add_argument_group("Generation parameters")
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=defaults.TEMPERATURE,
        help="Temperature for sampling (0.0 = deterministic)",
    )
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        default=defaults.MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=defaults.TOP_P,
        help="Top-p (nucleus) sampling parameter",
    )
    sampling_group.add_argument(
        "--top-k", type=int, default=defaults.TOP_K, help="Top-k sampling parameter"
    )

    if framework == "hf":
        # HuggingFace specific args
        benchmark_group.add_argument(
            "--output-path",
            type=str,
            default=defaults.OUTPUT_PATH_HF,
            help="Path to save the results",
        )
        sampling_group.add_argument(
            "--do-sample",
            action="store_true",
            default=defaults.DO_SAMPLE,
            help="Whether to use sampling (vs greedy decoding)",
        )
    elif framework == "vllm":
        # vLLM specific args
        benchmark_group.add_argument(
            "--output-path",
            type=str,
            default=defaults.OUTPUT_PATH_VLLM,
            help="Path to save the results",
        )
        benchmark_group.add_argument(
            "--batch-size",
            type=int,
            default=defaults.BATCH_SIZE,
            help="Batch size for inference",
        )

    return parser
