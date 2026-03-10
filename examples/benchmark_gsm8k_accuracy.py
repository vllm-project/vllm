#!/usr/bin/env python3
"""
Benchmark accuracy of a chat model on GSM8K test set using vLLM's OpenAI-compatible server.

Usage:
    python benchmark_gsm8k_accuracy.py --api-url http://localhost:8004/v1/chat/completions --model phi4-mini
"""

import argparse
import re
import requests
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K accuracy using vLLM chat completions.")
    parser.add_argument("--api-url", type=str, required=True,
                        help="Full URL of the vLLM chat completions endpoint, e.g., http://localhost:8004/v1/chat/completions")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name as served by vLLM (--served-model-name)")
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Number of prompts to evaluate (default: all test set)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate per request")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--hf-endpoint", type=str, default=None,
                        help="HuggingFace mirror endpoint, e.g., https://hf-mirror.com")
    return parser.parse_args()

def extract_answer(response_text):
    """
    Extract the final numeric answer from model output.
    GSM8K reference answers are formatted as "#### <number>".
    """
    # Try to find "####" followed by a number
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', response_text)
    if match:
        return match.group(1)
    # Fallback: last number in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response_text)
    if numbers:
        return numbers[-1]
    return None

def main():
    args = parse_args()

    if args.hf_endpoint:
        import os
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    print("Loading GSM8K test set...")
    ds = load_dataset("gsm8k", "main", split="test")
    problems = list(ds)
    if args.num_prompts:
        problems = problems[:args.num_prompts]
    total = len(problems)
    correct = 0

    print(f"Evaluating {total} examples...\n")

    for idx, problem in enumerate(problems):
        question = problem["question"]
        # Extract gold answer (after "####")
        gold_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', problem["answer"])
        if not gold_match:
            print(f"Warning: cannot parse gold answer for example {idx}, skipping")
            continue
        gold_answer = gold_match.group(1)

        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }

        try:
            response = requests.post(args.api_url, json=payload, timeout=60)
            response.raise_for_status()
            model_output = response.json()["choices"][0]["message"]["content"]
            pred_answer = extract_answer(model_output)

            if pred_answer == gold_answer:
                correct += 1
            else:
                print(f"Example {idx}:")
                print(f"  Q: {question[:60]}...")
                print(f"  Gold: {gold_answer}, Pred: {pred_answer}")
                print(f"  Model output snippet: {model_output[:120]}...\n")
        except Exception as e:
            print(f"Error on example {idx}: {e}")

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{total}, current accuracy: {correct/(idx+1)*100:.2f}%")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*40}")
    print(f"Final accuracy: {correct}/{total} = {accuracy:.2f}%")

if __name__ == "__main__":
    main()
