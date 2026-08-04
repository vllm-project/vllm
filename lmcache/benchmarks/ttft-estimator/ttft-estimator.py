# SPDX-License-Identifier: Apache-2.0
"""
Startup: 
- serving engine agnostic benchmark but example is with vllm, 
  where we can increase the max model length more by restricting
  the concurrency to 1 since this estimator will only send one request at a time
- load-format dummy to make weight loading faster and we don't care about the
  outputs only the prefill speed
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 130000 --port 8000 \
    --load-format dummy \
    --max-num-seqs 1

Example Usage: 
python ttft-estimator.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host localhost --port 8000\
    --context-lengths 2000,5000,10000,20000,30000,50000,75000,100000,128000
"""

# Standard
import argparse
import time

# Third Party
from openai import OpenAI
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

args = None
tokenizer = None
# the number of tokens in 10,000 "hi"s
hi_multiplier = None
context_length_ttfts = []

client = OpenAI(api_key="dummy-key", base_url="http://localhost:8000/v1")


def query_and_measure_ttft(prompt):
    start = time.perf_counter()
    ttft = None

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=args.model,
        temperature=0.7,
        stream=True,
        max_completion_tokens=5,
    )

    for chunk in chat_completion:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            if ttft is None:
                ttft = time.perf_counter()
            print(chunk_message, end="", flush=True)

    print("\n")
    return ttft - start


def main():
    # first send a dummy request
    # e.g. on vLLM the very first request sometimes has higher
    # TTFT due to kernel JIT compilation
    warm_up_prompt = "bye" * 50
    query_and_measure_ttft(warm_up_prompt)
    print("Warm up complete")

    for i, context_length in enumerate(
        map(int, (s.strip() for s in args.context_lengths.split(",")))
    ):
        number_of_his = context_length * hi_multiplier // 10_000
        # break the prefix with the enumeration
        prompt = f"{i}" + "hi" * number_of_his
        ttft = query_and_measure_ttft(prompt)
        print(f"Context length: {context_length}, TTFT: {ttft}")
        context_length_ttfts.append((context_length, ttft))
    draw_quadratic_interpolation()


def set_hi_multiplier():
    global hi_multiplier
    prompt = "hi" * 10000
    hi_multiplier = len(tokenizer.encode(prompt))
    print(f'number tokens in 10,000 "hi\'s": {hi_multiplier}')


def parse_args():
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, required=False, default="localhost")
    parser.add_argument("--port", type=str, required=False, default="8000")
    parser.add_argument("--context-lengths", type=str, required=False, default="1024")
    return parser.parse_args()


def draw_quadratic_interpolation():
    # Note: this interpolation is completely speculative and should NOT be trusted
    # This is to visualize under the assumption that prefill is quadratic complexity
    xs = [context_length for context_length, _ in context_length_ttfts]
    ys = [ttft for _, ttft in context_length_ttfts]
    # degree 2
    coeffs = np.polyfit(xs, ys, 2)
    print(
        f"Interpolation is: Prefill Time = {coeffs[0]} * Context Length^2 +"
        f"{coeffs[1]} * Context Length + {coeffs[2]}"
    )
    quadratic = np.poly1d(coeffs)

    x_smooth = np.linspace(min(xs), max(xs), 200)
    y_smooth = quadratic(x_smooth)

    plt.scatter(xs, ys, color="red", label="Data Points")
    plt.plot(x_smooth, y_smooth, color="pink", label="Quadratic Interpolation")
    plt.legend()
    plt.xlabel("Context Length")
    plt.ylabel("TTFT (s)")
    plt.title("Quadratic Prefill Estimation")
    plt.savefig("prefill-estimation.png")


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    set_hi_multiplier()
    main()
