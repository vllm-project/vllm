# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector


def load_prompts(dataset_path, num_prompts):
    if os.path.exists(dataset_path):
        prompts = []
        try:
            with open(dataset_path) as f:
                for line in f:
                    data = json.loads(line)
                    prompts.append(data["turns"][0])
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return []
    else:
        prompts = ["The future of AI is", "The president of the United States is"]

    return prompts[:num_prompts]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="./examples/data/gsm8k.jsonl",
        help="downloaded from the eagle repo "
        "https://github.com/SafeAILab/EAGLE/blob/main/eagle/data/",
    )
    parser.add_argument(
        "--method", type=str, default="eagle", choices=["eagle", "eagle3"]
    )
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--num_prompts", type=int, default=80)
    parser.add_argument("--num_spec_tokens", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--draft_tp", type=int, default=1)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--enable_chunked_prefill", action="store_true")
    parser.add_argument("--max_num_batched_tokens", type=int, default=2048)
    parser.add_argument("--temp", type=float, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = "meta-llama/Llama-3.1-8B-Instruct"

    if args.method == "eagle":
        eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
    elif args.method == "eagle3":
        eagle_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    else:
        raise ValueError(f"unknown method: {args.method}")

    max_model_len = 2048

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    prompts = load_prompts(args.dataset, args.num_prompts)

    prompt_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        for prompt in prompts
    ]

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.8,
        speculative_config={
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
            "draft_tensor_parallel_size": args.draft_tp,
            "max_model_len": max_model_len,
        },
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=256)

    outputs = llm.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)

    # print the generated text
    for output in outputs:
        print("-" * 50)
        print(f"prompt: {output.prompt}")
        print(f"generated text: {output.outputs[0].text}")
        print("-" * 50)

    try:
        metrics = llm.get_metrics()
    except AssertionError:
        print("Metrics are not supported in the V0 engine.")
        return

    num_drafts = num_accepted = 0
    acceptance_counts = [0] * args.num_spec_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    print(f"mean acceptance length: {1 + (num_accepted / num_drafts):.2f}")
    print("-" * 50)

    # print acceptance at each token position
    for i in range(len(acceptance_counts)):
        print(f"acceptance at token {i}:{acceptance_counts[i] / num_drafts:.2f}")


if __name__ == "__main__":
    print(
        "[WARNING] Use examples/offline_inference/spec_decode.py"
        " instead of this script."
    )
    main()
