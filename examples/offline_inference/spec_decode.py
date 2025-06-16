# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.v1.metrics.reader import Counter, Vector

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        default="./examples/data/gsm8k.jsonl",
        help="downloaded from the eagle repo "
        "https://github.com/SafeAILab/EAGLE/blob/main/eagle/data/",
    )
    parser.add_argument(
        "--method", type=str, default="eagle", choices=["ngram", "eagle", "eagle3"]
    )
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--num-spec-tokens", type=int, default=2)
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--output-len", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    args.endpoint_type = "openai-chat"

    model_dir = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    max_model_len = 2048

    prompts = get_samples(args, tokenizer)
    # add_special_tokens is False to avoid adding bos twice when using chat templates
    prompt_ids = [
        tokenizer.encode(prompt.prompt, add_special_tokens=False) for prompt in prompts
    ]

    if args.method == "eagle" or args.method == "eagle3":
        if args.method == "eagle":
            eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
        elif args.method == "eagle3":
            eagle_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        speculative_config = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
            "draft_tensor_parallel_size": args.draft_tp,
            "max_model_len": max_model_len,
        }
    elif args.method == "ngram":
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": args.num_spec_tokens,
            "prompt_lookup_max": args.prompt_lookup_max,
            "prompt_lookup_min": args.prompt_lookup_min,
            "max_model_len": max_model_len,
        }
    else:
        raise ValueError(f"unknown method: {args.method}")

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
        speculative_config=speculative_config,
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.output_len)
    outputs = llm.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)

    # print the generated text
    if args.print_output:
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
    main()
