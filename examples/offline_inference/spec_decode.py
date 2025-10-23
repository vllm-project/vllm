# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
import pathlib
import itertools

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Vector, Histogram

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

# create output directory
from datetime import datetime
outputs_dir = pathlib.Path("outputs/") / datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir.mkdir(parents=True, exist_ok=True)
(outputs_dir / "drafter.csv").touch()
(outputs_dir / "target.csv").touch()

def read_stats(path):
    forward_times, shapes = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            forward_times.append(float(parts[0]))
            shapes.append(parts[1])
    return forward_times, shapes

def print_dict(stats, file=None, newlines=[]):
  if file is None:
    for i, (k, v) in enumerate(stats.items()):
        print(f"{k:<50}{v}")
        if i in newlines: print()
  else:
    file.touch()
    with open(file, 'a') as f:
      for k, v in stats.items():
        f.write(json.dumps({k: v}) + '\n')

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/2/26/Ultramarine_Flycatcher_%28Ficedula_superciliaris%29_Naggar%2C_Himachal_Pradesh%2C_2013_%28cropped%29.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg/2560px-Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d4/Starfish%2C_Caswell_Bay_-_geograph.org.uk_-_409413.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/69/Grapevinesnail_01.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Texas_invasive_Musk_Thistle_1.jpg/1920px-Texas_invasive_Musk_Thistle_1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Huskiesatrest.jpg/2880px-Huskiesatrest.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/1920px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/30/George_the_amazing_guinea_pig.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Oryctolagus_cuniculus_Rcdo.jpg/1920px-Oryctolagus_cuniculus_Rcdo.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/98/Horse-and-pony.jpg",
]


def get_custom_mm_prompts(num_prompts):
    prompts = []
    for url in IMAGE_URLS:
        prompts.append(
            [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": QUESTION},
            ]
        )
    if num_prompts > len(IMAGE_URLS):
        prompts = prompts * (num_prompts // len(IMAGE_URLS) + 1)

    return [[{"role": "user", "content": prompt}] for prompt in prompts[:num_prompts]]


def parse_args():
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)
    parser.add_argument(
        "--method",
        type=str,
        default="eagle",
        choices=["ngram", "eagle", "eagle3", "mtp"],
    )
    parser.add_argument("--num-spec-tokens", type=int, default=2)
    parser.add_argument("--spec-token-tree", type=str, default=None)
    parser.add_argument("--spec-token-tree-depth", type=int, default=None)
    parser.add_argument("--spec-token-tree-branching", type=int, default=None)
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--eagle-dir", type=str, default=None)
    parser.add_argument("--custom-mm-prompts", action="store_true")
    parser.add_argument("--draft-vocab-frequency-path", type=str, default=None)
    parser.add_argument("--draft-vocab-frequency-keep-threshold", type=str, default=None)
    parser.add_argument("--compilation-config", type=str, default="")
    return parser.parse_args()

def main():
    args = parse_args()
    args.endpoint_type = "openai-chat"

    model_dir = args.model_dir
    if args.model_dir is None:
        if args.custom_mm_prompts:
            raise ValueError(
                "custom_mm_prompts requires mm based models"
                "default llama3.1-8b-instruct is not mm based"
                "please specify model_dir to give a mm based model"
            )
        model_dir = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    args.custom_skip_chat_template = True

    if not args.custom_mm_prompts:
        prompts = get_samples(args, tokenizer)
        # add_special_tokens is False to avoid adding bos twice
        # when using chat templates
        prompt_ids = [
            tokenizer.encode(prompt.prompt, add_special_tokens=False)
            for prompt in prompts
        ]
    else:
        prompts = get_custom_mm_prompts(args.num_prompts)
    ic(len(prompts), prompts)

    # manually specify the speculative token tree
    if args.spec_token_tree is not None:
        assert args.spec_token_tree_depth is None and args.spec_token_tree_branching is None, \
            "If using spec_token_tree, cannot also use spec token tree depth+branching"
        spec_token_tree = ast.literal_eval(args.spec_token_tree)
        assert args.num_spec_tokens == len(spec_token_tree), f'expected `len(spec_token_tree) == num_spec_tokens` but got {len(spec_token_tree)=} and {args.num_spec_tokens=}'
        spec_token_tree_str = str(sorted(spec_token_tree, key=lambda t: (len(t), t)))
    # construct a complete speculative token tree from depth, branch args
    elif args.spec_token_tree_depth is not None or args.spec_token_tree_branching is not None and not (args.spec_token_tree_depth is None and args.spec_token_tree_branching is None):
        assert args.spec_token_tree is None, "If using spec token tree depth+branching, cannot also use spec_token_tree"
        if args.spec_token_tree_depth is None: args.spec_token_tree_depth = 1
        if args.spec_token_tree_branching is None: args.spec_token_tree_branching = 1
        spec_token_tree = []
        depth, branching = args.spec_token_tree_depth, args.spec_token_tree_branching
        for d in range(1, depth + 1):
            for path in itertools.product(range(branching), repeat=d):
                spec_token_tree.append(path)
        if args.num_spec_tokens is None:
            args.num_spec_tokens = len(spec_token_tree)
        print(spec_token_tree)
        assert args.num_spec_tokens == len(spec_token_tree), f'expected `len(spec_token_tree) == num_spec_tokens` but got {len(spec_token_tree)=} and {args.num_spec_tokens=}'
        spec_token_tree_str = str(sorted(spec_token_tree, key=lambda t: (len(t), t)))
    else:
        spec_token_tree_str = None
    ic(args.num_spec_tokens, spec_token_tree_str)

    # vanilla inference if num_spec_tokens == 0
    if args.num_spec_tokens == 0:
        speculative_config = None
        print('Ignore speculative decoding when `args.num_spec_tokens == 0`.')
    elif args.method == "eagle":
        eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B" if args.eagle_dir is None else args.eagle_dir
        speculative_config = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
            "spec_token_tree": spec_token_tree_str,
            "draft_vocab_frequency_path": args.draft_vocab_frequency_path,
            "draft_vocab_frequency_keep_threshold": args.draft_vocab_frequency_keep_threshold,
        }
    elif args.method == "eagle3":
        eagle_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" if args.eagle_dir is None else args.eagle_dir
        speculative_config = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
            "spec_token_tree": spec_token_tree_str,
        }
    elif args.method == "ngram":
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": args.num_spec_tokens,
            "prompt_lookup_max": args.prompt_lookup_max,
            "prompt_lookup_min": args.prompt_lookup_min,
        }
    else:
        raise ValueError(f"unknown method: {args.method}")

    # save args
    print_dict({str(k): str(v) for k, v in vars(args).items()}, outputs_dir / "args.jsonl")

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=8192,
        seed=0,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 5},
        disable_chunked_mm_input=True,
        compilation_config=(
            json.loads(args.compilation_config) if args.compilation_config else None
        ),
    )

    # print out batch size
    scheduler_config = llm.llm_engine.vllm_config.scheduler_config
    ic(scheduler_config.max_num_seqs, scheduler_config.max_num_batched_tokens, scheduler_config.max_model_len)

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.output_len)
    if not args.custom_mm_prompts:
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=x) for x in prompt_ids],
            sampling_params=sampling_params,
        )
    else:
        outputs = llm.chat(prompts, sampling_params=sampling_params)

    # import Counter in the function b/c vllm has a seperate Counter object
    def get_finish_reason_counts(outputs):
        from collections import Counter
        finish_reasons = [output.outputs[0].finish_reason for output in outputs]
        return Counter(finish_reasons)
    finish_reason_counts = get_finish_reason_counts(outputs)
    print(f"Finish Reasons: {finish_reason_counts}")

    # print the generated text
    if args.print_output:
        for i, output in enumerate(outputs):
            ic(output)
            prompt = tokenizer.decode(output.prompt_token_ids)
            print("*" * 150)
            print(f"Output {i}:")
            print(f"---Finish reason---\n{output.outputs[0].finish_reason}")
            print(f"---Prompt ({len(output.prompt_token_ids)} tokens)---\n{prompt}")
            print(f"---Generated Text ({len(output.outputs[0].token_ids)})---\n{output.outputs[0].text}")
            print("*" * 150 + '\n')

    try:
        metrics = llm.get_metrics()
    except AssertionError:
        print("Metrics are not supported in the V0 engine.")
        return

    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    input_time = 0.0
    output_time = 0.0
    drafts = 0
    draft_tokens = 0
    accepted_tokens = 0
    input_tokens = 0
    requests = 0
    acceptance_counts = [0] * args.num_spec_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]
        elif metric.name == "vllm:prompt_tokens":
            assert isinstance(metric, Counter)
            input_tokens += metric.value
        elif metric.name == "vllm:request_prefill_time_seconds":
            assert isinstance(metric, Histogram)
            input_time += metric.sum
        elif metric.name == "vllm:request_decode_time_seconds":
            assert isinstance(metric, Histogram)
            output_time += metric.sum
        elif metric.name == "vllm:request_success":
            assert isinstance(metric, Counter)
            requests += metric.value

    # Calculate metrics
    tokens = input_tokens + output_tokens
    total_time = input_time + output_time # measured in seconds

    input_throughput = input_tokens / input_time if input_time > 0 else 0
    output_throughput = output_tokens / output_time if output_time > 0 else 0
    total_throughput = tokens / total_time

    mean_acceptance_length = 1 + (accepted_tokens / drafts) if drafts > 0 else 1
    draft_utilization_rate = accepted_tokens / draft_tokens * 100 if draft_tokens > 0 else 0

    drafter_forward_times, _ = read_stats(outputs_dir / "drafter.csv")
    target_forward_times, _ = read_stats(outputs_dir / "target.csv")

    drafter_forward_time = sum(drafter_forward_times)
    target_forward_time = sum(target_forward_times)
    forward_ratio = drafter_forward_time / target_forward_time if target_forward_time > 0 else 0

    stats = {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "input_time": input_time, "output_time": output_time, "total_time": total_time,
        "drafter_forward_time": drafter_forward_time, "target_forward_time": target_forward_time, "forward_ratio": forward_ratio,
        "input_throughput": input_throughput, "output_throughput": output_throughput, "total_throughput": total_throughput,
        "drafts": drafts, "draft_tokens": draft_tokens, "draft_utilization_rate": draft_utilization_rate,
        "accepted_tokens": accepted_tokens, "mean_acceptance_length": mean_acceptance_length
    }

    # print stats to stdout and save to file
    print_dict(stats, newlines=[1, 4, 7, 10, 13, 16])
    print_dict(stats, file=outputs_dir / "stats.jsonl")


if __name__ == "__main__":
    main()
