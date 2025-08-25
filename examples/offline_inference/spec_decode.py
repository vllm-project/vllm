# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.v1.metrics.reader import Counter, Vector

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


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
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--eagle-dir", type=str, default=None)
    parser.add_argument("--custom-mm-prompts", action="store_true")
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

    if args.method == "eagle" or args.method == "eagle3":
        eagle_dir = args.eagle_dir
        if args.method == "eagle" and eagle_dir is None:
            eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

        elif args.method == "eagle3" and eagle_dir is None:
            eagle_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        speculative_config = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
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

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=16384,
        limit_mm_per_prompt={"image": 5},
        disable_chunked_mm_input=True,
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.output_len)
    if not args.custom_mm_prompts:
        outputs = llm.generate(
            prompt_token_ids=prompt_ids, sampling_params=sampling_params
        )
    else:
        outputs = llm.chat(prompts, sampling_params=sampling_params)

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

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * args.num_spec_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # print acceptance at each token position
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
