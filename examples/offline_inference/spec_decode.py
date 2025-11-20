# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Vector

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser
from icecream import install
install()

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/lion.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/flycatcher.jpeg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/somefish.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/starfish.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/snail.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/thistle.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/husky.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/orangetabbycat.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/guineapig.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/rabbit.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/horsepony.jpg",
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


def run_mtbench_multiturn(llm, sampling_params, num_prompts, max_num_seqs):
    from datasets import load_dataset
    from tqdm import tqdm
    ds = load_dataset("philschmid/mt-bench", split="train")

    # from fr-spec https://github.com/thunlp/FR-Spec/blob/29d0136b43d372d7d48806db8702cc9c813fdccf/evaluation/mt_bench/eval.py#L97
    MTBENCH_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    )

    outputs = []
    assert max_num_seqs == 1, "only works for max_num_seqs==1 right now"
    total_samples = min(sum([len(data["turns"]) for data in ds]), num_prompts if num_prompts is not None else float('inf'))
    print(f'Running on {total_samples} samples.')

    for i, data in tqdm(enumerate(ds), total=total_samples):
        if i >= total_samples: break
        messages = [{"role": "system", "content": MTBENCH_SYSTEM_PROMPT}]
        for i in range(len(data["turns"])):
            qs = data["turns"][i]
            messages.append({"role": "user", "content": qs})
            output = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)[0]
            outputs.append(output)
            messages.append({
                "role": "assistant",
                "content": output.outputs[0].text
            })
    return outputs


def parse_args():
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)
    parser.add_argument("--test", action="store_true")
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
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--eagle-dir", type=str, default=None)
    parser.add_argument("--custom-mm-prompts", action="store_true")
    return parser.parse_args()


def main(args):
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

    if args.dataset_path == "philschmid/mt-bench-multiturn":
        prompts = None
    elif not args.custom_mm_prompts:
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
    elif args.method == "mtp":
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": args.num_spec_tokens,
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
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 5},
        disable_chunked_mm_input=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.output_len)
    if args.dataset_path == "philschmid/mt-bench-multiturn":
        # warmup run
        print('Warmup start')
        for _ in range(3):
            run_mtbench_multiturn(llm, sampling_params, 1, 1)
        # actually run everything
        print('Warmup end\nStart the actual run')
        outputs = run_mtbench_multiturn(llm, sampling_params, args.num_prompts, args.max_num_seqs)
    elif not args.custom_mm_prompts:
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=x) for x in prompt_ids],
            sampling_params=sampling_params,
        )
    else:
        outputs = llm.chat(prompts, sampling_params=sampling_params)

    # print the generated text
    if args.print_output:
        for i, output in enumerate(outputs):
            prompt = tokenizer.decode(output.prompt_token_ids)
            print("*" * 80)
            print(f"Output {i}:")
            print(f"---Finish reason---\n{output.outputs[0].finish_reason}")
            print(f"---Prompt ({len(output.prompt_token_ids)} tokens)---\n{prompt}")
            print(f"---Generated Text ({len(output.outputs[0].token_ids)} tokens)---\n{output.outputs[0].text}")
            print("*" * 80 + '\n')

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

    return acceptance_length


if __name__ == "__main__":
    args = parse_args()
    acceptance_length = main(args)

    if args.test:
        # takes ~30s to run on 1xH100
        assert args.method in ["eagle", "eagle3"]
        assert args.tp == 1
        assert args.num_spec_tokens == 3
        assert args.dataset_name == "hf"
        assert args.dataset_path == "philschmid/mt-bench"
        assert args.num_prompts == 80
        assert args.temp == 0
        assert args.top_p == 1.0
        assert args.top_k == -1
        assert args.enable_chunked_prefill

        # check acceptance length is within 2% of expected value
        rtol = 0.02
        expected_acceptance_length = 2.296 if args.method == "eagle" else 2.811

        assert (
            acceptance_length <= (1 + rtol) * expected_acceptance_length
            and acceptance_length >= (1 - rtol) * expected_acceptance_length
        ), (
            f"acceptance_length {acceptance_length} is not "
            f"within {rtol * 100}% of {expected_acceptance_length}"
        )

        print(
            f"Test passed! Expected AL: "
            f"{expected_acceptance_length}, got {acceptance_length}"
        )
