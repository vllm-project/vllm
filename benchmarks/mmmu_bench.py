r"""Benchmark offline inference throughput with MMMU-PRO Vision

e.g, 
python3 benchmarks/mmmu_bench.py \
    --model mistralai/Pixtral-12B-2409 \
    --tokenizer-model mistral \
    --num-prompts 1000

python3 benchmarks/mmmu_bench.py \
    --model allenai/Molmo-72B-0924 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --num-prompts 1000

"""
import argparse
import base64
import dataclasses
import io
import random
import time

from datasets import load_dataset
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.utils import FlexibleArgumentParser


def sample_mmmu_pro_vision_requests(
    dataset,
    num_requests: int,
    image_hit_rate: float,
):

    sampled_requests = []

    num_unique_images = int(num_requests * (1 - image_hit_rate))
    print(
        f"Total {num_requests} requests with {num_unique_images} unique images"
    )
    dataset = dataset.take(num_unique_images)

    for data in dataset:
        if len(sampled_requests) == num_requests:
            break

        # MMMU-Pro vision direct prompt
        # Ref: https://github.com/MMMU-Benchmark/MMMU/blob/6ce42f4d8f70c1841c67867152648974415b5cac/mmmu-pro/prompts.yaml#L5
        prompt = (
            "Answer with the option letter from the given choices directly. "
            "The last line of your response should be of the following "
            "format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
            "options.")

        image: Image = data["image"]
        image = image.convert("RGB")
        image_data = io.BytesIO()
        image.save(image_data, format='JPEG')
        image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        mm_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }

        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                mm_content,
            ],
        }]
        sampled_requests.append(messages)

    return sampled_requests


def sample_hf_requests(
    num_requests: int,
    random_seed: int,
    image_hit_rate: float,
):

    dataset = load_dataset('MMMU/MMMU_Pro',
                           name='vision',
                           split="test",
                           streaming=True)
    dataset = dataset.shuffle(seed=random_seed)
    return sample_mmmu_pro_vision_requests(dataset, num_requests,
                                           image_hit_rate)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    engine_args = EngineArgs.from_cli_args(args)
    sampled = sample_hf_requests(args.num_prompts, args.seed,
                                 args.image_hit_rate)
    llm = LLM(**dataclasses.asdict(engine_args))
    sampling_params = SamplingParams(max_tokens=args.output_len, temperature=0)
    st = time.perf_counter()
    outputs = llm.chat(sampled, sampling_params=sampling_params)
    duration = time.perf_counter() - st

    total_generated_tokens = 0
    for output in outputs:
        total_generated_tokens += len(output.outputs[0].token_ids)

    print(f"Request throughput: {args.num_prompts / duration:.2f} req/s")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(
        f"Token generation rate: {total_generated_tokens / duration:.2f} tok/s"
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--output-len",
                        type=int,
                        default=128,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--image-hit-rate",
                        type=float,
                        default=0.0,
                        help="Image hit rate between 0 and 1.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    main(args)
