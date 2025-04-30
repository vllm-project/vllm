# SPDX-License-Identifier: Apache-2.0
r"""Benchmark offline throughput with structured outputs.

Usage:
    python benchmarks/benchmark_offline_structured_output.py \
        --model <your_model> \
        --dataset json \
        --structured-output-ratio 1.0 \
        --structured-output-backend auto \
        --request-rate 10 \
        --num-prompts 1000

"""
import argparse
import random

import numpy as np
from benchmark_serving_structured_output import (
    SampleRequest, fill_structure_type, get_structured_output_argparser,
    get_tokenizer_from_args, sample_requests)
from transformers import PreTrainedTokenizerBase

from vllm import LLM
from vllm.sampling_params import GuidedDecodingParams, SamplingParams


def benchmark_sync(args: argparse.Namespace,
                   tokenizer: PreTrainedTokenizerBase,
                   input_requests: list[SampleRequest]) -> list[str]:
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer if args.tokenizer is not None else args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    def _to_vllm_guided_decoding_params(
            args: argparse.Namespace,
            request: SampleRequest) -> GuidedDecodingParams:
        kwargs = {
            "backend": args.structured_output_backend,
            args.dataset: request.schema
        }
        return GuidedDecodingParams(**kwargs)

    outputs = llm.generate(
        prompts=[req.prompt for req in input_requests],
        sampling_params=[
            SamplingParams(guided_decoding=_to_vllm_guided_decoding_params(
                args, req), ) for req in input_requests
        ],
        use_tqdm=not args.disable_tqdm)
    return [output.outputs[0].text for output in outputs]


def benchmark_async(backend: str, api_url: str, base_url: str, model_id: str):
    pass


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = get_tokenizer_from_args(args)
    fill_structure_type(args)

    input_requests = sample_requests(tokenizer, args)

    benchmark_sync(args=args,
                   tokenizer=tokenizer,
                   input_requests=input_requests)


if __name__ == "__main__":
    parser = get_structured_output_argparser()
    parser.add_argument("--async",
                        action="store_true",
                        help="Run the benchmark in async mode.")
    args = parser.parse_args()
    main(args)
