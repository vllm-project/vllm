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
from time import perf_counter

import numpy as np
from backend_request_func import RequestFuncOutput
from benchmark_serving_structured_output import (
    SampleRequest, calculate_metrics, evaluate, fill_structure_type,
    get_structured_output_argparser, get_tokenizer_from_args,
    print_metrics_to_console, sample_requests)
from transformers import PreTrainedTokenizerBase

from vllm import LLM
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams, SamplingParams


def _to_request_output(
        vllm_outputs: list[RequestOutput]) -> list[RequestFuncOutput]:
    return [
        RequestFuncOutput(
            generated_text=output.outputs[0].text,
            success=output.finished,
            latency=(output.metrics.finished_time -
                     output.metrics.arrival_time) if output.metrics else 0.0,
            output_tokens=len(output.outputs[0].token_ids),
            ttft=(output.metrics.first_token_time -
                  output.metrics.arrival_time) if output.metrics else 0.0,
        ) for output in vllm_outputs
    ]


def benchmark_sync(
    args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest]
) -> tuple[list[RequestFuncOutput], float]:
    """
    Benchmark synchronous offline vLLM with structured outputs.

    Returns:
        A tuple of (output texts, benchmark duration in seconds).
    """
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

    start = perf_counter()
    outputs = llm.generate(
        prompts=[req.prompt for req in input_requests],
        sampling_params=[
            SamplingParams(
                ignore_eos=args.ignore_eos,
                guided_decoding=_to_vllm_guided_decoding_params(args, req),
            ) for req in input_requests
        ],
        use_tqdm=not args.disable_tqdm)
    end = perf_counter()
    return outputs, end - start


def benchmark_async(backend: str, api_url: str, base_url: str, model_id: str):
    pass


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = get_tokenizer_from_args(args)
    fill_structure_type(args)

    input_requests = sample_requests(tokenizer, args)
    selected_percentile_metrics = []

    outputs, benchmark_duration = benchmark_sync(args=args,
                                                 tokenizer=tokenizer,
                                                 input_requests=input_requests)

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=_to_request_output(outputs),
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=[],
        goodput_config_dict=None,
    )

    print_metrics_to_console(metrics, benchmark_duration,
                             selected_percentile_metrics)

    score = evaluate([{
        'generated': output.outputs[0].text,
        'expected': None
    } for output in outputs], args)
    print("correct_rate(%)", score, '\n')


if __name__ == "__main__":
    parser = get_structured_output_argparser()
    parser.add_argument("--async",
                        action="store_true",
                        help="Run the benchmark in async mode.")
    args = parser.parse_args()
    main(args)
