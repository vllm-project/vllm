# SPDX-License-Identifier: Apache-2.0
r"""Benchmark offline throughput with structured outputs.

Usage:
    python benchmarks/benchmark_offline_structured_output.py \
        --model <your_model>

Run benchmark in async mode:
    python benchmarks/benchmark_offline_structured_output.py \
        --model <your_model> \
        --num-prompts 100 \
        --async-mode

Benchmark EBNF decoding on xgrammar backend in async mode:
    python benchmarks/benchmark_offline_structured_output.py \
        --model <your_model> \
        --structured-output-backend xgrammar \
        --dataset grammar \
        --num-prompts 1000 \
        --async-mode
"""
import argparse
import asyncio
import random
from time import perf_counter

import numpy as np
from backend_request_func import RequestFuncOutput
from benchmark_serving_structured_output import (
    SampleRequest, calculate_metrics, evaluate, fill_structure_type,
    get_structured_output_argparser, get_tokenizer_from_args,
    print_metrics_to_console, sample_requests)
from transformers import PreTrainedTokenizerBase

import vllm
from vllm import LLM, AsyncEngineArgs
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


def _to_vllm_guided_decoding_params(
        args: argparse.Namespace,
        request: SampleRequest) -> GuidedDecodingParams:
    if args.dataset in ["grammar", "regex", "json", "choice"]:
        field_name = args.dataset
    elif "grammar" in args.dataset:
        field_name = "grammar"
    elif "json" in args.dataset:
        field_name = "json"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    kwargs = {
        "backend": args.structured_output_backend,
        field_name: request.schema
    }
    return GuidedDecodingParams(**kwargs)


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
        tokenizer=args.tokenizer if args.tokenizer else args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
        guided_decoding_backend=args.structured_output_backend,
    )
    structured_output_request_indices = set(
        random.sample(range(len(input_requests)),
                      int(len(input_requests) * args.structured_output_ratio)))
    if args.profile:
        llm.start_profile()

    start = perf_counter()
    outputs = llm.generate(
        prompts=[req.prompt for req in input_requests],
        sampling_params=[
            SamplingParams(
                ignore_eos=args.ignore_eos,
                guided_decoding=_to_vllm_guided_decoding_params(args, req)
                if i in structured_output_request_indices else None,
            ) for i, req in enumerate(input_requests)
        ],
        use_tqdm=not args.disable_tqdm)
    end = perf_counter()

    if args.profile:
        llm.stop_profile()
    return outputs, end - start


async def benchmark_async(
    args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest]
) -> tuple[list[RequestFuncOutput], float]:
    """
    Benchmark asynchronous offline vLLM with structured outputs.

    Returns:
        A tuple of (output texts, benchmark duration in seconds).
    """
    engine = vllm.AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            tokenizer=args.tokenizer if args.tokenizer else args.model,
            tokenizer_mode=args.tokenizer_mode,
            trust_remote_code=args.trust_remote_code,
            guided_decoding_backend=args.structured_output_backend,
        ))
    structured_output_request_indices = set(
        random.sample(range(len(input_requests)),
                      int(len(input_requests) * args.structured_output_ratio)))
    if args.profile:
        await engine.start_profile()

    import uuid
    start = perf_counter()
    outputs = []
    streams = []
    if vllm.envs.VLLM_USE_V1:
        for i, req in enumerate(input_requests):
            streams.append(
                engine.generate(
                    request_id=str(uuid.uuid4()),
                    prompt=input_requests[0].prompt,
                    sampling_params=SamplingParams(
                        ignore_eos=args.ignore_eos,
                        guided_decoding=_to_vllm_guided_decoding_params(
                            args, req)
                        if i in structured_output_request_indices else None,
                    )))
    else:
        for i, req in enumerate(input_requests):
            streams.append(await engine.add_request(
                request_id=str(uuid.uuid4()),
                prompt=input_requests[0].prompt,
                params=SamplingParams(
                    ignore_eos=args.ignore_eos,
                    guided_decoding=_to_vllm_guided_decoding_params(args, req)
                    if i in structured_output_request_indices else None)))
    for stream in streams:
        async for output in stream:
            if output.finished:
                outputs.append(output)
    end = perf_counter()

    if args.profile:
        await engine.stop_profile()

    return outputs, end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = get_tokenizer_from_args(args)
    fill_structure_type(args)

    input_requests = sample_requests(tokenizer, args)
    selected_percentile_metrics = []

    if args.async_mode:
        outputs, benchmark_duration = asyncio.run(
            benchmark_async(args=args,
                            tokenizer=tokenizer,
                            input_requests=input_requests))
    else:
        outputs, benchmark_duration = benchmark_sync(
            args=args, tokenizer=tokenizer, input_requests=input_requests)

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
    parser.add_argument("--async-mode",
                        action="store_true",
                        help="Run the benchmark in async mode.")
    args = parser.parse_args()
    main(args)
